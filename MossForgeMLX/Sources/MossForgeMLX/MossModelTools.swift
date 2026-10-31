// Quant encoders: e4m3/e5m2/e2m1/nvfp4 encoders and the int8/int4 ConvRot fold. MOSS changes: canonical bare keys
// (fused qkv), the block-linear set from `MossExport`, per-tensor `comfy_quant` blobs (the Comfy-Org
// MOSS layout) instead of `_quantization_metadata`, and fp32 Turbo overrides. See NOTES.md.

import Foundation
import MLX
import MLXRandom
import MossForgeCore
import MossForgeEngine

/// Model tooling on the GPU: convert / quantize a DiT, fuse LoRAs into it (fp32 maths — the only way
/// a distillation LoRA survives), and merge DiTs. Output follows `MossExport`'s rules, so MossForge,
/// ComfyUI and Forge all read it.
public enum MossModelTools {
    public struct Result: Sendable {
        public var output: URL
        public var bytes: Int
        public var seconds: Double
    }

    // MARK: operations

    /// Re-encode a DiT at `precision`.
    @discardableResult
    public static func convert(
        input: URL, precision: WeightPrecision, output: URL, progress: ProgressHandler? = nil
    ) throws -> Result {
        let t0 = Date()
        progress?(GenerationProgress(step: 0, totalSteps: 2, message: "loading \(input.lastPathComponent)"))
        let weights = try withBoundedCache { try WeightLoader.load(url: input, prefix: MossDiTConfig.keyPrefix(of: input), working: .float32).arrays }
        progress?(GenerationProgress(step: 1, totalSteps: 2, message: "encoding \(precision.rawValue)"))
        let bytes = try write(weights, precision: precision, to: output, op: "convert")
        return Result(output: output, bytes: bytes, seconds: Date().timeIntervalSince(t0))
    }

    /// Fuse LoRAs (in order) into a DiT in fp32 and write it at `precision`. `overrides` are fp32
    /// replacement tensors (lodestones' `time_embedding.safetensors`) applied first and kept fp32.
    /// The source must be bf16 / fp16 / fp32 (`MossExport.checkFuseBase`).
    @discardableResult
    public static func fuse(
        input: URL, loras: [LoRASpec], overrides: URL? = nil, precision: WeightPrecision, output: URL,
        progress: ProgressHandler? = nil
    ) throws -> Result {
        let t0 = Date()
        if case .dit(_, let p) = try MossInspector.inspect(input) { try MossExport.checkFuseBase(p) }
        let steps = loras.count + 2
        progress?(GenerationProgress(step: 0, totalSteps: steps, message: "loading \(input.lastPathComponent)"))
        let bytes: Int = try withBoundedCache {
            var w = try WeightLoader.load(url: input, prefix: MossDiTConfig.keyPrefix(of: input), working: .float32).arrays
            if let overrides { try override(&w, from: overrides) }
            let config = try canonicalConfig(input)
            for (i, spec) in loras.enumerated() {
                progress?(GenerationProgress(step: i + 1, totalSteps: steps, message: "fusing \(spec.url.lastPathComponent)"))
                try fuseLoRA(spec, into: &w, config: config)
            }
            progress?(GenerationProgress(step: steps - 1, totalSteps: steps, message: "encoding \(precision.rawValue)"))
            return try write(w, precision: precision, to: output, op: "fuse", fp32: overrides != nil)
        }
        return Result(output: output, bytes: bytes, seconds: Date().timeIntervalSince(t0))
    }

    /// Merge DiTs: `models` = [A, B] (weighted sum) or [A, B, C] (add difference), then optional LoRAs.
    @discardableResult
    public static func merge(
        models: [URL], mode: ModelMergeMode, alpha: Double, loras: [LoRASpec] = [], precision: WeightPrecision, output: URL,
        progress: ProgressHandler? = nil
    ) throws -> Result {
        let t0 = Date()
        guard models.count == (mode == .weightedSum ? 2 : 3) else {
            throw EngineError.invalidRequest(mode == .weightedSum ? "weighted sum needs models A and B" : "add difference needs A, B and C")
        }
        let total = models.count + loras.count + 1
        let bytes: Int = try withBoundedCache {
            func load(_ i: Int) throws -> [String: MLXArray] {
                progress?(GenerationProgress(step: i, totalSteps: total, message: "loading \(models[i].lastPathComponent)"))
                return try WeightLoader.load(url: models[i], prefix: MossDiTConfig.keyPrefix(of: models[i])).arrays
            }
            var a = try load(0)
            var b = try load(1)
            var c = mode == .addDifference ? try load(2) : nil
            let s = Float(alpha)
            var acc: [String: MLXArray] = [:]
            var batch: [MLXArray] = []
            for key in a.keys.sorted() {
                let av = a.removeValue(forKey: key)!.asType(.float32)
                let bv = (b.removeValue(forKey: key) ?? av).asType(.float32)
                let r: MLXArray
                switch mode {
                case .weightedSum: r = av * (1 - s) + bv * s
                case .addDifference: r = av + (bv - (c?.removeValue(forKey: key) ?? av).asType(.float32)) * s
                }
                acc[key] = r
                batch.append(r)
                if batch.count >= 24 { eval(batch); batch.removeAll(keepingCapacity: true) }
            }
            eval(batch)
            let config = try canonicalConfig(models[0])
            for (i, spec) in loras.enumerated() {
                progress?(GenerationProgress(step: models.count + i, totalSteps: total, message: "fusing \(spec.url.lastPathComponent)"))
                try fuseLoRA(spec, into: &acc, config: config)
            }
            progress?(GenerationProgress(step: total - 1, totalSteps: total, message: "encoding \(precision.rawValue)"))
            return try write(acc, precision: precision, to: output, op: "merge")
        }
        return Result(output: output, bytes: bytes, seconds: Date().timeIntervalSince(t0))
    }

    // MARK: LoRA fusion

    /// `current + delta`, or `delta` added to the `rows` slice (a q|k|v part of the fused qkv).
    static func adding(_ delta: MLXArray, to current: MLXArray, rows: Range<Int>?) -> MLXArray {
        guard let rows else { return current + delta.reshaped(current.shape) }
        let n = current.shape[0]
        var pieces: [MLXArray] = []
        if rows.lowerBound > 0 { pieces.append(current[0..<rows.lowerBound]) }
        pieces.append(current[rows.lowerBound..<rows.upperBound] + delta)
        if rows.upperBound < n { pieces.append(current[rows.upperBound..<n]) }
        return concatenated(pieces, axis: 0)
    }

    /// `W[target][rows] += scale·strength·(up @ down)` (and `.diff` / `.diff_b` added), fp32 — the
    /// same plan `MossDiT.attachLoRA` applies unmerged at inference.
    static func fuseLoRA(_ spec: LoRASpec, into w: inout [String: MLXArray], config: MossDiTConfig) throws {
        let plan = try MossLoRAKeyMap.plan(url: spec.url, config: config, targetShapes: w.mapValues(\.shape))
        guard plan.unmapped.isEmpty else { throw EngineError.unmappedKeys(Array(plan.unmapped.prefix(8))) }
        let source = try TensorSource(url: spec.url)
        defer { source.close() }
        func read(_ name: String) throws -> MLXArray {
            try WeightLoader.materialize(.init(key: name, dequant: .passthrough(rawKey: name), shape: []), from: source).asType(.float32)
        }
        var touched: [MLXArray] = []
        for part in plan.parts {
            let strength = Float(spec.strength(forBlock: part.block))
            guard strength != 0, let current = w[part.target]?.asType(.float32) else { continue }
            let delta: MLXArray
            switch part.kind {
            case .lowRank: delta = matmul(try read(part.up), try read(part.down)) * (part.scale * strength)
            case .kronecker:
                // Dense w1 ⊗ w2 (fp32, one module at a time).
                let (w1, w2) = (try read(part.down), try read(part.up))
                let (a, b, c, d) = (w1.dim(0), w1.dim(1), w2.dim(0), w2.dim(1))
                delta = (w1.reshaped([a, 1, b, 1]) * w2.reshaped([1, c, 1, d])).reshaped([a * c, b * d]) * (part.scale * strength)
            case .weightDiff, .biasDiff: delta = try read(part.down) * strength
            case .hadamard:
                delta = matmul(try read(part.up), try read(part.down)) * matmul(try read(part.up2), try read(part.down2))
                    * (part.scale * strength)
            }
            let updated = adding(delta, to: current, rows: part.rows)
            w[part.target] = updated
            touched.append(updated)
            if touched.count >= 16 { eval(touched); touched.removeAll(keepingCapacity: true) }
        }
        eval(touched)
    }

    static func override(_ w: inout [String: MLXArray], from url: URL) throws {
        for (key, value) in try MLX.loadArrays(url: url) {
            guard let current = w[key], current.shape == value.shape else {
                throw EngineError.invalidRequest("override \(key) \(value.shape) doesn't match the DiT")
            }
            w[key] = value.asType(.float32)
        }
    }

    static func canonicalConfig(_ url: URL) throws -> MossDiTConfig {
        var config = try MossDiTConfig(descriptors: SafetensorsHeader.readDescriptors(from: url).tensors)
        config.qkvLayout = .fused
        return config
    }

    /// A DiT load/merge churns tens of GB of transient arrays; bound MLX's buffer cache meanwhile.
    static func withBoundedCache<T>(_ body: () throws -> T) rethrows -> T {
        let prior = GPU.cacheLimit
        GPU.set(cacheLimit: 2 << 30)
        defer {
            GPU.set(cacheLimit: prior)
            GPU.clearCache()
        }
        return try body()
    }

    // MARK: writing

    /// Encode every tensor and write the file. `fp32` keeps the Turbo overrides fp32.
    @discardableResult
    static func write(
        _ weights: [String: MLXArray], precision: WeightPrecision, to url: URL, op: String, fp32: Bool = false
    ) throws -> Int {
        var w = weights
        var tensors: [SafetensorsWriter.Tensor] = []
        tensors.reserveCapacity(w.count * 2)
        for key in w.keys.sorted() {
            guard let value = w.removeValue(forKey: key) else { continue }
            tensors.append(contentsOf: try encode(key, value, precision: precision, keepFP32: fp32 && MossExport.keepsFP32(key)))
        }
        if precision == .int8ConvRot {
            // The Comfy-Org files carry this runtime hint per block (ComfyUI's int8 attention); harmless elsewhere.
            let blocks = Set(tensors.compactMap { t -> Int? in
                let p = t.name.split(separator: ".")
                return p.count > 1 && p[0] == "layers" ? Int(p[1]) : nil
            })
            for b in blocks.sorted() {
                let blob = Data(#"{"attention": "comfy_kitchen_int8"}"#.utf8)
                tensors.append(.init(name: "layers.\(b).comfy_attention.config", dtype: .uint8, shape: [blob.count], data: blob))
            }
        }
        return try SafetensorsWriter.write(tensors, metadata: ["produced_by": "MossForge", "op": op, "precision": precision.rawValue], to: url)
    }

    /// One canonical tensor → its output tensor(s).
    static func encode(_ key: String, _ w: MLXArray, precision: WeightPrecision, keepFP32: Bool = false) throws -> [SafetensorsWriter.Tensor] {
        let shape = w.shape
        if keepFP32 { return [bytes(key, w.asType(.float32), .float32, shape)] }
        guard MossExport.quantizes(key, shape: shape, as: precision) else {
            switch precision {
            case .float32: return [bytes(key, w.asType(.float32), .float32, shape)]
            case .float16: return [bytes(key, w.asType(.float16), .float16, shape)]
            default: return [bytes(key, w.asType(.bfloat16), .bfloat16, shape)]
            }
        }
        let layer = String(key.dropLast(".weight".count))
        let blob = MossExport.comfyQuantBlob(for: precision)!
        let marker = SafetensorsWriter.Tensor(name: layer + ".comfy_quant", dtype: .uint8, shape: [blob.count], data: blob)
        let wf = w.asType(.float32)
        switch precision {
        case .float8:
            let scale = Swift.max(wf.abs().max().item(Float.self) / 448.0, .leastNormalMagnitude)
            return [bytes(key, e4m3Encode(wf / scale), .float8E4M3, shape), scalar(key + "_scale", scale), marker]
        case .float8E5M2:
            let scale = Swift.max(wf.abs().max().item(Float.self) / 57344.0, .leastNormalMagnitude)
            return [bytes(key, e5m2Encode(wf / scale), .float8E5M2, shape), scalar(key + "_scale", scale), marker]
        case .nvfp4:
            let (packed, blockScale, global) = encodeNVFP4(wf)
            return [
                bytes(key, packed, .uint8, [shape[0], shape[1] / 2]),
                bytes(key + "_scale", blockScale, .float8E4M3, [shape[0], shape[1] / 16]),
                scalar(key + "_scale_2", global), marker,
            ]
        case .int8ConvRot, .int4ConvRot:
            // Fold the group-wise regular-Hadamard rotation (W_rot = W @ H, H a symmetric involution),
            // then quantize per output row. The reader dequants and un-rotates with the same H.
            let g = MossExport.convRotGroupSize
            let wRot = matmul(wf.reshaped([shape[0], shape[1] / g, g]), WeightLoader.regularHadamard(g)).reshaped(shape)
            let amax = wRot.abs().max(axis: 1, keepDims: true)
            let int4 = precision == .int4ConvRot
            let scale = clip(amax / (int4 ? 7.0 : 127.0), min: MLXArray(Float.leastNormalMagnitude), max: MLXArray(Float.greatestFiniteMagnitude))
            let scaleT = bytes(key + "_scale", scale.asType(.float32), .float32, [shape[0], 1])
            if !int4 {
                let q = clip((wRot / scale).round(), min: -127, max: 127).asType(.int8)
                return [bytes(key, q, .int8, shape), scaleT, marker]
            }
            let q = clip((wRot / scale).round(), min: -8, max: 7).asType(.int32)
            let code = which(q .< 0, q + 16, q)  // signed nibble → 0…15
            let packed = (code.reshaped([shape[0], shape[1] / 2, 2]) * MLXArray([Int32(1), Int32(16)])).sum(axis: 2).asType(.uint8)
            return [bytes(key, packed, .int8, [shape[0], shape[1] / 2]), scaleT, marker]
        default:
            throw EngineError.invalidRequest("can't write \(precision.rawValue)")
        }
    }

    /// Encode a random block linear at each precision, write it, load it back through the real
    /// planner + loader and report the correlation (fp8 ≈ 0.999, int8·cr ≈ 0.9999, nvfp4 ≈ 0.99,
    /// int4·cr ≈ 0.99).
    public static func selfTest() -> [(WeightPrecision, Float)] {
        let key = "layers.0.proj.qkv.weight"
        let w = MLXRandom.normal([256, 512]) * 0.05
        eval(w)
        return MossExport.precisions.map { p in
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("mf-rt-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: url) }
            guard (try? write([key: w], precision: p, to: url, op: "selftest")) != nil,
                let back = (try? WeightLoader.load(url: url))?.arrays[key]
            else { return (p, .nan) }
            return (p, corr(w, back.asType(.float32)))
        }
    }

    static func corr(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
        let xc = x - x.mean(), yc = y - y.mean()
        return ((xc * yc).sum() / ((xc * xc).sum() * (yc * yc).sum()).sqrt()).item(Float.self)
    }

    // MARK: GPU quant encoders (inverses of the loader's decoders)

    static func e4m3Encode(_ v: MLXArray) -> MLXArray {
        let neg = v .< 0
        let a = clip(v.abs(), min: 0, max: 448)
        let isZero = a .<= 0
        let sub = a .< Float(0.015625)
        let e = clip(a.log2().floor(), min: -6, max: 8)
        let twoE = MLXArray(Float(2)).pow(e)
        var m = ((a / twoE - 1) * 8).round()
        let carry = m .>= 8
        let eC = e + which(carry, MLXArray(Float(1)), MLXArray(Float(0)))
        m = which(carry, MLXArray(Float(0)), m)
        let field = clip(eC + 7, min: 1, max: 15)
        let byteNorm = clip(field * 8 + m, min: 8, max: 126)
        let mSub = clip((a * 512).round(), min: 0, max: 8)
        let mag = which(isZero, MLXArray(Float(0)), which(sub, mSub, byteNorm))
        return (which(neg, MLXArray(Float(128)), MLXArray(Float(0))) + mag).asType(.uint8)
    }

    static func e5m2Encode(_ v: MLXArray) -> MLXArray {
        let neg = v .< 0
        let a = clip(v.abs(), min: 0, max: 57344)
        let isZero = a .<= 0
        let sub = a .< Float(0.00006103515625)
        let e = clip(a.log2().floor(), min: -14, max: 15)
        let twoE = MLXArray(Float(2)).pow(e)
        var m = ((a / twoE - 1) * 4).round()
        let carry = m .>= 4
        let eC = e + which(carry, MLXArray(Float(1)), MLXArray(Float(0)))
        m = which(carry, MLXArray(Float(0)), m)
        let field = clip(eC + 15, min: 1, max: 30)
        let byteNorm = clip(field * 4 + m, min: 4, max: 123)
        let mSub = clip((a * 65536).round(), min: 0, max: 4)
        let mag = which(isZero, MLXArray(Float(0)), which(sub, mSub, byteNorm))
        return (which(neg, MLXArray(Float(128)), MLXArray(Float(0))) + mag).asType(.uint8)
    }

    static func e2m1Encode(_ x: MLXArray) -> MLXArray {
        let a = x.abs()
        let mag =
            which(a .< 0.25, MLXArray(Int32(0)),
                which(a .< 0.75, MLXArray(Int32(1)),
                    which(a .< 1.25, MLXArray(Int32(2)),
                        which(a .< 1.75, MLXArray(Int32(3)),
                            which(a .< 2.5, MLXArray(Int32(4)),
                                which(a .< 3.5, MLXArray(Int32(5)),
                                    which(a .< 5.0, MLXArray(Int32(6)), MLXArray(Int32(7)))))))))
        return (which(x .< 0, MLXArray(Int32(8)), MLXArray(Int32(0))) + mag).asType(.uint8)
    }

    static func e4m3LUT() -> MLXArray { MLXArray((0..<256).map { Float8E4M3.decode(UInt8($0)) }) }

    /// Packed fp4 codes (high-nibble-first) + cutlass-128×4-swizzled E4M3 block scales + f32 global.
    static func encodeNVFP4(_ w: MLXArray) -> (packed: MLXArray, blockScale: MLXArray, global: Float) {
        let out = w.shape[0], inn = w.shape[1]
        let global = Swift.max(w.abs().max().item(Float.self) / 2688.0, .leastNormalMagnitude)  // 6 · 448
        let blocks = w.reshaped([out, inn / 16, 16])
        let blockScaleCodes = e4m3Encode(blocks.abs().max(axis: 2) / (6 * global))
        let blockScaleF = e4m3LUT().take(blockScaleCodes.asType(.int32), axis: 0)
        let denom = broadcast(blockScaleF.reshaped([out, inn / 16, 1]), to: [out, inn / 16, 16]).reshaped([out, inn]) * global
        let fp4 = e2m1Encode(w / denom)
        let packed = (fp4.reshaped([out, inn / 2, 2]).asType(.int32) * MLXArray([Int32(16), Int32(1)])).sum(axis: 2).asType(.uint8)
        let sw = blockScaleCodes.reshaped([out / 128, 4, 32, inn / 64, 4]).transposed(0, 3, 2, 1, 4).reshaped([out, inn / 16])
        return (packed, sw, global)
    }

    static func bytes(_ key: String, _ a: MLXArray, _ dt: TensorDType, _ shape: [Int]) -> SafetensorsWriter.Tensor {
        eval(a)
        return SafetensorsWriter.Tensor(name: key, dtype: dt, shape: shape, data: a.asData().data)
    }

    static func scalar(_ key: String, _ v: Float) -> SafetensorsWriter.Tensor {
        var f = v
        return SafetensorsWriter.Tensor(name: key, dtype: .float32, shape: [], data: Data(bytes: &f, count: 4))
    }
}
