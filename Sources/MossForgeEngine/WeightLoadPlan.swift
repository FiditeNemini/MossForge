import Foundation
import MossForgeCore

// MOSS key geometry in `validate`, per-tensor `comfy_quant` ConvRot params, prefix filtering
// (the Qwen3.5 file is a whole VLM) and split→fused qkv. See NOTES.md.

/// How a single logical weight is materialized into a working-precision `MLXArray`.
/// The planner picks this from header descriptors alone; the MLX backend executes it on-device.
public enum DequantOp: Sendable, Equatable {
    /// Cast the raw tensor straight to the working dtype (bf16 / f16 / f32 / norms / biases).
    case passthrough(rawKey: String)
    /// fp8-scaled: `w_f8.asWorking * scale`, where `scale` is a per-tensor F32 scalar. An empty
    /// `scaleKey` is a raw fp8 cast with an implicit scale of 1.0.
    case fp8Scaled(weightKey: String, scaleKey: String)
    /// nvfp4: `unpack4bit(w_u8) * blockScale(e4m3) * globalScale(f32)`, scales applied
    /// over contiguous runs of `blockSize` along the (unpacked) inner dim.
    case nvfp4(weightKey: String, blockScaleKey: String, globalScaleKey: String, blockSize: Int)
    /// INT8 ConvRot (comfy-kitchen W8A8): dequant the per-output-channel int8 weight
    /// (`w_i8.asWorking * scale`, `scale` shape `[out,1]`), then — when `groupSize > 0` —
    /// undo the folded group-wise regular-Hadamard rotation to return to the original basis:
    /// `W = W_rot @ H` block-diagonally over `groupSize` columns (H orthogonal ⇒ `Hᵀ⁻¹ = H`).
    /// `groupSize == 0` is a plain per-channel int8 weight with no rotation.
    case int8ConvRot(weightKey: String, scaleKey: String, groupSize: Int)
    /// INT4 ConvRot (`convrot_w4a4`): the `I8` bytes hold two signed 4-bit codes each, so the
    /// logical inner dim is twice the on-disk width. Unpack, scale per row, un-rotate as INT8.
    case int4ConvRot(weightKey: String, scaleKey: String, groupSize: Int)
}

/// One materialized weight: its native module key (on-disk prefix stripped) plus the
/// op + logical (post-dequant) shape.
public struct WeightLoadEntry: Sendable, Equatable {
    public var key: String
    public var dequant: DequantOp
    /// Logical shape after dequant (nvfp4 / int4 restore the packed inner dim).
    public var shape: [Int]

    public init(key: String, dequant: DequantOp, shape: [Int]) {
        self.key = key
        self.dequant = dequant
        self.shape = shape
    }
}

/// Several materialized entries concatenated along axis 0 into one logical weight, the parts
/// then dropped — how a split `proj.{q,k,v}` file reaches the canonical fused `proj.qkv`.
public struct WeightFusion: Sendable, Equatable {
    public var key: String
    public var parts: [String]

    public init(key: String, parts: [String]) {
        self.key = key
        self.parts = parts
    }
}

/// The full plan for one safetensors file: what to materialize, which raw tensors were folded in
/// as quant auxiliaries (scales / ComfyUI blobs) rather than emitted, and which entries fuse.
public struct WeightLoadPlan: Sendable, Equatable {
    public var entries: [WeightLoadEntry]
    public var consumedAuxiliaries: Set<String>
    public var fusions: [WeightFusion]

    public init(entries: [WeightLoadEntry], consumedAuxiliaries: Set<String>, fusions: [WeightFusion] = []) {
        self.entries = entries
        self.consumedAuxiliaries = consumedAuxiliaries
        self.fusions = fusions
    }

    /// Logical shape of a key after load — a fused key's rows are the sum of its parts'.
    public func shape(of key: String) -> [Int]? {
        if let fusion = fusions.first(where: { $0.key == key }) {
            let parts = fusion.parts.compactMap { shape(of: $0) }
            guard parts.count == fusion.parts.count, let first = parts.first else { return nil }
            return [parts.reduce(0) { $0 + $1[0] }] + first.dropFirst()
        }
        return entries.first { $0.key == key }?.shape
    }
}

/// Builds a `WeightLoadPlan` from header descriptors — the pure-logic half of the weight loader.
/// Per-tensor detection (no whole-file precision assumption) so a mixed checkpoint still plans.
public enum MossWeightPlan {
    /// Auxiliary suffixes that are never standalone weights. `.comfy_attention.config` is the
    /// Comfy int8 DiT's per-block runtime hint (`{"attention": "comfy_kitchen_int8"}`).
    private static let auxSuffixes = [".weight_scale", ".weight_scale_2", ".comfy_quant", ".comfy_attention.config"]

    /// Plan a file straight from disk: header descriptors, `_quantization_metadata` **and** the
    /// per-tensor `comfy_quant` blobs (the only ConvRot source in the Comfy-Org MOSS files).
    public static func build(url: URL, prefix: String = "") throws -> WeightLoadPlan {
        let reader = try SafetensorsReader(url: url)
        defer { reader.close() }
        var blobs: [String: Data] = [:]
        for d in reader.descriptors where d.name.hasSuffix(".comfy_quant") && d.name.hasPrefix(prefix) {
            blobs[d.name] = try reader.bytes(d.name)
        }
        let (_, metadata) = try SafetensorsHeader.readDescriptors(from: url)
        let convRot = ConvRotMetadata.layers(from: metadata)
            .merging(ComfyQuantBlob.convRotLayers(blobs: blobs)) { meta, _ in meta }
        return build(descriptors: reader.descriptors, prefix: prefix, convRot: convRot)
    }

    /// - Parameters:
    ///   - prefix: only tensors under this prefix are planned, and it is stripped from their keys
    ///     (`""` = the whole file).
    ///   - convRot: layer key (prefixed or stripped, `.weight` dropped) → ConvRot decode params.
    ///     An I8 weight listed here is un-rotated on load (nibble-unpacked first when `isInt4`);
    ///     other I8 weights are plain per-channel int8.
    public static func build(
        descriptors: [TensorDescriptor], prefix: String,
        convRot: [String: ConvRotMetadata.ConvRotLayer] = [:]
    ) -> WeightLoadPlan {
        let scoped = prefix.isEmpty ? descriptors : descriptors.filter { $0.name.hasPrefix(prefix) }
        let byName = Dictionary(scoped.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        var entries: [WeightLoadEntry] = []
        var consumed: Set<String> = []

        for d in scoped {
            if auxSuffixes.contains(where: { d.name.hasSuffix($0) }) {
                // Blobs/scales are consumed by their parent weight; orphan runtime hints too.
                if d.name.hasSuffix(".comfy_attention.config") { consumed.insert(d.name) }
                continue
            }
            let key = stripPrefix(d.name, prefix)

            guard d.name.hasSuffix(".weight") else {
                entries.append(.init(key: key, dequant: .passthrough(rawKey: d.name), shape: d.shape))
                continue
            }

            let scaleKey = d.name + "_scale"
            let scale2Key = d.name + "_scale_2"
            let comfyKey = String(d.name.dropLast(".weight".count)) + ".comfy_quant"
            func consumeComfy() { if byName[comfyKey] != nil { consumed.insert(comfyKey) } }

            // nvfp4: U8-packed weight + per-block e4m3 scale + global f32 scale.
            if d.dtype == TensorDType.uint8.rawValue, byName[scale2Key] != nil, byName[scaleKey] != nil {
                consumed.formUnion([scaleKey, scale2Key])
                consumeComfy()
                entries.append(
                    .init(
                        key: key,
                        dequant: .nvfp4(
                            weightKey: d.name, blockScaleKey: scaleKey, globalScaleKey: scale2Key, blockSize: 16),
                        shape: unpackedShape(d.shape)))
                continue
            }

            // fp8-scaled (E4M3 or E5M2; the materializer picks the LUT from the on-disk dtype).
            if isFP8(d.dtype), byName[scaleKey] != nil {
                consumed.insert(scaleKey)
                consumeComfy()
                entries.append(.init(key: key, dequant: .fp8Scaled(weightKey: d.name, scaleKey: scaleKey), shape: d.shape))
                continue
            }

            // INT4/INT8 ConvRot: I8 weight + per-output-row F32 `weight_scale` `[out,1]`.
            if d.dtype == TensorDType.int8.rawValue, byName[scaleKey] != nil {
                consumed.insert(scaleKey)
                consumeComfy()
                let layer = convRotLayer(convRot, weightName: d.name, prefix: prefix)
                if let layer, layer.isInt4 {
                    entries.append(
                        .init(
                            key: key,
                            dequant: .int4ConvRot(weightKey: d.name, scaleKey: scaleKey, groupSize: layer.groupSize),
                            shape: unpackedShape(d.shape)))
                } else {
                    entries.append(
                        .init(
                            key: key,
                            dequant: .int8ConvRot(weightKey: d.name, scaleKey: scaleKey, groupSize: layer?.groupSize ?? 0),
                            shape: d.shape))
                }
                continue
            }

            // Raw fp8 (no per-tensor scale): ComfyUI casts with an implicit 1.0.
            if isFP8(d.dtype) {
                entries.append(.init(key: key, dequant: .fp8Scaled(weightKey: d.name, scaleKey: ""), shape: d.shape))
                continue
            }

            entries.append(.init(key: key, dequant: .passthrough(rawKey: d.name), shape: d.shape))
        }

        return WeightLoadPlan(entries: entries, consumedAuxiliaries: consumed, fusions: qkvFusions(entries))
    }

    /// Split `layers.N.proj.{q,k,v}.weight` triples → one fused `layers.N.proj.qkv.weight`
    /// (rows q|k|v, matching `MossDiTConfig.qkvRowRanges`). Incomplete triples are left alone.
    private static func qkvFusions(_ entries: [WeightLoadEntry]) -> [WeightFusion] {
        let keys = Set(entries.map(\.key))
        var fusions: [(Int, WeightFusion)] = []
        for key in keys where key.hasSuffix(".proj.q.weight") {
            guard let (block, _) = blockLinear(key), let index = Int(block.dropFirst("layers.".count)) else { continue }
            let parts = ["q", "k", "v"].map { "\(block).proj.\($0).weight" }
            guard parts.allSatisfy(keys.contains), !keys.contains("\(block).proj.qkv.weight") else { continue }
            fusions.append((index, WeightFusion(key: "\(block).proj.qkv.weight", parts: parts)))
        }
        return fusions.sorted { $0.0 < $1.0 }.map(\.1)
    }

    /// Look one weight's ConvRot entry up under either key convention: prefix-stripped
    /// (our exports) or full (comfy-kitchen / Forge metadata, Comfy-Org blobs).
    private static func convRotLayer(
        _ convRot: [String: ConvRotMetadata.ConvRotLayer], weightName: String, prefix: String
    ) -> ConvRotMetadata.ConvRotLayer? {
        guard weightName.hasSuffix(".weight") else { return nil }
        let raw = String(weightName.dropLast(".weight".count))
        return convRot[stripPrefix(raw, prefix)] ?? convRot[raw]
    }

    private static func isFP8(_ dtype: String) -> Bool {
        dtype == TensorDType.float8E4M3.rawValue || dtype == TensorDType.float8E5M2.rawValue
    }

    private static func stripPrefix(_ name: String, _ prefix: String) -> String {
        guard !prefix.isEmpty, name.hasPrefix(prefix) else { return name }
        return String(name.dropFirst(prefix.count))
    }

    /// The innermost dim of a 4-bit-packed weight holds two values per byte.
    private static func unpackedShape(_ shape: [Int]) -> [Int] {
        guard let last = shape.last else { return shape }
        return shape.dropLast() + [last * 2]
    }

    // MARK: - Validation

    /// Block linears fed directly by the (modulated) hidden stream: inner dim == hidden size.
    private static let hiddenInputLinears: Set<String> = ["proj.qkv", "proj.q", "proj.k", "proj.v", "proj.ffn_up_gate"]

    /// Check a plan's DiT block geometry for self-consistency, throwing rather than letting a bad
    /// shape reach the GPU. MLX reports matmul shape errors through a handler that `fatalError`s,
    /// so a misdetected quant format (packed INT4 read as INT8 keeps half the width) would
    /// otherwise crash in the first projection. Plans with no DiT blocks (VAE, encoder) pass.
    public static func validate(_ plan: WeightLoadPlan) throws {
        var shapes: [String: [Int]] = [:]
        for entry in plan.entries { shapes[entry.key] = entry.shape }
        guard let hidden = shapes["layers.0.norm1.weight"]?.first else { return }

        for key in shapes.keys.sorted() {
            guard let shape = shapes[key], shape.count == 2, let (block, suffix) = blockLinear(key) else { continue }
            let expected: Int?
            if hiddenInputLinears.contains(suffix) {
                expected = hidden
            } else if suffix == "proj.o" {
                expected = shapes["\(block).proj.qkv.weight"].map { $0[0] / 3 } ?? shapes["\(block).proj.q.weight"]?.first
            } else if suffix == "proj.ffn_down" {
                expected = shapes["\(block).proj.ffn_up_gate.weight"].map { $0[0] / 2 }
            } else {
                continue
            }
            guard let want = expected, shape[1] != want else { continue }
            throw EngineError.backendFailure(
                "checkpoint layer \(key) has inner dim \(shape[1]), expected \(want). "
                    + "This usually means the quantization format was misdetected — most often a "
                    + "packed INT4 ConvRot weight read as INT8, which keeps half the true width.")
        }
    }

    /// `layers.12.proj.qkv.weight` → `("layers.12", "proj.qkv")`; `nil` for anything else.
    private static func blockLinear(_ key: String) -> (block: String, suffix: String)? {
        guard key.hasPrefix("layers."), key.hasSuffix(".weight") else { return nil }
        let body = key.dropFirst("layers.".count).dropLast(".weight".count)
        guard let dot = body.firstIndex(of: "."), Int(body[body.startIndex..<dot]) != nil else { return nil }
        return ("layers." + body[body.startIndex..<dot], String(body[body.index(after: dot)...]))
    }
}
