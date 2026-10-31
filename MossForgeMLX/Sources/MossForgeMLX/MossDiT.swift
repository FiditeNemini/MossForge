import Foundation
import MLX
import MLXFast
import MLXNN
import MossForgeCore
import MossForgeEngine

/// The MOSS-Image DiT (5.08B single-stream), ported from `comfyui/moss-image/model.py`
/// (Apache-2.0; transcribed in `Scripts/dit_ref.py`):
///
/// ```
/// t    = MLP(sinusoid(1000σ))                       time_embedding.{0,2} (SiLU between)
/// mod  = shared_block_modulation(t) → 6×dim         ONE modulation for all blocks, no SiLU
/// head = video_head_modulation(t)   → (shift, scale)
/// x    = [video_embedder(latent tokens) | TextBlock×2(text_embedder.proj(context))]   image first
/// 42 × block:  x += gate_a · o(SDPA(rope(qnorm(q)), rope(knorm(k)), v))   q|k|v from fused qkv on
///                    RMSNorm(x)·(1+scale_a)+shift_a
///              x += gate_f · ffn_down(SiTU(hidden, gate))  [hidden | gate] = ffn_up_gate(…f)
/// v    = final_linear(RMSNorm(x_img)·(1+scale)+shift) → [1,128,H,W]
/// ```
///
/// RMSNorms here use plain `w` (eps 1e-6). The residual stream, norms, modulation, time MLP and
/// RoPE stay F32; block/adapter matmuls run in the working dtype. Weights are the canonical fused
/// `layers.N.proj.qkv` layout (`WeightLoader` fuses split files).
public final class MossDiT {
    public let config: MossDiTConfig
    public let dtype: DType
    private var w: [String: MLXArray]
    /// Runtime (unmerged) LoRA updates per canonical weight key.
    private var loras: [String: [LoRAUpdate]] = [:]
    /// Originals of tensors modified by `.diff` / `.diff_b` LoRA parts.
    private var baseParams: [String: MLXArray] = [:]
    /// LoRA training: adapters being optimised, keyed by canonical weight (`layers.N.proj.qkv.weight`).
    /// The trainer sets them from the parameters `valueAndGrad` differentiates, inside the loss
    /// closure, so gradients reach `down`/`up` and the frozen weights stay constants.
    public var trainable: [String: TrainableLoRA] = [:]
    /// Training: recompute each block in the backward pass instead of keeping its activations
    /// (gradient checkpointing — less memory, ~30% more time). Only used while `trainable` is set.
    public var checkpointBlocks = false
    /// Edit LoRAs (community method): clean reference latents `[1,128,h,w]` on RoPE r = 1, 2, …,
    /// conditioned at timestep 0. Empty = plain text-to-image. How they join the sequence is `editLayout`.
    public var references: [MLXArray] = [] { didSet { referenceKV = nil } }
    /// Joint layouts append the references after each branch's text; isolated ones run them once on
    /// their own and append every block's reference keys/values to each branch's attention.
    public var editLayout: MossEditLayout = .mossForge { didSet { referenceKV = nil } }
    /// Isolated references' post-RoPE keys and values per block — they depend only on the references
    /// and the weights, so one pass serves every step (dropped when either changes).
    private var referenceKV: [(k: MLXArray, v: MLXArray)]?

    public struct TrainableLoRA {
        public var down: MLXArray  // [rank, in] F32 — LoKr: w1 [a, b]
        public var up: MLXArray  // [out, rank] F32 — LoKr: w2 [c, d]
        public var scale: Float
        public var kronecker: Bool
        public init(down: MLXArray, up: MLXArray, scale: Float, kronecker: Bool = false) {
            self.down = down
            self.up = up
            self.scale = scale
            self.kronecker = kronecker
        }
    }

    struct LoRAUpdate {
        var down: MLXArray  // [rank, in] F32 — LoKr: w1
        var up: MLXArray  // [out-slice, rank] F32 — LoKr: w2
        var scale: Float
        var rows: Range<Int>?
        var kronecker = false
    }

    /// `x · (w1 ⊗ w2)ᵀ` without forming the Kronecker product: with `x` as `[N, b, d]`,
    /// `y[n, i·c+k] = Σ_{j,l} w1[i,j] · x[n,j,l] · w2[k,l]` — two small matmuls.
    public static func kroneckerApply(_ x: MLXArray, w1: MLXArray, w2: MLXArray) -> MLXArray {
        let (a, b, c, d) = (w1.dim(0), w1.dim(1), w2.dim(0), w2.dim(1))
        let n = x.dim(0)
        let t = matmul(x.reshaped([n, b, d]), w2.T)  // [N, b, c]
        return matmul(w1, t).reshaped([n, a * c])  // [a,b]·[N,b,c] → [N, a, c]
    }
    private let fused: FusedOps
    /// F32-output twin of `fused`, used by blocks that carry a runtime LoRA (its delta takes F32 input).
    private let fused32: FusedOps

    public init(weights: [String: MLXArray], config: MossDiTConfig, dtype: DType = .float16) {
        self.config = config
        self.dtype = dtype
        var prepared: [String: MLXArray] = [:]
        for (key, value) in weights {
            let f32 = key.hasPrefix("time_embedding") || key.hasSuffix("modulation.weight")
                || key.hasSuffix("modulation.bias") || key.contains("norm") || key.hasSuffix(".bias")
                || key.hasPrefix("rope.")
            prepared[key] = value.asType(f32 ? .float32 : dtype)
        }
        w = prepared
        fused = FusedOps(dtype: dtype, ffnDim: config.ffnDim)
        fused32 = FusedOps(dtype: .float32, ffnDim: config.ffnDim)
    }

    /// Attach a LoRA: low-rank parts **unmerged** (`y = Wx + s·up(down(x))` in F32, so deltas below
    /// bf16 rounding survive — lodestones' DMD); ComfyUI `.diff` / `.diff_b` full deltas and LoHa
    /// products are added onto a saved copy of the target (restored by `detachLoRAs`). Each part's strength is
    /// `spec.strength(forBlock:)` — per-block weights; zero-strength parts are skipped.
    public func attachLoRA(_ spec: LoRASpec) throws {
        let shapes = w.mapValues(\.shape)
        let plan = try MossLoRAKeyMap.plan(url: spec.url, config: config, targetShapes: shapes)
        guard plan.unmapped.isEmpty else {
            throw EngineError.unmappedKeys(Array(plan.unmapped.prefix(8)))
        }
        let source = try TensorSource(url: spec.url)
        defer { source.close() }
        func read(_ name: String) throws -> MLXArray {
            try WeightLoader.materialize(.init(key: name, dequant: .passthrough(rawKey: name), shape: []), from: source).asType(.float32)
        }
        for part in plan.parts {
            let strength = Float(spec.strength(forBlock: part.block))
            guard strength != 0 else { continue }
            switch part.kind {
            case .lowRank, .kronecker:
                loras[part.target, default: []].append(LoRAUpdate(
                    down: try read(part.down), up: try read(part.up), scale: part.scale * strength, rows: part.rows,
                    kronecker: part.kind == .kronecker))
            case .weightDiff, .biasDiff:
                if baseParams[part.target] == nil { baseParams[part.target] = w[part.target]! }
                w[part.target] = w[part.target]!.asType(.float32) + strength * (try read(part.down))
            case .hadamard:
                // LoHa has no low-rank form: the dense delta is merged in F32 and stored back in the
                // weight's dtype (an F32 copy of every touched projection would double the DiT).
                if baseParams[part.target] == nil { baseParams[part.target] = w[part.target]! }
                let delta = matmul(try read(part.up), try read(part.down)) * matmul(try read(part.up2), try read(part.down2))
                let current = w[part.target]!
                w[part.target] = MossModelTools.adding(delta * (part.scale * strength), to: current.asType(.float32), rows: part.rows)
                    .asType(current.dtype)
                eval(w[part.target]!)
            }
        }
        eval(loras.values.flatMap { $0.flatMap { [$0.down, $0.up] } } + baseParams.keys.compactMap { w[$0] })
        referenceKV = nil
    }

    public func attachLoRA(url: URL, strength: Float) throws {
        try attachLoRA(LoRASpec(url: url, strength: Double(strength)))
    }

    public func detachLoRAs() {
        loras = [:]
        for (key, original) in baseParams { w[key] = original }
        baseParams = [:]
        referenceKV = nil
    }

    /// Replace parameters with F32 tensors (lodestones' `time_embedding.safetensors`: the step embedder
    /// plus `shared_block_modulation.bias` / `video_head_modulation.bias`, which only work together).
    public func overrideParameters(_ tensors: [String: MLXArray]) throws {
        for (key, value) in tensors {
            guard let current = w[key], current.shape == value.shape else {
                throw EngineError.invalidRequest("override \(key) \(value.shape) doesn't match the DiT")
            }
            w[key] = value.asType(.float32)
        }
        referenceKV = nil
    }

    public static func load(url: URL, dtype: DType = .float16) throws -> MossDiT {
        let (descriptors, _) = try SafetensorsHeader.readDescriptors(from: url)
        var config = try MossDiTConfig(descriptors: descriptors)
        config.qkvLayout = .fused  // the loader fuses split files
        let prefix = MossDiTConfig.keyPrefix(descriptors.map(\.name))
        return MossDiT(weights: try WeightLoader.load(url: url, prefix: prefix).arrays, config: config, dtype: dtype)
    }

    /// One conditioning branch of a packed forward: text states `[S,4096]` (valid tokens only) and
    /// the blocks this branch skips (uncond-skip / SLG).
    public struct Branch {
        public var context: MLXArray
        public var skipLayers: Set<Int>
        public init(context: MLXArray, skipLayers: Set<Int> = []) {
            self.context = context
            self.skipLayers = skipLayers
        }
    }

    /// Velocity for one sample. `latent` `[1,128,H,W]` (normalised DiT tokens), `context` `[S,4096]`
    /// (valid text states only), `timestep` = 1000σ. Returns F32 `[1,128,H,W]`.
    public func callAsFunction(
        _ latent: MLXArray, timestep: Double, context: MLXArray,
        skipLayers: Set<Int> = [], tap: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        velocities(latent, timestep: timestep, branches: [Branch(context: context, skipLayers: skipLayers)], tap: tap)[0]
    }

    /// Velocities for several branches (cond, uncond, SLG) of the same latent and timestep in ONE
    /// packed forward: each branch's `[image | text]` rows are concatenated, so every matmul, norm
    /// and modulation runs once over all rows, and attention runs per branch (one SDPA each over
    /// its own rows — no padding, no mask, nothing leaks between branches). Numerically the same
    /// as separate forwards; the image embedding and modulation are shared.
    public func velocities(
        _ latent: MLXArray, timestep: Double, branches: [Branch], tap: ((String, MLXArray) -> Void)? = nil
    ) -> [MLXArray] {
        let (h, wd) = (latent.dim(2), latent.dim(3))
        let n = h * wd

        let (stepMod, head) = modulation(timestep: timestep)
        tap?("mod", stepMod)

        let video = linear(latent.reshaped([config.latentChannels, n]).T, "video_embedder")
        let isolated = editLayout.isolatedReferences && !references.isEmpty
        if isolated, referenceKV == nil { referenceKV = isolatedReferenceKV() }
        let refs = isolated ? [] : references.map { r in linear(r.reshaped([config.latentChannels, r.dim(2) * r.dim(3)]).T, "video_embedder") }
        let refKV = isolated ? referenceKV : nil
        var segments: [Segment] = []
        var rows: [MLXArray] = []
        var modParts: [MLXArray] = []  // per-row modulation when references ride along
        let mod0 = refs.isEmpty ? nil : modulation(timestep: 0).0
        var start = 0
        for branch in branches {
            let s = branch.context.dim(0)
            let placement = MossRoPEIndex.referenceSequence(layout: editLayout, textLength: s)
            var rope = ropeTable(
                gridHeight: h, gridWidth: wd, textLength: s, sequenceOffset: references.isEmpty ? 0 : placement.offset)
            var text = linear(branch.context, "text_embedder.proj")
            let textRope = (cos: rope.cos[n...], sin: rope.sin[n...])
            for b in 0..<config.textBlocks { text = textBlock(text, rope: textRope, "text_embedder.blocks.\(b)") }
            if segments.isEmpty { tap?("text", text) }
            rows += [video, text]
            var count = n + s
            if let mod0 {
                modParts.append(broadcast(stepMod[0..., .newAxis, 0...], to: [6, n + s, config.dim]))
                var cosParts = [rope.cos], sinParts = [rope.sin]
                for (k, ref) in references.enumerated() {
                    let (rh, rw) = (ref.dim(2), ref.dim(3))
                    let a = MLXArray(
                        MossRoPEIndex.referenceAngles(
                            index: k + 1, gridHeight: rh, gridWidth: rw, sequencePosition: placement.reference,
                            headDim: config.headDim, base: config.ropeBase),
                        [rh * rw, config.headDim / 2])
                    let full = concatenated([a, a], axis: 1)
                    cosParts.append(cos(full))
                    sinParts.append(sin(full))
                    rows.append(refs[k])
                    modParts.append(broadcast(mod0[0..., .newAxis, 0...], to: [6, rh * rw, config.dim]))
                    count += rh * rw
                }
                rope = (concatenated(cosParts, axis: 0), concatenated(sinParts, axis: 0))
            }
            segments.append(Segment(start: start, count: count, rope: rope, skip: branch.skipLayers))
            start += count
        }
        // Rows conditioned at the step's timestep, reference rows at t = 0: modulation becomes [6, rows, dim].
        let mod = modParts.isEmpty ? stepMod : concatenated(modParts, axis: 1)

        var x = concatenated(rows, axis: 0)  // [Σ(N+S), dim] F32
        for i in 0..<config.layers {
            let active = segments.filter { !$0.skip.contains(i) }
            if active.count == segments.count {
                x = checkpointBlocks && !trainable.isEmpty
                    ? checkpointedBlock(x, segments: segments, mod: mod, "layers.\(i)")
                    : block(x, segments: segments, mod: mod, "layers.\(i)", referenceKV: refKV?[i])
            } else if !active.isEmpty {
                // Run the block on the active branches only and splice the skipped ones back in.
                var packed: [Segment] = []
                var offset = 0
                for seg in active {
                    packed.append(Segment(start: offset, count: seg.count, rope: seg.rope, skip: seg.skip))
                    offset += seg.count
                }
                let activeMod = mod.ndim == 3 ? concatenated(active.map { mod[0..., $0.range] }, axis: 1) : mod
                let y = block(
                    concatenated(active.map { x[$0.range] }, axis: 0), segments: packed, mod: activeMod, "layers.\(i)",
                    referenceKV: refKV?[i])
                var parts: [MLXArray] = []
                var k = 0
                for seg in segments {
                    if seg.skip.contains(i) {
                        parts.append(x[seg.range])
                    } else {
                        parts.append(y[packed[k].range])
                        k += 1
                    }
                }
                x = concatenated(parts, axis: 0)
            }
            if tap != nil, i == 0 || i == 21 {
                eval(x)
                tap?("layer\(i)", x[segments[0].range])
            }
        }
        let images = concatenated(segments.map { x[$0.start..<($0.start + n)] }, axis: 0)
        var out = rmsNorm(images, "final_norm_video") * (1 + head[1]) + head[0]
        out = linear(out, "final_linear_video", bias: false)  // [B·N, 128]
        return (0..<segments.count).map { b in
            out[(b * n)..<((b + 1) * n)].T.reshaped([1, config.latentChannels, h, wd])
        }
    }

    /// Isolated references (ai-toolkit kv_cache, Moss-Image-Edit): all references in one sequence at
    /// timestep 0, t = 0, attending only to each other through every block (LoRAs included); each
    /// block's post-RoPE keys and values are kept (ComfyUI `preprocess_refs`).
    private func isolatedReferenceKV() -> [(k: MLXArray, v: MLXArray)] {
        var tokens: [MLXArray] = [], cosParts: [MLXArray] = [], sinParts: [MLXArray] = []
        for (k, ref) in references.enumerated() {
            let (rh, rw) = (ref.dim(2), ref.dim(3))
            tokens.append(linear(ref.reshaped([config.latentChannels, rh * rw]).T, "video_embedder"))
            let a = MLXArray(
                MossRoPEIndex.referenceAngles(
                    index: k + 1, gridHeight: rh, gridWidth: rw, sequencePosition: 0, headDim: config.headDim, base: config.ropeBase),
                [rh * rw, config.headDim / 2])
            let full = concatenated([a, a], axis: 1)
            cosParts.append(cos(full))
            sinParts.append(sin(full))
        }
        var x = concatenated(tokens, axis: 0)
        let segment = Segment(
            start: 0, count: x.dim(0), rope: (concatenated(cosParts, axis: 0), concatenated(sinParts, axis: 0)), skip: [])
        let mod0 = modulation(timestep: 0).0
        var out: [(k: MLXArray, v: MLXArray)] = []
        for i in 0..<config.layers {
            var kv: (k: MLXArray, v: MLXArray)?
            x = block(x, segments: [segment], mod: mod0, "layers.\(i)", capture: { kv = ($0, $1) })
            eval(x, kv!.k, kv!.v)
            out.append(kv!)
        }
        return out
    }

    /// A branch's rows in the packed sequence.
    private struct Segment {
        var start: Int
        var count: Int
        var rope: (cos: MLXArray, sin: MLXArray)
        var skip: Set<Int>
        var range: Range<Int> { start..<(start + count) }
    }

    /// `(mod [6, dim], head [2, dim])` from the timestep — shared by every block and the head.
    public func modulation(timestep: Double) -> (MLXArray, MLXArray) {
        let embedding = MLXArray(MossTimestepEmbedding.sinusoid(timestep: timestep, dim: config.freqDim))[.newAxis]
        let t = linear(silu(linear(embedding, "time_embedding.0", f32: true)), "time_embedding.2", f32: true)
        let mod = linear(t, "shared_block_modulation", f32: true).reshaped([6, config.dim])
        let head = linear(t, "video_head_modulation", f32: true).reshaped([2, config.dim])
        return (mod, head)
    }

    // MARK: - Blocks

    /// `referenceKV` (isolated edit references) joins every segment's keys and values; `capture` sees
    /// this block's post-RoPE keys and values (building that cache).
    private func block(
        _ x: MLXArray, segments: [Segment], mod: MLXArray, _ p: String,
        referenceKV: (k: MLXArray, v: MLXArray)? = nil, capture: ((MLXArray, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let (dim, heads) = (config.dim, config.heads)
        let n = x.dim(0)
        let rope = segments.count == 1
            ? segments[0].rope
            : (cos: concatenated(segments.map(\.rope.cos), axis: 0), sin: concatenated(segments.map(\.rope.sin), axis: 0))
        let (cosT, sinT) = (rope.cos[0..., .newAxis, 0...], rope.sin[0..., .newAxis, 0...])
        // A block with a runtime LoRA feeds its projections F32 (the delta's input, as in `linear`).
        let adapted = loras.keys.contains(where: { $0.hasPrefix("\(p).") }) || trainable.keys.contains(where: { $0.hasPrefix("\(p).") })
        let fused = adapted ? fused32 : self.fused

        // Matmuls return the working dtype; the F32 norm/modulation/RoPE/SiTU/residual math runs
        // inside fused (compiled) kernels so no F32 intermediate of the big activations hits memory.
        let qkv = project(fused.modulate([x, w["\(p).norm1.weight"]!, mod[1], mod[0]])[0], "\(p).proj.qkv")
        let q = fused.qkNormRope([qkv[0..., ..<dim].reshaped([n, heads, -1]), w["\(p).q_norm.weight"]!, cosT, sinT])[0]
        let k = fused.qkNormRope([qkv[0..., dim..<(2 * dim)].reshaped([n, heads, -1]), w["\(p).k_norm.weight"]!, cosT, sinT])[0]
        let v = qkv[0..., (2 * dim)...].reshaped([n, heads, -1])
        capture?(k, v)
        let attn = concatenated(segments.map { seg in
            guard let r = referenceKV else { return sdpa(q[seg.range], k[seg.range], v[seg.range]) }
            return sdpa(q[seg.range], concatenated([k[seg.range], r.k], axis: 0), concatenated([v[seg.range], r.v], axis: 0))
        }, axis: 0)
        var out = fused.gatedResidual([x, project(attn, "\(p).proj.o"), mod[2]])[0]

        let upGate = project(fused.modulate([out, w["\(p).norm2.weight"]!, mod[4], mod[3]])[0], "\(p).proj.ffn_up_gate")
        out = fused.gatedResidual([out, project(fused.situ([upGate])[0], "\(p).proj.ffn_down"), mod[5]])[0]
        return out
    }

    /// `block` as an MLX custom function whose VJP re-runs the block (MLX's `checkpoint`): only the
    /// block's input is kept for the backward pass. The block's adapters are inputs too, so their
    /// gradients flow; modulation and RoPE are constants (nothing trains there).
    private func checkpointedBlock(_ x: MLXArray, segments: [Segment], mod: MLXArray, _ p: String) -> MLXArray {
        let saved = trainable
        let keys = saved.keys.filter { $0.hasPrefix("\(p).") }.sorted()
        let run: ([MLXArray]) -> [MLXArray] = { [unowned self] inputs in
            var t = saved
            for (j, k) in keys.enumerated() {
                let a = saved[k]!
                t[k] = TrainableLoRA(down: inputs[1 + 2 * j], up: inputs[2 + 2 * j], scale: a.scale, kronecker: a.kronecker)
            }
            let previous = self.trainable
            self.trainable = t
            defer { self.trainable = previous }
            return [self.block(inputs[0], segments: segments, mod: mod, p)]
        }
        let f = CustomFunction {
            Forward(run)
            VJP { primals, cotangents in vjp(run, primals: primals, cotangents: cotangents).1 }
        }
        return f([x] + keys.flatMap { [saved[$0]!.down, saved[$0]!.up] })[0]
    }

    /// Unmasked SDPA over one branch's rows; q, k (already normed + rotated), v `[N, heads, D]` in
    /// the working dtype → `[N, dim]`.
    private func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let n = q.dim(0)
        func prep(_ x: MLXArray) -> MLXArray { x.transposed(1, 0, 2)[.newAxis].asType(dtype) }  // [1,H,N,D]
        let o = MLXFast.scaledDotProductAttention(
            queries: prep(q), keys: prep(k), values: prep(v), scale: 1 / Float(config.headDim).squareRoot(), mask: .none)
        return o[0].transposed(1, 0, 2).reshaped([n, config.dim])
    }

    /// `x @ Wᵀ` in the weight's dtype, no bias, result left in that dtype — or, with a runtime LoRA on
    /// this weight, F32 with the unmerged delta added (exactly `linear`'s math).
    private func project(_ x: MLXArray, _ p: String) -> MLXArray {
        let weight = w["\(p).weight"]!
        var y = matmul(x.asType(weight.dtype), weight.T)
        if let updates = loras["\(p).weight"] {
            y = y.asType(.float32) + loraDelta(x.asType(.float32), updates, outRows: weight.dim(0))
        }
        if let t = trainable["\(p).weight"] {
            let xf = x.asType(.float32)
            y = y.asType(.float32)
                + (t.kronecker ? Self.kroneckerApply(xf, w1: t.down, w2: t.up) : matmul(matmul(xf, t.down.T), t.up.T)) * t.scale
        }
        return y
    }

    /// Text adapter block: pre-norm, no modulation, fused qkv **with bias**, SiTU FFN with separate
    /// `ffn_gate` / `ffn_up` (+bias).
    private func textBlock(_ x: MLXArray, rope: (cos: MLXArray, sin: MLXArray), _ p: String) -> MLXArray {
        let (dim, heads) = (config.dim, config.heads)
        let n = x.dim(0)
        let qkv = linear(rmsNorm(x, "\(p).norm1"), "\(p).qkv")
        let q = rmsNorm(qkv[0..., ..<dim].reshaped([n, heads, -1]), "\(p).q_norm")
        let k = rmsNorm(qkv[0..., dim..<(2 * dim)].reshaped([n, heads, -1]), "\(p).k_norm")
        let v = qkv[0..., (2 * dim)...].reshaped([n, heads, -1])
        let out = x + linear(attention(q, k, v, rope: rope), "\(p).proj")
        let y = rmsNorm(out, "\(p).norm2")
        return out + linear(Self.situ(hidden: linear(y, "\(p).ffn_up"), gate: linear(y, "\(p).ffn_gate")), "\(p).ffn_down")
    }

    /// Full (unmasked) attention over the joint sequence with the 4D RoPE. q, k, v `[N, heads, 128]`.
    private func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let n = q.dim(0)
        func prep(_ x: MLXArray) -> MLXArray { x.transposed(1, 0, 2)[.newAxis].asType(dtype) }  // [1,H,N,D]
        let o = MLXFast.scaledDotProductAttention(
            queries: prep(Self.rotate(q, rope)), keys: prep(Self.rotate(k, rope)), values: prep(v),
            scale: 1 / Float(config.headDim).squareRoot(), mask: .none)
        return o[0].transposed(1, 0, 2).reshaped([n, config.dim])
    }

    // MARK: - Primitives

    /// `x @ Wᵀ (+ b)` → F32. Block/adapter weights are in the working dtype; `f32` forces F32 math.
    private func linear(_ x: MLXArray, _ p: String, bias: Bool = true, f32: Bool = false) -> MLXArray {
        let weight = w["\(p).weight"]!
        let t: DType = f32 ? .float32 : weight.dtype
        var y = matmul(x.asType(t), weight.asType(t).T).asType(.float32)
        if bias, let b = w["\(p).bias"] { y = y + b }
        if let updates = loras["\(p).weight"] { y = y + loraDelta(x.asType(.float32), updates, outRows: weight.dim(0)) }
        return y
    }

    /// Σ scale·up(down(x)) over the updates, row slices (fused q|k|v) placed with zero fill.
    private func loraDelta(_ x: MLXArray, _ updates: [LoRAUpdate], outRows: Int) -> MLXArray {
        var full: MLXArray?
        var sliced: [(Range<Int>, MLXArray)] = []
        for u in updates {
            let d = (u.kronecker ? Self.kroneckerApply(x, w1: u.down, w2: u.up) : matmul(matmul(x, u.down.T), u.up.T)) * u.scale
            if let rows = u.rows { sliced.append((rows, d)) } else { full = full.map { $0 + d } ?? d }
        }
        guard !sliced.isEmpty else { return full! }
        // Merge same-range slices, then lay them out across all output rows.
        var byRange: [Range<Int>: MLXArray] = [:]
        for (r, d) in sliced { byRange[r] = byRange[r].map { $0 + d } ?? d }
        var pieces: [MLXArray] = []
        var cursor = 0
        for r in byRange.keys.sorted(by: { $0.lowerBound < $1.lowerBound }) {
            if r.lowerBound > cursor { pieces.append(MLXArray.zeros([x.dim(0), r.lowerBound - cursor])) }
            pieces.append(byRange[r]!)
            cursor = r.upperBound
        }
        if cursor < outRows { pieces.append(MLXArray.zeros([x.dim(0), outRows - cursor])) }
        let delta = concatenated(pieces, axis: 1)
        return full.map { $0 + delta } ?? delta
    }

    private func rmsNorm(_ x: MLXArray, _ p: String) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: w["\(p).weight"]!, eps: 1e-6)
    }

    /// SiTU-GLU: `(4·tanh(g/4)·σ(g)) ⊙ (25·tanh(h/25))` (paper eq. 1, constants from the code).
    static func situ(hidden: MLXArray, gate: MLXArray) -> MLXArray {
        (4 * tanh(gate / 4) * sigmoid(gate)) * (25 * tanh(hidden / 25))
    }

    /// Rotate-half with the 4D table: `x·cos + [−x₂, x₁]·sin`, `x` `[N, heads, D]`, cos/sin `[N, D]`.
    static func rotate(_ x: MLXArray, _ rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let half = x.dim(2) / 2
        let rotated = concatenated([-x[0..., 0..., half...], x[0..., 0..., ..<half]], axis: 2)
        return x * rope.cos[0..., .newAxis, 0...] + rotated * rope.sin[0..., .newAxis, 0...]
    }

    /// cos/sin `[N+S, headDim]` = `[angles | angles]` from `MossRoPEIndex.angles` (F32).
    func ropeTable(gridHeight: Int, gridWidth: Int, textLength: Int, sequenceOffset: Int = 0) -> (cos: MLXArray, sin: MLXArray) {
        let angles = MossRoPEIndex.angles(
            gridHeight: gridHeight, gridWidth: gridWidth, textLength: textLength, headDim: config.headDim,
            base: config.ropeBase, sequenceOffset: Float(sequenceOffset))
        let a = MLXArray(angles, [gridHeight * gridWidth + textLength, config.headDim / 2])
        let full = concatenated([a, a], axis: 1)
        return (cos(full), sin(full))
    }
}

/// The block's elementwise chains, each compiled into one kernel (`MLX.compile`). Inputs come in
/// the working dtype or F32; the math is F32 exactly as in the unfused reference (`rmsNorm`,
/// `situ`, `rotate`), and only the result is cast back — so fusing changes no numerics, it just
/// keeps the F32 intermediates of `[tokens, 3·dim]` / `[tokens, 2·ffn]` activations out of memory.
private struct FusedOps {
    /// `(RMSNorm(x)·(1+scale)+shift)` → working dtype. `[x f32, normWeight, scale, shift]`.
    let modulate: ([MLXArray]) -> [MLXArray]
    /// `rotate(RMSNorm(t))` → working dtype. `[t [N,H,D], normWeight, cos [N,1,D], sin [N,1,D]]`.
    let qkNormRope: ([MLXArray]) -> [MLXArray]
    /// `x + f32(y)·gate` (F32 residual). `[x, y, gate]`.
    let gatedResidual: ([MLXArray]) -> [MLXArray]
    /// SiTU-GLU of `[hidden | gate]` → working dtype. `[upGate]`.
    let situ: ([MLXArray]) -> [MLXArray]

    init(dtype: DType, ffnDim: Int) {
        modulate = compile { a in
            [(MLXFast.rmsNorm(a[0].asType(.float32), weight: a[1], eps: 1e-6) * (1 + a[2]) + a[3]).asType(dtype)]
        }
        qkNormRope = compile { a in
            let t = MLXFast.rmsNorm(a[0].asType(.float32), weight: a[1], eps: 1e-6)
            let half = t.dim(2) / 2
            let rotated = concatenated([-t[0..., 0..., half...], t[0..., 0..., ..<half]], axis: 2)
            return [(t * a[2] + rotated * a[3]).asType(dtype)]
        }
        gatedResidual = compile { a in [a[0] + a[1].asType(.float32) * a[2]] }
        situ = compile { a in
            let x = a[0].asType(.float32)
            return [MossDiT.situ(hidden: x[0..., ..<ffnDim], gate: x[0..., ffnDim...]).asType(dtype)]
        }
    }
}
