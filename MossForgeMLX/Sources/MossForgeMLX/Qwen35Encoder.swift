import Foundation
import MLX
import MLXFast
import MLXNN
import MossForgeCore
import MossForgeEngine
import Tokenizers

/// The MOSS-Image text encoder: the Qwen3.5-9B text stack (24 Gated DeltaNet + 8 gated full-
/// attention layers) returning **final-norm** hidden states `[T, 4096]` for the chat-wrapped
/// prompt — ComfyUI's `layer="last"`. Ported from the semantics transcribed in
/// `Scripts/encoder_ref.py`; geometry from `Qwen35TextConfig`.
///
/// Matmuls run in the working dtype; norms, the DeltaNet gates/recurrence and the residual stream
/// run in F32. **Precision:** F32 matches the torch reference to rel-err 2e-4 at 4096 tokens. bf16
/// (weights rounded to bf16) is fine on natural prompts (1k tokens: rel-err 0.011, token cosine
/// ≥ 0.997) but the model amplifies the rounding on long repetitive input (synthetic 4k fixture:
/// rel-err 0.2) — bf16-rounded weights *alone* with F32 compute already reach rel-err 0.07 there, so
/// it is model sensitivity, not a kernel bug (`moss-mlx encoder-diverge`). The int8 file needs
/// ~32 GB in F32 / 16 GB in bf16.
public final class Qwen35Encoder {
    public let config: Qwen35TextConfig
    public let dtype: DType
    private let w: [String: MLXArray]

    public init(weights: [String: MLXArray], config: Qwen35TextConfig, dtype: DType = .bfloat16) {
        self.config = config
        self.dtype = dtype
        var prepared: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasSuffix("norm.weight") || key.hasSuffix("A_log") || key.hasSuffix("dt_bias") {
                prepared[key] = value.asType(.float32)
            } else {
                prepared[key] = value.asType(dtype)
            }
        }
        w = prepared
    }

    /// Load the language-model half of a Qwen3.5 VLM file (vision tower, lm_head and MTP skipped).
    public static func load(url: URL, dtype: DType = .bfloat16) throws -> Qwen35Encoder {
        let prefix = "model.language_model."
        let plan = try MossWeightPlan.build(url: url, prefix: prefix)
        let config = try Qwen35TextConfig(
            descriptors: plan.entries.map { TensorDescriptor(name: $0.key, dtype: "", shape: $0.shape, byteCount: 0) })
        let weights = try WeightLoader.load(url: url, prefix: prefix, working: dtype)
        return Qwen35Encoder(weights: weights.arrays, config: config, dtype: dtype)
    }

    /// Token ids → final-norm hidden states `[T, hidden]` (F32).
    public func encode(_ ids: [Int], taps: Set<Int> = [], tapped: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        forward(ids, cache: nil, taps: taps, tapped: tapped)
    }

    /// Per-layer decoding state for autoregressive generation (`Qwen35Generator`): full-attention
    /// layers keep their K/V; DeltaNet layers keep the last `convKernel−1` conv inputs and the F32
    /// recurrent state. `offset` = tokens consumed so far (the RoPE position of the next one).
    public final class Cache {
        var keys: [Int: MLXArray] = [:]
        var values: [Int: MLXArray] = [:]
        var convState: [Int: MLXArray] = [:]
        var recurrent: [Int: MLXArray] = [:]
        public internal(set) var offset = 0
        /// M-RoPE: the RoPE position of the next token minus `offset` (an image's tokens take fewer
        /// positions than slots — `QwenMRoPEIndex.positions(...).next − T` after a multimodal prefill).
        public internal(set) var positionDelta = 0
        public init() {}
    }

    /// Token ids → final-norm hidden states `[T, hidden]` (F32). With a `cache`, the ids continue
    /// the sequence it holds (prefill once, then one token per step) and the cache is advanced.
    public func forward(_ ids: [Int], cache: Cache?, taps: Set<Int> = [], tapped: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        forward(embeddings: embed(ids), positions: nil, cache: cache, taps: taps, tapped: tapped)
    }

    /// Token embeddings `[T, hidden]` (F32) — the multimodal path splices image tokens into these.
    public func embed(_ ids: [Int]) -> MLXArray {
        w["embed_tokens.weight"]!.take(MLXArray(ids.map(Int32.init)), axis: 0).asType(.float32)
    }

    /// Embeddings `[T, hidden]` → final-norm hidden states `[T, hidden]` (F32). `positions` are
    /// M-RoPE `[3][T]` ids (temporal, height, width) for a sequence with an image; `nil` = text
    /// positions continuing the cache. With a cache and `next` (the position after the sequence),
    /// later text tokens continue from `next`.
    public func forward(
        embeddings: MLXArray, positions: (ids: [[Int]], next: Int)?, cache: Cache?,
        taps: Set<Int> = [], tapped: ((Int, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let t = embeddings.dim(0)
        var x = embeddings.asType(.float32)[.newAxis]  // [1,T,H]
        let rope = positions.map { mropeCosSin($0.ids) }
        for i in 0..<config.layers {
            let p = "layers.\(i)."
            let h = norm(x, p + "input_layernorm.weight")
            x = x + (config.isFullAttention(i)
                ? attention(h, p + "self_attn.", length: t, layer: i, cache: cache, rope: rope)
                : deltaNet(h, p + "linear_attn.", layer: i, cache: cache))
            let m = norm(x, p + "post_attention_layernorm.weight").asType(dtype)
            let mlp = linear(silu(linear(m, p + "mlp.gate_proj.weight")) * linear(m, p + "mlp.up_proj.weight"), p + "mlp.down_proj.weight")
            x = x + mlp.asType(.float32)
            if taps.contains(i) {
                eval(x)
                tapped?(i, x[0])
            }
        }
        if let cache {
            if let positions { cache.positionDelta = positions.next - (cache.offset + t) }
            cache.offset += t
        }
        return norm(x, "norm.weight")[0]
    }

    /// Qwen3.5 interleaved M-RoPE (`apply_interleaved_mrope`, section [11, 11, 10] over the 32
    /// frequencies of the 64 rotary dims): frequency j follows the height axis for j ≡ 1 (mod 3),
    /// j < 33, the width axis for j ≡ 2 (mod 3), j < 30, else the temporal axis. Text tokens (all
    /// axes equal) reduce to the plain split-half RoPE. Returns cos/sin `[1, 1, T, rotaryDim]`.
    func mropeCosSin(_ ids: [[Int]]) -> (cos: MLXArray, sin: MLXArray) {
        let half = config.rotaryDim / 2
        let section = Self.mropeSection
        var axisOf = [Int](repeating: 0, count: half)
        for (axis, offset) in [(1, 1), (2, 2)] {
            for j in stride(from: offset, to: min(section[axis] * 3, half), by: 3) { axisOf[j] = axis }
        }
        let theta = Double(config.ropeTheta)
        let invFreq = (0..<half).map { Float(1 / pow(theta, Double(2 * $0) / Double(config.rotaryDim))) }
        let t = ids[0].count
        var freqs = [Float](repeating: 0, count: t * half)
        for i in 0..<t {
            for j in 0..<half { freqs[i * half + j] = Float(ids[axisOf[j]][i]) * invFreq[j] }
        }
        let f = MLXArray(freqs, [t, half])
        let emb = concatenated([f, f], axis: -1).reshaped([1, 1, t, config.rotaryDim])
        return (cos(emb), sin(emb))
    }

    static let mropeSection = [11, 11, 10]

    /// Split-half rotation of the first `rotaryDim` dims of `x` `[B, heads, T, D]` with explicit cos/sin.
    private func applyRoPE(_ x: MLXArray, _ rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let r = config.rotaryDim
        let rot = x[.ellipsis, ..<r].asType(.float32)
        let half = r / 2
        let rotated = concatenated([-rot[.ellipsis, half...], rot[.ellipsis, ..<half]], axis: -1)
        let out = rot * rope.cos + rotated * rope.sin
        return concatenated([out.asType(x.dtype), x[.ellipsis, r...]], axis: -1)
    }

    // MARK: - Blocks

    private func linear(_ x: MLXArray, _ key: String) -> MLXArray { matmul(x.asType(dtype), w[key]!.T) }

    /// Qwen3.5 RMSNorm: scale by `(1 + w)`, F32.
    private func norm(_ x: MLXArray, _ key: String) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: 1 + w[key]!, eps: config.rmsEpsilon)
    }

    /// Gated DeltaNet: causal depthwise conv (k=4) + SiLU over q|k|v, l2-normalised q/k (q·Dk^-½),
    /// decay `g = exp(−exp(A_log)·softplus(a + dt_bias))`, `β = σ(b)`, the delta-rule recurrence,
    /// then `RMSNorm(y)·w · SiLU(z)` per value head and the output projection.
    private func deltaNet(_ x: MLXArray, _ p: String, layer: Int, cache: Cache?) -> MLXArray {
        let c = config
        let (b, t) = (x.dim(0), x.dim(1))
        let proj = linear(x, p + "in_proj_qkv.weight").asType(.float32)  // [B,T,conv]
        let z = linear(x, p + "in_proj_z.weight").asType(.float32)
        let beta = sigmoid(linear(x, p + "in_proj_b.weight").asType(.float32))
        let a = linear(x, p + "in_proj_a.weight").asType(.float32)
        let g = exp(-exp(w[p + "A_log"]!) * softplus(a + w[p + "dt_bias"]!))

        // Causal depthwise conv: out[t] = Σ_j w[c, j] · x[t + j − (K−1)], with the cached history
        // (zeros at the start of a sequence) in front.
        let kernel = c.convKernel
        let convW = w[p + "conv1d.weight"]!.asType(.float32).reshaped([c.linearConvDim, kernel])  // [C,K]
        let history = cache?.convState[layer] ?? MLXArray.zeros([b, kernel - 1, c.linearConvDim])
        let padded = concatenated([history, proj], axis: 1)
        if let cache { cache.convState[layer] = padded[0..., (padded.dim(1) - (kernel - 1))..., 0...] }
        var conv = MLXArray.zeros(like: proj)
        for j in 0..<kernel { conv = conv + padded[0..., j..<(j + t), 0...] * convW[0..., j] }
        conv = silu(conv)

        let parts = split(conv, indices: [c.linearKeyDim, 2 * c.linearKeyDim], axis: -1)
        var q = parts[0].reshaped([b, t, c.linearKeyHeads, c.linearKeyHeadDim])
        var k = parts[1].reshaped([b, t, c.linearKeyHeads, c.linearKeyHeadDim])
        let v = parts[2].reshaped([b, t, c.linearValueHeads, c.linearValueHeadDim])
        // F.normalize: x / max(‖x‖, 1e-12).
        q = q / maximum(sqrt((q * q).sum(axis: -1, keepDims: true)), 1e-12) * Float(c.linearKeyHeadDim).squareRootInverse
        k = k / maximum(sqrt((k * k).sum(axis: -1, keepDims: true)), 1e-12)

        let (y, state) = GatedDelta.run(q: q, k: k, v: v, g: g, beta: beta, state: cache?.recurrent[layer])  // [B,T,Hv,Dv] F32
        if let cache { cache.recurrent[layer] = state }
        let gated = MLXFast.rmsNorm(y, weight: w[p + "norm.weight"]!, eps: c.rmsEpsilon)
            * silu(z.reshaped([b, t, c.linearValueHeads, c.linearValueHeadDim]))
        return linear(gated.reshaped([b, t, c.linearValueDim]), p + "out_proj.weight").asType(.float32)
    }

    /// Gated full attention: q_proj → per head [q | gate]; (1+w) q/k RMSNorm; split-half RoPE on
    /// the first `rotaryDim` dims (θ 1e7); causal GQA SDPA; `· σ(gate)`; o_proj.
    private func attention(
        _ x: MLXArray, _ p: String, length t: Int, layer: Int, cache: Cache?, rope: (cos: MLXArray, sin: MLXArray)?
    ) -> MLXArray {
        let c = config
        let b = x.dim(0)
        let qg = linear(x, p + "q_proj.weight").reshaped([b, t, c.heads, 2 * c.headDim])
        let gate = qg[0..., 0..., 0..., c.headDim...].reshaped([b, t, c.heads * c.headDim]).asType(.float32)
        var q = norm(qg[0..., 0..., 0..., ..<c.headDim], p + "q_norm.weight").transposed(0, 2, 1, 3)
        var k = norm(linear(x, p + "k_proj.weight").reshaped([b, t, c.kvHeads, c.headDim]), p + "k_norm.weight")
            .transposed(0, 2, 1, 3)
        var v = linear(x, p + "v_proj.weight").reshaped([b, t, c.kvHeads, c.headDim]).transposed(0, 2, 1, 3)
        if let rope {
            q = applyRoPE(q, rope)
            k = applyRoPE(k, rope)
        } else {
            let offset = (cache?.offset ?? 0) + (cache?.positionDelta ?? 0)
            q = MLXFast.RoPE(q, dimensions: c.rotaryDim, traditional: false, base: Float(c.ropeTheta), scale: 1, offset: offset)
            k = MLXFast.RoPE(k, dimensions: c.rotaryDim, traditional: false, base: Float(c.ropeTheta), scale: 1, offset: offset)
        }
        k = k.asType(dtype)
        if let cache {
            if let pk = cache.keys[layer], let pv = cache.values[layer] {
                k = concatenated([pk, k], axis: 2)
                v = concatenated([pv, v], axis: 2)
            }
            cache.keys[layer] = k
            cache.values[layer] = v
        }
        // Causal over the new rows; a single new token attends to everything cached.
        let o = MLXFast.scaledDotProductAttention(
            queries: q.asType(dtype), keys: k, values: v,
            scale: Float(c.headDim).squareRootInverse, mask: t > 1 ? .causal : .none)
        let gated = o.transposed(0, 2, 1, 3).reshaped([b, t, c.heads * c.headDim]).asType(.float32) * sigmoid(gate)
        return linear(gated, p + "o_proj.weight").asType(.float32)
    }
}

extension Float {
    fileprivate var squareRootInverse: Float { 1 / squareRoot() }
}

/// The Qwen3.5 tokenizer (`tokenizer.json` + `tokenizer_config.json` from Qwen/Qwen3.5-9B) plus the
/// MOSS prompt wrapping: chat template, no BOS/EOS/padding, truncation to 4096 tokens.
public struct MossTokenizer: Sendable {
    private let tokenizer: any Tokenizer

    /// Load from a local folder. Never call from the main thread (swift-transformers' loader can
    /// deadlock there).
    public static func load(folder: URL) async throws -> MossTokenizer {
        MossTokenizer(tokenizer: try await AutoTokenizer.from(modelFolder: folder))
    }

    /// Raw ids for already-templated text (special tokens like `<|im_start|>` are recognised).
    public func ids(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// Text for generated ids (special tokens dropped).
    public func decode(_ ids: [Int]) -> String { tokenizer.decode(tokens: ids, skipSpecialTokens: true) }

    /// The id of a special token such as `<|im_end|>`.
    public func id(of token: String) -> Int? { tokenizer.convertTokenToId(token) }

    /// The ids the DiT conditioning sees for a user prompt.
    public func promptIDs(_ prompt: String) -> [Int] {
        Array(ids(MossPromptTemplate.wrap(prompt)).prefix(MossPromptTemplate.maxTokens))
    }
}
