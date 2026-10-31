import Foundation
import MossForgeCore

/// Geometry of the Qwen3.5 text stack (the MOSS-Image conditioning encoder), derived from a
/// header whose keys are prefix-stripped (`MossWeightPlan` with `prefix: "model.language_model."`).
/// Grounded in ComfyUI `comfy/text_encoders/qwen35.py` (`qwen35_9b`) and Qwen/Qwen3.5-9B
/// `config.json`; see `docs/dev/technical-reference.md`.
///
/// Per layer: pre-norm → token mixer → residual → pre-norm → SwiGLU MLP → residual, where the
/// mixer is **Gated DeltaNet** linear attention on 3 of every 4 layers and gated full attention
/// (GQA, q/k RMSNorm, partial split-half RoPE) on every 4th. All RMSNorms scale by `(1 + w)`
/// except the DeltaNet's gated output norm (plain `w`). Encoder output = final-norm hidden states.
public struct Qwen35TextConfig: Sendable, Equatable {
    public var hidden: Int
    public var layers: Int
    public var intermediate: Int
    public var vocab: Int
    public var fullAttentionLayers: [Int]
    // Full attention
    public var heads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var rotaryDim: Int
    public var ropeTheta: Double
    // Gated DeltaNet
    public var linearKeyHeads: Int
    public var linearValueHeads: Int
    public var linearKeyHeadDim: Int
    public var linearValueHeadDim: Int
    public var convKernel: Int
    public var rmsEpsilon: Float

    public var linearKeyDim: Int { linearKeyHeads * linearKeyHeadDim }
    public var linearValueDim: Int { linearValueHeads * linearValueHeadDim }
    /// Channels through the causal depthwise conv: q | k | v.
    public var linearConvDim: Int { 2 * linearKeyDim + linearValueDim }

    public func isFullAttention(_ layer: Int) -> Bool { fullAttentionLayers.contains(layer) }

    /// Qwen3.5-9B as released (config.json: full_attention_interval 4, rope θ 1e7, partial 0.25).
    public static let qwen35_9b = Qwen35TextConfig(
        hidden: 4096, layers: 32, intermediate: 12288, vocab: 248320,
        fullAttentionLayers: [3, 7, 11, 15, 19, 23, 27, 31],
        heads: 16, kvHeads: 4, headDim: 256, rotaryDim: 64, ropeTheta: 1e7,
        linearKeyHeads: 16, linearValueHeads: 32, linearKeyHeadDim: 128, linearValueHeadDim: 128,
        convKernel: 4, rmsEpsilon: 1e-6)

    public init(
        hidden: Int, layers: Int, intermediate: Int, vocab: Int, fullAttentionLayers: [Int],
        heads: Int, kvHeads: Int, headDim: Int, rotaryDim: Int, ropeTheta: Double,
        linearKeyHeads: Int, linearValueHeads: Int, linearKeyHeadDim: Int, linearValueHeadDim: Int,
        convKernel: Int, rmsEpsilon: Float
    ) {
        self.hidden = hidden
        self.layers = layers
        self.intermediate = intermediate
        self.vocab = vocab
        self.fullAttentionLayers = fullAttentionLayers
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.rotaryDim = rotaryDim
        self.ropeTheta = ropeTheta
        self.linearKeyHeads = linearKeyHeads
        self.linearValueHeads = linearValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.convKernel = convKernel
        self.rmsEpsilon = rmsEpsilon
    }

    /// From prefix-stripped header descriptors. Constants the header can't carry (θ 1e7, partial
    /// rotary 0.25, eps 1e-6, key head dim = value head dim) are the Qwen3.5 family's.
    public init(descriptors: [TensorDescriptor]) throws {
        let shapes = Dictionary(descriptors.map { ($0.name, $0.shape) }, uniquingKeysWith: { a, _ in a })
        guard let embed = shapes["embed_tokens.weight"], embed.count == 2 else {
            throw EngineError.invalidRequest("not a Qwen3.5 language model (no embed_tokens)")
        }
        var layers = 0
        while shapes["layers.\(layers).input_layernorm.weight"] != nil { layers += 1 }
        let full = (0..<layers).filter { shapes["layers.\($0).self_attn.q_proj.weight"] != nil }
        guard layers > 0, let f = full.first, let l = (0..<layers).first(where: { !full.contains($0) }) else {
            throw EngineError.invalidRequest("Qwen3.5 header needs both linear and full attention layers")
        }
        func rows(_ key: String) -> Int { shapes[key]?.first ?? 0 }
        let headDim = rows("layers.\(f).self_attn.q_norm.weight")
        let valueHeads = rows("layers.\(l).linear_attn.A_log")
        let valueHeadDim = rows("layers.\(l).linear_attn.norm.weight")
        let keyDim = (rows("layers.\(l).linear_attn.in_proj_qkv.weight") - valueHeads * valueHeadDim) / 2
        guard headDim > 0, valueHeads > 0, valueHeadDim > 0, keyDim > 0 else {
            throw EngineError.invalidRequest("implausible Qwen3.5 attention dims")
        }
        self.init(
            hidden: embed[1], layers: layers, intermediate: rows("layers.0.mlp.gate_proj.weight"), vocab: embed[0],
            fullAttentionLayers: full,
            heads: rows("layers.\(f).self_attn.q_proj.weight") / (2 * headDim),  // q | output gate
            kvHeads: rows("layers.\(f).self_attn.k_proj.weight") / headDim,
            headDim: headDim, rotaryDim: headDim / 4, ropeTheta: 1e7,
            linearKeyHeads: keyDim / valueHeadDim, linearValueHeads: valueHeads,
            linearKeyHeadDim: valueHeadDim, linearValueHeadDim: valueHeadDim,
            convKernel: shapes["layers.\(l).linear_attn.conv1d.weight"]?.last ?? 4, rmsEpsilon: 1e-6)
    }

    /// `inv_freq_i = θ^(−2i / rotaryDim)`, i < rotaryDim/2 — applied split-half (rotate-half) to
    /// the first `rotaryDim` dims of each q/k head; the rest pass through. Text-only positions make
    /// the interleaved M-RoPE sections collapse to plain 1D RoPE.
    public var ropeInverseFrequencies: [Float] {
        (0..<(rotaryDim / 2)).map { Float(pow(ropeTheta, -2.0 * Double($0) / Double(rotaryDim))) }
    }
}
