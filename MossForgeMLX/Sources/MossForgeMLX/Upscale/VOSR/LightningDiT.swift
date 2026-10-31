
import Foundation
import MossForgeCore
import MossForgeEngine
import MLX
import MLXFast
import MLXNN

/// The VOSR2 restoration DiT (LightningDiT backbone, 1.4B; VOSR is Apache-2.0, cswry/VOSR).
/// Independent MLX port of the inference forward in Forge's `modules/vosr/lightningdit.py`:
/// a square latent tile `[1, 32, s, s]` (LQ latent ‖ noise) + timestep + DINOv2 tokens →
/// velocity `[1, 16, s, s]`. adaLN-single modulation, RMSNorm + QK-norm, SwiGLU, EVA-02 2D
/// rotary embeddings rescaled to the 32×32 training grid, and an un-gated cross-attention to
/// the DINOv2 features after every self-attention. Verified via `moss-mlx vosr-parity`.
public struct LightningDiT {
    public struct Config: Sendable {
        public var hidden = 1536
        public var depth = 36
        public var heads = 24
        public var patch = 2
        public var outChannels = 16
        /// `resolution / 8 / patch` of training (512 px) — RoPE positions are rescaled to it.
        public var trainGrid = 32
        public init() {}
    }

    public let config: Config
    private let w: [String: MLXArray]
    private let xEmbedWeight: MLXArray

    public init(weights: [String: MLXArray], config: Config = .init()) {
        w = weights
        self.config = config
        xEmbedWeight = weights["x_embedder.proj.weight"]!.reshaped([config.hidden, -1])
    }

    /// Load `ema_model.safetensors`. Strict: every block's tensors must be present.
    public static func load(url: URL, config: Config = .init(), dtype: DType = .float32) throws -> LightningDiT {
        var weights: [String: MLXArray] = [:]
        for (k, v) in try MLX.loadArrays(url: url) where !k.hasPrefix("feat_rope.") {
            weights[k] = v.asType(dtype)
        }
        let required = ["x_embedder.proj.weight", "t_block.1.weight", "mlp_ca.fc1.weight",
            "final_layer.linear.weight", "blocks.\(config.depth - 1).cross_attn.proj.weight"]
        for k in required where weights[k] == nil {
            throw EngineError.backendFailure("VOSR: \(url.lastPathComponent) has no \(k) — not a VOSR2 1.4B checkpoint")
        }
        return LightningDiT(weights: weights, config: config)
    }

    // MARK: forward

    /// - Parameters:
    ///   - x: `[B, 32, s, s]` (LQ latent ‖ noise), s a multiple of the patch size.
    ///   - t: timestep (1 for VOSR2's one-step flow).
    ///   - features: DINOv2 tokens `[B, M, 1024]` (one row per tile).
    public func callAsFunction(_ x: MLXArray, t: Float, features: MLXArray) -> MLXArray {
        let c = config
        let b = x.dim(0), s = x.dim(2)
        let hw = s / c.patch
        let dtype = xEmbedWeight.dtype
        var h = linear(Self.patchify(x.asType(dtype), patch: c.patch), xEmbedWeight, w["x_embedder.proj.bias"])

        let temb = timestepEmbedding(t).asType(dtype)  // [1, hidden]
        let c0 = linear(silu(temb), "t_block.1")  // [1, 6·hidden]
        let z = linear(geluApproximate(linear(
            MLXFast.layerNorm(features.asType(dtype), weight: w["layer_norm.weight"], bias: w["layer_norm.bias"], eps: 1e-5),
            "mlp_ca.fc1")), "mlp_ca.fc2")

        let (cos, sin) = Self.rope(halfHeadDim: c.hidden / c.heads / 2, grid: hw, trainGrid: c.trainGrid)
        let rope = (cos.asType(dtype), sin.asType(dtype))
        for i in 0..<c.depth { h = block(h, c0: c0, z: z, rope: rope, "blocks.\(i)") }

        let mod = linear(silu(temb), "final_layer.adaLN_modulation.1")
        let shift = mod[0..., ..<c.hidden].expandedDimensions(axis: 1)
        let scale = mod[0..., c.hidden...].expandedDimensions(axis: 1)
        h = rms(h, "final_layer.norm_final") * (1 + scale) + shift
        h = linear(h, "final_layer.linear")  // [B, hw², p·p·16]
        // nhwpqc → nchpwq
        return h.reshaped([b, hw, hw, c.patch, c.patch, c.outChannels])
            .transposed(0, 5, 1, 3, 2, 4)
            .reshaped([b, c.outChannels, s, s])
            .asType(.float32)
    }

    /// Sinusoidal embedding (cos first, 256 dims) → MLP.
    func timestepEmbedding(_ t: Float) -> MLXArray {
        let half = 128
        let freqs = (0..<half).map { Foundation.exp(-Foundation.log(10_000) * Float($0) / Float(half)) }
        let args = freqs.map { t * $0 }
        let emb = MLXArray(args.map(Foundation.cos) + args.map(Foundation.sin), [1, 2 * half])
        return linear(silu(linear(emb.asType(xEmbedWeight.dtype), "t_embedder.mlp.0")), "t_embedder.mlp.2")
    }

    /// EVA-02 2D rotary table for a `grid×grid` token grid: `[grid², 4·halfHeadDim/2…]` = the
    /// row frequencies then the column frequencies, each repeated pairwise (interleaved rotate).
    public static func rope(halfHeadDim: Int, grid: Int, trainGrid: Int) -> (MLXArray, MLXArray) {
        let nf = halfHeadDim / 2
        let freqs = (0..<nf).map { 1 / pow(Float(10_000), Float(2 * $0) / Float(halfHeadDim)) }
        let pos = (0..<grid).map { Float($0) / Float(grid) * Float(trainGrid) }
        // per-axis table [grid, halfHeadDim] with each frequency twice
        let axis: [[Float]] = pos.map { p in freqs.flatMap { [p * $0, p * $0] } }
        var cos = [Float](), sin = [Float]()
        cos.reserveCapacity(grid * grid * 2 * halfHeadDim)
        sin.reserveCapacity(grid * grid * 2 * halfHeadDim)
        for i in 0..<grid {
            for j in 0..<grid {
                let row = axis[i] + axis[j]
                cos += row.map(Foundation.cos)
                sin += row.map(Foundation.sin)
            }
        }
        return (MLXArray(cos, [grid * grid, 2 * halfHeadDim]), MLXArray(sin, [grid * grid, 2 * halfHeadDim]))
    }

    // MARK: pieces

    private func linear(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray?) -> MLXArray {
        let y = x.matmul(weight.transposed())
        return bias.map { y + $0 } ?? y
    }

    private func linear(_ x: MLXArray, _ p: String) -> MLXArray { linear(x, w["\(p).weight"]!, w["\(p).bias"]) }

    private func rms(_ x: MLXArray, _ p: String) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: w["\(p).weight"]!.asType(.float32), eps: 1e-6).asType(x.dtype)
    }

    /// `rotate_half` on interleaved pairs: (x0, x1) → (−x1, x0).
    private func rotateInterleaved(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = x.reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
        let x1 = pairs[.ellipsis, 0 ..< 1], x2 = pairs[.ellipsis, 1 ..< 2]
        return concatenated([-x2, x1], axis: -1).reshaped(shape)
    }

    private func block(
        _ x: MLXArray, c0: MLXArray, z: MLXArray, rope: (MLXArray, MLXArray), _ p: String
    ) -> MLXArray {
        let c = config
        let b = x.dim(0), n = x.dim(1), hd = c.hidden / c.heads
        let mod = (w["\(p).scale_shift_table"]! + c0.reshaped([6, c.hidden])).expandedDimensions(axis: 1)  // [6,1,C]
        let (shiftMSA, scaleMSA, gateMSA) = (mod[0], mod[1], mod[2])
        let (shiftMLP, scaleMLP, gateMLP) = (mod[3], mod[4], mod[5])

        // Self-attention with QK-norm + RoPE.
        let y = rms(x, "\(p).norm1") * (1 + scaleMSA) + shiftMSA
        let qkv = linear(y, "\(p).attn.qkv").reshaped([b, n, 3, c.heads, hd]).transposed(2, 0, 3, 1, 4)
        var q = rms(qkv[0], "\(p).attn.q_norm"), k = rms(qkv[1], "\(p).attn.k_norm")
        q = q * rope.0 + rotateInterleaved(q) * rope.1
        k = k * rope.0 + rotateInterleaved(k) * rope.1
        let a = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: qkv[2], scale: 1 / Float(hd).squareRoot(), mask: .none)
            .transposed(0, 2, 1, 3).reshaped([b, n, c.hidden])
        var h = x + gateMSA * linear(a, "\(p).attn.proj")

        // Cross-attention to the DINOv2 tokens (no norm, no gate).
        let m = z.dim(1)
        func heads(_ t: MLXArray, _ len: Int) -> MLXArray { t.reshaped([b, len, c.heads, hd]).transposed(0, 2, 1, 3) }
        let cq = rms(heads(linear(h, "\(p).cross_attn.q_linear"), n), "\(p).cross_attn.q_norm")
        let ck = rms(heads(linear(z, "\(p).cross_attn.k_linear"), m), "\(p).cross_attn.k_norm")
        let cv = heads(linear(z, "\(p).cross_attn.v_linear"), m)
        let ca = MLXFast.scaledDotProductAttention(
            queries: cq, keys: ck, values: cv, scale: 1 / Float(hd).squareRoot(), mask: .none)
            .transposed(0, 2, 1, 3).reshaped([b, n, c.hidden])
        h = h + linear(ca, "\(p).cross_attn.proj")

        // SwiGLU MLP.
        let u = linear(rms(h, "\(p).norm2") * (1 + scaleMLP) + shiftMLP, "\(p).mlp.w12")
        let half = u.dim(-1) / 2
        h = h + gateMLP * linear(silu(u[.ellipsis, ..<half]) * u[.ellipsis, half...], "\(p).mlp.w3")
        return h
    }
}

extension LightningDiT {
    /// `[B, C, H, W]` → `[B, (H/p)·(W/p), C·p·p]` tokens, channel-major within a patch
    static func patchify(_ latent: MLXArray, patch: Int) -> MLXArray {
        let s = latent.shape
        let (b, c, h, w) = (s[0], s[1], s[2] / patch, s[3] / patch)
        return latent
            .reshaped([b, c, h, patch, w, patch])
            .transposed(0, 2, 4, 1, 3, 5)  // b h w c ph pw
            .reshaped([b, h * w, c * patch * patch])
    }
}
