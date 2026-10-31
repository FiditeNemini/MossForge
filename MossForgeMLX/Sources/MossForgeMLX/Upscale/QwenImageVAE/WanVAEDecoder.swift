
import Foundation
import MossForgeCore
import MossForgeEngine
import MLX
import MLXFast
import MLXNN

/// The Qwen Image (Wan-style) VAE **decoder**, ported to MLX for single-image
/// latents (T=1). For one temporal frame the Wan `feat_cache` is inert: every
/// causal conv uses plain full padding and the `upsample3d` `time_conv` is skipped,
/// so both upsample modes reduce to spatial-only (nearest-exact 2× + Conv2d). All
/// math is grounded on diffusers' `AutoencoderKLWan` source; weights use the
/// original Wan naming (`conv1`/`middle`/`upsamples`/`head`) as on disk.
///
/// Verified end-to-end against a torch reference decode of the real weights via
/// `moss-mlx vae-decode`.
public struct WanVAEDecoder {
    private let w: [String: MLXArray]

    /// - Parameter weights: native-keyed VAE tensors (float32) from `WeightLoader`.
    public init(weights: [String: MLXArray]) { self.w = weights }

    /// Load the decoder weights from a VAE safetensors file (bf16 → float32).
    public static func load(url: URL) throws -> WanVAEDecoder {
        let loaded = try WeightLoader.load(url: url, working: .float32)
        return WanVAEDecoder(weights: loaded.arrays)
    }

    /// Decode a latent `z` (NCDHW, 16 channels, T=1) to an image in `[-1, 1]`
    /// (NCDHW, 3 channels, 8× spatial).
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var x = conv(z, "conv2")  // post_quant_conv (1×1×1)
        x = conv(x, "decoder.conv1")  // CausalConv3d 16→384
        x = residual(x, "decoder.middle.0")
        x = attention(x, "decoder.middle.1")
        x = residual(x, "decoder.middle.2")
        for i in 0..<15 {
            let p = "decoder.upsamples.\(i)"
            x = w["\(p).residual.0.gamma"] != nil ? residual(x, p) : resample(x, p)
        }
        x = silu(WanRMSNorm(gamma: w["decoder.head.0.gamma"]!)(x))
        x = conv(x, "decoder.head.2")  // CausalConv3d 96→3
        return clip(x, min: MLXArray(Float(-1)), max: MLXArray(Float(1)))
    }

    // MARK: blocks

    private func conv(_ x: MLXArray, _ p: String) -> MLXArray {
        CausalConv3d(weightPyTorch: w["\(p).weight"]!, bias: w["\(p).bias"])(x)
    }

    private func residual(_ x: MLXArray, _ p: String) -> MLXArray {
        let shortcut = w["\(p).shortcut.weight"].map {
            CausalConv3d(weightPyTorch: $0, bias: w["\(p).shortcut.bias"])
        }
        let block = ResidualBlock3D(
            norm0: WanRMSNorm(gamma: w["\(p).residual.0.gamma"]!),
            conv2: CausalConv3d(weightPyTorch: w["\(p).residual.2.weight"]!, bias: w["\(p).residual.2.bias"]),
            norm3: WanRMSNorm(gamma: w["\(p).residual.3.gamma"]!),
            conv6: CausalConv3d(weightPyTorch: w["\(p).residual.6.weight"]!, bias: w["\(p).residual.6.bias"]),
            shortcut: shortcut)
        return block(x)
    }

    /// Spatial self-attention (single head) over H·W, per the Wan middle block.
    private func attention(_ x: MLXArray, _ p: String) -> MLXArray {
        let s = x.shape
        let (b, c, d, h, wd) = (s[0], s[1], s[2], s[3], s[4])
        let bt = b * d
        var y = x.reshaped([bt, c, h, wd])
        // channel RMS norm (gamma [c,1,1])
        let gamma = w["\(p).norm.gamma"]!.reshaped([1, c, 1, 1])
        let denom = sqrt((y * y).sum(axis: 1, keepDims: true))
        y = y / maximum(denom, MLXArray(Float(1e-12))) * Float(c).squareRoot() * gamma
        var yl = y.transposed(0, 2, 3, 1)  // [bt,h,w,c]
        let qkvW = w["\(p).to_qkv.weight"]!.transposed(0, 2, 3, 1)  // [3c,1,1,c]
        var qkv = conv2d(yl, qkvW) + w["\(p).to_qkv.bias"]!  // [bt,h,w,3c]
        qkv = qkv.reshaped([bt, h * wd, 3 * c])
        let parts = qkv.split(parts: 3, axis: -1)  // each [bt,hw,c]
        let (q, k, v) = (parts[0], parts[1], parts[2])
        let scale = 1.0 / Float(c).squareRoot()
        // Fused single-head SDPA — avoids the [hw, hw] score matrix (17 GB at 2048²).
        let hw = h * wd
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: q.reshaped([bt, 1, hw, c]), keys: k.reshaped([bt, 1, hw, c]),
            values: v.reshaped([bt, 1, hw, c]), scale: scale, mask: .none)
        var o = sdpa.reshaped([bt, h, wd, c])
        let projW = w["\(p).proj.weight"]!.transposed(0, 2, 3, 1)
        o = conv2d(o, projW) + w["\(p).proj.bias"]!
        o = o.transposed(0, 3, 1, 2).reshaped([b, c, d, h, wd])
        _ = yl
        return o + x
    }

    /// Spatial upsample: nearest-exact 2× (pixel replication) + Conv2d(k3, pad1),
    /// halving channels. The `time_conv` (upsample3d) is skipped for T=1 images.
    private func resample(_ x: MLXArray, _ p: String) -> MLXArray {
        let s = x.shape
        let (b, c, d, h, wd) = (s[0], s[1], s[2], s[3], s[4])
        let bt = b * d
        var y = x.reshaped([bt, c, h, wd]).transposed(0, 2, 3, 1)  // [bt,h,w,c]
        y = repeated(y, count: 2, axis: 1)  // 2× H (nearest)
        y = repeated(y, count: 2, axis: 2)  // 2× W
        let cw = w["\(p).resample.1.weight"]!.transposed(0, 2, 3, 1)  // [c/2,3,3,c]
        y = conv2d(y, cw, stride: 1, padding: 1) + w["\(p).resample.1.bias"]!
        let co = y.shape[3]
        return y.transposed(0, 3, 1, 2).reshaped([b, co, d, h * 2, wd * 2])
    }
}
