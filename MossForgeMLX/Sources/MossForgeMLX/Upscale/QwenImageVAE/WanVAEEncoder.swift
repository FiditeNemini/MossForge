
import Foundation
import MossForgeCore
import MossForgeEngine
import MLX
import MLXFast
import MLXNN

/// The Qwen Image (Wan-style) VAE **encoder** — the mirror of `WanVAEDecoder`, ported to
/// MLX for single-image latents (T=1). For one temporal frame the Wan `feat_cache` is
/// inert: every causal conv uses plain full padding and the `downsample3d` `time_conv` is
/// skipped (the sole chunk just caches `x`), so both downsample modes reduce to a spatial
/// `ZeroPad2d((0,1,0,1))` + `Conv2d(stride 2)`. Grounded on diffusers' `AutoencoderKLWan`;
/// original Wan key names (`encoder.conv1`/`downsamples`/`middle`/`head` + the top-level
/// `conv1` quant conv). Verified against a torch reference via `moss-mlx vae-encode`.
public struct WanVAEEncoder {
    private let w: [String: MLXArray]

    /// - Parameter weights: native-keyed VAE tensors (float32) from `WeightLoader`.
    public init(weights: [String: MLXArray]) { self.w = weights }

    /// Load the encoder weights from a VAE safetensors file (bf16 → float32).
    public static func load(url: URL) throws -> WanVAEEncoder {
        let loaded = try WeightLoader.load(url: url, working: .float32)
        return WanVAEEncoder(weights: loaded.arrays)
    }

    /// Encode an image `x` (NCDHW, 3 channels, T=1, values in `[-1, 1]`) to the latent
    /// distribution **mean** `mu` (NCDHW, 16 channels, 8× smaller spatially) — the
    /// deterministic latent used for img2img.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x, "encoder.conv1")  // CausalConv3d 3→96
        for i in 0..<11 {
            let p = "encoder.downsamples.\(i)"
            h = w["\(p).residual.0.gamma"] != nil ? residual(h, p) : downsample(h, p)
        }
        h = residual(h, "encoder.middle.0")
        h = attention(h, "encoder.middle.1")
        h = residual(h, "encoder.middle.2")
        h = silu(WanRMSNorm(gamma: w["encoder.head.0.gamma"]!)(h))
        h = conv(h, "encoder.head.2")  // CausalConv3d 384→32
        let enc = conv(h, "conv1")  // quant_conv 32→32 (1×1×1)
        return enc[0..., 0..<16]  // distribution mean (channels 0–15); logvar is 16–31
    }

    // MARK: blocks (mirror of the decoder)

    private func conv(_ x: MLXArray, _ p: String) -> MLXArray {
        CausalConv3d(weightPyTorch: w["\(p).weight"]!, bias: w["\(p).bias"])(x)
    }

    private func residual(_ x: MLXArray, _ p: String) -> MLXArray {
        let shortcut = w["\(p).shortcut.weight"].map {
            CausalConv3d(weightPyTorch: $0, bias: w["\(p).shortcut.bias"])
        }
        return ResidualBlock3D(
            norm0: WanRMSNorm(gamma: w["\(p).residual.0.gamma"]!),
            conv2: CausalConv3d(weightPyTorch: w["\(p).residual.2.weight"]!, bias: w["\(p).residual.2.bias"]),
            norm3: WanRMSNorm(gamma: w["\(p).residual.3.gamma"]!),
            conv6: CausalConv3d(weightPyTorch: w["\(p).residual.6.weight"]!, bias: w["\(p).residual.6.bias"]),
            shortcut: shortcut)(x)
    }

    /// Spatial self-attention (single head) over H·W — identical to the decoder's Wan
    /// middle block.
    private func attention(_ x: MLXArray, _ p: String) -> MLXArray {
        let s = x.shape
        let (b, c, d, h, wd) = (s[0], s[1], s[2], s[3], s[4])
        let bt = b * d
        var y = x.reshaped([bt, c, h, wd])
        let gamma = w["\(p).norm.gamma"]!.reshaped([1, c, 1, 1])
        let denom = sqrt((y * y).sum(axis: 1, keepDims: true))
        y = y / maximum(denom, MLXArray(Float(1e-12))) * Float(c).squareRoot() * gamma
        let yl = y.transposed(0, 2, 3, 1)  // [bt,h,w,c]
        let qkvW = w["\(p).to_qkv.weight"]!.transposed(0, 2, 3, 1)  // [3c,1,1,c]
        var qkv = conv2d(yl, qkvW) + w["\(p).to_qkv.bias"]!  // [bt,h,w,3c]
        qkv = qkv.reshaped([bt, h * wd, 3 * c])
        let parts = qkv.split(parts: 3, axis: -1)
        let (q, k, v) = (parts[0], parts[1], parts[2])
        let scale = 1.0 / Float(c).squareRoot()
        let hw = h * wd
        let sdpa = MLXFast.scaledDotProductAttention(
            queries: q.reshaped([bt, 1, hw, c]), keys: k.reshaped([bt, 1, hw, c]),
            values: v.reshaped([bt, 1, hw, c]), scale: scale, mask: .none)
        var o = sdpa.reshaped([bt, h, wd, c])
        let projW = w["\(p).proj.weight"]!.transposed(0, 2, 3, 1)
        o = conv2d(o, projW) + w["\(p).proj.bias"]!
        return o.transposed(0, 3, 1, 2).reshaped([b, c, d, h, wd]) + x
    }

    /// Spatial downsample: `ZeroPad2d((0,1,0,1))` + `Conv2d(k3, stride 2)`, channels
    /// preserved. The `time_conv` (downsample3d) is skipped for T=1 images.
    private func downsample(_ x: MLXArray, _ p: String) -> MLXArray {
        let s = x.shape
        let (b, c, d, h, wd) = (s[0], s[1], s[2], s[3], s[4])
        let bt = b * d
        var y = x.reshaped([bt, c, h, wd]).transposed(0, 2, 3, 1)  // [bt,h,w,c]
        // Asymmetric pad: W-right +1, H-bottom +1 (torch ZeroPad2d((0,1,0,1))).
        y = MLX.padded(
            y, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))])
        let cw = w["\(p).resample.1.weight"]!.transposed(0, 2, 3, 1)  // [c,3,3,c]
        y = conv2d(y, cw, stride: 2, padding: 0) + w["\(p).resample.1.bias"]!
        let (co, nh, nw) = (y.shape[3], y.shape[1], y.shape[2])
        return y.transposed(0, 3, 1, 2).reshaped([b, co, d, nh, nw])
    }
}
