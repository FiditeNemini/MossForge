
import Foundation
import MLX

/// **SPAN** (Swift Parameter-free Attention Network) super-resolution — the architecture behind
/// the "ClearReality" upscalers. Ported to MLX from spandrel's reference and verified bit-exact
/// (`moss-mlx esrgan-parity`). Each `Conv3XC` reparam block collapses to its fused `eval_conv` (a
/// single 3×3) at inference, so the converter exports just those; here every conv is plain.
///
/// Forward (from spandrel `SPAN.forward`):
///   x  = (img − rgb_mean) · 255                                   (internal normalization)
///   f  = conv_1(x)
///   for b in 1…6: (out_b, out1_b) = SPAB(prev)                    (out1 = block's first conv)
///   c2 = conv_2(out_b6)
///   out = conv_cat([f, c2, out_b1, out1_b6])                      (1×1, 4·C → C)
///   img = PixelShuffle(upsampler(out), scale)
/// SPAB(x): o1=c1(x); o2=c2(silu(o1)); o3=c3(silu(o2)); att=sigmoid(o3)−0.5; return (o3+x)·att, o1
public struct SPANUpscaler: Upscaler {
    private let w: [String: MLXArray]
    public let scale: Int
    /// SPAN's fixed input normalization constants (ImageNet-ish mean, img_range 255).
    private static let rgbMean: [Float] = [0.4488, 0.4371, 0.4040]

    init(weights: [String: MLXArray]) {
        w = weights
        // upsampler out-channels = 3·scale² ⇒ scale = √(out/3).
        let out = weights["upsampler.weight"]!.shape[0]
        scale = Int((Double(out) / 3.0).squareRoot().rounded())
    }

    public func upscale(_ image: MLXArray) -> MLXArray {
        let mean = MLXArray(Self.rgbMean).reshaped([1, 1, 1, 3])
        let x = (image - mean) * 255.0

        let f = conv("conv_1", x)
        let (b1, _) = spab(f, "block_1")
        let (b2, _) = spab(b1, "block_2")
        let (b3, _) = spab(b2, "block_3")
        let (b4, _) = spab(b3, "block_4")
        let (b5, _) = spab(b4, "block_5")
        let (b6, b6o1) = spab(b5, "block_6")
        let c2 = conv("conv_2", b6)
        let cat = concatenated([f, c2, b1, b6o1], axis: 3)
        let out = conv("conv_cat", cat)  // 1×1
        let up = pixelShuffle(conv("upsampler", out), scale)  // 3×3 conv → PixelShuffle
        eval(up)
        return up
    }

    private func conv(_ p: String, _ x: MLXArray) -> MLXArray {
        upConv(x, w["\(p).weight"]!, w["\(p).bias"]!)
    }
    private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// One SPAB: returns `(attention-gated output, silu(first-conv output))`. The second value is
    /// the block's `out1` — spandrel activates it in place (`SiLU(inplace=True)`), so the value
    /// that reaches the final concat is `silu(c1_r(x))`, not the raw conv output.
    private func spab(_ x: MLXArray, _ p: String) -> (MLXArray, MLXArray) {
        let o1 = conv("\(p).c1_r", x)
        let o1a = silu(o1)
        let o2 = conv("\(p).c2_r", o1a)
        let o3 = conv("\(p).c3_r", silu(o2))
        let att = sigmoid(o3) - 0.5
        return ((o3 + x) * att, o1a)
    }

    /// `nn.PixelShuffle(scale)` in NHWC: `[N,H,W,C·s²]→[N,H·s,W·s,C]`, channel `c·s²+i·s+j`.
    private func pixelShuffle(_ x: MLXArray, _ s: Int) -> MLXArray {
        let sh = x.shape
        let (n, h, wd, c) = (sh[0], sh[1], sh[2], sh[3] / (s * s))
        return x.reshaped([n, h, wd, c, s, s])
            .transposed(0, 1, 4, 2, 5, 3)  // → [N, H, s_i, W, s_j, C]
            .reshaped([n, h * s, wd * s, c])
    }
}
