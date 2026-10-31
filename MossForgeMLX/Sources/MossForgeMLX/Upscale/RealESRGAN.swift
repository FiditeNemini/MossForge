
import Foundation
import MossForgeCore
import MLX

/// A super-resolution upscaler for the MrFlow sampler's pixel-space stage. Implementations are
/// ported from the standard `.pth` architectures and verified bit-for-bit against spandrel (the
/// canonical arch zoo) — see `Scripts/upscaler_convert.py` + `moss-mlx esrgan-parity`.
public protocol Upscaler {
    /// Native integer upscale factor (2 or 4).
    var scale: Int { get }
    /// Upscale an `NHWC` image `[1, H, W, 3]` in `[0,1]` → `[1, scale·H, scale·W, 3]`. The result
    /// may slightly exceed `[0,1]`; the caller clips. H·W must be even for a ×2 RRDBNet.
    func upscale(_ image: MLXArray) -> MLXArray
}

/// Builds the right `Upscaler` for a converted safetensors by sniffing its canonical keys.
public enum UpscalerFactory {
    public static func load(url: URL) throws -> any Upscaler {
        let w = try MLX.loadArrays(url: url).mapValues { $0.asType(.float32) }
        if w["conv_1.weight"] != nil, w["block_1.c1_r.weight"] != nil, w["upsampler.weight"] != nil {
            return SPANUpscaler(weights: w)
        }
        if w["conv_first.weight"] != nil {
            return RRDBNetUpscaler(weights: w)
        }
        throw EngineError.backendFailure(
            "unrecognized upscaler format: \(url.lastPathComponent)")
    }
}

/// 2D conv from a torch-layout weight `[O,I,kH,kW]` (Swift transposes to MLX `[O,kH,kW,I]`),
/// same-size padding derived from the kernel, bias broadcast over the trailing channel dim.
func upConv(_ x: MLXArray, _ weightPyTorch: MLXArray, _ bias: MLXArray) -> MLXArray {
    let pad = (weightPyTorch.shape[2] - 1) / 2
    return conv2d(x, weightPyTorch.transposed(0, 2, 3, 1), stride: 1, padding: IntOrPair((pad, pad))) + bias
}

/// Nearest-neighbour ×2 upsample on the spatial dims of an `NHWC` tensor (pixel replication),
/// matching `F.interpolate(mode: "nearest", scale_factor: 2)`.
func upNearest2x(_ x: MLXArray) -> MLXArray {
    repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
}

/// **RRDBNet** (ESRGAN / Real-ESRGAN, both old-arch `model.X` and new-arch keys — normalized to
/// canonical keys by the converter). Handles ×2 (12-channel pixel-unshuffled input) and ×4
/// (3-channel direct input); both run the body then two nearest ×2 upsamples. Verified bit-exact
/// vs spandrel (RealESRGAN_x2plus, 4x_foolhardy_Remacri).
public struct RRDBNetUpscaler: Upscaler {
    private let w: [String: MLXArray]
    public let scale: Int
    private let numBlocks: Int

    init(weights: [String: MLXArray]) {
        w = weights
        // conv_first in-channels: 12 ⇒ ×2 (pixel-unshuffle), 3 ⇒ ×4 (direct).
        scale = w["conv_first.weight"]!.shape[1] == 12 ? 2 : 4
        var n = 0
        while w["body.\(n).rdb1.conv1.weight"] != nil { n += 1 }
        numBlocks = n
    }

    public func upscale(_ image: MLXArray) -> MLXArray {
        var feat = conv("conv_first", scale == 2 ? pixelUnshuffle2(image) : image)
        let trunk = feat
        var body = feat
        for i in 0..<numBlocks {
            body = rrdb(body, "body.\(i)")
            eval(body)  // bound memory across blocks
        }
        feat = trunk + conv("conv_body", body)
        feat = lrelu(conv("conv_up1", upNearest2x(feat)))
        feat = lrelu(conv("conv_up2", upNearest2x(feat)))
        let out = conv("conv_last", lrelu(conv("conv_hr", feat)))
        eval(out)
        return out
    }

    private func conv(_ p: String, _ x: MLXArray) -> MLXArray {
        upConv(x, w["\(p).weight"]!, w["\(p).bias"]!)
    }
    private func lrelu(_ x: MLXArray) -> MLXArray { maximum(x, x * 0.2) }

    private func rdb(_ x: MLXArray, _ p: String) -> MLXArray {
        let x1 = lrelu(conv("\(p).conv1", x))
        let x2 = lrelu(conv("\(p).conv2", concatenated([x, x1], axis: 3)))
        let x3 = lrelu(conv("\(p).conv3", concatenated([x, x1, x2], axis: 3)))
        let x4 = lrelu(conv("\(p).conv4", concatenated([x, x1, x2, x3], axis: 3)))
        let x5 = conv("\(p).conv5", concatenated([x, x1, x2, x3, x4], axis: 3))
        return x5 * 0.2 + x
    }
    private func rrdb(_ x: MLXArray, _ p: String) -> MLXArray {
        let o = rdb(rdb(rdb(x, "\(p).rdb1"), "\(p).rdb2"), "\(p).rdb3")
        return o * 0.2 + x
    }

    /// `F.pixel_unshuffle(x, 2)` in NHWC: `[N,H,W,C]→[N,H/2,W/2,C*4]`, channel `c*4+i*2+j`.
    private func pixelUnshuffle2(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        return x.reshaped([s[0], s[1] / 2, 2, s[2] / 2, 2, s[3]])
            .transposed(0, 1, 3, 5, 2, 4)
            .reshaped([s[0], s[1] / 2, s[2] / 2, s[3] * 4])
    }
}
