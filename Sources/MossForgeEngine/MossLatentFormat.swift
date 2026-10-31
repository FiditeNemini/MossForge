import Foundation
import MossForgeCore

/// The bridge between FLUX.2 VAE latents and MOSS DiT tokens, grounded in ComfyUI
/// `comfy/ldm/models/autoencoder.py` (`AutoencodingEngine` with `batch_norm_latent`) — the path
/// the Comfy MOSS node takes (`latent_formats.Flux2`: 128 ch, ×16, no extra scaling):
///
/// - **encode**: VAE posterior **mean** `[32, H/8, W/8]` → 2×2 patchify
///   `"c (i pi) (j pj) -> (c pi pj) i j"` → `[128, H/16, W/16]` → BatchNorm (eval) with the
///   file's `bn.running_mean/var`, eps **1e-4**, no affine.
/// - **decode**: the exact inverse, then `post_quant_conv` → decoder.
///
/// The DiT flattens the normalised `[128, H/16, W/16]` row-major into tokens, so a token's 128
/// values are channel-major then (dy, dx). These CPU versions are the index-level reference the
/// MLX port is checked against (layout `[C, H, W]`, one image).
public enum MossLatentFormat {
    public static let bnEpsilon: Float = 1e-4
    public static let vaeChannels = 32
    public static let tokenChannels = 128
    /// Pixels per VAE latent cell / per DiT token.
    public static let vaeStride = 8
    public static let tokenStride = 16

    /// Channel of `(c, dy, dx)` in the patchified latent: `(c pi pj)` ⇒ `c·4 + dy·2 + dx`.
    public static func patchedChannel(c: Int, dy: Int, dx: Int) -> Int { c * 4 + dy * 2 + dx }

    public static func vaeLatentShape(width: Int, height: Int) -> [Int] {
        [vaeChannels, height / vaeStride, width / vaeStride]
    }

    public static func tokenLatentShape(width: Int, height: Int) -> [Int] {
        [tokenChannels, height / tokenStride, width / tokenStride]
    }

    /// `[C, H, W]` → `[4C, H/2, W/2]`.
    public static func patchify(_ x: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let (ph, pw) = (height / 2, width / 2)
        var out = [Float](repeating: 0, count: x.count)
        for c in 0..<channels {
            for i in 0..<ph {
                for dy in 0..<2 {
                    for j in 0..<pw {
                        for dx in 0..<2 {
                            out[(patchedChannel(c: c, dy: dy, dx: dx) * ph + i) * pw + j] =
                                x[(c * height + 2 * i + dy) * width + 2 * j + dx]
                        }
                    }
                }
            }
        }
        return out
    }

    /// `[4C, H/2, W/2]` → `[C, H, W]` (`channels`, `height`, `width` describe the output).
    public static func unpatchify(_ p: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let (ph, pw) = (height / 2, width / 2)
        var out = [Float](repeating: 0, count: p.count)
        for c in 0..<channels {
            for i in 0..<ph {
                for dy in 0..<2 {
                    for j in 0..<pw {
                        for dx in 0..<2 {
                            out[(c * height + 2 * i + dy) * width + 2 * j + dx] =
                                p[(patchedChannel(c: c, dy: dy, dx: dx) * ph + i) * pw + j]
                        }
                    }
                }
            }
        }
        return out
    }

    /// Eval-mode BatchNorm over channel-major `[C, …]`: `(x − mean) / √(var + eps)`.
    public static func normalize(_ x: [Float], mean: [Float], variance: [Float]) -> [Float] {
        let spatial = x.count / max(mean.count, 1)
        return x.enumerated().map { n, v in
            let c = n / spatial
            return (v - mean[c]) / (variance[c] + bnEpsilon).squareRoot()
        }
    }

    /// Inverse of `normalize`: `x · √(var + eps) + mean`.
    public static func denormalize(_ x: [Float], mean: [Float], variance: [Float]) -> [Float] {
        let spatial = x.count / max(mean.count, 1)
        return x.enumerated().map { n, v in
            let c = n / spatial
            return v * (variance[c] + bnEpsilon).squareRoot() + mean[c]
        }
    }
}

/// Geometry of an LDM `AutoencoderKL` (BFL key names: `encoder.down.N.block.M`, `decoder.up.N`,
/// `encoder.quant_conv`, `decoder.post_quant_conv`, `bn.*`), derived from the header the way
/// ComfyUI's `sd.py` does: `ch = conv_in` width, `ch_mult` from each level's block width,
/// `z = decoder.conv_in` input. GroupNorm 32 (eps 1e-6), mid attention in both halves.
public struct FluxVAEConfig: Sendable, Equatable {
    public var baseChannels: Int
    public var channelMultipliers: [Int]
    /// Res blocks per encoder level (the decoder has one more).
    public var resBlocks: Int
    public var zChannels: Int
    public var batchNormLatent: Bool

    public var downscale: Int { 1 << (channelMultipliers.count - 1) }

    /// FLUX.2 VAE (klein / dev): 128 · [1,2,4,4], 2 res blocks, 32 latent channels, BN latent.
    public static let flux2 = FluxVAEConfig(
        baseChannels: 128, channelMultipliers: [1, 2, 4, 4], resBlocks: 2, zChannels: 32, batchNormLatent: true)

    public init(baseChannels: Int, channelMultipliers: [Int], resBlocks: Int, zChannels: Int, batchNormLatent: Bool) {
        self.baseChannels = baseChannels
        self.channelMultipliers = channelMultipliers
        self.resBlocks = resBlocks
        self.zChannels = zChannels
        self.batchNormLatent = batchNormLatent
    }

    public init(descriptors: [TensorDescriptor]) throws {
        let shapes = Dictionary(descriptors.map { ($0.name, $0.shape) }, uniquingKeysWith: { a, _ in a })
        guard let ch = shapes["encoder.conv_in.weight"]?.first, ch > 0,
            let z = shapes["decoder.conv_in.weight"].flatMap({ $0.count > 1 ? $0[1] : nil })
        else { throw EngineError.invalidRequest("not an LDM AutoencoderKL (no encoder/decoder conv_in)") }
        var mults: [Int] = []
        while let width = shapes["encoder.down.\(mults.count).block.0.conv1.weight"]?.first {
            mults.append(width / ch)
        }
        var blocks = 0
        while shapes["encoder.down.0.block.\(blocks).conv1.weight"] != nil { blocks += 1 }
        guard !mults.isEmpty, blocks > 0 else { throw EngineError.invalidRequest("VAE has no encoder levels") }
        self.init(
            baseChannels: ch, channelMultipliers: mults, resBlocks: blocks, zChannels: z,
            batchNormLatent: shapes["bn.running_mean"] != nil)
    }
}
