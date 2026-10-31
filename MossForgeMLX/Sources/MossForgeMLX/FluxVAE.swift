import Foundation
import MLX
import MLXFast
import MLXNN
import MossForgeCore
import MossForgeEngine

/// FLUX.2 VAE (LDM `AutoencoderKL`, BFL key names) + the MOSS latent bridge, ported from the
/// ComfyUI modules transcribed in `Scripts/vae_ref.py`:
///
/// - `encode`: pixels `[1,3,H,W]` in [0,1] → `×2−1` → encoder → `quant_conv` → posterior **mean**
///   `[1,32,H/8,W/8]` → 2×2 patchify `(c pi pj)` → BatchNorm (eps 1e-4) → DiT tokens `[1,128,H/16,W/16]`.
/// - `decode`: the inverse → `post_quant_conv` → decoder → `(x+1)/2` clamped.
///
/// Public tensors are NCHW (torch layout, so fixtures compare directly); compute is NHWC inside.
/// GroupNorm 32 (eps 1e-6) and attention run in F32 whatever the working dtype.
public final class FluxVAE {
    public let config: FluxVAEConfig
    public let dtype: DType
    private var w: [String: MLXArray]
    private let bnMean: MLXArray  // [1,128,1,1] F32
    private let bnStd: MLXArray  // √(var + eps)

    public init(weights: [String: MLXArray], config: FluxVAEConfig = .flux2, dtype: DType = .float32) throws {
        self.config = config
        self.dtype = dtype
        guard let mean = weights["bn.running_mean"], let variance = weights["bn.running_var"] else {
            throw WeightLoaderError.missingTensors(["bn.running_mean", "bn.running_var"])
        }
        bnMean = mean.asType(.float32).reshaped([1, -1, 1, 1])
        bnStd = sqrt(variance.asType(.float32) + MossLatentFormat.bnEpsilon).reshaped([1, -1, 1, 1])
        var prepared: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix("encoder.") || key.hasPrefix("decoder.") {
            // torch conv [O,I,kh,kw] → MLX [O,kh,kw,I]; norms stay F32, convs take the working dtype.
            if value.ndim == 4 {
                prepared[key] = value.transposed(0, 2, 3, 1).asType(dtype)
            } else if key.contains("norm") {
                prepared[key] = value.asType(.float32)
            } else {
                prepared[key] = value.asType(dtype)
            }
        }
        w = prepared
    }

    public static func load(url: URL, dtype: DType = .float32) throws -> FluxVAE {
        let (descriptors, _) = try SafetensorsHeader.readDescriptors(from: url)
        let config = try FluxVAEConfig(descriptors: descriptors)
        return try FluxVAE(weights: try WeightLoader.load(url: url).arrays, config: config, dtype: dtype)
    }

    // MARK: - Public pipeline (NCHW)

    /// Pixels `[1,3,H,W]` in [0,1] → normalised DiT tokens `[1,128,H/16,W/16]` (F32).
    public func encode(_ image: MLXArray) -> MLXArray {
        normalize(encodeMean(image))
    }

    /// Pixels `[1,3,H,W]` in [0,1] → VAE posterior mean `[1,32,H/8,W/8]` (F32).
    public func encodeMean(_ image: MLXArray) -> MLXArray {
        let x = (image.asType(.float32) * 2 - 1).transposed(0, 2, 3, 1)
        return encoder(x).transposed(0, 3, 1, 2)
    }

    /// Normalised DiT tokens `[1,128,h,w]` → pixels `[1,3,16h,16w]` in [0,1].
    public func decode(_ tokens: MLXArray) -> MLXArray {
        clip((decodeRaw(tokens) + 1) / 2, min: 0, max: 1)
    }

    /// Decoder output before the `(x+1)/2` clamp — ≈[-1, 1].
    public func decodeRaw(_ tokens: MLXArray) -> MLXArray {
        decoder(denormalize(tokens).transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }

    /// Posterior mean `[1,32,H,W]` → patchify → BatchNorm → `[1,128,H/2,W/2]`.
    public func normalize(_ z: MLXArray) -> MLXArray {
        (Self.patchify(z.asType(.float32)) - bnMean) / bnStd
    }

    /// Tokens `[1,128,h,w]` → BN-denormalise → unpatchify → `[1,32,2h,2w]`.
    public func denormalize(_ tokens: MLXArray) -> MLXArray {
        Self.unpatchify(tokens.asType(.float32) * bnStd + bnMean)
    }

    /// `"b c (i pi) (j pj) -> b (c pi pj) i j"` — matches `MossLatentFormat.patchify`.
    public static func patchify(_ z: MLXArray) -> MLXArray {
        let (b, c, h, wd) = (z.dim(0), z.dim(1), z.dim(2), z.dim(3))
        return z.reshaped([b, c, h / 2, 2, wd / 2, 2]).transposed(0, 1, 3, 5, 2, 4).reshaped([b, c * 4, h / 2, wd / 2])
    }

    /// `"b (c pi pj) i j -> b c (i pi) (j pj)"`.
    public static func unpatchify(_ p: MLXArray) -> MLXArray {
        let (b, c4, h, wd) = (p.dim(0), p.dim(1), p.dim(2), p.dim(3))
        return p.reshaped([b, c4 / 4, 2, 2, h, wd]).transposed(0, 1, 4, 2, 5, 3).reshaped([b, c4 / 4, h * 2, wd * 2])
    }

    // MARK: - Encoder / decoder (NHWC)

    private func encoder(_ x: MLXArray) -> MLXArray {
        var h = conv(x.asType(dtype), "encoder.conv_in")
        let levels = config.channelMultipliers.count
        for i in 0..<levels {
            for b in 0..<config.resBlocks { h = resblock(h, "encoder.down.\(i).block.\(b)") }
            if i != levels - 1 {
                // Asymmetric (0,1,0,1) zero pad, then a stride-2 valid conv.
                h = padded(h, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
                h = conv(h, "encoder.down.\(i).downsample.conv", stride: 2, padding: 0)
            }
        }
        h = resblock(h, "encoder.mid.block_1")
        h = attention(h, "encoder.mid.attn_1")
        h = resblock(h, "encoder.mid.block_2")
        h = conv(silu(groupNorm(h, "encoder.norm_out")), "encoder.conv_out")
        let moments = conv(h, "encoder.quant_conv", padding: 0).asType(.float32)
        return moments[0..., 0..., 0..., ..<(moments.dim(3) / 2)]  // posterior mode = mean
    }

    private func decoder(_ z: MLXArray) -> MLXArray {
        var h = conv(z.asType(dtype), "decoder.post_quant_conv", padding: 0)
        h = conv(h, "decoder.conv_in")
        h = resblock(h, "decoder.mid.block_1")
        h = attention(h, "decoder.mid.attn_1")
        h = resblock(h, "decoder.mid.block_2")
        for i in (0..<config.channelMultipliers.count).reversed() {
            for b in 0...config.resBlocks { h = resblock(h, "decoder.up.\(i).block.\(b)") }
            if i != 0 {
                h = repeated(repeated(h, count: 2, axis: 1), count: 2, axis: 2)  // nearest ×2
                h = conv(h, "decoder.up.\(i).upsample.conv")
            }
            // Large images: one command buffer per level, so none runs into the GPU watchdog.
            if h.dim(1) * h.dim(2) > Self.chunkAbove { eval(h) }
        }
        return conv(silu(groupNorm(h, "decoder.norm_out")), "decoder.conv_out").asType(.float32)
    }

    // MARK: - Blocks

    private func conv(_ x: MLXArray, _ p: String, stride: Int = 1, padding: Int = 1) -> MLXArray {
        conv2d(x, w["\(p).weight"]!, stride: .init(stride), padding: .init(padding)) + w["\(p).bias"]!
    }

    /// GroupNorm(32, eps 1e-6, affine) over NHWC, statistics in F32.
    private func groupNorm(_ x: MLXArray, _ p: String, groups: Int = 32) -> MLXArray {
        let (b, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let g = x.asType(.float32).reshaped([b, h, wd, groups, c / groups])
        let mean = g.mean(axes: [1, 2, 4], keepDims: true)
        let variance = g.variance(axes: [1, 2, 4], keepDims: true)
        let normed = ((g - mean) * rsqrt(variance + 1e-6)).reshaped([b, h, wd, c])
        return (normed * w["\(p).weight"]! + w["\(p).bias"]!).asType(dtype)
    }

    private func resblock(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = conv(silu(groupNorm(x, "\(p).norm1")), "\(p).conv1")
        h = conv(silu(groupNorm(h, "\(p).norm2")), "\(p).conv2")
        let shortcut = w["\(p).nin_shortcut.weight"] != nil ? conv(x, "\(p).nin_shortcut", padding: 0) : x
        return shortcut + h
    }

    /// Above this many tokens (a 1024² image's mid block), attention runs in `attentionChunk`-row
    /// query chunks, one command buffer each: one SDPA over the 65 536 tokens of a 2048² mid block runs
    /// long enough for macOS's GPU watchdog to kill it ("Impacting Interactivity"; 16 384-row chunks
    /// still did). Chunking the queries is exact — each row's softmax still spans every key.
    static let chunkAbove = 16384
    static let attentionChunk = 2048

    /// Single-head self-attention over H·W (scale c^-½), fused SDPA so the score matrix never
    /// materialises (it would be 1 GB at 1024², 16 GB at 2048²).
    private func attention(_ x: MLXArray, _ p: String) -> MLXArray {
        let (b, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let y = groupNorm(x, "\(p).norm")
        func project(_ n: String) -> MLXArray {
            conv(y, "\(p).\(n)", padding: 0).asType(.float32).reshaped([b, 1, h * wd, c])
        }
        let (q, k, v) = (project("q"), project("k"), project("v"))
        let n = h * wd
        let o: MLXArray
        if n <= Self.chunkAbove {
            o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(c).squareRoot(), mask: .none)
        } else {
            eval(k, v)
            var parts: [MLXArray] = []
            for start in stride(from: 0, to: n, by: Self.attentionChunk) {
                let part = MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., start..<min(start + Self.attentionChunk, n)], keys: k, values: v,
                    scale: 1 / Float(c).squareRoot(), mask: .none)
                eval(part)
                parts.append(part)
            }
            o = concatenated(parts, axis: 2)
        }
        return x + conv(o.reshaped([b, h, wd, c]).asType(dtype), "\(p).proj_out", padding: 0)
    }
}
