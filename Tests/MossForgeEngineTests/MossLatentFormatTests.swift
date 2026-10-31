import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// The FLUX.2 latent ↔ DiT-token bridge, grounded in ComfyUI `comfy/ldm/models/autoencoder.py`
/// (`batch_norm_latent`): encode = posterior mean → `rearrange("c (i pi) (j pj) -> (c pi pj) i j")`
/// → `batch_norm(running_mean, running_var, eps 1e-4)`; decode is the exact inverse. The MLX port
/// must match these CPU references index for index.
@Suite("MossLatentFormat — 2×2 patchify + BN normalise (Comfy Flux2 parity)")
struct MossLatentFormatTests {
    /// [C, H, W] with every value distinct, so any index slip shows.
    private func ramp(_ c: Int, _ h: Int, _ w: Int) -> [Float] { (0..<(c * h * w)).map(Float.init) }

    @Test func patchedChannelOrderIsChannelMajorThenRowThenColumn() {
        #expect(MossLatentFormat.patchedChannel(c: 0, dy: 0, dx: 1) == 1)
        #expect(MossLatentFormat.patchedChannel(c: 0, dy: 1, dx: 0) == 2)
        #expect(MossLatentFormat.patchedChannel(c: 5, dy: 1, dx: 1) == 23)
    }

    @Test func patchifyMatchesTheComfyRearrange() {
        let (c, h, w) = (3, 4, 6)
        let x = ramp(c, h, w)
        let p = MossLatentFormat.patchify(x, channels: c, height: h, width: w)
        #expect(p.count == x.count)
        let (ph, pw) = (h / 2, w / 2)
        for ci in 0..<c {
            for dy in 0..<2 {
                for dx in 0..<2 {
                    for i in 0..<ph {
                        for j in 0..<pw {
                            let out = (MossLatentFormat.patchedChannel(c: ci, dy: dy, dx: dx) * ph + i) * pw + j
                            let src = (ci * h + 2 * i + dy) * w + 2 * j + dx
                            #expect(p[out] == x[src])
                        }
                    }
                }
            }
        }
    }

    @Test func unpatchifyInvertsPatchify() {
        let x = ramp(32, 6, 4)
        let p = MossLatentFormat.patchify(x, channels: 32, height: 6, width: 4)
        #expect(MossLatentFormat.unpatchify(p, channels: 32, height: 6, width: 4) == x)
    }

    @Test func normalizeUsesBatchNormWithEps1em4AndDenormalizeInverts() {
        let x: [Float] = [1, 2, 3, 4, 10, 20, 30, 40]  // 2 channels × 2×2
        let mean: [Float] = [1, 10]
        let variance: [Float] = [4, 0]
        let n = MossLatentFormat.normalize(x, mean: mean, variance: variance)
        #expect(MossLatentFormat.bnEpsilon == 1e-4)
        #expect(abs(n[1] - (2 - 1) / (4 + 1e-4).squareRoot()) < 1e-6)
        #expect(abs(n[5] - (20 - 10) / Float(1e-4).squareRoot()) < 1e-2)
        let back = MossLatentFormat.denormalize(n, mean: mean, variance: variance)
        for (a, b) in zip(back, x) { #expect(abs(a - b) < 1e-4) }
    }

    @Test func latentShapesForAPixelSize() {
        #expect(MossLatentFormat.vaeLatentShape(width: 1024, height: 768) == [32, 96, 128])
        #expect(MossLatentFormat.tokenLatentShape(width: 1024, height: 768) == [128, 48, 64])
    }
}

@Suite("FluxVAEConfig — LDM AutoencoderKL geometry from the header")
struct FluxVAEConfigTests {
    private func vaeDescriptors() -> [TensorDescriptor] {
        func d(_ n: String, _ s: [Int]) -> TensorDescriptor { TensorDescriptor(name: n, dtype: "F32", shape: s, byteCount: 0) }
        var out = [
            d("encoder.conv_in.weight", [128, 3, 3, 3]), d("encoder.conv_out.weight", [64, 512, 3, 3]),
            d("decoder.conv_in.weight", [512, 32, 3, 3]), d("decoder.conv_out.weight", [3, 128, 3, 3]),
            d("bn.running_mean", [128]), d("bn.running_var", [128]),
        ]
        for (level, ch) in [128, 256, 512, 512].enumerated() {
            for b in 0..<2 { out.append(d("encoder.down.\(level).block.\(b).conv1.weight", [ch, ch, 3, 3])) }
            for b in 0..<3 { out.append(d("decoder.up.\(level).block.\(b).conv1.weight", [ch, ch, 3, 3])) }
        }
        return out
    }

    @Test func derivesTheFlux2Geometry() throws {
        let c = try FluxVAEConfig(descriptors: vaeDescriptors())
        #expect(c == FluxVAEConfig.flux2)
        #expect(c.baseChannels == 128 && c.channelMultipliers == [1, 2, 4, 4])
        #expect(c.resBlocks == 2 && c.zChannels == 32 && c.batchNormLatent)
        #expect(c.downscale == 8)
    }

    @Test func rejectsANonVAE() {
        #expect(throws: EngineError.self) { try FluxVAEConfig(descriptors: ditDescriptors(layers: 1)) }
    }

    @Test func realHeaderMatches() throws {
        let url = modelsDir.appendingPathComponent("vae/flux2-vae.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let (descriptors, _) = try SafetensorsHeader.readDescriptors(from: url)
        #expect(try FluxVAEConfig(descriptors: descriptors) == .flux2)
    }
}
