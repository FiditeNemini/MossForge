import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import MossForgeCore
@testable import MossForgeEngine

/// The upscaler is an engine capability like the rest: the app, CLI and server ask the
/// `MossEngine` facade, and the stub answers without weights.
@Suite struct UpscaleCapabilityTests {
    @Test func everyBackendAdvertisesUpscaling() {
        #expect(EngineCapabilities.all.contains(.upscaling))
        #expect(MossEngine().capabilities.contains(.upscaling))
    }

    @Test func settingsDefaultToForgesMeasuredValues() {
        let s = UpscaleSettings()
        #expect(s.tile == 512 && s.overlap == 256 && s.vaeTile == 1024)
        #expect(s.soften == .medium && s.colorFix == .wavelet && s.seed == 42)
    }

    @Test func settingsAreClampedToWorkableTiles() {
        #expect(UpscaleSettings(tile: 2000).tile == 1024)
        #expect(UpscaleSettings(tile: 300).tile == 256)
        // Overlap must leave a tile stride.
        #expect(UpscaleSettings(tile: 256, overlap: 384).overlap == 192)
        #expect(UpscaleSettings(overlap: 10).overlap == 64)
    }

    /// VOSR tiles per DiT call: clamped to a sane range; settings saved before it existed decode.
    @Test func tileBatchIsClampedAndOptionalInSavedSettings() throws {
        #expect(UpscaleSettings(batch: 0).batch == 1)
        #expect(UpscaleSettings(batch: 99).batch == UpscaleSettings.batchRange.upperBound)
        let old = #"{"seed":7,"tile":512,"overlap":256,"vaeTile":1024,"soften":"medium","colorFix":"wavelet"}"#
        let s = try JSONDecoder().decode(UpscaleSettings.self, from: Data(old.utf8))
        #expect(s.seed == 7 && s.batch == UpscaleSettings().batch)
    }

    /// Tiles are visited row-major and grouped into DiT batches; the last batch may be short.
    @Test func tileBatchesCoverTheGridInOrder() {
        let b = UpscaleTiling.batches(rows: [0, 32], cols: [0, 32, 64], size: 4)
        #expect(b.count == 2)
        #expect(b[0].map { [$0.0, $0.1] } == [[0, 0], [0, 32], [0, 64], [32, 0]])
        #expect(b[1].map { [$0.0, $0.1] } == [[32, 32], [32, 64]])
        #expect(UpscaleTiling.batches(rows: [0], cols: [0], size: 4).map(\.count) == [1])
    }

    @Test func defaultOutputSitsBesideTheInput() {
        let r = UpscaleRequest(image: URL(fileURLWithPath: "/pics/fox.png"), scale: 4)
        #expect(r.resolvedOutput.path == "/pics/fox_x4_vosr.png")
        let e = UpscaleRequest(image: URL(fileURLWithPath: "/pics/fox.jpg"), scale: 2, method: .esrgan)
        #expect(e.resolvedOutput.path == "/pics/fox_x2.png")
        #expect(UpscaleRequest(image: URL(fileURLWithPath: "/a.png"), scale: 99).scale == 8)
    }

    @Test func stubUpscaleReportsTheTargetSize() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mf-up-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let input = dir.appendingPathComponent("in.png")
        let ctx = CGContext(
            data: nil, width: 101, height: 77, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let dest = CGImageDestinationCreateWithURL(input as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        #expect(CGImageDestinationFinalize(dest))
        let result = try await MossEngine().upscale(
            UpscaleRequest(image: input, scale: 4, output: dir.appendingPathComponent("out.png")))
        #expect(result.width == 400 && result.height == 304)
        #expect(FileManager.default.fileExists(atPath: result.output.path))
    }

    /// VOSR2 is self-contained under `vosr/`: its DiT, DINOv2-L and the Qwen-Image VAE it was
    /// trained in (not MOSS's FLUX.2 VAE).
    @Test func vosrWeightsLiveUnderTheirOwnFolder() throws {
        #expect(UpscaleModels.vosrFiles == [
            "vosr/VOSR2/checkpoints/ema_model.safetensors", "vosr/dinov2-large/model.safetensors",
            "vosr/qwen_image_vae.safetensors",
        ])
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mf-up-models-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(UpscaleModels.missingVOSR(in: dir) == UpscaleModels.vosrFiles)
        let dino = dir.appendingPathComponent(UpscaleModels.vosrFiles[1])
        try FileManager.default.createDirectory(at: dino.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: dino)
        #expect(UpscaleModels.missingVOSR(in: dir) == [UpscaleModels.vosrFiles[0], UpscaleModels.vosrFiles[2]])
    }

    /// ESRGAN-family models are the `.safetensors` in `models/esrgan`; the default prefers
    /// RealESRGAN ×2plus, else the first by name.
    @Test func esrganModelsAreListedWithAStableDefault() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mf-up-esr-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(UpscaleModels.esrganModels(in: dir).isEmpty)
        #expect(UpscaleModels.defaultESRGAN(in: dir) == nil)
        let esr = dir.appendingPathComponent("esrgan")
        try FileManager.default.createDirectory(at: esr, withIntermediateDirectories: true)
        for name in ["4x-UltraSharp.safetensors", "4x-UltraSharp.pth", "2x-b.safetensors"] {
            try Data().write(to: esr.appendingPathComponent(name))
        }
        #expect(UpscaleModels.esrganModels(in: dir) == ["2x-b.safetensors", "4x-UltraSharp.safetensors"])
        #expect(UpscaleModels.defaultESRGAN(in: dir) == "2x-b.safetensors")
        try Data().write(to: esr.appendingPathComponent("RealESRGAN_x2plus.safetensors"))
        #expect(UpscaleModels.defaultESRGAN(in: dir) == "RealESRGAN_x2plus.safetensors")
    }
}
