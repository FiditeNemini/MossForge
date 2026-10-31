import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

// The VOSR2 and ESRGAN paths, pulled out of the backend so `MossBackend` stays a thin router. See NOTES.md.

/// Runs one `UpscaleRequest`: VOSR2 (generative, loaded lazily, ~7 GB f32) or an ESRGAN-family
/// model from `models/esrgan` (cached by filename). Not thread-safe; `MossBackend` (an actor) owns it.
public final class MossUpscaler: @unchecked Sendable {
    private let models: URL
    private var vosr: VOSRUpscaler?
    private var esrgan: [String: any Upscaler] = [:]

    public init(models: URL) { self.models = models }

    /// Drop VOSR2 and the cached ESRGAN models.
    public func release() {
        vosr = nil
        esrgan = [:]
        MLX.GPU.clearCache()
    }

    public func upscale(_ request: UpscaleRequest, progress: ProgressHandler?) throws -> UpscaleResult {
        let image = try RGB8.load(url: request.image)
        let (tw, th) = UpscaleTiling.targetSize(width: image.width, height: image.height, scale: request.scale)
        let out = request.resolvedOutput
        let result: RGB8
        switch request.method {
        case .vosr:
            if vosr == nil {
                let missing = UpscaleModels.missingVOSR(in: models)
                guard missing.isEmpty else {
                    throw EngineError.backendFailure(
                        "VOSR2 weights not installed (models/\(missing.joined(separator: ", models/"))). "
                            + "See docs/user/upscaling.md, or use method esrgan.")
                }
                progress?(GenerationProgress(step: 0, totalSteps: 1, message: "Loading VOSR2…"))
                vosr = try VOSRUpscaler.load(modelsDirectory: models)
            }
            result = try vosr!.upscale(image, targetWidth: tw, targetHeight: th, settings: request.settings, progress: { done, total in
                progress?(GenerationProgress(step: done, totalSteps: total, message: "VOSR tile \(done)/\(total)"))
            })
        case .esrgan:
            let model = try esrganModel(named: request.esrganModel)
            progress?(GenerationProgress(step: 0, totalSteps: 1, message: "upscaling ×\(model.scale)…"))
            // Native passes until the image covers the target, then PIL-bicubic to the exact size.
            var x = MLXArray(image.pixels, [1, image.height, image.width, 3]).asType(.float32) / 255
            var (w, h) = (image.width, image.height)
            repeat {
                if Task.isCancelled { throw EngineError.cancelled }
                x = clip(model.upscale(x), min: 0, max: 1)
                eval(x)
                (w, h) = (w * model.scale, h * model.scale)
            } while w < tw || h < th
            let v = (x[0] * 255).round().asType(.uint8)
            eval(v)
            result = RGB8(width: tw, height: th, pixels: PILResample.bicubic(
                v.asArray(UInt8.self), width: w, height: h, channels: 3, toWidth: tw, toHeight: th))
        }
        try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        try result.savePNG(to: out)
        return UpscaleResult(output: out, width: tw, height: th)
    }

    /// An ESRGAN/SPAN model by filename under `models/esrgan`; `nil` picks `UpscaleModels.defaultESRGAN`.
    private func esrganModel(named name: String?) throws -> any Upscaler {
        guard let file = name ?? UpscaleModels.defaultESRGAN(in: models) else {
            throw EngineError.backendFailure(
                "no ESRGAN model in models/esrgan (e.g. RealESRGAN_x2plus.safetensors; see docs/user/upscaling.md)")
        }
        if let cached = esrgan[file] { return cached }
        let url = models.appendingPathComponent(UpscaleModels.esrganFolder).appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EngineError.backendFailure("ESRGAN model not found: models/esrgan/\(file)")
        }
        let model = try UpscalerFactory.load(url: url)
        esrgan[file] = model
        return model
    }
}
