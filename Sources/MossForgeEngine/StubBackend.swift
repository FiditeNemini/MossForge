import CoreGraphics
import Foundation
import ImageIO
import MossForgeCore
import UniformTypeIdentifiers

/// Canned backend so the UI, CLI, server and tests run with no weights: emits one progress
/// update per step and returns a gradient PNG at the requested size.
public struct StubBackend: EngineBackend {
    public init() {}

    public var name: String { "stub" }
    public var capabilities: EngineCapabilities { [.imageGeneration, .promptEnhancing, .modelInspection, .upscaling, .captioning] }

    public func generate(_ request: GenerationRequest, progress: ProgressHandler?) async throws -> GenerationResult {
        for step in 1...request.steps {
            try Task.checkCancellation()
            progress?(GenerationProgress(step: step, totalSteps: request.steps, message: "stub step \(step)"))
        }
        let seed = request.seed ?? UInt64.random(in: 1...UInt64(UInt32.max))
        return GenerationResult(
            imagePNG: try Self.gradientPNG(width: request.width, height: request.height, seed: seed),
            width: request.width, height: request.height, seed: seed)
    }

    public func enhancePrompt(_ request: PromptEnhanceRequest, progress: ProgressHandler?) async throws -> String {
        guard request.layout else { return "\(request.prompt), detailed composition, legible text, balanced layout" }
        let c = request.canvas
        let layout = LayoutPrompt(
            summary: .init(meta: ["poster"], canvas: "single", caption: request.prompt),
            items: [
                .init(
                    label: .text, caption: "Bold title across the top.", text: request.prompt,
                    bbox: LayoutBox.fromTopLeft(x: c.width / 8, y: c.height / 16, width: c.width * 3 / 4, height: c.height / 8, canvas: c).yxyx),
                .init(
                    label: .object, caption: "The main subject, centred.",
                    bbox: LayoutBox.fromTopLeft(x: c.width / 4, y: c.height / 4, width: c.width / 2, height: c.height / 2, canvas: c).yxyx),
            ])
        return try layout.jsonString()
    }

    /// Canned upscale: reports the target size and copies the input (no weights, no resampling).
    public func upscale(_ request: UpscaleRequest, progress: ProgressHandler?) async throws -> UpscaleResult {
        guard let src = CGImageSourceCreateWithURL(request.image as CFURL, nil),
            let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
            let w = props[kCGImagePropertyPixelWidth] as? Int, let h = props[kCGImagePropertyPixelHeight] as? Int
        else { throw EngineError.inputNotFound(request.image) }
        let (tw, th) = UpscaleTiling.targetSize(width: w, height: h, scale: request.scale)
        let out = request.resolvedOutput
        if FileManager.default.fileExists(atPath: out.path) { try FileManager.default.removeItem(at: out) }
        try FileManager.default.copyItem(at: request.image, to: out)
        progress?(GenerationProgress(step: 1, totalSteps: 1, message: "upscaled (stub)"))
        return UpscaleResult(output: out, width: tw, height: th)
    }

    /// Canned caption naming the file (no weights).
    /// Training needs real weights; the stub doesn't advertise it.
    public func trainLoRA(_ request: TrainingRequest, samples: [TrainingSample], progress: ProgressHandler?) async throws -> TrainingResult {
        throw EngineError.notImplemented(capability: "LoRA training")
    }

    /// Model tooling needs real weights; the stub doesn't advertise it.
    public func runModelTool(_ request: ModelToolRequest, progress: ProgressHandler?) async throws -> ModelToolResult {
        throw EngineError.notImplemented(capability: "model tooling")
    }

    public func caption(_ request: CaptionRequest, progress: ProgressHandler?) async throws -> MossCaption {
        guard CGImageSourceCreateWithURL(request.image as CFURL, nil).flatMap({ CGImageSourceCreateImageAtIndex($0, 0, nil) }) != nil
        else { throw EngineError.inputNotFound(request.image) }
        progress?(GenerationProgress(step: 1, totalSteps: 1, message: "captioned (stub)"))
        let name = request.image.lastPathComponent
        return MossCaption(
            long: "A placeholder long caption from the stub VLM for \(name), describing composition, colours and text.",
            medium: "A placeholder caption from the stub VLM for \(name).", short: "Stub caption of \(name).")
    }

    static func gradientPNG(width: Int, height: Int, seed: UInt64) throws -> Data {
        let hue = CGFloat(seed % 360) / 360
        guard
            let ctx = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
            let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: [CGColor(red: hue, green: 0.4, blue: 1 - hue, alpha: 1), CGColor(red: 0.1, green: 0.1, blue: 0.12, alpha: 1)] as CFArray,
                locations: [0, 1])
        else { throw EngineError.backendFailure("stub: could not create a bitmap") }
        ctx.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: width, y: height), options: [])
        guard let image = ctx.makeImage() else { throw EngineError.backendFailure("stub: no image") }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw EngineError.backendFailure("stub: no PNG encoder")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw EngineError.backendFailure("stub: PNG encode failed") }
        return data as Data
    }
}
