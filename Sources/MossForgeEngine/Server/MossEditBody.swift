import Foundation
import MossForgeCore

// Sources are base64 or
// a job, never a file path) and source-following size; MOSS has no task presets, so the client's edit
// LoRA (`loras`) is required and the one source becomes `GenerationRequest.references`. See NOTES.md.

/// The edit-specific half of `POST /v1/moss/edit` and `/v1/moss/edit/jobs`. The same JSON also parses
/// as a `MossGenerationBody` (prompt = the instruction, variant, steps, seed, loras, …), so an edit takes
/// every generation field plus a source image, which runs as an edit-LoRA reference (experimental:
/// clean reference tokens on RoPE r = 1, the Ostris AI Toolkit method).
public struct MossEditBody: Decodable, Sendable {
    public var image: String?
    public var job: String?
    /// Which of the job's images (default 0).
    public var index: Int?

    /// Output long-side bounds when the size follows the source: large photos scale down, thumbnails
    /// scale up so the edit has pixels to work with.
    public static let maxFittedSide = 2048
    public static let minFittedSide = 512

    /// Exactly one source.
    public func validateSource() throws {
        let hasImage = !(image ?? "").isEmpty, hasJob = !(job ?? "").isEmpty
        guard hasImage != hasJob else {
            throw EngineError.invalidRequest("give exactly one of 'image' (base64 PNG/JPEG) or 'job' (a job id)")
        }
    }

    /// The source's size and aspect, long side kept in `minFittedSide…maxFittedSide`, snapped to 16 px.
    public static func fittedSize(width: Int, height: Int) -> (Int, Int) {
        let long = max(width, height, 1)
        let target = min(max(long, minFittedSide), maxFittedSide)
        let k = Double(target) / Double(long)
        func snap(_ v: Int) -> Int {
            let s = Int((Double(v) * k / 16).rounded()) * 16
            return min(max(s, MossGenerationBody.sizeRange.lowerBound), maxFittedSide)
        }
        return (snap(width), snap(height))
    }

    /// The edit request: the generation body with `reference` as its edit reference, at the source's
    /// size unless the client gave one. Without an active LoRA the reference would only steer a plain
    /// model, so that is refused.
    public func makeRequest(
        body: MossGenerationBody, modelsDirectory: URL, reference: URL, sourceWidth: Int, sourceHeight: Int
    ) throws -> GenerationRequest {
        guard body.loras.contains(where: { $0.strength != 0 }) else {
            throw EngineError.invalidRequest(
                "an edit needs an edit LoRA in 'loras' (one trained in Train → Edit, or a community MOSS edit LoRA)")
        }
        var sized = body
        if body.width == nil, body.height == nil {
            (sized.width, sized.height) = Self.fittedSize(width: sourceWidth, height: sourceHeight)
        }
        var r = try sized.makeRequest(modelsDirectory: modelsDirectory)
        r.references = [reference]
        r.editLayout = body.editLayout
        try r.validate()
        return r
    }
}
