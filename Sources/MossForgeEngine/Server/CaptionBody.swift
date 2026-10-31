import Foundation
import MossForgeCore

/// The JSON body of `POST /v1/moss/caption`: one source — `image` (base64 or a `data:` URL) or `job`
/// (a finished job's image, by id) — like `UpscaleBody`, plus the caption knobs. No file paths: a
/// LAN-exposed server must not read arbitrary files.
public struct CaptionBody: Decodable, Sendable {
    public var image: String?
    public var job: String?
    /// Which of the job's images (default 0).
    public var index: Int?
    /// `long` (default) | `medium` | `short` — which caption `caption` holds (all three are returned).
    public var detail: String?
    /// An existing caption for the model to correct.
    public var reference: String?
    /// Vision pixel budget, 65536…4194304 (default 1048576 = 1024²).
    public var max_pixels: Int?
    public var seed: UInt64?

    /// Exactly one source, and known choices.
    public func validate() throws {
        guard (image ?? "").isEmpty != (job ?? "").isEmpty else {
            throw EngineError.invalidRequest("give exactly one of 'image' (base64 PNG/JPEG) or 'job' (a job id)")
        }
        _ = try UpscaleBody.choice(detail, default: CaptionDetail.long, name: "detail")
        if let max_pixels, !(CaptionRequest.minPixels...CaptionRequest.maxPixelsLimit).contains(max_pixels) {
            throw EngineError.invalidRequest("'max_pixels' must be \(CaptionRequest.minPixels)…\(CaptionRequest.maxPixelsLimit)")
        }
    }

    public func makeRequest(image url: URL) throws -> CaptionRequest {
        try validate()
        return CaptionRequest(
            image: url, detail: try UpscaleBody.choice(detail, default: CaptionDetail.long, name: "detail"),
            reference: reference, maxPixels: max_pixels ?? CaptionRequest.defaultMaxPixels, seed: seed ?? 0)
    }
}
