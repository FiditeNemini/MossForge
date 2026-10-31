
import Foundation
import MossForgeCore

/// The JSON body of `POST /v1/moss/upscale` and `/v1/moss/upscale/jobs`: one source — `image`
/// (base64 or a `data:` URL) or `job` (a finished server job's image, by id) — plus the
/// upscale knobs. There is deliberately no file-path source: a LAN-exposed server must not
/// read arbitrary files, and a job id covers "upscale what you just made".
public struct UpscaleBody: Decodable, Sendable {
    public var image: String?
    public var job: String?
    /// Which of the job's images (default 0).
    public var index: Int?
    /// Output size factor, 1…8 (default 2).
    public var scale: Double?
    /// `vosr` (default — generative, rebuilds texture, slow) or `esrgan` (fast, sharp).
    public var method: String?
    /// ESRGAN filename under `<models>/esrgan`; omitted → the backend's default.
    public var esrgan_model: String?
    /// VOSR pre-filter: `off` | `light` | `medium` (default) | `strong`.
    public var softening: String?
    /// `wavelet` (default) | `adain` | `none`.
    public var color_fix: String?
    public var seed: UInt64?
    /// Tile overlap in px, 64…384 (default 256).
    public var tile_overlap: Int?

    /// Bigger outputs are refused: VOSR at 8K per side is minutes of GPU and gigabytes of PNG.
    public static let maxOutputSide = 8192

    /// Exactly one source.
    public func validateSource() throws {
        let hasImage = !(image ?? "").isEmpty, hasJob = !(job ?? "").isEmpty
        guard hasImage != hasJob else {
            throw EngineError.invalidRequest("give exactly one of 'image' (base64 PNG/JPEG) or 'job' (a job id)")
        }
    }

    /// The `image` bytes (whitespace and a `data:…;base64,` prefix are tolerated).
    public func imageData() throws -> Data {
        var text = image ?? ""
        if text.hasPrefix("data:"), let comma = text.firstIndex(of: ",") { text = String(text[text.index(after: comma)...]) }
        text.removeAll { $0.isWhitespace }
        guard let data = Data(base64Encoded: text), !data.isEmpty else {
            throw EngineError.invalidRequest("'image' must be base64 (or a data: URL)")
        }
        return data
    }

    /// The engine request for an input already written to `input`.
    public func makeRequest(input: URL, output: URL?) throws -> UpscaleRequest {
        let scale = self.scale ?? 2
        guard UpscaleRequest.scaleRange.contains(scale) else {
            throw EngineError.invalidRequest("'scale' must be 1…8")
        }
        let method = try Self.choice(self.method, default: UpscaleMethod.vosr, name: "method")
        let soften = try Self.choice(softening, default: UpscaleSoftening.medium, name: "softening")
        let colorFix = try Self.choice(color_fix, default: UpscaleColorFix.wavelet, name: "color_fix")
        let d = UpscaleSettings()
        if let tile_overlap, !UpscaleSettings.overlapRange.contains(tile_overlap) {
            throw EngineError.invalidRequest(
                "'tile_overlap' must be \(UpscaleSettings.overlapRange.lowerBound)…\(UpscaleSettings.overlapRange.upperBound)")
        }
        if let esrgan_model, !Self.isBareFileName(esrgan_model) {
            throw EngineError.invalidRequest(
                "'esrgan_model' must be a .safetensors filename in models/esrgan (see /v1/moss/options)")
        }
        return UpscaleRequest(
            image: input, scale: scale, method: method, esrganModel: esrgan_model,
            settings: UpscaleSettings(
                seed: seed ?? d.seed, tile: d.tile, overlap: tile_overlap ?? d.overlap, vaeTile: d.vaeTile,
                soften: soften, colorFix: colorFix),
            output: output)
    }

    /// A `.safetensors` name with no directory part — it can't escape `models/esrgan`.
    static func isBareFileName(_ name: String) -> Bool {
        name.hasSuffix(".safetensors") && !name.contains("/") && !name.hasPrefix(".")
    }

    static func choice<T: RawRepresentable & CaseIterable>(_ value: String?, default d: T, name: String) throws -> T
    where T.RawValue == String {
        guard let value, !value.isEmpty else { return d }
        guard let v = T(rawValue: value.lowercased()) else {
            let names = T.allCases.map(\.rawValue).joined(separator: " | ")
            throw EngineError.invalidRequest("unknown \(name) '\(value)' (\(names))")
        }
        return v
    }
}
