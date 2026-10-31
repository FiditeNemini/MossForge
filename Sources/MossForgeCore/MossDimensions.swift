import Foundation

/// An output image size in pixels.
public struct Dimensions: Sendable, Equatable, Codable, Hashable {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}

/// MOSS-Image sizing: a pixel budget (0.25 / 1 / 1.5 / 2 / 4 MP) times an aspect ratio, each edge
/// rounded to a multiple of 16 (the FLUX.2 VAE's ×8 times the DiT's 2×2 patch). Mirrors the
/// reference app's `gradio_presets.fit_size` so presets produce identical sizes.
public enum MossDimensions {
    /// Effective image stride: one DiT token per 16×16 pixels.
    public static let stride = 16
    /// Preset pixel budgets (USERGUIDE recommends 1 M, 1.5 M, or 2 M).
    public static let pixelBudgets = [262_144, 1_048_576, 1_572_864, 2_097_152, 4_194_304]
    /// Preset ratios, width:height.
    public static let ratios = [
        "1:8", "1:4", "1:3", "1:2", "9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9", "2:1", "3:1", "4:1", "8:1",
    ]

    /// The size for a pixel budget and a `"w:h"` ratio string (`"1:1"` if unparsable).
    public static func fit(pixels: Int, ratio: String) -> Dimensions {
        let parts = ratio.split(separator: ":").compactMap { Double($0) }
        let r = parts.count == 2 && parts[1] > 0 ? parts[0] / parts[1] : 1
        let w = (Double(pixels) * r).squareRoot()
        let h = (Double(pixels) / r).squareRoot()
        func snap(_ edge: Double) -> Int { max(stride, Int(floor(edge / Double(stride) + 0.5)) * stride) }
        return Dimensions(width: snap(w), height: snap(h))
    }

    /// The preset ratio closest to an image's aspect (in log space), e.g. an edit reference's.
    public static func nearestRatio(width: Int, height: Int) -> String {
        let target = log(Double(max(width, 1)) / Double(max(height, 1)))
        func aspect(_ r: String) -> Double {
            let p = r.split(separator: ":").compactMap { Double($0) }
            return log(p[0] / p[1])
        }
        return ratios.min { abs(aspect($0) - target) < abs(aspect($1) - target) } ?? "1:1"
    }

    public static func isValid(width: Int, height: Int) -> Bool {
        width > 0 && height > 0 && width % stride == 0 && height % stride == 0
    }

    /// DiT image tokens: (H/16)·(W/16).
    public static func imageTokens(width: Int, height: Int) -> Int {
        (width / stride) * (height / stride)
    }
}
