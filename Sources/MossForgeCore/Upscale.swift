
import Foundation

/// Which model enlarges the image.
public enum UpscaleMethod: String, Sendable, Codable, CaseIterable, Identifiable {
    /// VOSR2 1.4B one-step generative SR: rebuilds texture (pores, fibres, foliage). Slow.
    case vosr
    /// A Real-ESRGAN-family model from `models/esrgan/`: fast, sharp, invents less.
    case esrgan

    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .vosr: "VOSR2 (generative)"
        case .esrgan: "ESRGAN (fast)"
        }
    }
}

/// Pre-filter for VOSR's input. Few-step generations carry fine sampler/VAE grain that VOSR
/// misreads as fabric-weave or scale texture on smooth skin; a light edge-preserving filter
/// removes it and VOSR rebuilds plausible detail instead. Colour is matched to the unsoftened input.
public enum UpscaleSoftening: String, Sendable, Codable, CaseIterable, Identifiable {
    case off, light, medium, strong
    public var id: String { rawValue }
    public var displayName: String { rawValue.capitalized }
}

/// How the upscaled colours are matched back to the input.
public enum UpscaleColorFix: String, Sendable, Codable, CaseIterable, Identifiable {
    /// Low frequencies (σ 5 blur) from the input, high frequencies from VOSR.
    case wavelet
    /// Per-channel mean/std matched to the input.
    case adain
    case none
    public var id: String { rawValue }
}

/// VOSR knobs. Defaults are the ones measured in Forge: 512 px tiles (the training size) with a
/// 256 px raised-cosine overlap, 1024 px VAE tiles, medium softening, wavelet colour fix.
public struct UpscaleSettings: Sendable, Codable, Equatable {
    /// VOSR starts from noise; a fixed seed makes repeat upscales identical.
    public var seed: UInt64
    public var tile: Int
    public var overlap: Int
    public var vaeTile: Int
    public var soften: UpscaleSoftening
    public var colorFix: UpscaleColorFix
    /// Tiles per DiT call. Same result as one at a time (within one 8-bit level); 4 is ~18% faster
    /// than 1 for a ×4 upscale (896→3584: 81 s → 66.5 s, +2 GB peak) and 8 adds nothing more.
    public var batch: Int

    public static let tileRange = 256...1024
    public static let overlapRange = 64...384
    public static let batchRange = 1...8
    public static let defaultBatch = 4

    public init(
        seed: UInt64 = 42, tile: Int = 512, overlap: Int = 256, vaeTile: Int = 1024,
        soften: UpscaleSoftening = .medium, colorFix: UpscaleColorFix = .wavelet,
        batch: Int = UpscaleSettings.defaultBatch
    ) {
        self.batch = min(max(batch, Self.batchRange.lowerBound), Self.batchRange.upperBound)
        self.seed = seed
        self.tile = min(max(tile / 64 * 64, Self.tileRange.lowerBound), Self.tileRange.upperBound)
        self.overlap = min(max(overlap, Self.overlapRange.lowerBound), min(Self.overlapRange.upperBound, self.tile - 64))
        self.vaeTile = max(0, vaeTile)
        self.soften = soften
        self.colorFix = colorFix
    }

    enum CodingKeys: String, CodingKey { case seed, tile, overlap, vaeTile, soften, colorFix, batch }

    /// Tolerant decode: settings saved before a field existed keep its default.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = UpscaleSettings()
        self.init(
            seed: (try? c.decode(UInt64.self, forKey: .seed)) ?? d.seed,
            tile: (try? c.decode(Int.self, forKey: .tile)) ?? d.tile,
            overlap: (try? c.decode(Int.self, forKey: .overlap)) ?? d.overlap,
            vaeTile: (try? c.decode(Int.self, forKey: .vaeTile)) ?? d.vaeTile,
            soften: (try? c.decode(UpscaleSoftening.self, forKey: .soften)) ?? d.soften,
            colorFix: (try? c.decode(UpscaleColorFix.self, forKey: .colorFix)) ?? d.colorFix,
            batch: (try? c.decode(Int.self, forKey: .batch)) ?? d.batch)
    }
}

/// Enlarge one image.
public struct UpscaleRequest: Sendable {
    public var image: URL
    /// Output size factor (e.g. 2 or 4); the result is rounded down to a multiple of 8.
    public var scale: Double
    public var method: UpscaleMethod
    /// ESRGAN model filename under `models/esrgan/`; `nil` picks the default.
    public var esrganModel: String?
    public var settings: UpscaleSettings
    /// Where to write the PNG; `nil` lets the backend choose (next to the input, `_x<scale>`).
    public var output: URL?

    public static let scaleRange = 1.0...8.0

    public init(
        image: URL, scale: Double = 2, method: UpscaleMethod = .vosr, esrganModel: String? = nil,
        settings: UpscaleSettings = UpscaleSettings(), output: URL? = nil
    ) {
        self.image = image
        self.scale = min(max(scale, Self.scaleRange.lowerBound), Self.scaleRange.upperBound)
        self.method = method
        self.esrganModel = esrganModel
        self.settings = settings
        self.output = output
    }

    /// The default output path: `<name>_x<scale>[_vosr].png` beside the input.
    public var resolvedOutput: URL {
        if let output { return output }
        let base = image.deletingPathExtension().lastPathComponent
        let s = scale == scale.rounded() ? "\(Int(scale))" : String(format: "%.2g", scale)
        let tag = method == .vosr ? "_vosr" : ""
        return image.deletingLastPathComponent().appendingPathComponent("\(base)_x\(s)\(tag).png")
    }
}

public struct UpscaleResult: Sendable {
    public var output: URL
    public var width: Int
    public var height: Int
    public init(output: URL, width: Int, height: Int) {
        self.output = output
        self.width = width
        self.height = height
    }
}

/// Where the upscalers' weights live under the models folder. Users download them (VOSR2 and
/// DINOv2 are Apache-2.0, as is the Qwen-Image VAE; ESRGAN-family files carry their own licences).
public enum UpscaleModels {
    /// VOSR2 is self-contained: its DiT, the DINOv2-L feature tower, and the Qwen-Image VAE whose
    /// 16-channel latent space VOSR2 was trained in (MOSS's FLUX.2 VAE won't do).
    public static let vosrDiT = "vosr/VOSR2/checkpoints/ema_model.safetensors"
    public static let vosrDINOv2 = "vosr/dinov2-large/model.safetensors"
    public static let vosrVAE = "vosr/qwen_image_vae.safetensors"
    public static let vosrFiles = [vosrDiT, vosrDINOv2, vosrVAE]
    /// ESRGAN / SPAN safetensors (converted with upstream key names) sit here.
    public static let esrganFolder = "esrgan"

    /// The VOSR files not present under `models`.
    public static func missingVOSR(in models: URL) -> [String] {
        vosrFiles.filter { !FileManager.default.fileExists(atPath: models.appendingPathComponent($0).path) }
    }

    /// The `.safetensors` under `models/esrgan`, sorted for a stable list.
    public static func esrganModels(in models: URL) -> [String] {
        let dir = models.appendingPathComponent(esrganFolder)
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        return files.filter { $0.hasSuffix(".safetensors") }.sorted()
    }

    /// RealESRGAN ×2plus when installed, else the first by name.
    public static func defaultESRGAN(in models: URL) -> String? {
        let files = esrganModels(in: models)
        return files.first { $0.contains("RealESRGAN_x2plus") } ?? files.first
    }
}
