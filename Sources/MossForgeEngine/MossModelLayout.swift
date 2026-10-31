import Foundation
import MossForgeCore

/// Where each MOSS-Image component lives under a `models/` folder (Comfy-style names). One table for
/// the MLX backend's resolver and the app's Models screen, so the two can't drift apart. Paths are
/// relative to the models folder; the first existing candidate wins.
public enum MossModelLayout {
    public enum Component: String, Sendable, CaseIterable, Identifiable {
        case sftDiT, baseDiT, textEncoder, tokenizer, vae
        case turboLoRA
        case vosrDiT, vosrDINOv2, vosrVAE

        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .sftDiT: "MOSS-Image SFT DiT"
            case .baseDiT: "MOSS-Image Base DiT"
            case .textEncoder: "Qwen3.5-9B text encoder"
            case .tokenizer: "Qwen3.5 tokenizer"
            case .vae: "FLUX.2 VAE"
            case .turboLoRA: "Turbo LoRA (akhaliq 4-step)"
            case .vosrDiT: "VOSR2 upscaler"
            case .vosrDINOv2: "DINOv2-L (VOSR2)"
            case .vosrVAE: "Qwen-Image VAE (VOSR2)"
            }
        }

        /// What needs it.
        public var usedBy: String {
            switch self {
            case .sftDiT: "SFT, Turbo"
            case .baseDiT: "Base"
            case .textEncoder, .tokenizer: "generation, enhancer, captioner"
            case .vae: "generation"
            case .turboLoRA: "Turbo"
            case .vosrDiT, .vosrDINOv2, .vosrVAE: "VOSR2 upscale"
            }
        }

        /// Candidate paths, preferred first.
        public var candidates: [String] {
            switch self {
            case .sftDiT: MossModelLayout.ditCandidates("sft")
            case .baseDiT: MossModelLayout.ditCandidates("base")
            case .textEncoder: ["text_encoders/qwen3.5_9b_int8_convrot.safetensors"]
            case .tokenizer: ["text_encoders/qwen3.5-9b-tokenizer"]
            case .vae: ["vae/flux2-vae.safetensors"]
            // akhaliq 4-step only; lodestones' SFT data-space files are no longer looked for (2026-10-31).
            case .turboLoRA: [MossVariant.turboLoRAPath]
            case .vosrDiT: [UpscaleModels.vosrDiT]
            case .vosrDINOv2: [UpscaleModels.vosrDINOv2]
            case .vosrVAE: [UpscaleModels.vosrVAE]
            }
        }

        /// The file whose presence proves a candidate (a folder component is proved by a file inside it).
        func probe(_ candidate: String) -> String {
            self == .tokenizer ? candidate + "/tokenizer.json" : candidate
        }
    }

    /// The pre-2026-10-20 folder of lodestones' data-space adapter files (still hidden from the LoRA list).
    public static let dmdFolder = "lora/lodestones-dmd"

    static func ditCandidates(_ name: String) -> [String] {
        ["bf16", "int8_convrot"].map { "diffusion_models/moss_image_\(name)_\($0).safetensors" }
    }

    /// Settings → Models: explicit files for the shared components (each falls back to the layout
    /// when unset or missing).
    public struct Overrides: Sendable, Equatable, Codable {
        public var vae: URL?
        public var textEncoder: URL?
        public var tokenizer: URL?
        public init(vae: URL? = nil, textEncoder: URL? = nil, tokenizer: URL? = nil) {
            self.vae = vae
            self.textEncoder = textEncoder
            self.tokenizer = tokenizer
        }

        func url(for c: Component) -> URL? {
            let u: URL? = switch c {
            case .vae: vae
            case .textEncoder: textEncoder
            case .tokenizer: tokenizer
            default: nil
            }
            guard let u else { return nil }
            let probe = c == .tokenizer ? u.appendingPathComponent("tokenizer.json") : u
            return FileManager.default.fileExists(atPath: probe.path) ? u : nil
        }
    }

    /// The override for `component` if set and present, else the first existing layout candidate.
    public static func find(_ component: Component, in models: URL, overrides: Overrides) -> URL? {
        overrides.url(for: component) ?? find(component, in: models)
    }

    /// The first existing candidate for `component`, or `nil`.
    public static func find(_ component: Component, in models: URL) -> URL? {
        component.candidates.first {
            FileManager.default.fileExists(atPath: models.appendingPathComponent(component.probe($0)).path)
        }.map { models.appendingPathComponent($0) }
    }

    /// The DiT a variant runs (Turbo distils SFT).
    public static func dit(for variant: MossVariant) -> Component {
        variant == .base ? .baseDiT : .sftDiT
    }

    /// Everything a variant needs, in load order.
    public static func required(for variant: MossVariant) -> [Component] {
        [dit(for: variant), .vae, .textEncoder, .tokenizer]
            + (variant == .turbo ? [.turboLoRA] : [])
    }

    /// The components `variant` needs that aren't under `models`.
    public static func missing(for variant: MossVariant, in models: URL) -> [Component] {
        required(for: variant).filter { find($0, in: models) == nil }
    }

    /// A name for a file `MossInspector` can't classify (upscaler weights, the Turbo step embedder),
    /// from its place in the layout; `nil` when the path means nothing to us.
    public static func label(for relativePath: String) -> String? {
        if let c = Component.allCases.first(where: { $0.candidates.contains(relativePath) }) { return c.title }
        if relativePath.hasPrefix(UpscaleModels.esrganFolder + "/") { return "ESRGAN / SPAN upscaler" }
        return nil
    }

    /// Turbo adapter files are applied by the Turbo variant itself, not picked as user LoRAs.
    public static func isTurboAdapter(_ relativePath: String) -> Bool {
        relativePath.hasPrefix(dmdFolder + "/") || relativePath.contains("moss_8step_dmd_turbo")
            || ModelSidecar.inferred(fileName: (relativePath as NSString).lastPathComponent, isLoRA: true, siblings: []).distill != nil
    }
}
