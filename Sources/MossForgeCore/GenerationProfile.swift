import Foundation

/// A named model + Turbo LoRA + steps combination for Generate's Profile menu. Paths are relative to
/// the models folder (the `lora/<user>/<user>-<steps>-<family>` layout); a profile whose files are
/// missing is shown but can't be picked.
public struct GenerationProfile: Sendable, Codable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var detail: String
    /// The DiT.
    public var model: String
    /// The Turbo LoRA, or `nil` for the model's full sampling (50 steps, CFG).
    public var turbo: String?
    /// A steps override; `nil` = the Turbo recipe's (or the model's default).
    public var steps: Int?

    public init(id: String, name: String, detail: String, model: String, turbo: String?, steps: Int? = nil) {
        self.id = id
        self.name = name
        self.detail = detail
        self.model = model
        self.turbo = turbo
        self.steps = steps
    }

    public var isBuiltIn: Bool { Self.builtIn.contains { $0.id == id } }

    /// Files this profile needs that aren't in `library` (relative paths).
    public func missing(in library: Set<String>) -> [String] {
        ([model] + (turbo.map { [$0] } ?? [])).filter { !library.contains($0) }
    }

    /// The first profile describing exactly this selection.
    public static func matching(model: String?, turbo: String?, steps: Int?, in profiles: [GenerationProfile]) -> GenerationProfile? {
        profiles.first { $0.model == model && $0.turbo == turbo && $0.steps == steps }
    }

    static let sft = "diffusion_models/moss_image_sft_bf16.safetensors"
    static let base = "diffusion_models/moss_image_base_bf16.safetensors"

    /// Draft → full quality, measured on this machine (512²: 5.4 / 6.5 / ~10 s; 1024² × 50 ≈ 4 min).
    public static let builtIn: [GenerationProfile] = [
        .init(id: "draft", name: "Draft — fastest", detail: "SFT + akhaliq 2-step. Previews and prompt iteration; softer detail.",
            model: sft, turbo: "lora/akhaliq/akhaliq-2step-sft.safetensors"),
        .init(id: "fast", name: "Fast — 4 steps", detail: "SFT + akhaliq 4-step. The everyday pick: quick, best text and numbers. 6–8 steps above 1024².",
            model: sft, turbo: MossVariant.turboLoRAPath),
        .init(id: "sft-8step", name: "8 steps — lodestones", detail: "SFT + lodestones lora_latest at 0.7 (flow, 8 steps). The 8-step alternative to Fast; a little slower.",
            model: sft, turbo: "lora/lodestones/lodestones-8step-sft-latest.safetensors"),
        .init(id: "full", name: "Full quality — 50 steps", detail: "SFT without Turbo: CFG 3, negative prompt, SLG. Slowest; the reference.",
            model: sft, turbo: nil),
        .init(id: "base-turbo", name: "Base — Turbo", detail: "Base + lodestones 8-step Base (data space). Quick checks of Base and Base LoRAs.",
            model: base, turbo: "lora/lodestones/lodestones-8step-base.safetensors"),
        .init(id: "base-full", name: "Base — full", detail: "Base without Turbo: the raw pretrain, for LoRA training checks.",
            model: base, turbo: nil),
    ]
}
