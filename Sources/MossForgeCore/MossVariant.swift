/// The MOSS-Image checkpoints MossForge knows how to run.
public enum MossVariant: String, Sendable, Codable, CaseIterable, Identifiable {
    /// MOSS-Image-Base: raw pretrain EMA. Best for LoRA training; needs ≥ 50 steps + CFG.
    case base
    /// MOSS-Image-SFT: post-trained release.
    case sft
    /// SFT + akhaliq's 4-step distillation LoRA on stock flow Euler, CFG 1 (`turboLoRAPath`). lodestones'
    /// SFT data-space Turbo was retired in 0.2.12 (an A/B showed washed-out images and stray text); the data-space
    /// sampler stays for Base 8-step and fused models.
    case turbo

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .base: "Base"
        case .sft: "SFT"
        case .turbo: "Turbo (4-step)"
        }
    }

    /// Recommended inference settings (USERGUIDE for base/SFT; model cards for Turbo).
    public var defaults: SamplingDefaults {
        switch self {
        case .base, .sft:
            SamplingDefaults(steps: 50, cfg: 3.0, shift: 5.0, sampler: .flowEuler)
        case .turbo:
            SamplingDefaults(steps: 4, cfg: 1.0, shift: 5.0, sampler: .flowEuler)
        }
    }

    /// The Turbo variant's LoRA, relative to the models folder (akhaliq MOSS-Image-SFT-4-steps v2.0).
    public static let turboLoRAPath = "lora/akhaliq/akhaliq-4step-sft.safetensors"
}

/// How the denoising loop runs.
public enum SamplerKind: String, Sendable, Codable, CaseIterable {
    /// Shifted flow-matching Euler (the release sampler) with optional CFG / SLG.
    case flowEuler
    /// Data-space DMD (Turbo): every step re-mixes fixed noise `z` with the previous clean
    /// estimate through `latent_mix.Q`, conditioned on learned `step_codes`.
    case dataSpaceDMD
}

public struct SamplingDefaults: Sendable, Equatable {
    public var steps: Int
    public var cfg: Double
    public var shift: Double
    public var sampler: SamplerKind
}

/// The USERGUIDE's recommended negative tags.
public enum MossNegativePrompt {
    public static let recommended = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
}

/// The Qwen3.5 chat wrapping the official tokenizer applies before encoding
/// (comfyui/moss-image/text_encoder.py).
public enum MossPromptTemplate {
    public static let maxTokens = 4096

    public static func wrap(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n"
    }
}
