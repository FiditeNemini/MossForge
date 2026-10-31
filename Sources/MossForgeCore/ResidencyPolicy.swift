/// Settings → Memory: how many models stay on the GPU between jobs.
public enum MossResidencyMode: String, Sendable, Codable, CaseIterable, Identifiable {
    /// Decided from the Mac's recommended GPU working set (`MossResidencyPolicy.resolve`).
    case auto
    /// Keep everything loaded: the text encoder, the DiT and the enhancer / captioner / Help / VOSR2 together.
    case keepLoaded
    /// One big model at a time; the text encoder loads per generation (the small-Mac rules).
    case saveMemory

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .auto: "Automatic"
        case .keepLoaded: "Keep models loaded"
        case .saveMemory: "Save memory"
        }
    }
}

/// What stays resident. Pure, so the decision is unit-tested; the MLX backend only reads the two flags.
///
/// Footprints (bf16/fp16): DiT ~10 GB; one Qwen3.5-9B shared by the text encoder, enhancer, captioner and
/// Help (`SharedQwen35`): ~16 GB transformer + ~2 GB `lm_head` + ~0.9 GB vision tower; VOSR2 ~7 GB; plus
/// a few GB of activations at 1024² (60 GB process footprint with everything loaded on an M2 Ultra,
/// MLX's buffer cache included).
/// Idle unload still frees everything whichever policy is active.
public struct MossResidencyPolicy: Sendable, Equatable {
    /// Keep the text encoder between generations (saves its reload, ~2.7 s per image on an M2 Ultra).
    public var keepEncoderResident: Bool
    /// Don't evict the DiT for the enhancer, captioner, Help or VOSR2 (and vice versa).
    public var modelsCoexist: Bool

    public init(keepEncoderResident: Bool, modelsCoexist: Bool) {
        self.keepEncoderResident = keepEncoderResident
        self.modelsCoexist = modelsCoexist
    }

    public static let saveMemory = MossResidencyPolicy(keepEncoderResident: false, modelsCoexist: false)
    public static let keepLoaded = MossResidencyPolicy(keepEncoderResident: true, modelsCoexist: true)

    /// DiT + encoder + activations (~30 GB) with headroom: 96 GB Macs (~72 GB working set) and up;
    /// 64 GB Macs (~48 GB) keep the per-generation reload.
    public static let encoderThreshold = 64 << 30
    /// Every model at once (~36 GB + activations and cache): 128 GB Macs (~96–107 GB) and up.
    public static let coexistThreshold = 96 << 30

    /// - Parameter recommendedWorkingSet: Metal's `recommendedMaxWorkingSetSize` in bytes (0 = unknown).
    public static func resolve(_ mode: MossResidencyMode, recommendedWorkingSet: Int) -> MossResidencyPolicy {
        switch mode {
        case .keepLoaded: return .keepLoaded
        case .saveMemory: return .saveMemory
        case .auto:
            return MossResidencyPolicy(
                keepEncoderResident: recommendedWorkingSet >= encoderThreshold,
                modelsCoexist: recommendedWorkingSet >= coexistThreshold)
        }
    }

    /// One line for Settings and diagnostics.
    public var summary: String {
        switch (keepEncoderResident, modelsCoexist) {
        case (true, true): "The text encoder, DiT, enhancer, captioner and upscaler stay loaded together."
        case (true, false): "The text encoder and DiT stay loaded; other models load one at a time."
        case (false, true): "Models stay loaded together; the text encoder loads per generation."
        case (false, false): "One model at a time; the text encoder loads per generation."
        }
    }
}
