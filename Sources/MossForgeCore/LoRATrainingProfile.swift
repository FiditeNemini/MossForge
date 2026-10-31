import Foundation

// LoRA training profiles: MOSS layer sets (all-linear = attention + FFN), a run-length scale for the Train page's Quick /
// Standard / Detailed presets, and a checkpoint interval.

/// What kind of LoRA an image dataset is for. Each type fills the Train page's hyperparameters
/// with a starting point for that class of adapter; Custom leaves them as they are. The Quick /
/// Standard / Detailed presets stay the run-length (and resolution) axis beside it.
///
/// The numbers follow common community practice for Flux / Qwen-Image class DiTs, scaled from
/// MossForge's defaults (rank 16, 1e-4) — starting points, not yet measured on MOSS-Image:
/// - Character: identity lives in fine detail, so attention + FFN at rank 32, ~80 steps per image.
/// - Concept (an object, pose, outfit, action): composition, so attention only at rank 16.
/// - Style: texture and palette everywhere, so attention + FFN, and the longest runs.
public enum LoRATrainingProfile: String, Sendable, Codable, CaseIterable, Identifiable {
    case custom, character, concept, style

    public var id: String { rawValue }

    public struct Settings: Sendable, Equatable {
        public var rank: Int
        public var learningRate: Double
        public var steps: Int
        public var targets: LoRATargetSet
        public var factorization: LoRAFactorization
        /// About four checkpoints per run (each gets the test images).
        public var saveEvery: Int
    }

    public var title: String {
        switch self {
        case .custom: "Custom"
        case .character: "Character"
        case .concept: "Concept"
        case .style: "Style"
        }
    }

    public var summary: String {
        switch self {
        case .custom: "Every hyperparameter by hand."
        case .character:
            "One person or character, consistent from close-up to full body. Attention + FFN at rank 32 so "
                + "the face and build come through."
        case .concept:
            "An object, outfit, pose or action you can place in new scenes. Attention only at rank 16, so "
                + "it learns the idea without the training images' look."
        case .style:
            "A look — medium, brushwork, palette, lighting — applied to any subject. Attention + FFN and the "
                + "longest run, so texture transfers."
        }
    }

    /// How to caption this kind of dataset (the part people most often get wrong).
    public var captionTip: String {
        switch self {
        case .custom: ""
        case .character:
            "Start each caption with the trigger word, then describe the shot, pose, outfit and setting — "
                + "not the face or hair you want the LoRA to learn."
        case .concept:
            "Name the concept with the trigger word and describe everything else in the image, so only the "
                + "concept is tied to the trigger."
        case .style:
            "Describe the subject and scene plainly and leave the style out (or use only the trigger for it), "
                + "so the look attaches to the trigger rather than to the subjects."
        }
    }

    /// The dataset size this type is tuned for.
    public var recommendedImages: ClosedRange<Int> {
        switch self {
        case .custom, .character: 15...30
        case .concept: 15...40
        case .style: 20...50
        }
    }

    var rank: Int { self == .concept ? 16 : 32 }
    var targets: LoRATargetSet { self == .concept ? .attention : .attentionAndFFN }
    var stepsPerImage: Int { self == .concept ? 60 : 80 }

    /// Steps stay within this band whatever the dataset size (before the run-length scale).
    public var stepBand: ClosedRange<Int> {
        switch self {
        case .custom, .character: 1000...2500
        case .concept: 800...2000
        case .style: 1500...3000
        }
    }

    /// The run length before a dataset is picked.
    public var typicalSteps: Int {
        switch self {
        case .custom, .character: 2000
        case .concept: 1500
        case .style: 2500
        }
    }

    /// The settings for a dataset of `imageCount` images (0 = not picked yet); `nil` for Custom.
    /// `runScale` is the preset's run length (Quick 0.1, Standard 1, Detailed 1.5), applied after
    /// the band so a quick check stays quick.
    public func settings(imageCount: Int, runScale: Double = 1) -> Settings? {
        guard self != .custom else { return nil }
        let raw = imageCount > 0 ? imageCount * stepsPerImage : typicalSteps
        let banded = min(max(raw, stepBand.lowerBound), stepBand.upperBound)
        let steps = max(50, Self.round50(Double(banded) * runScale))
        return Settings(
            rank: rank, learningRate: 1e-4, steps: steps, targets: targets, factorization: .lora,
            saveEvery: max(50, Self.round50(Double(steps) / 4)))
    }

    static func round50(_ v: Double) -> Int { Int((v / 50).rounded()) * 50 }
}
