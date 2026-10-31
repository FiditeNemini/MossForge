import Foundation

/// Who the assistant is speaking for. The only product-specific input the prompts and the
/// feedback report need, so the rest of the template stays unchanged between apps.
public struct HelpProduct: Sendable, Equatable {
    /// "MossForge".
    public var name: String
    /// Completes "<name> is …": "a Mac app that runs the MOSS-Image text-to-image model on device
    /// (generation, LoRAs, upscaling, training, a local server)".
    public var summary: String
    /// Shown in the UI: "Qwen3-VL", "Qwen3.5".
    public var modelName: String
    /// Suggested first questions on the empty Ask screen.
    public var starters: [String]

    public init(name: String, summary: String, modelName: String, starters: [String] = []) {
        self.name = name
        self.summary = summary
        self.modelName = modelName
        self.starters = starters
    }
}

extension HelpProduct {
    /// MossForge's Help and Feedback identity (app, `moss-mlx ask`).
    public static let mossForge = HelpProduct(
        name: "MossForge",
        summary: "a Mac app that runs the MOSS-Image text-to-image model on device (generation, layouts, Turbo LoRAs, "
            + "upscaling, captioning, LoRA training, model tools, a local server)",
        modelName: "Qwen3.5",
        starters: [
            "What settings give the best quality?",
            "How do I put exact text on a poster?",
            "When should I turn Turbo off?",
            "How do I train a LoRA on my own images?",
            "Why is my generation slow?",
            "How do I share an image?",
        ])
}
