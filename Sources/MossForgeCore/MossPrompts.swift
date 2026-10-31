import Foundation

/// The instructions MossForge sends to Qwen3.5 — the enhancer's rules and answer formats, the
/// recaptioner and the slider designer — editable in Settings → Prompts without a rebuild. Only edits are stored, so an improved built-in reaches anyone who never
/// changed that prompt. A blank edit stays blank in the editor but sends the built-in.
public struct MossPrompts: Sendable, Equatable, Codable {
    public enum Kind: String, Sendable, CaseIterable, Identifiable {
        case enhancerRules, enhancerCaption, enhancerLayout, caption, slider
        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .enhancerRules: "Prompt enhancer — rules"
            case .enhancerCaption: "Prompt enhancer — caption answer"
            case .enhancerLayout: "Prompt enhancer — layout answer"
            case .caption: "Captioner"
            case .slider: "Slider concept"
            }
        }

        /// Where it's used, and the placeholders it fills.
        public var detail: String {
            switch self {
            case .enhancerRules: "System prompt for Enhance and Layout from prompt, before the answer format below."
            case .enhancerCaption: "Answer format for Enhance (plain prompt). Must ask for {\"caption\": …} JSON."
            case .enhancerLayout: "Answer format for Layout from prompt. {width} and {height} become the canvas size."
            case .caption: "Caption and Train → caption the dataset. {{reference}} becomes the existing caption, or (none)."
            case .slider: "Train → Slider → Suggest. Must answer the four labelled lines."
            }
        }

        public var builtIn: String {
            switch self {
            case .enhancerRules: MossEnhancerPrompt.common
            case .enhancerCaption: MossEnhancerPrompt.caption
            case .enhancerLayout: MossEnhancerPrompt.layoutTemplate
            case .caption: MossCaptionPrompt.template
            case .slider: SliderConceptDraft.systemPrompt
            }
        }
    }

    private var edits: [Kind: String] = [:]

    public init() {}

    /// The text the editor shows: the edit, or the built-in. Setting the built-in clears the edit.
    public subscript(kind: Kind) -> String {
        get { edits[kind] ?? kind.builtIn }
        set { edits[kind] = newValue == kind.builtIn ? nil : newValue }
    }

    /// The text sent to the model: a blank edit falls back to the built-in.
    public func resolved(_ kind: Kind) -> String {
        let text = self[kind]
        return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? kind.builtIn : text
    }

    public func isDefault(_ kind: Kind) -> Bool { edits[kind] == nil }
    public var isDefault: Bool { edits.isEmpty }
    public mutating func reset(_ kind: Kind) { edits[kind] = nil }
    public mutating func resetAll() { edits.removeAll() }

    /// The enhancer's system prompt for a caption or a Layout draft on `canvas`.
    public func enhancerSystem(layout: Bool, canvas: Dimensions) -> String {
        let format = layout
            ? resolved(.enhancerLayout).replacingOccurrences(of: "{width}", with: "\(canvas.width)")
                .replacingOccurrences(of: "{height}", with: "\(canvas.height)")
            : resolved(.enhancerCaption)
        return resolved(.enhancerRules) + "\n\n" + format
    }

    // Stored as {kindRawValue: text}; unknown keys (a later release's prompts) are ignored.
    public init(from decoder: any Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: String].self)
        for (key, text) in raw { if let kind = Kind(rawValue: key) { self[kind] = text } }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(Dictionary(uniqueKeysWithValues: edits.map { ($0.key.rawValue, $0.value) }))
    }
}
