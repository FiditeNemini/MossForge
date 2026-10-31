import Foundation

// The wizard request is a Qwen3.5 chat (`messages(for:)`).

/// The apply-weight span a Concept-Slider LoRA is trained for. Each training step draws a LoRA
/// multiplier from this span (both sides of zero when it straddles it) and asks for a
/// proportional push along the concept direction, so the adapter behaves linearly across the
/// whole range and its far end reaches full concept strength. Recorded in the saved LoRA as its
/// recommended weight range.
public struct SliderRange: Sendable, Equatable, Codable {
    public var min: Double
    public var max: Double

    public static let `default` = SliderRange(min: -1, max: 1)

    public init(min: Double, max: Double) {
        self.min = min
        self.max = max
    }

    /// `"min,max"`, as saved in a trained slider's `mossforge_slider_range` metadata; `nil` unless valid.
    public init?(parsing text: String) {
        let parts = text.split(separator: ",").map { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard parts.count == 2, let lo = parts[0], let hi = parts[1] else { return nil }
        self.init(min: lo, max: hi)
        guard isValid else { return nil }
    }

    /// Weight 0 is always "off", so the range has to contain it (and not be empty).
    public var isValid: Bool { min <= 0 && max >= 0 && min < max }

    /// The weight band saved with a trained slider: the trained negative end, and twice the
    /// trained positive end. Slider round 2 (2026-10-17) found η 1 sliders trained on ±1 stay
    /// clean but need +1.5…+2 to age clearly, so the positive side is worth overdriving.
    public var recommendedBand: SliderRange { SliderRange(min: min, max: max * 2) }

    /// The magnitude of the range's far end.
    public var peak: Double { Swift.max(abs(min), abs(max)) }

    /// Concept push per unit of LoRA weight, chosen so the far end equals `full`.
    public func unitStrength(full: Double) -> Double { full / peak }

    /// Multipliers closer to zero than this teach nothing (the target ≈ the base) — skipped.
    private var deadZone: Double { 0.1 * peak }

    /// Map `u ∈ [0, 1)` onto a training multiplier across the range, minus the dead zone
    /// around 0: the negative side `[min, −dz]` first, then the positive side `[dz, max]`.
    public func multiplier(at u: Double) -> Double {
        let dz = deadZone
        let negative = Swift.max(0, -min - dz)
        let positive = Swift.max(0, max - dz)
        let x = Swift.min(Swift.max(u, 0), 1) * (negative + positive)
        return x < negative ? min + x : dz + (x - negative)
    }
}

/// A slider concept drafted by the on-device LLM from a plain description ("a digital to
/// analog slider"): the fields the Train screen's slider mode needs.
public struct SliderConceptDraft: Sendable, Equatable {
    public var name: String
    public var target: String
    public var positive: String
    public var negative: String

    public init(name: String, target: String, positive: String, negative: String) {
        self.name = name
        self.target = target
        self.positive = positive
        self.negative = negative
    }

    /// Instructions for Qwen3.5. A worked example pins the reply format, and the
    /// "X to Y" convention matches the slider: +weight pushes toward positive (Y).
    public static let systemPrompt = """
        You design Concept Slider LoRAs for a text-to-image model. A slider has a TARGET (the \
        neutral subject the slider applies to), a POSITIVE prompt (what +weight pushes toward) \
        and a NEGATIVE prompt (what −weight pushes toward). For "X to Y", X is negative and Y is \
        positive. Positive and negative are short comma-separated visual descriptors that differ \
        ONLY in the concept, mirror each other, and share no other content. When a request is \
        ambiguous, read it as the image's overall look (photography, medium, style, mood) rather \
        than a specific object, and keep the target generic so the slider works on many images. \
        Reply with exactly four lines and nothing else:
        Name: <snake_case_name>
        Target: <neutral subject>
        Positive: <descriptors>
        Negative: <descriptors>

        Example request: a young to old slider
        Name: young_to_old
        Target: a portrait of a person
        Positive: elderly, old, wrinkled skin, grey hair
        Negative: young, youthful, smooth skin, vibrant hair
        """

    /// The chat for a description: the slider instructions, then the user's words.
    public static func messages(for description: String, prompts: MossPrompts = MossPrompts()) -> [ChatMessage] {
        [ChatMessage(.system, prompts.resolved(.slider)), ChatMessage(.user, description)]
    }

    /// Read the four labelled lines out of a model reply. Tolerates markdown bullets/bold,
    /// quotes, a `<think>` block, label case and parentheticals (`Positive (+):`). `nil` if any
    /// field is missing — e.g. the stub backend's plain prompt expansion.
    public static func parse(_ reply: String) -> SliderConceptDraft? {
        var text = reply
        if let open = text.range(of: "<think>"), let close = text.range(of: "</think>") {
            text.removeSubrange(open.lowerBound..<close.upperBound)
        }
        var fields: [String: String] = [:]
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.replacingOccurrences(of: "**", with: "")
                .trimmingCharacters(in: CharacterSet(charactersIn: "-*•> \t"))
            guard let colon = line.firstIndex(of: ":") else { continue }
            let label = line[..<colon].lowercased().filter(\.isLetter)
            var value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            value = value.trimmingCharacters(in: CharacterSet(charactersIn: "\"'`“”"))
            guard !value.isEmpty else { continue }
            let key: String? =
                label.hasPrefix("name") || label.hasPrefix("slider") ? "name"
                : label.hasPrefix("target") || label.hasPrefix("subject") ? "target"
                : label.hasPrefix("pos") ? "positive"
                : label.hasPrefix("neg") ? "negative" : nil
            if let key, fields[key] == nil { fields[key] = value }
        }
        guard let name = fields["name"], let target = fields["target"],
            let positive = fields["positive"], let negative = fields["negative"]
        else { return nil }
        return SliderConceptDraft(name: slug(name), target: target, positive: positive, negative: negative)
    }

    /// Offline fallback for "X to Y" phrasing (no LLM, or an unreadable reply).
    public static func heuristic(_ description: String) -> SliderConceptDraft? {
        let text = description.lowercased().trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
        guard let sep = [" to ", " → ", " -> ", " vs ", " versus "].lazy.compactMap({ text.range(of: $0) }).first
        else { return nil }
        let filler: Set<String> = [
            "a", "an", "the", "like", "want", "make", "create", "build", "need", "please", "me", "i'd", "i",
            "id", "would", "slider", "lora", "for", "from",
        ]
        // Left concept: the trailing words before the separator, up to the first filler word.
        let leftWords = text[..<sep.lowerBound].split(separator: " ").map(String.init)
        let left = leftWords.reversed().prefix { !filler.contains($0) }.reversed().joined(separator: " ")
        // Right concept: everything after, minus a trailing "slider"/"lora".
        var rightWords = text[sep.upperBound...].split(separator: " ").map(String.init)
        while let last = rightWords.last, filler.contains(last) { rightWords.removeLast() }
        let right = rightWords.joined(separator: " ")
        guard !left.isEmpty, !right.isEmpty else { return nil }
        return SliderConceptDraft(
            name: slug("\(left) to \(right)"), target: "an image", positive: right, negative: left)
    }

    /// A file-safe snake_case LoRA name.
    public static func slug(_ s: String) -> String {
        var out = ""
        for ch in s.lowercased() {
            if ch.isASCII, ch.isLetter || ch.isNumber {
                out.append(ch)
            } else if !out.isEmpty, out.last != "_" {
                out.append("_")
            }
        }
        while out.hasSuffix("_") { out.removeLast() }
        return out.isEmpty ? "slider" : out
    }
}
