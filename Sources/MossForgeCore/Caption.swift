import Foundation

/// How much of the recaption to use as "the" caption.
public enum CaptionDetail: String, Sendable, Codable, CaseIterable, Identifiable {
    case long, medium, short
    public var id: String { rawValue }
}

/// Caption one image with the on-device VLM (Qwen3.5-9B's vision tower + language model).
public struct CaptionRequest: Sendable {
    public var image: URL
    public var detail: CaptionDetail
    /// An existing caption to correct (the prompt's `{{reference}}`); `nil` = none.
    public var reference: String?
    /// Pixel budget for the vision tower (Qwen `smart_resize` max); 1024² ≈ 1024 image tokens.
    public var maxPixels: Int
    public var seed: UInt64
    /// The recaption instruction (Settings → Prompts); built-in by default.
    public var prompts: MossPrompts

    /// Qwen3.5's processor minimum (`shortest_edge` 65536 = 256²).
    public static let minPixels = 256 * 256
    public static let defaultMaxPixels = 1024 * 1024
    public static let maxPixelsLimit = 2048 * 2048

    public init(
        image: URL, detail: CaptionDetail = .long, reference: String? = nil,
        maxPixels: Int = CaptionRequest.defaultMaxPixels, seed: UInt64 = 0, prompts: MossPrompts = MossPrompts()
    ) {
        self.prompts = prompts
        self.image = image
        self.detail = detail
        self.reference = reference
        self.maxPixels = min(max(maxPixels, Self.minPixels), Self.maxPixelsLimit)
        self.seed = seed
    }
}

/// The recaption answer: the official schema's three captions, watermarks and borders.
public struct MossCaption: Sendable, Equatable, Codable {
    public struct Watermark: Sendable, Equatable, Codable {
        /// Normalised 0–1000 `[y_min, x_min, y_max, x_max]`.
        public var box: [Double]
        public var label: String
        public init(box: [Double], label: String) {
            self.box = box
            self.label = label
        }
    }

    public var long: String
    public var medium: String
    public var short: String
    public var watermarks: [Watermark]
    public var borders: [String]

    public init(long: String, medium: String, short: String, watermarks: [Watermark] = [], borders: [String] = ["none"]) {
        self.long = long
        self.medium = medium
        self.short = short
        self.watermarks = watermarks
        self.borders = borders
    }

    public func text(_ detail: CaptionDetail) -> String {
        switch detail {
        case .long: long
        case .medium: medium
        case .short: short
        }
    }

    /// The JSON object in a model answer (fences or chatter around it are tolerated). Throws
    /// `invalidRequest` unless all three captions are non-empty strings.
    public static func parse(_ answer: String) throws -> MossCaption {
        guard let json = MossPromptCompiler.extractJSON(from: answer),
            let obj = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any]
        else { throw EngineError.invalidRequest("no JSON object in the caption answer") }
        let caption = obj["caption"] as? [String: Any] ?? [:]
        func field(_ name: String) throws -> String {
            guard let s = (caption[name] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else {
                throw EngineError.invalidRequest("caption.\(name) must be a non-empty string")
            }
            return s
        }
        let marks = (obj["watermark"] as? [[String: Any]] ?? []).compactMap { m -> Watermark? in
            guard let box = m["box_2d"] as? [NSNumber], box.count == 4 else { return nil }
            return Watermark(box: box.map(\.doubleValue), label: m["label"] as? String ?? "")
        }
        return MossCaption(
            long: try field("long"), medium: try field("medium"), short: try field("short"),
            watermarks: marks, borders: obj["borders"] as? [String] ?? ["none"])
    }
}

/// The official recaption prompt (`Reference/MOSS-Image/USERGUIDE.md`, "Image Captioning") and its
/// Qwen3.5 chat wrapping: the instruction, then the image (the USERGUIDE request order), with a
/// non-thinking generation prompt.
public enum MossCaptionPrompt {
    public static let imagePad = "<|image_pad|>"
    public static let stopToken = "<|im_end|>"

    /// The prompt with `{{reference}}` filled in (`(none)` when there is no reference).
    public static func instruction(reference: String?, prompts: MossPrompts = MossPrompts()) -> String {
        let ref = reference?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return prompts.resolved(.caption).replacingOccurrences(of: "{{reference}}", with: ref.isEmpty ? "(none)" : ref)
    }

    /// The full chat for an image of `imageTokens` merged tokens.
    public static func chat(imageTokens: Int, reference: String?, prompts: MossPrompts = MossPrompts()) -> String {
        "<|im_start|>user\n" + instruction(reference: reference, prompts: prompts)
            + "<|vision_start|>" + String(repeating: imagePad, count: imageTokens) + "<|vision_end|>"
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    static let template = """
        You are an image recaptioner. Return exactly one JSON object with this schema:
        {
          "caption": {
            "long": "a detailed image description, including all non-watermark visible text",
            "medium": "a one-paragraph image description",
            "short": "a one-sentence image description"
          },
          "watermark": [
            {"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}
          ],
          "borders": ["none"]
        }

        Rules:
        - caption.long:
            - Describe the composition, key subjects, actions, setting, colors, lighting, and artistic style in detail. Include only information visible in the image.
            - Preserve visible text exactly, excluding watermarks. Do not translate, paraphrase, romanize, or normalize image text.
            - Describe text layout as much as possible: line breaks, ordering, relative position, alignment, size hierarchy, font/style, color, spacing, and how text blocks are arranged.
        - caption.medium: compress long into one paragraph.
        - caption.short: summarize the main subject, action, or relationship in one sentence.
        - Do not describe watermarks in any caption field. Record them only in watermark.
        - watermark:
            - Use normalized 0–1000 [y_min, x_min, y_max, x_max] coordinates relative to the full image.
            - Set label to the visible watermark text, or a short description if unreadable.
            - Use [] if there are no watermarks.
        - borders:
            - Record evident borders or margins using one or more of "left", "right", "up", and "down".
            - Use ["none"] if there are no evident borders or margins. Do not combine "none" with other directions.
        - All three caption fields must be non-empty strings.
        - Treat the original caption as a reference. Correct any claims that are not supported by the image.
        - Return only the JSON object, without Markdown or explanations.

        Here is the original caption reference:
        {{reference}}
        """
}
