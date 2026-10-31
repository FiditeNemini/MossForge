import Foundation

/// The on-device prompt enhancer's instructions: the official `moss-image-prompt-enhancer` skill
/// (SKILL.md + references/layout.md, Apache-2.0, OpenMOSS) condensed into a system prompt for Qwen3.5-9B,
/// asked to answer with a **draft** that `MossPromptCompiler` validates and compiles. Rendered with the
/// Qwen3.5 chat template in non-thinking mode (`<think>\n\n</think>\n\n` — `enable_thinking=False`).
public enum MossEnhancerPrompt {
    static let common = """
        You expand a short image idea into a detailed prompt for MOSS-Image, a text-to-image model that \
        renders exact visible text well. Preserve the user's subjects, counts, relationships, medium, style, \
        exclusions and intended visible wording; resolve ordinary creative gaps yourself. Describe visible \
        choices: subject and action, appearance, setting, framing, medium, lighting, materials, palette and \
        mood. Avoid quality tags and invisible process notes. Default to English prose; keep visible text \
        verbatim in its original language, punctuation, capitalization and line breaks. Write real copy, never \
        placeholders. Answer with ONE JSON object and nothing else — no code fences, no commentary.
        """

    static let caption = """
        Output exactly {"caption": "..."} — cohesive prose with spatial relations in words and any visible \
        wording quoted exactly. No coordinates.
        """

    /// `{width}` / `{height}` are filled with the canvas (`MossPrompts.enhancerSystem`).
    static let layoutTemplate = """
        Output a Layout draft for a {width}x{height} canvas:
        {"width": {width}, "height": {height}, "bbox_coord_space": "abs", "bbox_format": "xyxy",
         "summary": {"meta": [genre tags], "canvas": "single", "caption": "global description"},
         "items": [{"label": "object"|"character"|"text", "caption": "visible appearance", "text": "exact text (text items only)", "bbox": [x_min, y_min, x_max, y_max]}]}
        Boxes are actual pixels from the canvas top-left: x in 0..{width}, y in 0..{height}, x_min < x_max, \
        y_min < y_max. Use boxes for coherent visible units; overlap is allowed; a box is a placement, not a \
        drawn border. Text items need the exact visible characters in "text" (use \\n for intended line \
        breaks) and boxes large enough for the glyphs at this size. Establish reading order and prominence. \
        Use only these fields.
        """

    /// System prompt for a caption or a Layout draft.
    public static func system(layout: Bool, canvas: Dimensions, prompts: MossPrompts = MossPrompts()) -> String {
        prompts.enhancerSystem(layout: layout, canvas: canvas)
    }

    /// The full chat transcript, ready for the tokenizer (special tokens recognised there).
    /// `retry` adds the previous answer and its validation error so the model can fix the draft.
    public static func chat(
        prompt: String, layout: Bool, canvas: Dimensions, retry: (answer: String, error: String)? = nil, prompts: MossPrompts = MossPrompts()
    ) -> String {
        var s = "<|im_start|>system\n\(system(layout: layout, canvas: canvas, prompts: prompts))<|im_end|>\n"
        s += "<|im_start|>user\n\(prompt)<|im_end|>\n"
        if let retry {
            s += "<|im_start|>assistant\n<think>\n\n</think>\n\n\(retry.answer)<|im_end|>\n"
            s += "<|im_start|>user\nThat draft is invalid: \(retry.error). Answer again with a corrected JSON object only.<|im_end|>\n"
        }
        return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// Generation stops at the end of the assistant turn.
    public static let stopToken = "<|im_end|>"
}
