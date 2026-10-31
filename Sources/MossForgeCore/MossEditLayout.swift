import Foundation

/// How an edit LoRA was trained to see its reference images. Every layout puts reference `i` on
/// RoPE r = i + 1 with its own zero-centred h/w grid, conditioned at timestep 0; they differ in
/// what the references attend to, where they sit on the sequence axis and whether the text encoder
/// sees them too. These mirror ComfyUI's `EDIT_FORMATS` (PR #17296, `nodes_moss_image.py`).
public enum MossEditLayout: String, CaseIterable, Sendable, Codable, Identifiable {
    /// MossForge's Train → Edit: references in the sequence, text-only encoder.
    case mossForge
    /// ostris ai-toolkit, kv_cache on (`index_timestep_zero_isolated`) — akhaliq/MOSS-Image-Edit-HQ.
    case aiToolkit
    /// ostris ai-toolkit, kv_cache off (`index_timestep_zero`).
    case aiToolkitNoCache
    /// chinoll/Moss-Image-Edit's `moss-edit-lora` (`index_timestep_zero_text_start`).
    case mossImageEdit

    public var id: String { rawValue }

    /// A raw value or a ComfyUI `edit_format` name (`ai-toolkit`, `Moss-Image-Edit`, …), any case.
    public init?(name: String) {
        let n = name.trimmingCharacters(in: .whitespaces).lowercased()
        let comfy: [String: MossEditLayout] = [
            "ai-toolkit": .aiToolkit, "ai-toolkit (kv_cache off)": .aiToolkitNoCache, "moss-image-edit": .mossImageEdit,
            "mossforge": .mossForge,
        ]
        guard let layout = comfy[n] ?? Self.allCases.first(where: { $0.rawValue.lowercased() == n }) else { return nil }
        self = layout
    }

    public var displayName: String {
        switch self {
        case .mossForge: "MossForge (Train → Edit)"
        case .aiToolkit: "ai-toolkit"
        case .aiToolkitNoCache: "ai-toolkit (kv_cache off)"
        case .mossImageEdit: "Moss-Image-Edit"
        }
    }

    /// References attend only to each other, so their keys and values are computed once per image
    /// and appended to every block's attention; otherwise they ride along in the sequence.
    public var isolatedReferences: Bool { self == .aiToolkit || self == .mossImageEdit }

    /// References at the text's first sequence position instead of the image's.
    public var referencesAtTextStart: Bool { self == .mossImageEdit }

    /// The text encoder also sees each reference ("Picture N:" + Qwen3.5 vision tokens, ≤ 384²).
    public var encoderSeesReferences: Bool { self == .aiToolkit || self == .aiToolkitNoCache }

    /// The layout for a request: an explicit choice, else an edit LoRA's, else any detected one,
    /// else MossForge's (what references meant before layouts existed).
    public static func resolve(explicit: MossEditLayout?, loras: [MossEditLoRAInfo]) -> MossEditLayout {
        explicit ?? loras.first(where: \.isEdit)?.layout ?? loras.compactMap(\.layout).first ?? .mossForge
    }
}

extension GenerationRequest {
    /// An edit (references) on the data-space DMD loop, which re-mixes the clean estimate with fixed
    /// noise and doesn't carry reference tokens — the backend moves it to flow Euler.
    public var editRunsOnFlow: Bool { !references.isEmpty && sampler == .dataSpaceDMD }

    /// A Turbo as Edit-HQ stacks it for edits: the LoRA alone on flow Euler, 8 steps at most, CFG 1,
    /// shift 5, no SLG (the caller moves the Turbo LoRA onto `loras` and drops its data-space adapters).
    public mutating func moveTurboToFlow() {
        sampler = .flowEuler
        steps = min(steps, 8)
        cfg = 1
        shift = 5
        slg = .off
        negativePrompt = ""
    }
}

/// What a LoRA file's safetensors metadata (and name) says about editing.
public struct MossEditLoRAInfo: Sendable, Equatable {
    /// The layout it was trained with, when the metadata tells.
    public var layout: MossEditLayout?
    /// An edit LoRA (wants reference images), not just one from an edit-capable trainer.
    public var isEdit: Bool

    public init(layout: MossEditLayout?, isEdit: Bool) {
        self.layout = layout
        self.isEdit = isEdit
    }

    public init(metadata: [String: String], filename: String) {
        let adapterFormat = metadata["adapter_config"]
            .flatMap { $0.data(using: .utf8) }
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["format"] as? String
        if let raw = metadata["mossforge_edit_layout"], let ours = MossEditLayout(rawValue: raw) {
            layout = ours
            isEdit = true
        } else if adapterFormat == "moss-edit-lora" {
            layout = .mossImageEdit
            isEdit = true
        } else {
            layout = (metadata["software"] ?? "").contains("ai-toolkit") ? .aiToolkit
                : metadata["produced_by"] == "MossForge" ? .mossForge : nil
            isEdit = filename.lowercased().contains("edit")
        }
    }
}

/// The edit instruction as the text encoder sees it when the layout shows it the references.
public enum MossEditPrompt {
    public static let visionStart = "<|vision_start|>"
    public static let visionEnd = "<|vision_end|>"
    public static let imagePad = "<|image_pad|>"
    /// The encoder sees each reference at an area of at most 384² (ComfyUI).
    public static let visionPixels = 384 * 384

    /// `"Picture 1: <|vision_start|><|image_pad|>…<|vision_end|>…" + prompt` (trimmed), one picture
    /// per reference with its merged-token count of `<|image_pad|>`.
    public static func instruction(_ prompt: String, pictureTokens: [Int]) -> String {
        pictureTokens.enumerated().map { i, n in
            "Picture \(i + 1): " + visionStart + String(repeating: imagePad, count: n) + visionEnd
        }.joined() + prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
