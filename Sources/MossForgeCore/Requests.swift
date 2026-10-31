import Foundation

/// Skip-layer guidance, exactly as the official pipeline exposes it (OpenMOSS-Team/MOSS-Image-SFT
/// `pipeline_moss_image.py`, defaults from `gradio_app.py`):
///
/// - **uncond skip** (`uncondSkipLayers`, default `[22]`): the CFG negative-branch forward runs with
///   these DiT blocks skipped — still two forwards per step.
/// - **standard SLG** (`layers` + `scale`): an extra positive forward with `layers` skipped, adding
///   `scale·(cond − cond_skip)`. USERGUIDE: scale 3, window 0.01–1.0.
///
/// Both are active only where `start ≤ step/steps < stop`; outside it, plain CFG.
public struct SkipLayerGuidance: Sendable, Equatable, Codable {
    public var uncondSkipLayers: [Int]
    public var layers: [Int]
    public var scale: Double
    public var start: Double
    public var stop: Double

    public init(uncondSkipLayers: [Int], layers: [Int], scale: Double, start: Double, stop: Double) {
        self.uncondSkipLayers = uncondSkipLayers
        self.layers = layers
        self.scale = scale
        self.start = start
        self.stop = stop
    }

    /// No skipping at all: plain CFG ("more photoreal, but details may collapse" — USERGUIDE).
    public static let off = SkipLayerGuidance(uncondSkipLayers: [], layers: [], scale: 0, start: 0, stop: 1)
    /// The release default: skip block 22 in the negative branch on every step.
    public static let official = SkipLayerGuidance(uncondSkipLayers: [22], layers: [], scale: 0, start: 0, stop: 1)

    /// Standard SLG (no negative-branch skipping).
    public static func standard(layers: [Int], scale: Double = 3, start: Double = 0.01, stop: Double = 1) -> SkipLayerGuidance {
        SkipLayerGuidance(uncondSkipLayers: [], layers: layers, scale: scale, start: start, stop: stop)
    }

    /// `start ≤ index/steps < stop`.
    public func isActive(step index: Int, of steps: Int) -> Bool {
        let f = Double(index) / Double(steps)
        return start <= f && f < stop
    }

    func validate(depth: Int) throws {
        for (name, l) in [("uncond skip", uncondSkipLayers), ("SLG", layers)] {
            guard l.allSatisfy({ (0..<depth).contains($0) }), Set(l).count == l.count else {
                throw EngineError.invalidRequest("\(name) layers must be unique block indices in 0..<\(depth) (got \(l))")
            }
        }
        guard scale.isFinite, scale >= 0 else { throw EngineError.invalidRequest("SLG scale must be finite and ≥ 0") }
        guard 0 <= start, start <= stop, stop <= 1 else { throw EngineError.invalidRequest("SLG window needs 0 ≤ start ≤ stop ≤ 1") }
    }
}

/// Per-block LoRA strength multipliers over the DiT's 42 single-stream blocks (`layers.N`), plus one
/// multiplier for everything outside them (embedders, modulation, text adapter, head).
///
/// Spec syntax:
/// - bare ranges `"21-41"` / `"3,4,21-41"` — only those blocks, at 1; all other blocks and non-block
///   modules 0 (Kijai's SFT fix: the SFT LoRA's later blocks at −0.5 → `file@-0.5:21-41`);
/// - weighted `"0-20=0,21-41=0.5"` — listed blocks get their value, unlisted stay 1; `*=x` sets the
///   unlisted blocks, `other=x` the non-block modules.
public struct LoRABlockWeights: Sendable, Equatable, Codable {
    public var blocks: [Int: Double]
    public var unlisted: Double
    public var other: Double

    public init(blocks: [Int: Double], unlisted: Double, other: Double) {
        self.blocks = blocks
        self.unlisted = unlisted
        self.other = other
    }

    /// Multiplier for a block index, or for a non-block module when `block` is nil.
    public func weight(block: Int?) -> Double {
        guard let block else { return other }
        return blocks[block] ?? unlisted
    }

    public init?(parsing spec: String) {
        let items = spec.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard !items.isEmpty else { return nil }
        let weighted = items.contains { $0.contains("=") }
        var blocks: [Int: Double] = [:]
        var unlisted: Double = weighted ? 1 : 0
        var other: Double = weighted ? 1 : 0
        for item in items {
            let kv = item.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            if weighted && kv.count != 2 { return nil }
            let value: Double
            if weighted { guard let v = Double(kv[1]) else { return nil }; value = v } else { value = 1 }
            switch kv[0] {
            case "*": unlisted = value
            case "other": other = value
            default:
                let bounds = kv[0].split(separator: "-").map { Int($0) }
                guard (1...2).contains(bounds.count), bounds.allSatisfy({ $0 != nil && $0! >= 0 }) else { return nil }
                let lo = bounds[0]!, hi = bounds.last!!
                guard lo <= hi else { return nil }
                for b in lo...hi { blocks[b] = value }
            }
        }
        self.init(blocks: blocks, unlisted: unlisted, other: other)
    }
}

/// A LoRA to apply at inference.
public struct LoRASpec: Sendable, Equatable, Codable {
    public var url: URL
    public var strength: Double
    /// Optional per-block multipliers (nil = uniform).
    public var blocks: LoRABlockWeights?

    public init(url: URL, strength: Double = 1.0, blocks: LoRABlockWeights? = nil) {
        self.url = url
        self.strength = strength
        self.blocks = blocks
    }

    /// Effective strength for a module in `block` (nil = non-block module).
    public func strength(forBlock block: Int?) -> Double {
        strength * (blocks?.weight(block: block) ?? 1)
    }

    /// CLI syntax `path[@strength[:blocks]]`, e.g. `kijai.safetensors@-0.5:21-41`.
    public init?(parsing s: String) {
        let parts = s.split(separator: "@", maxSplits: 1).map(String.init)
        guard let path = parts.first, !path.isEmpty else { return nil }
        var strength = 1.0
        var blocks: LoRABlockWeights?
        if parts.count == 2 {
            let sb = parts[1].split(separator: ":", maxSplits: 1).map(String.init)
            guard let v = Double(sb[0]) else { return nil }
            strength = v
            if sb.count == 2 { guard let b = LoRABlockWeights(parsing: sb[1]) else { return nil }; blocks = b }
        }
        self.init(url: URL(fileURLWithPath: path), strength: strength, blocks: blocks)
    }
}

/// One text-to-image request. Unset sampling fields fall back to the variant's defaults.
public struct GenerationRequest: Sendable, Equatable {
    public var prompt: String
    public var negativePrompt: String
    public var variant: MossVariant
    public var width: Int
    public var height: Int
    public var steps: Int
    public var cfg: Double
    public var shift: Double
    public var sampler: SamplerKind
    public var slg: SkipLayerGuidance
    public var seed: UInt64?
    public var loras: [LoRASpec]
    /// A specific DiT file (a library model); `nil` = the variant's default under `models/`.
    public var model: URL?
    /// A few-step distillation LoRA and its sampling (set with `apply(_:)`).
    public var distill: DistillSpec?
    /// Edit in Layout: redraw only the masked area of `edit.source`, pinning the rest to it.
    public var edit: ImageEditSpec?
    /// Reference images for an edit LoRA (community method: clean tokens on the RoPE r axis). Up to 4.
    public var references: [URL] = []
    /// How the edit LoRA sees `references`; `nil` = detect from the LoRAs' metadata.
    public var editLayout: MossEditLayout?

    public init(
        prompt: String, negativePrompt: String = "", variant: MossVariant = .sft,
        width: Int = 1024, height: Int = 1024, steps: Int? = nil, cfg: Double? = nil,
        shift: Double? = nil, sampler: SamplerKind? = nil, slg: SkipLayerGuidance = .official,
        seed: UInt64? = nil, loras: [LoRASpec] = []
    ) {
        let d = variant.defaults
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.variant = variant
        self.width = width
        self.height = height
        self.steps = steps ?? d.steps
        self.cfg = cfg ?? d.cfg
        self.shift = shift ?? d.shift
        self.sampler = sampler ?? d.sampler
        self.slg = slg
        self.seed = seed
        self.loras = loras
    }

    /// Edit LoRAs take up to four references (chinoll / ComfyUI's multi-reference workflows).
    public static let maxReferences = 4

    /// Reject requests the model cannot run, before any weights load.
    public func validate() throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EngineError.invalidRequest("prompt is empty")
        }
        guard MossDimensions.isValid(width: width, height: height) else {
            throw EngineError.invalidRequest("width and height must be positive multiples of 16 (got \(width)×\(height))")
        }
        guard steps > 0 else { throw EngineError.invalidRequest("steps must be positive") }
        guard cfg >= 1 else { throw EngineError.invalidRequest("cfg must be ≥ 1") }
        try slg.validate(depth: 42)
        guard references.count <= Self.maxReferences else {
            throw EngineError.invalidRequest("at most \(Self.maxReferences) reference images")
        }
        if let edit {
            guard edit.mask.width == width / 16, edit.mask.height == height / 16, edit.mask.values.count == edit.mask.width * edit.mask.height else {
                throw EngineError.invalidRequest("the edit mask is \(edit.mask.width)×\(edit.mask.height) tokens but the image is \(width / 16)×\(height / 16)")
            }
            guard edit.mask.coverage > 0 else {
                throw EngineError.invalidRequest("nothing to redraw — change, add or remove a box first")
            }
        }
    }
}

public struct GenerationResult: Sendable {
    public var imagePNG: Data
    public var width: Int
    public var height: Int
    public var seed: UInt64

    public init(imagePNG: Data, width: Int, height: Int, seed: UInt64) {
        self.imagePNG = imagePNG
        self.width = width
        self.height = height
        self.seed = seed
    }
}

/// Expand a short request into a detailed prompt, or (with `layout`) a JSON layout prompt.
public struct PromptEnhanceRequest: Sendable, Equatable {
    public var prompt: String
    public var layout: Bool
    public var canvas: Dimensions
    /// The enhancer's instructions (Settings → Prompts); built-in by default.
    public var prompts: MossPrompts

    public init(
        prompt: String, layout: Bool = false, canvas: Dimensions = Dimensions(width: 1024, height: 1024), prompts: MossPrompts = MossPrompts()
    ) {
        self.prompts = prompts
        self.prompt = prompt
        self.layout = layout
        self.canvas = canvas
    }
}
