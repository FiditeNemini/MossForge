import Foundation

/// The two MOSS-Image DiT families. A finetune or merge belongs to one; a LoRA works with one or both.
public enum MossFamily: String, Sendable, Codable, CaseIterable, Identifiable, Comparable {
    case sft, base

    public var id: String { rawValue }
    public var displayName: String { self == .sft ? "SFT" : "Base" }
    /// The variant whose recommended sampling (50 steps, CFG 3, shift 5) applies.
    public var variant: MossVariant { self == .sft ? .sft : .base }

    public static func < (a: MossFamily, b: MossFamily) -> Bool { a == .sft && b == .base }
}

/// How a few-step distillation LoRA is sampled.
public enum DistillRoute: String, Sendable, Codable, CaseIterable, Identifiable {
    /// lodestones' data-space DMD loop: needs the LoRA's `dataspace` (Q + step codes) and fp32
    /// `time_embedding` files.
    case dataSpace
    /// The LoRA alone on the stock flow-Euler sampler at CFG 1 (akhaliq's 4/2-step, lodestones' Base).
    case flow

    public var id: String { rawValue }
    public var displayName: String { self == .dataSpace ? "Data-space DMD" : "Flow Euler" }
}

/// A distillation LoRA's sampling recipe. Adapter files are names beside the LoRA.
public struct DistillInfo: Sendable, Codable, Equatable {
    public var steps: Int
    public var route: DistillRoute
    public var cfg: Double
    public var shift: Double
    public var dataSpace: String?
    public var timeEmbedding: String?

    public init(
        steps: Int, route: DistillRoute, cfg: Double = 1, shift: Double = 5, dataSpace: String? = nil, timeEmbedding: String? = nil
    ) {
        self.steps = steps
        self.route = route
        self.cfg = cfg
        self.shift = shift
        self.dataSpace = dataSpace
        self.timeEmbedding = timeEmbedding
    }

    /// A DiT that is already few-step but has no sidecar recipe (Generate → Turbo → "None — model is
    /// already 8-step"): 8 flow-Euler steps at CFG 1, no Turbo LoRA stacked on it.
    public static let alreadyDistilled = DistillInfo(steps: 8, route: .flow)
}

/// Per-file properties the safetensors header can't carry, in `<file>.safetensors.mfmeta.json` (moved
/// and trashed with the file by `ModelFileOps`). Without one, `inferred` guesses from the filename.
public struct ModelSidecar: Sendable, Codable, Equatable {
    /// A DiT's family (one), or the families a LoRA works with.
    public var families: [MossFamily] { didSet { families = Array(Set(families)).sorted() } }
    /// Set for a few-step distillation LoRA (shown as a Turbo option, not a normal LoRA).
    public var distill: DistillInfo?
    /// Default strength when the LoRA is added in Generate.
    public var strength: Double?
    /// The span of Generate's strength slider for this LoRA (`nil` = ±2). Sliders trained for wider
    /// weights (e.g. ±10) set it; a MossForge-trained slider's header range fills it in `load`.
    public var strengthRange: SliderRange?
    public var source: String?
    public var notes: String?

    /// Generate's slider span when no (valid) range is set.
    public static let defaultStrengthRange = SliderRange(min: -2, max: 2)

    public init(
        families: [MossFamily], distill: DistillInfo? = nil, strength: Double? = nil, strengthRange: SliderRange? = nil,
        source: String? = nil, notes: String? = nil
    ) {
        self.families = Array(Set(families)).sorted()
        self.distill = distill
        self.strength = strength
        self.strengthRange = strengthRange
        self.source = source
        self.notes = notes
    }

    public func supports(_ family: MossFamily) -> Bool { families.contains(family) }

    /// The strength slider's span: `strengthRange` when it's valid, else ±2.
    public var sliderRange: SliderRange {
        guard let r = strengthRange, r.isValid else { return Self.defaultStrengthRange }
        return r
    }

    public static func url(for model: URL) -> URL { URL(fileURLWithPath: model.path + ".mfmeta.json") }

    public static func read(for model: URL) -> ModelSidecar? {
        guard let data = try? Data(contentsOf: url(for: model)) else { return nil }
        return try? JSONDecoder().decode(ModelSidecar.self, from: data)
    }

    public func write(for model: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: Self.url(for: model), options: .atomic)
    }

    /// The saved sidecar, else `inferred`. A LoRA with no saved range takes its header's
    /// `mossforge_slider_range` (written by MossForge's slider trainer), when `metadata` has one.
    public static func load(for model: URL, isLoRA: Bool, metadata: [String: String] = [:]) -> ModelSidecar {
        var sidecar = read(for: model) ?? {
            let siblings = (try? FileManager.default.contentsOfDirectory(atPath: model.deletingLastPathComponent().path)) ?? []
            return inferred(fileName: model.lastPathComponent, isLoRA: isLoRA, siblings: siblings)
        }()
        if isLoRA, sidecar.strengthRange == nil, let raw = metadata[sliderRangeMetadataKey] {
            sidecar.strengthRange = SliderRange(parsing: raw)
        }
        return sidecar
    }

    /// The safetensors metadata key a trained slider's recommended weight band is saved under ("min,max").
    public static let sliderRangeMetadataKey = "mossforge_slider_range"

    /// Defaults from the name: `base` / `sft` pick the family (a LoRA naming neither works with both;
    /// a DiT naming neither is taken as SFT); `<n>step` or `dmd`/`turbo` marks a distillation LoRA,
    /// whose `<stem>-dataspace` + `<stem>-time_embedding` siblings select the data-space route.
    public static func inferred(fileName: String, isLoRA: Bool, siblings: [String]) -> ModelSidecar {
        let stem = (fileName as NSString).deletingPathExtension
        let name = stem.lowercased()
        let tokens = Set(name.split { !$0.isLetter && !$0.isNumber }.map(String.init))
        let saysBase = tokens.contains("base"), saysSFT = tokens.contains("sft")
        let families: [MossFamily] =
            saysBase && !saysSFT ? [.base] : saysSFT && !saysBase ? [.sft] : isLoRA ? [.sft, .base] : [.sft]
        guard isLoRA else { return ModelSidecar(families: families) }

        var steps: Int?
        if let r = name.range(of: #"(\d+)[-_ ]?steps?"#, options: .regularExpression) {
            steps = Int(name[r].prefix { $0.isNumber })
        } else if tokens.contains("dmd") || tokens.contains("turbo") {
            steps = 8
        }
        guard let steps else { return ModelSidecar(families: families) }
        let ds = "\(stem)-dataspace.safetensors", te = "\(stem)-time_embedding.safetensors"
        let info = siblings.contains(ds) && siblings.contains(te)
            ? DistillInfo(steps: steps, route: .dataSpace, dataSpace: ds, timeEmbedding: te)
            : DistillInfo(steps: steps, route: .flow)
        return ModelSidecar(families: families, distill: info)
    }
}

/// A distillation ready to run: the LoRA (or none, when it's fused into the DiT) plus its sampling
/// recipe, adapters as absolute URLs.
public struct DistillSpec: Sendable, Equatable, Codable {
    public var lora: LoRASpec?
    public var route: DistillRoute
    public var steps: Int
    public var cfg: Double
    public var shift: Double
    public var dataSpace: URL?
    public var timeEmbedding: URL?

    public init(lora: LoRASpec?, route: DistillRoute, steps: Int, cfg: Double = 1, shift: Double = 5, dataSpace: URL? = nil, timeEmbedding: URL? = nil) {
        self.lora = lora
        self.route = route
        self.steps = steps
        self.cfg = cfg
        self.shift = shift
        self.dataSpace = dataSpace
        self.timeEmbedding = timeEmbedding
    }

    /// `nil` when the data-space route is missing an adapter name.
    public init?(lora: URL, info: DistillInfo, strength: Double = 1) {
        let dir = lora.deletingLastPathComponent()
        let ds = info.dataSpace.map { dir.appendingPathComponent($0) }
        let te = info.timeEmbedding.map { dir.appendingPathComponent($0) }
        if info.route == .dataSpace && (ds == nil || te == nil) { return nil }
        self.init(
            lora: LoRASpec(url: lora, strength: strength), route: info.route, steps: info.steps, cfg: info.cfg,
            shift: info.shift, dataSpace: ds, timeEmbedding: te)
    }
}

extension DistillSpec {
    /// A Turbo fused into the DiT (its sidecar's `distill`): no LoRA, no time embedding (baked in);
    /// the data-space route still needs its `dataspace` file beside the model.
    public init?(builtInto model: URL, info: DistillInfo) {
        let ds = info.dataSpace.map { model.deletingLastPathComponent().appendingPathComponent($0) }
        if info.route == .dataSpace && ds == nil { return nil }
        self.init(lora: nil, route: info.route, steps: info.steps, cfg: info.cfg, shift: info.shift, dataSpace: ds, timeEmbedding: nil)
    }
}

extension GenerationRequest {
    /// Run with a distillation LoRA: its steps, CFG (1, so no negative pass) and sampler.
    public mutating func apply(_ spec: DistillSpec) {
        distill = spec
        steps = spec.steps
        cfg = spec.cfg
        shift = spec.shift
        sampler = spec.route == .dataSpace ? .dataSpaceDMD : .flowEuler
        negativePrompt = ""
    }
}
