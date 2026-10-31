import Foundation

/// The σ distribution MOSS-Image's final pretraining stage samples (technical report App. A.2, eq. 4,
/// Table 6, steps 274,001–310,000): a truncated logit-normal on `S = 1 − sigmoid(Z)`, `Z ~ N(μ, s²)`,
/// cut off at `a` and rescaled onto the grid σ_j = j/T, mixed with a `ρ` share of uniform sampling:
/// `p_j = (1−ρ)·[F(a·j/T) − F(a·(j−1)/T)] / F(a) + ρ/T`, `F(x) = 1 − Φ((log((1−x)/x) − μ)/s)`.
public struct MossNoiseSchedule: Sendable, Equatable {
    public let probabilities: [Double]
    private let cdf: [Double]

    public init(mu: Double, s: Double, cutoff a: Double, uniform rho: Double, steps T: Int = 1000) {
        func phi(_ x: Double) -> Double { 0.5 * erfc(-x / 2.0.squareRoot()) }
        func F(_ x: Double) -> Double {
            if x <= 0 { return 0 }
            if x >= 1 { return 1 }
            return 1 - phi((log((1 - x) / x) - mu) / s)
        }
        let Fa = F(a)
        var p: [Double] = []
        p.reserveCapacity(T)
        for j in 1...T {
            p.append((1 - rho) * (F(a * Double(j) / Double(T)) - F(a * Double(j - 1) / Double(T))) / Fa + rho / Double(T))
        }
        let total = p.reduce(0, +)
        probabilities = p.map { $0 / total }
        var running = 0.0
        cdf = probabilities.map { running += $0; return running }
    }

    /// The final-stage schedule: μ −1.09, s 0.95, cutoff 0.95, 1% uniform.
    public static let stage3 = MossNoiseSchedule(mu: -1.09, s: 0.95, cutoff: 0.95, uniform: 0.01)

    /// σ for a uniform draw `u ∈ [0, 1)` (inverse CDF over the 1000-step grid).
    public func sigma(u: Double) -> Double {
        var lo = 0, hi = cdf.count - 1
        while lo < hi {
            let mid = (lo + hi) / 2
            if cdf[mid] >= u { hi = mid } else { lo = mid + 1 }
        }
        return Double(lo + 1) / Double(cdf.count)
    }
}

/// Aspect-ratio buckets: each image trains at its own aspect, scaled to the pixel budget (never up)
/// and floored to multiples of 16 (the DiT's token stride).
public enum TrainingBuckets {
    public static func size(forWidth w: Int, height h: Int, pixels: Int) -> Dimensions {
        let area = Double(w * h)
        let scale = min(1, (Double(pixels) / area).squareRoot())
        func snap(_ v: Double) -> Int { max(16, Int(v * scale) / 16 * 16) }
        return Dimensions(width: snap(Double(w)), height: snap(Double(h)))
    }
}

/// Which DiT linears a LoRA trains (canonical, fused names under `layers.N.`).
public enum LoRATargetSet: String, Sendable, Codable, CaseIterable, Identifiable {
    /// Attention only — the usual choice for style / subject LoRAs; smallest file.
    case attention
    /// Attention + the SiTU-GLU feed-forward — more capacity (concepts, typography).
    case attentionAndFFN

    public var id: String { rawValue }
    public var modules: [String] {
        switch self {
        case .attention: ["proj.qkv", "proj.o"]
        case .attentionAndFFN: ["proj.qkv", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"]
        }
    }
    public var displayName: String { self == .attention ? "Attention" : "Attention + FFN" }
}

/// One image + caption pair.
public struct TrainingSample: Sendable, Equatable {
    public var image: URL
    public var caption: String
    /// Edit training: the source image this target was edited from (a reference token block).
    public var control: URL?
    public init(image: URL, caption: String, control: URL? = nil) {
        self.image = image
        self.caption = caption
        self.control = control
    }
}

/// Train a LoRA on a folder of images with `name.txt` captions. Defaults follow the MOSS recipe where
/// it applies to LoRAs (flow-match velocity loss, Stage-3 σ sampling, AdamW β (0.9, 0.95), wd 0.01,
/// gradient clip 1.0, frozen text encoder).
public struct TrainingRequest: Sendable, Equatable {
    public var baseModel: URL
    public var dataset: URL
    public var output: URL
    public var rank: Int
    public var alpha: Double
    public var learningRate: Double
    public var steps: Int
    /// Pixel budget per image (aspect kept, multiples of 16).
    public var pixels: Int
    public var targets: LoRATargetSet
    /// Write `<name>-step<N>.safetensors` every N steps (0 = only at the end).
    public var saveEvery: Int
    public var warmupSteps: Int
    public var gradientClip: Double
    /// Prepended to every caption as `"trigger, caption"`.
    public var trigger: String
    public var seed: UInt64?
    /// The family the result is marked for (its sidecar).
    public var family: MossFamily
    /// Test prompts rendered before training, at every checkpoint and at the end (max 3 used).
    public var samplePrompts: [String] = []
    /// Sampling for the test images: steps (CFG flow Euler, official SLG), pixel budget, CFG, fixed seed.
    public var sampleSteps = 20
    public var samplePixels = 512 * 512
    public var sampleCFG = 3.0
    public var sampleSeed: UInt64 = 42

    // Slider mode (Concept Sliders, image-free).
    public var mode: TrainingMode = .dataset
    /// The neutral subject ("a portrait of a person"); positive / negative are descriptors anchored to it.
    public var sliderTarget = ""
    public var sliderPositive = ""
    public var sliderNegative = ""
    /// The apply-weight span trained for (must contain 0); the saved band doubles the positive end.
    public var sliderRange = SliderRange.default
    /// Concept push at the range's far end: target = v(target) + m·η/peak·(v(pos) − v(neg)).
    public var sliderEta = TrainingRequest.defaultSliderEta
    public static let defaultSliderEta = 1.0
    /// Steps of the schedule each training latent is partly denoised along (k ∈ 1..<N, CFG flow).
    public var sliderSampleSteps = 10
    /// Attributes to keep out of the direction ("male, female" for an age slider).
    public var sliderPreserve: [String] = []

    /// LoRA (low-rank) or LoKr (Kronecker); `lokrFactor` sizes the split (≤ 0 = balanced).
    public var factorization: LoRAFactorization = .lora
    public var lokrFactor = -1
    /// Share of steps that train on an empty caption (0–0.5): keeps the LoRA usable without its
    /// exact captions and helps it generalise. 0 = off.
    public var captionDropout = 0.0
    /// Recompute DiT blocks in the backward pass (gradient checkpointing). On by default: measured
    /// at 1024², attention + FFN, rank 16 on an M2 Ultra — identical losses, peak memory 101.7 → 51.3 GB,
    /// and 62 → 9.8 s/step, because without it the activations push the GPU into memory pressure.
    public var gradientCheckpointing = true

    public init(
        baseModel: URL, dataset: URL, output: URL, rank: Int = 16, alpha: Double = 16, learningRate: Double = 1e-4,
        steps: Int = 1000, pixels: Int = 1024 * 1024, targets: LoRATargetSet = .attention, saveEvery: Int = 250,
        warmupSteps: Int = 50, gradientClip: Double = 1, trigger: String = "", seed: UInt64? = nil, family: MossFamily = .base
    ) {
        self.baseModel = baseModel
        self.dataset = dataset
        self.output = output
        self.rank = rank
        self.alpha = alpha
        self.learningRate = learningRate
        self.steps = steps
        self.pixels = pixels
        self.targets = targets
        self.saveEvery = saveEvery
        self.warmupSteps = warmupSteps
        self.gradientClip = gradientClip
        self.trigger = trigger
        self.seed = seed
        self.family = family
    }

    public func validate(sampleCount: Int) throws {
        if mode == .slider {
            guard !sliderPositive.trimmingCharacters(in: .whitespaces).isEmpty, !sliderNegative.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw EngineError.invalidRequest("a slider needs positive and negative descriptors")
            }
            guard sliderSettingsAreValid else {
                throw EngineError.invalidRequest("slider: η must be positive, latent steps ≥ 2 and the range must contain 0")
            }
        } else {
            guard sampleCount > 0 else {
                throw EngineError.invalidRequest(mode == .edit
                    ? "no edit pairs — put targets + name.txt instructions in the folder and the sources with the same names in control/"
                    : "the dataset folder has no images")
            }
        }
        guard (1...256).contains(rank) else { throw EngineError.invalidRequest("rank must be 1…256") }
        guard alpha > 0, learningRate > 0, learningRate < 1 else { throw EngineError.invalidRequest("alpha and learning rate must be positive") }
        guard steps > 0 else { throw EngineError.invalidRequest("steps must be positive") }
        guard pixels >= 256 * 256 else { throw EngineError.invalidRequest("train at 256² or more") }
        guard (0...0.5).contains(captionDropout) else { throw EngineError.invalidRequest("caption dropout must be 0…0.5") }
        if !preparedSamplePrompts.isEmpty {
            guard (1...100).contains(sampleSteps) else { throw EngineError.invalidRequest("test image steps must be 1…100") }
            guard samplePixels >= 256 * 256 else { throw EngineError.invalidRequest("test images need 256² or more") }
        }
    }

    /// The test prompts that will run: non-empty, at most three, with the trigger prefixed. (A slider's
    /// test images are one prompt across its range instead — `sliderTestPrompt` / `sliderTestWeights`.)
    public var preparedSamplePrompts: [String] {
        guard mode != .slider else { return [] }
        return Array(samplePrompts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.prefix(TrainingSamples.maxPrompts))
            .map { Self.caption($0, trigger: trigger) }
    }

    /// `"trigger, caption"` (not doubled when the caption already starts with it).
    public static func caption(_ text: String, trigger: String) -> String {
        let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let word = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !word.isEmpty else { return body }
        return body.lowercased().hasPrefix(word.lowercased()) ? body : "\(word), \(body)"
    }
}

/// What a finished (or stopped) training run produced.
public struct TrainingResult: Sendable, Equatable {
    public var output: URL
    public var steps: Int
    public var losses: [Double]
    public var seconds: Double
    public init(output: URL, steps: Int, losses: [Double], seconds: Double) {
        self.output = output
        self.steps = steps
        self.losses = losses
        self.seconds = seconds
    }
}

/// Where checkpoint test images go and how they're grouped for the Train screen.
public enum TrainingSamples {
    public static let maxPrompts = 3

    public struct ID: Sendable, Equatable {
        public var step: Int
        public var index: Int
    }

    /// One row of the Train screen: a step and its images in prompt order.
    public struct Row: Sendable, Equatable, Identifiable {
        public var step: Int
        public var images: [URL]
        public var id: Int { step }
    }

    /// Before training (0), every `every` steps, and the end.
    public static func steps(total: Int, every: Int) -> [Int] {
        var out = [0]
        if every > 0 { out += stride(from: every, to: total, by: every) }
        out.append(total)
        return out
    }

    /// `<dir>/<name>-checkpoints/samples/step00250-p2.png` (index 0-based → p1…p3).
    public static func url(output: URL, step: Int, index: Int) -> URL {
        let name = output.deletingPathExtension().lastPathComponent
        return output.deletingLastPathComponent().appendingPathComponent("\(name)-checkpoints/samples")
            .appendingPathComponent(String(format: "step%05d-p%d.png", step, index + 1))
    }

    public static func parse(_ url: URL) -> ID? {
        let name = url.deletingPathExtension().lastPathComponent
        guard url.pathExtension == "png", name.hasPrefix("step"), let dash = name.firstIndex(of: "-") else { return nil }
        let stepText = name[name.index(name.startIndex, offsetBy: 4)..<dash]
        let rest = name[name.index(after: dash)...]
        guard rest.hasPrefix("p"), let step = Int(stepText), let n = Int(rest.dropFirst()), n >= 1 else { return nil }
        return ID(step: step, index: n - 1)
    }

    public static func rows(_ urls: [URL]) -> [Row] {
        let parsed = urls.compactMap { u in parse(u).map { (u, $0) } }
        return Dictionary(grouping: parsed, by: { $0.1.step }).keys.sorted().map { step in
            Row(step: step, images: parsed.filter { $0.1.step == step }.sorted { $0.1.index < $1.1.index }.map(\.0))
        }
    }
}

/// Train → Suggest test prompts: Qwen3.5 reads a few dataset captions and proposes three prompts that
/// show whether the LoRA learned its subject / style (close to the data, a new scene, a stress test).
public enum TrainingPromptSuggester {
    public static func messages(captions: [String], trigger: String) -> [ChatMessage] {
        let system = """
            You write test prompts for checking a LoRA while it trains on an image model (MOSS-Image). \
            Given captions from its training set, write exactly 3 prompts, one per line, numbered 1–3, nothing else:
            1. close to the training images (same kind of subject and setting),
            2. the learned subject or style in a clearly different scene,
            3. a harder test (unusual composition, medium or lighting).
            Each prompt is one sentence of plain description, under 40 words.
            """
        var user = "Training captions:\n" + captions.prefix(12).map { "- \($0.prefix(300))" }.joined(separator: "\n")
        let word = trigger.trimmingCharacters(in: .whitespaces)
        user += word.isEmpty ? "\n\nNo trigger word." : "\n\nStart every prompt with the trigger word \(word)."
        return [ChatMessage(.system, system), ChatMessage(.user, user)]
    }

    /// Numbered / bulleted lines → up to three prompts without markers, quotes or bold.
    public static func parse(_ answer: String) -> [String] {
        var out: [String] = []
        for raw in answer.components(separatedBy: "\n") {
            var line = raw.trimmingCharacters(in: .whitespaces)
            guard let r = line.range(of: #"^(\d+[.)]|[-*•])\s+"#, options: .regularExpression) else { continue }
            line = String(line[r.upperBound...]).replacingOccurrences(of: "**", with: "")
            line = line.trimmingCharacters(in: CharacterSet(charactersIn: "\"“”' ").union(.whitespaces))
            if !line.isEmpty { out.append(line) }
            if out.count == TrainingSamples.maxPrompts { break }
        }
        return out
    }

    /// No language model: three captions spread across the dataset, else generic prompts.
    public static func fallback(captions: [String], trigger: String) -> [String] {
        let clean = captions.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        let picks: [String]
        if clean.isEmpty {
            picks = ["a photo", "an illustration", "close-up portrait"]
        } else {
            let n = min(TrainingSamples.maxPrompts, clean.count)
            picks = (0..<n).map { clean[n == 1 ? 0 : $0 * (clean.count - 1) / (n - 1)] }
        }
        return picks.map { TrainingRequest.caption($0, trigger: trigger) }
    }
}

/// Dataset (image/caption pairs) or an image-free Concept Slider.
public enum TrainingMode: String, Sendable, Codable, CaseIterable, Identifiable {
    case dataset, slider
    /// Experimental: an instruction-edit LoRA from source → target pairs (sources as reference tokens).
    case edit
    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .dataset: "From images"
        case .slider: "Slider"
        case .edit: "Edit (experimental)"
        }
    }
}

/// One target/positive/negative prompt triple a slider trains on.
public struct SliderPromptSet: Sendable, Equatable {
    public var target: String
    public var positive: String
    public var negative: String

    public init(target: String, positive: String, negative: String) {
        self.target = target
        self.positive = positive
        self.negative = negative
    }

    /// Split free text (commas, semicolons or newlines) into trimmed, non-empty attributes.
    public static func attributes(from text: String) -> [String] {
        text.split(whereSeparator: { ",;\n".contains($0) })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    static func appending(_ attribute: String, to prompt: String) -> String {
        prompt.isEmpty ? attribute : "\(prompt), \(attribute)"
    }
}

extension TrainingRequest {
    /// `sliderPreserve` trimmed, without blanks or case-insensitive duplicates.
    public var sliderPreservedAttributes: [String] {
        var seen = Set<String>()
        return sliderPreserve
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && seen.insert($0.lowercased()).inserted }
    }

    /// The anchored triple, then one variant per preserved attribute (appended to all three).
    public var sliderPromptSets: [SliderPromptSet] {
        let base = SliderPromptSet(
            target: sliderTarget.trimmingCharacters(in: .whitespacesAndNewlines),
            positive: sliderPositivePrompt, negative: sliderNegativePrompt)
        return [base] + sliderPreservedAttributes.map { a in
            SliderPromptSet(
                target: SliderPromptSet.appending(a, to: base.target),
                positive: SliderPromptSet.appending(a, to: base.positive),
                negative: SliderPromptSet.appending(a, to: base.negative))
        }
    }

    public var sliderSettingsAreValid: Bool {
        sliderEta > 0 && sliderEta.isFinite && sliderSampleSteps >= 2 && sliderRange.isValid
    }

    /// "target, positive descriptors" (verbatim when it already names the target).
    public var sliderPositivePrompt: String { Self.anchored(sliderPositive, to: sliderTarget) }
    public var sliderNegativePrompt: String { Self.anchored(sliderNegative, to: sliderTarget) }

    static func anchored(_ descriptors: String, to target: String) -> String {
        let t = target.trimmingCharacters(in: .whitespacesAndNewlines)
        let d = descriptors.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, d.range(of: t, options: .caseInsensitive) == nil else { return d }
        return d.isEmpty ? t : "\(t), \(d)"
    }

    /// A slider's test images: this prompt (the first test prompt, else the target) at each weight.
    public var sliderTestPrompt: String {
        samplePrompts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.first { !$0.isEmpty }
            ?? sliderTarget.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The range's ends and its middle (0 when it straddles zero) — "off" sits between the two pushes.
    public var sliderTestWeights: [Double] {
        let mid = sliderRange.min < 0 && sliderRange.max > 0 ? 0 : (sliderRange.min + sliderRange.max) / 2
        return [sliderRange.min, mid, sliderRange.max]
    }
}

/// How a trained adapter factorizes its weight delta.
public enum LoRAFactorization: String, Sendable, Codable, CaseIterable, Identifiable {
    /// Classic low-rank LoRA: `ΔW = up · down`.
    case lora
    /// LyCORIS LoKr: `ΔW = w1 ⊗ w2` (Kronecker product of two small factors).
    case lokr

    public var id: String { rawValue }
    public var displayName: String { self == .lora ? "LoRA" : "LoKr" }

    /// LyCORIS-style `dimension = m · n`, `m ≤ n`, `m` the largest divisor ≤ `factor` (or the most
    /// balanced split when `factor <= 0`). Split `out` and `in` independently: `w1 = [out.m, in.m]`,
    /// `w2 = [out.n, in.n]`.
    public static func split(_ dimension: Int, factor: Int = -1) -> (m: Int, n: Int) {
        precondition(dimension > 0, "dimension must be positive")
        if factor > 0, dimension % factor == 0 {
            let other = dimension / factor
            return factor <= other ? (factor, other) : (other, factor)
        }
        let cap = factor > 0 ? factor : dimension
        var m = 1
        var candidate = 2
        while candidate * candidate <= dimension {
            if dimension % candidate == 0, candidate <= cap { m = candidate }
            candidate += 1
        }
        return (m, dimension / m)
    }
}
