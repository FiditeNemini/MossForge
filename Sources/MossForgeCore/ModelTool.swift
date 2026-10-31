import Foundation

/// How two or three DiTs combine.
public enum ModelMergeMode: String, Sendable, Codable, CaseIterable, Identifiable {
    /// `(1−α)·A + α·B`
    case weightedSum
    /// `A + α·(B − C)`: transplant what B learned on top of C into A.
    case addDifference

    public var id: String { rawValue }
    public var displayName: String { self == .weightedSum ? "Weighted sum" : "Add difference" }
}

/// How Extract LoRA picks each layer's rank (never above the requested rank, never below 1).
public enum LoRARankCutoff: Sendable, Equatable, Hashable, CustomStringConvertible {
    /// Every layer at the requested rank.
    case fixed
    /// The smallest rank that keeps this fraction (0…1) of the layer difference's energy `‖ΔW‖²`.
    case energy(Double)
    /// Singular values at least this fraction (0…1) of the layer's largest.
    case threshold(Double)

    /// `fixed`, `energy:0.9`, `threshold:0.02` (the CLI's `--cutoff`); nil for a fraction outside (0, 1].
    public init?(parsing text: String) {
        let parts = text.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
        if parts == ["fixed"] { self = .fixed; return }
        guard parts.count == 2, let v = Double(parts[1]), v > 0, v <= 1 else { return nil }
        switch parts[0] {
        case "energy": self = .energy(v)
        case "threshold": self = .threshold(v)
        default: return nil
        }
    }

    public var description: String {
        switch self {
        case .fixed: "fixed"
        case .energy(let f): "energy:\(f)"
        case .threshold(let f): "threshold:\(f)"
        }
    }

    /// The rank to keep from the top singular values (descending). `totalEnergy` is the full
    /// `‖ΔW‖²_F` — larger than `Σσ²` of a truncated SVD, so an unreachable energy target yields `maxRank`.
    public func rank(singularValues s: [Float], totalEnergy: Double, maxRank: Int) -> Int {
        let cap = Swift.max(1, Swift.min(maxRank, s.count))
        switch self {
        case .fixed:
            return cap
        case .energy(let f):
            var kept = 0.0
            for (i, v) in s.prefix(cap).enumerated() {
                kept += Double(v) * Double(v)
                if kept >= f * totalEnergy { return i + 1 }
            }
            return cap
        case .threshold(let f):
            guard let top = s.first else { return 1 }
            return Swift.max(1, s.prefix(cap).filter { Double($0) >= f * Double(top) }.count)
        }
    }
}

public enum ModelToolOperation: Sendable, Equatable {
    /// Re-encode at the request's precision.
    case convert
    /// Fuse LoRAs (fp32 maths) — `overrides` are fp32 replacement tensors (a data-space Turbo's
    /// time embedding), applied first and kept fp32.
    case fuse(loras: [LoRASpec], overrides: URL?)
    /// The request's input is model A; `others` = [B] or [B, C].
    case merge(others: [URL], mode: ModelMergeMode, alpha: Double, loras: [LoRASpec])
    /// The request's input is a finetune / merge; `parent` is the model it came from. Writes a LoRA:
    /// truncated SVD of each 2-D weight's difference (`rank` at most), full diffs for norms and biases.
    case extractLoRA(parent: URL, rank: Int, cutoff: LoRARankCutoff)
}

/// Convert / fuse / merge one DiT into a new file.
public struct ModelToolRequest: Sendable, Equatable {
    public var input: URL
    public var operation: ModelToolOperation
    public var precision: WeightPrecision
    public var output: URL

    public init(input: URL, operation: ModelToolOperation, precision: WeightPrecision, output: URL) {
        self.input = input
        self.operation = operation
        self.precision = precision
        self.output = output
    }

    /// Refuse what can't run before any weights load.
    public func validate() throws {
        guard case .extractLoRA(let parent, let rank, _) = operation else { return }
        guard (1...1024).contains(rank) else { throw EngineError.invalidRequest("LoRA rank must be 1…1024 (got \(rank))") }
        guard [.bfloat16, .float16, .float32].contains(precision) else {
            throw EngineError.invalidRequest("an extracted LoRA is saved as bf16, fp16 or fp32 (not \(precision.rawValue))")
        }
        guard parent.standardizedFileURL != input.standardizedFileURL else {
            throw EngineError.invalidRequest("the parent model must differ from the source")
        }
    }

    /// A fuse request; a data-space Turbo LoRA among `loras` brings its time embedding as the overrides.
    public static func fuse(dit: URL, loras: [LoRASpec], precision: WeightPrecision, output: URL) -> ModelToolRequest {
        let overrides = loras.lazy.compactMap { spec -> URL? in
            guard let d = ModelSidecar.read(for: spec.url)?.distill, d.route == .dataSpace, let te = d.timeEmbedding else { return nil }
            return spec.url.deletingLastPathComponent().appendingPathComponent(te)
        }.first
        return ModelToolRequest(input: dit, operation: .fuse(loras: loras, overrides: overrides), precision: precision, output: output)
    }

    /// The output DiT's sidecar (family from the source; a fused Turbo LoRA becomes a built-in recipe)
    /// and the files to copy beside it (a data-space Turbo's `dataspace`, which the sampler still needs).
    public static func outputSidecar(
        source: ModelSidecar, operation: ModelToolOperation, precision: WeightPrecision, sourceName: String, output: URL? = nil
    ) -> (sidecar: ModelSidecar, copies: [(from: URL, to: URL)]) {
        var out = ModelSidecar(families: source.families, distill: source.distill)
        var copies: [(URL, URL)] = []
        var what: String
        switch operation {
        case .convert:
            what = "Converted from \(sourceName) to \(precision.rawValue)."
        case .merge(let others, let mode, let alpha, let loras):
            what = "\(mode.displayName) (α \(alpha)) of \(sourceName) + \(others.map { $0.deletingPathExtension().lastPathComponent }.joined(separator: " − "))"
                + (loras.isEmpty ? "" : " + " + loras.map { $0.url.deletingPathExtension().lastPathComponent }.joined(separator: ", "))
                + ", \(precision.rawValue)."
        case .fuse(let loras, _):
            what = "Fused \(loras.map { "\($0.url.deletingPathExtension().lastPathComponent)@\($0.strength)" }.joined(separator: ", ")) into \(sourceName) (fp32), \(precision.rawValue)."
            for spec in loras {
                guard let d = ModelSidecar.read(for: spec.url)?.distill else { continue }
                if d.route == .flow {
                    out.distill = DistillInfo(steps: d.steps, route: .flow, cfg: d.cfg, shift: d.shift)
                } else if let ds = d.dataSpace, let output {
                    let name = output.deletingPathExtension().lastPathComponent + "-dataspace.safetensors"
                    out.distill = DistillInfo(steps: d.steps, route: .dataSpace, cfg: d.cfg, shift: d.shift, dataSpace: name)
                    copies.append((spec.url.deletingLastPathComponent().appendingPathComponent(ds), output.deletingLastPathComponent().appendingPathComponent(name)))
                }
            }
        case .extractLoRA(let parent, let rank, let cutoff):
            // A LoRA works on its parent: the parent's family, full strength.
            let parentName = parent.deletingPathExtension().lastPathComponent
            out = ModelSidecar(families: ModelSidecar.load(for: parent, isLoRA: false).families, strength: 1)
            what = "Extracted from \(sourceName) against \(parentName): rank \(rank) (\(cutoff)), \(precision.rawValue). Load it on \(parentName)."
        }
        if let notes = source.notes, !notes.isEmpty { what += " Source notes: \(notes)" }
        out.notes = what
        return (out, copies)
    }
}

public struct ModelToolResult: Sendable, Equatable {
    public var output: URL
    public var bytes: Int
    public var seconds: Double
    /// What the tool wants the user to know (Extract LoRA: energy kept, tensors skipped).
    public var report: String
    public init(output: URL, bytes: Int, seconds: Double, report: String = "") {
        self.output = output
        self.bytes = bytes
        self.seconds = seconds
        self.report = report
    }
}
