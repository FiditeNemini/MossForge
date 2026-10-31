import Foundation
import MossForgeCore

/// Extract LoRA, the pure half: which tensors of `finetune − parent` the LoRA carries and under which
/// names. Every 2-D weight (block projections, text adapter, embedders, modulation, head) is a Linear
/// the DiT runs through its LoRA-aware `linear` / `project`, so it becomes a kohya low-rank pair
/// (truncated SVD). 1-D weights (RMSNorm scales) and biases can't be factorised, but they're tiny
/// (~0.3 M values in all), so they're stored whole as ComfyUI `.diff` / `.diff_b` deltas, which
/// `MossLoRAKeyMap` already applies — skipping them would drop part of what the finetune changed.
public enum MossLoRAExtraction {
    public struct Layer: Sendable, Equatable {
        public enum Kind: Sendable, Equatable { case lowRank, weightDiff, biasDiff }
        /// Canonical DiT key (`layers.3.proj.qkv.weight`, `time_embedding.0.bias`).
        public var key: String
        public var kind: Kind
        public var shape: [Int]
        public init(key: String, kind: Kind, shape: [Int]) {
            self.key = key
            self.kind = kind
            self.shape = shape
        }
    }

    public struct Plan: Sendable, Equatable {
        public var layers: [Layer]
        /// Keys only one side has (e.g. the official file's `rope.bands` buffer) — reported, not stored.
        public var sourceOnly: [String]
        public var parentOnly: [String]
        /// Keys whose shapes differ — not the same architecture there; reported, not stored.
        public var mismatched: [String]
    }

    /// Pair the two models' canonical (prefix-stripped, qkv-fused) tensors.
    public static func plan(source: [String: [Int]], parent: [String: [Int]]) -> Plan {
        var layers: [Layer] = []
        var mismatched: [String] = []
        for key in source.keys.sorted() {
            guard let shape = source[key], let other = parent[key] else { continue }
            guard shape == other else { mismatched.append(key); continue }
            let kind: Layer.Kind
            if key.hasSuffix(".bias") {
                kind = .biasDiff
            } else if key.hasSuffix(".weight") {
                guard shape.count <= 2 else { mismatched.append(key); continue }
                kind = shape.count == 2 ? .lowRank : .weightDiff
            } else {
                continue  // buffers
            }
            layers.append(Layer(key: key, kind: kind, shape: shape))
        }
        return Plan(
            layers: layers,
            sourceOnly: source.keys.filter { parent[$0] == nil }.sorted(),
            parentOnly: parent.keys.filter { source[$0] == nil }.sorted(),
            mismatched: mismatched)
    }

    /// File names for a layer: kohya `lora_down` / `lora_up` / `alpha` (the trainer's names), or a single
    /// `.diff` / `.diff_b` in `down` (`up` / `alpha` empty).
    public static func names(for layer: Layer) -> (down: String, up: String, alpha: String) {
        let module = String(layer.key.dropLast(layer.kind == .biasDiff ? ".bias".count : ".weight".count))
        switch layer.kind {
        case .lowRank: return MossLoRAOutput.keys(module: module)
        case .weightDiff: return ("diffusion_model." + module + ".diff", "", "")
        case .biasDiff: return ("diffusion_model." + module + ".diff_b", "", "")
        }
    }

    public static func metadata(source: String, parent: String, rank: Int, cutoff: LoRARankCutoff) -> [String: String] {
        ["produced_by": "MossForge", "op": "extract-lora", "source": source, "parent": parent, "rank": "\(rank)", "cutoff": cutoff.description]
    }

    public static let outputFolder = "extracted"

    public static func outputStem(source: String, rank: Int) -> String { "\(source)-lora-r\(rank)" }

    /// `<models>/lora/extracted/<stem>.safetensors`.
    public static func outputURL(stem: String, models: URL) -> URL {
        models.appendingPathComponent(ModelLibrary.loraFolder).appendingPathComponent(outputFolder)
            .appendingPathComponent(stem + ".safetensors")
    }

    /// The model a finetune is extracted against: the official `moss_image_<family>_*` file of its family
    /// (bf16 / fp32 before quantized ones), else any other DiT of the family; never the source itself.
    public static func defaultParent(for source: URL, family: MossFamily, among dits: [ModelItem]) -> ModelItem? {
        let candidates = dits.filter { $0.url.standardizedFileURL != source.standardizedFileURL && $0.sidecar.supports(family) }
        func score(_ m: ModelItem) -> Int {
            let official = m.name.lowercased().hasPrefix("moss_image_\(family.rawValue)") ? 0 : 2
            let full = ["bf16", "fp32", "fp16"].contains { m.summary.hasSuffix($0) } ? 0 : 1
            return official + full
        }
        return candidates.min { (score($0), $0.name) < (score($1), $1.name) }
    }
}

extension WeightLoadPlan {
    /// Every logical key's shape after load: fused keys (`proj.qkv`) in, their parts out.
    public var logicalShapes: [String: [Int]] {
        let parts = Set(fusions.flatMap(\.parts))
        var out: [String: [Int]] = [:]
        for e in entries where !parts.contains(e.key) { out[e.key] = e.shape }
        for f in fusions { out[f.key] = shape(of: f.key) }
        return out
    }
}
