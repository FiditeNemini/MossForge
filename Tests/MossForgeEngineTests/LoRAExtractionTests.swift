import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// Extract LoRA: the pure half — which tensors become low-rank / full diffs, their file names (which
/// the existing LoRA loader must map back), the per-layer rank choice, and the output's sidecar/header.
@Suite("LoRA extraction — plan, keys, rank, sidecar")
struct LoRAExtractionTests {
    private func tmp() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("lx-\(UUID())")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    // MARK: rank choice

    @Test func fixedRankKeepsTheMaximum() {
        #expect(LoRARankCutoff.fixed.rank(singularValues: [3, 2, 1, 0.5], totalEnergy: 14.25, maxRank: 2) == 2)
        #expect(LoRARankCutoff.fixed.rank(singularValues: [3, 2], totalEnergy: 13, maxRank: 8) == 2)  // never more than it has
    }

    @Test func energyCutoffTakesTheSmallestRankThatReachesTheFraction() {
        // σ² = 9, 4, 1 of a total 14: 9/14 = 0.64, 13/14 = 0.93.
        #expect(LoRARankCutoff.energy(0.9).rank(singularValues: [3, 2, 1], totalEnergy: 14, maxRank: 3) == 2)
        #expect(LoRARankCutoff.energy(0.5).rank(singularValues: [3, 2, 1], totalEnergy: 14, maxRank: 3) == 1)
        // Unreachable within the cap (the true total includes the tail the SVD didn't compute) → the cap.
        #expect(LoRARankCutoff.energy(0.99).rank(singularValues: [3, 2, 1], totalEnergy: 100, maxRank: 2) == 2)
    }

    @Test func thresholdCutoffKeepsValuesAboveAFractionOfTheLargest() {
        #expect(LoRARankCutoff.threshold(0.5).rank(singularValues: [4, 3, 1, 0.1], totalEnergy: 26, maxRank: 4) == 2)
        #expect(LoRARankCutoff.threshold(0.01).rank(singularValues: [4, 3, 1, 0.1], totalEnergy: 26, maxRank: 3) == 3)
        #expect(LoRARankCutoff.threshold(0.9).rank(singularValues: [4, 3], totalEnergy: 25, maxRank: 2) == 1)  // at least 1
    }

    @Test func cutoffsRoundTripThroughText() {
        #expect(LoRARankCutoff(parsing: "energy:0.9") == .energy(0.9))
        #expect(LoRARankCutoff(parsing: "threshold:0.02") == .threshold(0.02))
        #expect(LoRARankCutoff(parsing: "fixed") == .fixed)
        #expect(LoRARankCutoff(parsing: "energy:2") == nil)
        #expect(LoRARankCutoff.energy(0.9).description == "energy:0.9")
    }

    // MARK: layer plan

    @Test func twoDWeightsGoLowRankOneDWeightsAndBiasesAsFullDiffs() {
        let source: [String: [Int]] = [
            "layers.0.proj.qkv.weight": [9216, 3072], "layers.0.norm1.weight": [3072],
            "shared_block_modulation.weight": [18432, 3072], "shared_block_modulation.bias": [18432],
            "text_embedder.proj.weight": [3072, 4096],
        ]
        var parent = source
        parent["rope.bands"] = [16]                           // a buffer only the official file has
        parent["text_embedder.proj.weight"] = [3072, 2048]   // shape mismatch
        let plan = MossLoRAExtraction.plan(source: source, parent: parent)
        #expect(plan.layers.map(\.key) == [
            "layers.0.norm1.weight", "layers.0.proj.qkv.weight", "shared_block_modulation.bias", "shared_block_modulation.weight",
        ])
        #expect(plan.layers.map(\.kind) == [.weightDiff, .lowRank, .biasDiff, .lowRank])
        #expect(plan.parentOnly == ["rope.bands"] && plan.sourceOnly.isEmpty)
        #expect(plan.mismatched == ["text_embedder.proj.weight"])
    }

    @Test func extractedNamesMapBackOntoTheDiT() throws {
        let c = MossDiTConfig.release
        let layers: [MossLoRAExtraction.Layer] = [
            .init(key: "layers.3.proj.qkv.weight", kind: .lowRank, shape: [9216, 3072]),
            .init(key: "layers.3.proj.ffn_down.weight", kind: .lowRank, shape: [3072, 8192]),
            .init(key: "text_embedder.blocks.0.qkv.weight", kind: .lowRank, shape: [9216, 3072]),
            .init(key: "layers.3.norm1.weight", kind: .weightDiff, shape: [3072]),
            .init(key: "time_embedding.0.bias", kind: .biasDiff, shape: [3072]),
        ]
        var descriptors: [TensorDescriptor] = []
        func add(_ name: String, _ shape: [Int]) {
            descriptors.append(TensorDescriptor(name: name, dtype: "BF16", shape: shape, byteCount: shape.reduce(2, *)))
        }
        for l in layers {
            let names = MossLoRAExtraction.names(for: l)
            if l.kind == .lowRank {
                add(names.down, [16, l.shape[1]])
                add(names.up, [l.shape[0], 16])
            } else {
                add(names.down, l.shape)
            }
        }
        #expect(MossLoRAExtraction.names(for: layers[0]).down == "diffusion_model.layers.3.proj.qkv.lora_down.weight")
        #expect(MossLoRAExtraction.names(for: layers[3]).down == "diffusion_model.layers.3.norm1.diff")
        #expect(MossLoRAExtraction.names(for: layers[4]).down == "diffusion_model.time_embedding.0.diff_b")
        let targets = Dictionary(uniqueKeysWithValues: layers.map { ($0.key, $0.shape) })
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: [:], config: c, targetShapes: targets)
        #expect(plan.unmapped.isEmpty)
        #expect(Set(plan.parts.map(\.target)) == Set(layers.map(\.key)))
        #expect(plan.parts.first { $0.target == "layers.3.proj.qkv.weight" }?.rows == nil)  // the whole fused qkv
        #expect(plan.parts.first { $0.target == "layers.3.norm1.weight" }?.kind == .weightDiff)
        #expect(plan.parts.first { $0.target == "time_embedding.0.bias" }?.kind == .biasDiff)
    }

    @Test func logicalShapesFoldFusionsAndDropTheirParts() {
        let plan = WeightLoadPlan(
            entries: [
                .init(key: "layers.0.proj.q.weight", dequant: .passthrough(rawKey: "a"), shape: [4, 4]),
                .init(key: "layers.0.proj.k.weight", dequant: .passthrough(rawKey: "b"), shape: [4, 4]),
                .init(key: "layers.0.proj.v.weight", dequant: .passthrough(rawKey: "c"), shape: [4, 4]),
                .init(key: "layers.0.norm1.weight", dequant: .passthrough(rawKey: "d"), shape: [4]),
            ],
            consumedAuxiliaries: [],
            fusions: [.init(key: "layers.0.proj.qkv.weight", parts: ["layers.0.proj.q.weight", "layers.0.proj.k.weight", "layers.0.proj.v.weight"])])
        #expect(plan.logicalShapes == ["layers.0.proj.qkv.weight": [12, 4], "layers.0.norm1.weight": [4]])
    }

    // MARK: request, sidecar, header

    @Test func theRequestRefusesQuantizedOutputsBadRanksAndSelfParents() throws {
        let a = URL(fileURLWithPath: "/m/diffusion_models/ft.safetensors"), p = URL(fileURLWithPath: "/m/diffusion_models/base.safetensors")
        let out = URL(fileURLWithPath: "/m/lora/extracted/ft-lora.safetensors")
        func req(_ parent: URL, _ rank: Int, _ prec: WeightPrecision) -> ModelToolRequest {
            ModelToolRequest(input: a, operation: .extractLoRA(parent: parent, rank: rank, cutoff: .fixed), precision: prec, output: out)
        }
        try req(p, 64, .bfloat16).validate()
        #expect(throws: EngineError.self) { try req(p, 64, .int8ConvRot).validate() }
        #expect(throws: EngineError.self) { try req(p, 0, .bfloat16).validate() }
        #expect(throws: EngineError.self) { try req(a, 64, .bfloat16).validate() }
        try ModelToolRequest(input: a, operation: .convert, precision: .int8ConvRot, output: out).validate()  // others unaffected
    }

    @Test func theLoRAsSidecarTakesTheParentsFamilyStrengthOneAndNamesItsSources() throws {
        let d = try tmp()
        defer { try? FileManager.default.removeItem(at: d) }
        let parent = d.appendingPathComponent("moss_image_base_bf16.safetensors")
        let op = ModelToolOperation.extractLoRA(parent: parent, rank: 64, cutoff: .energy(0.9))
        let out = ModelToolRequest.outputSidecar(
            source: ModelSidecar(families: [.sft]), operation: op, precision: .bfloat16, sourceName: "DirtyMoss-BASE-BF16")
        #expect(out.sidecar.families == [.base])  // from the parent's (inferred) family, not the source's
        #expect(out.sidecar.strength == 1 && out.sidecar.distill == nil && out.copies.isEmpty)
        let notes = try #require(out.sidecar.notes)
        #expect(notes.contains("DirtyMoss-BASE-BF16") && notes.contains("moss_image_base_bf16") && notes.contains("rank 64") && notes.contains("energy:0.9"))
    }

    @Test func theHeaderSaysWhoMadeItFromWhat() {
        let m = MossLoRAExtraction.metadata(source: "DirtyMoss-BASE-BF16", parent: "moss_image_base_bf16", rank: 64, cutoff: .fixed)
        #expect(m["produced_by"] == "MossForge" && m["op"] == "extract-lora")
        #expect(m["source"] == "DirtyMoss-BASE-BF16" && m["parent"] == "moss_image_base_bf16" && m["rank"] == "64" && m["cutoff"] == "fixed")
    }

    @Test func theOutputLandsInLoRAExtractedNamedAfterTheSourceAndRank() {
        let models = URL(fileURLWithPath: "/m")
        #expect(MossLoRAExtraction.outputStem(source: "DirtyMoss-BASE-BF16", rank: 64) == "DirtyMoss-BASE-BF16-lora-r64")
        #expect(MossLoRAExtraction.outputURL(stem: "x", models: models).path == "/m/lora/extracted/x.safetensors")
    }

    @Test func theParentIsTheOfficialModelOfTheSourcesFamily() {
        func item(_ name: String, _ fam: MossFamily, summary: String = "DiT · 42 blocks · bf16") -> ModelItem {
            ModelItem(url: URL(fileURLWithPath: "/m/diffusion_models/\(name).safetensors"), relativePath: "diffusion_models/\(name).safetensors",
                kind: .dit, sidecar: ModelSidecar(families: [fam]), hasSavedSidecar: false, sizeBytes: 1, summary: summary)
        }
        let source = item("DirtyMoss-BASE-BF16", .base)
        let dits = [
            source, item("Kromoss-BASE-BF16", .base), item("moss_image_sft_bf16", .sft),
            item("moss_image_base_int8_convrot", .base, summary: "DiT · 42 blocks · int8-convrot"), item("moss_image_base_bf16", .base),
        ]
        #expect(MossLoRAExtraction.defaultParent(for: source.url, family: .base, among: dits)?.name == "moss_image_base_bf16")
        // No official file of the family → another DiT of the family, never the source itself.
        let few = [source, item("Kromoss-BASE-BF16", .base), item("moss_image_sft_bf16", .sft)]
        #expect(MossLoRAExtraction.defaultParent(for: source.url, family: .base, among: few)?.name == "Kromoss-BASE-BF16")
        #expect(MossLoRAExtraction.defaultParent(for: source.url, family: .base, among: [source]) == nil)
    }
}
