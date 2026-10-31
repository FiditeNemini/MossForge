import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// LoRA keys → MossForge's canonical DiT weights. Kohya (`diffusion_model.<m>.lora_down/lora_up/alpha`,
/// lodestones DMD, ComfyUI) and PEFT (`<m>.lora_A/lora_B`, lodestones `generator.safetensors`) both
/// name the **split** `proj.{q,k,v}`; they land on row slices of the fused `proj.qkv`
/// (`MossDiTConfig.qkvRowRanges`, q|k|v verified against the official weights).
@Suite("MossLoRAKeyMap — kohya / PEFT keys onto the fused-qkv DiT")
struct MossLoRAKeyMapTests {
    private func d(_ n: String, _ s: [Int]) -> TensorDescriptor { TensorDescriptor(name: n, dtype: "F32", shape: s, byteCount: 0) }

    private func kohya(_ module: String, out: Int, inp: Int, rank: Int = 128) -> [TensorDescriptor] {
        [d("diffusion_model.\(module).lora_down.weight", [rank, inp]), d("diffusion_model.\(module).lora_up.weight", [out, rank]),
         d("diffusion_model.\(module).alpha", [])]
    }

    @Test func kohyaSplitQKVLandsOnFusedRowSlices() throws {
        let descriptors = ["q", "k", "v"].flatMap { kohya("layers.3.proj.\($0)", out: 3072, inp: 3072) }
            + kohya("layers.3.proj.ffn_down", out: 3072, inp: 8192)
        let alphas = Dictionary(uniqueKeysWithValues: descriptors.filter { $0.name.hasSuffix(".alpha") }.map { ($0.name, Float(64)) })
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: alphas, config: .release)
        #expect(plan.unmapped.isEmpty)
        #expect(plan.parts.count == 4)
        let k = try #require(plan.parts.first { $0.down == "diffusion_model.layers.3.proj.k.lora_down.weight" })
        #expect(k.target == "layers.3.proj.qkv.weight")
        #expect(k.rows == 3072..<6144)
        #expect(k.up == "diffusion_model.layers.3.proj.k.lora_up.weight")
        #expect(k.rank == 128 && k.scale == 0.5)  // alpha / rank
        let down = try #require(plan.parts.first { $0.target == "layers.3.proj.ffn_down.weight" })
        #expect(down.rows == nil)
        // Grouped by target, q|k|v in row order — what the runtime concatenates.
        #expect(plan.parts(for: "layers.3.proj.qkv.weight").map(\.rows) == [0..<3072, 3072..<6144, 6144..<9216])
    }

    @Test func peftKeysAndMissingAlphaMeanScaleOne() throws {
        let descriptors = [
            d("layers.0.proj.o.lora_A.weight", [16, 3072]), d("layers.0.proj.o.lora_B.weight", [3072, 16]),
            d("transformer.layers.1.proj.q.lora_A.weight", [16, 3072]), d("transformer.layers.1.proj.q.lora_B.weight", [3072, 16]),
        ]
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: [:], config: .release)
        #expect(plan.parts.map(\.target).sorted() == ["layers.0.proj.o.weight", "layers.1.proj.qkv.weight"])
        #expect(plan.parts.allSatisfy { $0.scale == 1 && $0.rank == 16 })
    }

    @Test func unknownModulesAreReportedNotGuessed() throws {
        let plan = try MossLoRAKeyMap.plan(
            descriptors: kohya("layers.0.proj.mystery", out: 10, inp: 10) + kohya("layers.99.proj.o", out: 3072, inp: 3072),
            alphas: [:], config: .release)
        #expect(plan.parts.isEmpty)
        #expect(Set(plan.unmapped) == ["layers.0.proj.mystery", "layers.99.proj.o"])
    }

    @Test func shapeMismatchThrows() {
        #expect(throws: EngineError.self) {
            try MossLoRAKeyMap.plan(descriptors: kohya("layers.0.proj.o", out: 1024, inp: 3072), alphas: [:], config: .release)
        }
    }

    @Test func lodestonesTurboLoRAMapsCompletely() throws {
        let url = modelsDir.appendingPathComponent("lora/moss_8step_dmd_turbo.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let plan = try MossLoRAKeyMap.plan(url: url, config: .release)
        #expect(plan.unmapped.isEmpty)
        #expect(plan.parts.count == 42 * 6)  // q, k, v, o, ffn_up_gate, ffn_down per block
        #expect(plan.parts.allSatisfy { $0.rank == 128 && $0.scale == 1 })  // alpha 128 / rank 128
        #expect(Set(plan.parts.map(\.target)).count == 42 * 4)  // q|k|v share the fused target
    }
}

@Suite("lodestones data-space DMD constants")
struct DataSpaceConstantsTests {
    @Test func stepCodesAreTheOfficialEightStepTimesteps() throws {
        // The distilled sampler's `step_codes` are exactly 1000·σ of the official snapped grid at
        // 8 steps / shift 5 — an independent check of `MossSchedule.pipelineSigmas`.
        let url = modelsDir.appendingPathComponent("lora/lodestones-dmd/dataspace.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let reader = try SafetensorsReader(url: url)
        defer { reader.close() }
        let codes = try SafetensorsCodec.decodeFloats(reader.bytes("step_codes"), dtype: .float32)
        let sigmas = MossSchedule.pipelineSigmas(steps: 8, shift: 5).dropLast()
        #expect(codes.count == 8)
        for (c, s) in zip(codes, sigmas) { #expect(abs(Double(c) - 1000 * s) < 1e-3) }
    }
}

/// Kijai's SFT-minus-Base LoRAs (ComfyUI format): LoRA pairs on the fused qkv and on non-block linears,
/// `.diff` full deltas on norms, `.diff_b` bias deltas — every part tagged with its block for
/// per-block strength.
@Suite("MossLoRAKeyMap — ComfyUI diffs, non-block modules, block tags")
struct MossLoRAComfyFormatTests {
    private func d(_ n: String, _ s: [Int]) -> TensorDescriptor { TensorDescriptor(name: n, dtype: "BF16", shape: s, byteCount: 0) }
    private let ditShapes: [String: [Int]] = [
        "layers.21.norm1.weight": [3072], "layers.21.proj.qkv.weight": [9216, 3072], "final_linear_video.weight": [128, 3072],
        "video_embedder.weight": [3072, 128], "video_embedder.bias": [3072], "text_embedder.blocks.0.qkv.weight": [9216, 3072],
    ]

    @Test func mapsDiffsGenericModulesAndTagsBlocks() throws {
        let plan = try MossLoRAKeyMap.plan(
            descriptors: [
                d("diffusion_model.layers.21.norm1.diff", [3072]),
                d("diffusion_model.layers.21.proj.qkv.lora_down.weight", [512, 3072]), d("diffusion_model.layers.21.proj.qkv.lora_up.weight", [9216, 512]),
                d("diffusion_model.final_linear_video.lora_down.weight", [128, 3072]), d("diffusion_model.final_linear_video.lora_up.weight", [128, 128]),
                d("diffusion_model.video_embedder.diff_b", [3072]),
                d("diffusion_model.text_embedder.blocks.0.qkv.lora_down.weight", [8, 3072]), d("diffusion_model.text_embedder.blocks.0.qkv.lora_up.weight", [9216, 8]),
            ],
            alphas: [:], config: .release, targetShapes: ditShapes)
        #expect(plan.unmapped.isEmpty)
        let byTarget = Dictionary(grouping: plan.parts, by: \.target)
        #expect(byTarget["layers.21.norm1.weight"]?.first?.kind == .weightDiff)
        #expect(byTarget["layers.21.norm1.weight"]?.first?.block == 21)
        #expect(byTarget["layers.21.proj.qkv.weight"]?.first?.rows == nil)  // fused qkv LoRA covers all rows
        #expect(byTarget["layers.21.proj.qkv.weight"]?.first?.block == 21)
        #expect(byTarget["final_linear_video.weight"]?.first?.block == nil)
        #expect(byTarget["video_embedder.bias"]?.first?.kind == .biasDiff)
        #expect(byTarget["text_embedder.blocks.0.qkv.weight"]?.first?.block == nil)  // adapter ≠ DiT block
    }

    @Test func withoutTargetShapesNonBlockModulesAreUnmapped() throws {
        let plan = try MossLoRAKeyMap.plan(
            descriptors: [d("diffusion_model.final_norm_video.diff", [3072])], alphas: [:], config: .release)
        #expect(plan.parts.isEmpty && plan.unmapped == ["final_norm_video (diff)"])
    }

    @Test func kijaiLaterBlocksLoRAMapsCompletely() throws {
        let lora = modelsDir.appendingPathComponent("lora/kijai/moss_image_sft_lora_blocks_21_to_49_rank_512_bf16.safetensors")
        let dit = modelsDir.appendingPathComponent("diffusion_models/moss_image_sft_bf16.safetensors")
        guard FileManager.default.fileExists(atPath: lora.path), FileManager.default.fileExists(atPath: dit.path) else { return }
        let (ditDescriptors, _) = try SafetensorsHeader.readDescriptors(from: dit)
        let plan = try MossLoRAKeyMap.plan(
            url: lora, config: .release, targetShapes: Dictionary(uniqueKeysWithValues: ditDescriptors.map { ($0.name, $0.shape) }))
        #expect(plan.unmapped.isEmpty)
        #expect(Set(plan.parts.compactMap(\.block)) == Set([3, 4] + Array(21...41)))
        #expect(plan.parts.filter { $0.kind == .weightDiff }.count == 23 * 4)  // norm1, norm2, q_norm, k_norm
    }
}
