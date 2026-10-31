import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// Community adapters (2026-10): chinoll / Edit-HQ `moss-edit-lora` files (bare `lora_A` / `lora_B`,
/// alphas in the `adapter_config` metadata), ai-toolkit edit LoRAs, and comfyanonymous's hires LoHa
/// (`hada_w{1,2}_{a,b}` on the fused qkv). Shapes are the real files' (safetensors headers).
@Suite struct EditLoRAFormatsTests {
    private func d(_ n: String, _ s: [Int]) -> TensorDescriptor { TensorDescriptor(name: n, dtype: "BF16", shape: s, byteCount: 0) }

    @Test func mossEditLoRABareKeysLandOnTheFusedQKV() throws {
        let descriptors = [
            d("layers.0.proj.k.lora_A", [64, 3072]), d("layers.0.proj.k.lora_B", [3072, 64]),
            d("layers.0.proj.ffn_up_gate.lora_A", [64, 3072]), d("layers.0.proj.ffn_up_gate.lora_B", [16384, 64]),
        ]
        let config = #"{"format": "moss-edit-lora", "modules": {"layers.0.proj.k": {"rank": 64, "alpha": 32}, "layers.0.proj.ffn_up_gate": {"rank": 64, "alpha": 64}}}"#
        let alphas = MossLoRAKeyMap.adapterConfigAlphas(metadata: ["adapter_config": config])
        #expect(alphas == ["layers.0.proj.k.alpha": 32, "layers.0.proj.ffn_up_gate.alpha": 64])
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: alphas, config: .release)
        #expect(plan.unmapped.isEmpty)
        let k = try #require(plan.parts.first { $0.down == "layers.0.proj.k.lora_A" })
        #expect(k.target == "layers.0.proj.qkv.weight" && k.rows == 3072..<6144)
        #expect(k.up == "layers.0.proj.k.lora_B" && k.scale == 0.5)
        let up = try #require(plan.parts.first { $0.target == "layers.0.proj.ffn_up_gate.weight" })
        #expect(up.scale == 1 && up.block == 0)
    }

    /// Edit-HQ armV5: rank 192 on q/k/v, 64 elsewhere, no alpha tensors (scale 1).
    @Test func aiToolkitEditLoRAMaps() throws {
        let descriptors = [
            d("diffusion_model.layers.41.proj.q.lora_A.weight", [192, 3072]), d("diffusion_model.layers.41.proj.q.lora_B.weight", [3072, 192]),
            d("diffusion_model.layers.41.proj.ffn_down.lora_A.weight", [64, 8192]), d("diffusion_model.layers.41.proj.ffn_down.lora_B.weight", [3072, 64]),
        ]
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: [:], config: .release)
        #expect(plan.unmapped.isEmpty && plan.parts.count == 2)
        #expect(plan.parts.allSatisfy { $0.scale == 1 && $0.block == 41 })
        #expect(plan.parts.first { $0.rows == 0..<3072 }?.rank == 192)
    }

    /// LyCORIS LoHa: `ΔW = (w1_a·w1_b) ⊙ (w2_a·w2_b) · alpha/rank`, rank = w1_b rows.
    @Test func loHaMapsAsAHadamardPart() throws {
        let m = "diffusion_model.layers.2.proj.qkv"
        let descriptors = [
            d("\(m).hada_w1_a", [9216, 64]), d("\(m).hada_w1_b", [64, 3072]),
            d("\(m).hada_w2_a", [9216, 64]), d("\(m).hada_w2_b", [64, 3072]), d("\(m).alpha", []),
        ]
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: ["\(m).alpha": 32], config: .release)
        #expect(plan.unmapped.isEmpty)
        let part = try #require(plan.parts.first)
        #expect(plan.parts.count == 1)
        #expect(part.kind == .hadamard && part.target == "layers.2.proj.qkv.weight" && part.rows == nil)
        #expect(part.up == "\(m).hada_w1_a" && part.down == "\(m).hada_w1_b")
        #expect(part.up2 == "\(m).hada_w2_a" && part.down2 == "\(m).hada_w2_b")
        #expect(part.rank == 64 && part.scale == 0.5 && part.block == 2)
    }

    @Test func loHaWithTheWrongShapeIsRefused() {
        let m = "diffusion_model.layers.2.proj.o"
        let descriptors = [
            d("\(m).hada_w1_a", [9216, 16]), d("\(m).hada_w1_b", [16, 3072]),
            d("\(m).hada_w2_a", [9216, 16]), d("\(m).hada_w2_b", [16, 3072]),
        ]
        #expect(throws: EngineError.self) { try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: [:], config: .release) }
    }

    /// ComfyUI: area-scale the reference to ≤ 384², then Qwen2-VL `smart_resize` (×32, ≥ 3136 px).
    @Test func encoderReferenceSizeFollowsComfyUI() {
        let a = QwenVisionGeometry.editReferenceSize(height: 1024, width: 1024)
        #expect(a.height == 384 && a.width == 384)
        let b = QwenVisionGeometry.editReferenceSize(height: 768, width: 1360)  // → 289×511 → 288×512
        #expect(b.height == 288 && b.width == 512)
        let small = QwenVisionGeometry.editReferenceSize(height: 40, width: 40)  // ≥ 3136 px
        #expect(small.height == 64 && small.width == 64)
        #expect(QwenVisionGeometry.grid(height: a.height, width: a.width).tokens == 144)
    }

    /// Sequence (t) positions per layout, as ComfyUI's `rope(…, f_offset)`: ai-toolkit puts the references
    /// on the image's t (isolated: everything shifted so both are 0, where the cached keys were made),
    /// Moss-Image-Edit at the text's start.
    @Test func referenceSequencePositionsFollowTheLayout() {
        let s = 77
        #expect(MossRoPEIndex.referenceSequence(layout: .mossForge, textLength: s) == (reference: 77, offset: 0))
        #expect(MossRoPEIndex.referenceSequence(layout: .aiToolkitNoCache, textLength: s) == (reference: 77, offset: 0))
        #expect(MossRoPEIndex.referenceSequence(layout: .aiToolkit, textLength: s) == (reference: 0, offset: -77))
        #expect(MossRoPEIndex.referenceSequence(layout: .mossImageEdit, textLength: s) == (reference: 0, offset: 0))
        // The offset moves image and text together, so the branch's own attention is unchanged.
        let shifted = MossRoPEIndex.positions(gridHeight: 2, gridWidth: 2, textLength: 3, sequenceOffset: -3)
        #expect(shifted[0] == [0, 0, -0.5, -0.5] && shifted[4] == [0, -3, 0, 0] && shifted[6] == [0, -1, 0, 0])
        #expect(MossRoPEIndex.referencePositions(index: 2, gridHeight: 1, gridWidth: 2, sequencePosition: 0) == [[2, 0, 0, -0.5], [2, 0, 0, 0.5]])
    }

    /// The library lists only files the inspector calls LoRAs — chinoll's bare keys, LoHa and LoKr
    /// came back "unknown", so Character Studio never saw chinoll (2026-10-27).
    @Test func inspectorRecognisesEveryAdapterForm() {
        #expect(MossInspector.classify([d("layers.0.proj.q.lora_A", [64, 3072]), d("layers.0.proj.q.lora_B", [3072, 64])]) == .lora(rank: 64))
        #expect(MossInspector.classify([d("diffusion_model.layers.0.proj.qkv.hada_w1_a", [9216, 32]),
                                        d("diffusion_model.layers.0.proj.qkv.hada_w1_b", [32, 3072])]) == .lora(rank: 32))
        #expect(MossInspector.classify([d("diffusion_model.layers.0.proj.o.lokr_w1", [8, 8]),
                                        d("diffusion_model.layers.0.proj.o.lokr_w2", [384, 384])]) == .lora(rank: 0))
    }
}
