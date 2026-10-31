import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

// Dequant cases, MOSS key names, plus the per-tensor `comfy_quant` path, prefix filtering
// (the Qwen3.5 file also holds a vision tower / lm_head / mtp) and split→fused qkv.

private func isPassthrough(_ entry: WeightLoadEntry) -> Bool {
    if case .passthrough = entry.dequant { return true }
    return false
}

private func descriptors(_ entries: [(name: String, dtype: String, shape: [Int])]) -> [TensorDescriptor] {
    entries.map { TensorDescriptor(name: $0.name, dtype: $0.dtype, shape: $0.shape, byteCount: $0.shape.reduce(1, *)) }
}

private func plan(
    _ entries: [(name: String, dtype: String, shape: [Int])],
    prefix: String = "",
    convRot: [String: ConvRotMetadata.ConvRotLayer] = [:]
) -> WeightLoadPlan {
    MossWeightPlan.build(descriptors: descriptors(entries), prefix: prefix, convRot: convRot)
}

@Suite("MossWeightPlan — header → per-tensor dequant ops")
struct MossWeightPlanTests {
    @Test func bf16WeightsPassThroughAndComfyAttentionConfigIsSkipped() {
        // The int8 DiT carries a `layers.N.comfy_attention.config` U8 runtime hint per block; it is
        // not a weight and passthrough can't reinterpret U8 — it must be folded away, not emitted.
        let p = plan([
            ("layers.0.proj.qkv.weight", "BF16", [9216, 3072]),
            ("layers.0.norm1.weight", "BF16", [3072]),
            ("layers.0.comfy_attention.config", "U8", [36]),
            ("rope.bands", "F32", [16]),
        ])
        #expect(p.entries.map(\.key).sorted() == ["layers.0.norm1.weight", "layers.0.proj.qkv.weight", "rope.bands"])
        #expect(p.entries.allSatisfy(isPassthrough))
        #expect(p.consumedAuxiliaries == ["layers.0.comfy_attention.config"])
        #expect(p.fusions.isEmpty)
    }

    @Test func int8ConvRotFromComfyQuantBlobsConsumesScaleAndBlob() {
        // Comfy-Org MOSS int8·cr: no `_quantization_metadata`; the group size comes from the
        // per-tensor blob, keyed by the raw (unprefixed) layer name.
        let p = plan(
            [
                ("layers.0.proj.o.weight", "I8", [3072, 3072]),
                ("layers.0.proj.o.weight_scale", "F32", [3072, 1]),
                ("layers.0.proj.o.comfy_quant", "U8", [72]),
            ],
            convRot: ["layers.0.proj.o": .init(groupSize: 256, isInt4: false)])
        #expect(p.entries.count == 1)
        #expect(
            p.entries[0].dequant
                == .int8ConvRot(
                    weightKey: "layers.0.proj.o.weight", scaleKey: "layers.0.proj.o.weight_scale", groupSize: 256))
        #expect(p.entries[0].shape == [3072, 3072])
        #expect(p.consumedAuxiliaries == ["layers.0.proj.o.weight_scale", "layers.0.proj.o.comfy_quant"])
    }

    @Test func int8WeightWithoutConvRotEntryIsPlainPerChannel() {
        let p = plan([("layers.0.proj.o.weight", "I8", [3072, 3072]), ("layers.0.proj.o.weight_scale", "F32", [3072, 1])])
        #expect(
            p.entries[0].dequant
                == .int8ConvRot(weightKey: "layers.0.proj.o.weight", scaleKey: "layers.0.proj.o.weight_scale", groupSize: 0))
    }

    @Test func int4ConvRotDoublesInnerDim() {
        let p = plan(
            [("layers.0.proj.o.weight", "I8", [3072, 1536]), ("layers.0.proj.o.weight_scale", "F32", [3072, 1])],
            convRot: ["layers.0.proj.o": .init(groupSize: 256, isInt4: true)])
        #expect(
            p.entries[0].dequant
                == .int4ConvRot(weightKey: "layers.0.proj.o.weight", scaleKey: "layers.0.proj.o.weight_scale", groupSize: 256))
        #expect(p.entries[0].shape == [3072, 3072])
    }

    @Test func fp8ScaledAndRawFp8() {
        let p = plan([
            ("layers.0.proj.o.weight", "F8_E4M3", [3072, 3072]), ("layers.0.proj.o.weight_scale", "F32", []),
            ("layers.0.proj.ffn_down.weight", "F8_E5M2", [3072, 8192]),
        ])
        let byKey = Dictionary(uniqueKeysWithValues: p.entries.map { ($0.key, $0.dequant) })
        #expect(byKey["layers.0.proj.o.weight"] == .fp8Scaled(weightKey: "layers.0.proj.o.weight", scaleKey: "layers.0.proj.o.weight_scale"))
        #expect(byKey["layers.0.proj.ffn_down.weight"] == .fp8Scaled(weightKey: "layers.0.proj.ffn_down.weight", scaleKey: ""))
    }

    @Test func nvfp4ConsumesBothScales() {
        let p = plan([
            ("layers.0.proj.o.weight", "U8", [3072, 1536]),
            ("layers.0.proj.o.weight_scale", "F8_E4M3", [3072, 192]),
            ("layers.0.proj.o.weight_scale_2", "F32", []),
        ])
        #expect(p.entries.count == 1)
        #expect(p.entries[0].shape == [3072, 3072])
        #expect(p.consumedAuxiliaries.isSuperset(of: ["layers.0.proj.o.weight_scale", "layers.0.proj.o.weight_scale_2"]))
    }

    @Test func prefixKeepsOnlyThatSubtreeAndStripsIt() {
        // The Qwen3.5 file is a whole VLM; the text-conditioning path wants only the language
        // model, so `lm_head`, `model.visual.*` and `mtp.*` must not be materialized at all.
        let p = plan(
            [
                ("model.language_model.layers.0.input_layernorm.weight", "BF16", [4096]),
                ("model.language_model.embed_tokens.weight", "I8", [248320, 4096]),
                ("model.language_model.embed_tokens.weight_scale", "F32", [248320, 1]),
                ("model.visual.blocks.0.attn.qkv.weight", "BF16", [3456, 1152]),
                ("lm_head.weight", "I8", [248320, 4096]), ("lm_head.weight_scale", "F32", [248320, 1]),
                ("mtp.fc.weight", "BF16", [4096, 8192]),
            ],
            prefix: "model.language_model.",
            convRot: ["model.language_model.embed_tokens": .init(groupSize: 256, isInt4: false)])
        #expect(p.entries.map(\.key).sorted() == ["embed_tokens.weight", "layers.0.input_layernorm.weight"])
        #expect(
            p.entries.first { $0.key == "embed_tokens.weight" }?.dequant
                == .int8ConvRot(
                    weightKey: "model.language_model.embed_tokens.weight",
                    scaleKey: "model.language_model.embed_tokens.weight_scale", groupSize: 256))
    }

    @Test func splitQKVIsFusedInQKVOrder() throws {
        // Canonical in-memory layout is the fused `proj.qkv` (one matmul, the Comfy-Org layout);
        // diffusers-style split files are concatenated q|k|v on load. LoRAs target the split
        // names and map onto `MossDiTConfig.qkvRowRanges`.
        let p = MossWeightPlan.build(descriptors: ditDescriptors(layers: 2, fusedQKV: false), prefix: "")
        #expect(p.fusions.count == 2)
        let f = try #require(p.fusions.first { $0.key == "layers.1.proj.qkv.weight" })
        #expect(f.parts == ["layers.1.proj.q.weight", "layers.1.proj.k.weight", "layers.1.proj.v.weight"])
        #expect(p.shape(of: "layers.1.proj.qkv.weight") == [9216, 3072])
        // The fused file needs no fusion.
        #expect(MossWeightPlan.build(descriptors: ditDescriptors(layers: 2), prefix: "").fusions.isEmpty)
    }

    @Test func incompleteSplitQKVIsNotFused() {
        let p = plan([("layers.0.proj.q.weight", "BF16", [3072, 3072]), ("layers.0.proj.k.weight", "BF16", [3072, 3072])])
        #expect(p.fusions.isEmpty)
    }
}

@Suite("ComfyQuantBlob — per-tensor blobs → ConvRot layer map")
struct ComfyQuantLayerMapTests {
    @Test func rotatedLayersKeyedByLayerNameWithTheirGroupSizes() {
        let blobs: [String: Data] = [
            "layers.0.proj.qkv.comfy_quant": Data(#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#.utf8),
            "model.visual.blocks.0.mlp.linear_fc1.comfy_quant": Data(#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 16}"#.utf8),
            "layers.0.proj.o.comfy_quant": Data(#"{"format": "convrot_w4a4", "convrot_groupsize": 64}"#.utf8),
            "layers.0.proj.ffn_down.comfy_quant": Data(#"{"format": "int8_tensorwise", "convrot": false}"#.utf8),
            "layers.0.proj.ffn_up_gate.comfy_quant": Data(#"{"format": "float8_e4m3fn"}"#.utf8),
            "broken.comfy_quant": Data("not json".utf8),
        ]
        let layers = ComfyQuantBlob.convRotLayers(blobs: blobs)
        #expect(
            layers == [
                "layers.0.proj.qkv": .init(groupSize: 256, isInt4: false),
                "model.visual.blocks.0.mlp.linear_fc1": .init(groupSize: 16, isInt4: false),
                "layers.0.proj.o": .init(groupSize: 64, isInt4: true),
            ])
    }
}

@Suite("MossWeightPlan.validate — geometry check before any byte is read")
struct MossWeightPlanValidationTests {
    private func blockPlan(qkvInner: Int = 3072, downInner: Int = 8192, split: Bool = false) -> WeightLoadPlan {
        func e(_ key: String, _ shape: [Int]) -> WeightLoadEntry { .init(key: key, dequant: .passthrough(rawKey: key), shape: shape) }
        var entries = [
            e("layers.0.norm1.weight", [3072]),
            e("layers.0.proj.o.weight", [3072, 3072]),
            e("layers.0.proj.ffn_up_gate.weight", [16384, 3072]),
            e("layers.0.proj.ffn_down.weight", [3072, downInner]),
        ]
        if split {
            entries += ["q", "k", "v"].map { e("layers.0.proj.\($0).weight", [3072, qkvInner]) }
        } else {
            entries.append(e("layers.0.proj.qkv.weight", [9216, qkvInner]))
        }
        return WeightLoadPlan(entries: entries, consumedAuxiliaries: [])
    }

    @Test func acceptsConsistentFusedAndSplitPlans() throws {
        try MossWeightPlan.validate(blockPlan())
        try MossWeightPlan.validate(blockPlan(split: true))
    }

    @Test func rejectsHalfWidthProjection() {
        // The int4-read-as-int8 failure: inner dim stays at the packed width.
        #expect(throws: EngineError.self) { try MossWeightPlan.validate(blockPlan(qkvInner: 1536)) }
        #expect(throws: EngineError.self) { try MossWeightPlan.validate(blockPlan(qkvInner: 1536, split: true)) }
    }

    @Test func rejectsFFNDownThatMissesTheGLUWidthAndNamesIt() {
        do {
            try MossWeightPlan.validate(blockPlan(downInner: 4096))
            Issue.record("expected validate to throw")
        } catch {
            let text = "\(error)"
            #expect(text.contains("layers.0.proj.ffn_down.weight"))
            #expect(text.contains("4096") && text.contains("8192"))
        }
    }

    @Test func ignoresNonDiTPlans() throws {
        let p = WeightLoadPlan(
            entries: [.init(key: "decoder.conv_in.weight", dequant: .passthrough(rawKey: "x"), shape: [512, 32, 3, 3])],
            consumedAuxiliaries: [])
        try MossWeightPlan.validate(p)
    }
}

/// Real-header checks (skipped when `models/` is absent).
@Suite("MossWeightPlan — live headers")
struct LiveMossWeightPlanTests {
    private func url(_ path: String) -> URL? {
        let u = modelsDir.appendingPathComponent(path)
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }

    @Test func int8SFTPlansTheSameKeysAsBF16WithEveryBlockLinearUnRotated() throws {
        guard let bf16 = url("diffusion_models/moss_image_sft_bf16.safetensors"),
            let int8 = url("diffusion_models/moss_image_sft_int8_convrot.safetensors")
        else { return }
        let pb = try MossWeightPlan.build(url: bf16)
        let pq = try MossWeightPlan.build(url: int8)
        #expect(pb.entries.count == 379 && pb.entries.allSatisfy(isPassthrough))
        #expect(Set(pq.entries.map(\.key)) == Set(pb.entries.map(\.key)))
        let rotated = pq.entries.filter { $0.dequant.isInt8ConvRot(groupSize: 256) }
        #expect(rotated.count == 168)  // qkv, o, ffn_up_gate, ffn_down × 42
        for e in pq.entries { #expect(e.shape == pb.shape(of: e.key), "\(e.key)") }
        try MossWeightPlan.validate(pq)
    }

    @Test func qwenLanguageModelPlanSkipsVisionAndHeads() throws {
        guard let te = url("text_encoders/qwen3.5_9b_int8_convrot.safetensors") else { return }
        let p = try MossWeightPlan.build(url: te, prefix: "model.language_model.")
        #expect(!p.entries.isEmpty)
        #expect(p.entries.allSatisfy { !$0.key.hasPrefix("model.") && !$0.key.hasPrefix("visual") })
        #expect(p.entries.first { $0.key == "embed_tokens.weight" }?.dequant.isInt8ConvRot(groupSize: 256) == true)
        #expect(!p.entries.contains { if case .int8ConvRot(_, _, 0) = $0.dequant { return true } else { return false } })
    }
}

extension DequantOp {
    fileprivate func isInt8ConvRot(groupSize g: Int) -> Bool {
        if case let .int8ConvRot(_, _, groupSize) = self { return groupSize == g }
        return false
    }
}
