import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

// MARK: - Fixtures

/// A synthetic DiT header in either key layout, shaped like the real 42-layer checkpoint.
func ditDescriptors(layers: Int = 42, fusedQKV: Bool = true, dtype: String = "BF16") -> [TensorDescriptor] {
    func d(_ n: String, _ s: [Int], _ t: String = dtype) -> TensorDescriptor {
        TensorDescriptor(name: n, dtype: t, shape: s, byteCount: 0)
    }
    var out = [
        d("video_embedder.weight", [3072, 128]), d("video_embedder.bias", [3072]),
        d("text_embedder.proj.weight", [3072, 4096]), d("text_embedder.blocks.0.qkv.weight", [9216, 3072]), d("text_embedder.blocks.1.qkv.weight", [9216, 3072]),
        d("shared_block_modulation.weight", [18432, 3072]), d("time_embedding.0.bias", [3072]),
        d("time_embedding.0.weight", [3072, 256]), d("final_linear_video.weight", [128, 3072]),
        d("video_head_modulation.weight", [6144, 3072]), d("rope.bands", [16], "F32"),
    ]
    for i in 0..<layers {
        out.append(d("layers.\(i).q_norm.weight", [128]))
        out.append(d("layers.\(i).proj.ffn_up_gate.weight", [16384, 3072]))
        if fusedQKV {
            out.append(d("layers.\(i).proj.qkv.weight", [9216, 3072]))
        } else {
            for p in ["q", "k", "v"] { out.append(d("layers.\(i).proj.\(p).weight", [3072, 3072])) }
        }
    }
    return out
}

/// The repo's `models/` folder (real weights, gitignored) — tests that need it skip when absent.
let modelsDir = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("models")

// MARK: - DiT config

@Suite("MossDiTConfig — dims derived from the header like loaders.model_config()")
struct MossDiTConfigTests {
    @Test func derivesReleaseConfig() throws {
        let c = try MossDiTConfig(descriptors: ditDescriptors())
        #expect(c == MossDiTConfig.release)
        #expect(c.dim == 3072 && c.heads == 24 && c.headDim == 128)
        #expect(c.layers == 42 && c.ffnDim == 8192 && c.latentChannels == 128)
        #expect(c.qkvLayout == .fused)
    }

    @Test func detectsSplitQKV() throws {
        let c = try MossDiTConfig(descriptors: ditDescriptors(fusedQKV: false))
        #expect(c.qkvLayout == .split)
    }

    /// Forge's checkpoint merger saves a DiT-only file under `model.diffusion_model.` (DirtyMoss).
    @Test func acceptsAPrefixedDiTOnlyFile() throws {
        let prefixed = ditDescriptors().map {
            TensorDescriptor(name: "model.diffusion_model." + $0.name, dtype: $0.dtype, shape: $0.shape, byteCount: 0)
        }
        #expect(MossDiTConfig.keyPrefix(prefixed.map(\.name)) == "model.diffusion_model.")
        #expect(MossDiTConfig.keyPrefix(ditDescriptors().map(\.name)) == "")
        #expect(try MossDiTConfig(descriptors: prefixed) == .release)
        if case .dit = MossInspector.classify(prefixed) {} else { Issue.record("prefixed DiT not classified as a DiT") }
        let f16 = ditDescriptors(dtype: "F16").map {
            TensorDescriptor(name: "model.diffusion_model." + $0.name, dtype: $0.dtype, shape: $0.shape, byteCount: 0)
        }
        #expect(MossInspector.ditPrecision(f16) == .float16)
    }

    /// An all-in-one checkpoint (VAE or text encoder beside the DiT) stays a checkpoint to extract.
    @Test func allInOneCheckpointIsNotADiT() {
        let names = ditDescriptors().map { "model.diffusion_model." + $0.name } + ["vae.decoder.conv_in.weight"]
        #expect(MossDiTConfig.keyPrefix(names) == "")
    }

    @Test func missingKeysThrow() {
        #expect(throws: EngineError.self) {
            try MossDiTConfig(descriptors: [TensorDescriptor(name: "x", dtype: "BF16", shape: [1], byteCount: 2)])
        }
    }

    /// Fused qkv rows split q | k | v in dim-sized chunks (to be confirmed by dit-parity).
    @Test func qkvRowRanges() {
        let r = MossDiTConfig.release.qkvRowRanges
        #expect(r.q == 0..<3072 && r.k == 3072..<6144 && r.v == 6144..<9216)
    }

    @Test(.enabled(if: FileManager.default.fileExists(
        atPath: modelsDir.appendingPathComponent("diffusion_models/moss_image_sft_bf16.safetensors").path)))
    func realSFTCheckpointMatchesRelease() throws {
        let url = modelsDir.appendingPathComponent("diffusion_models/moss_image_sft_bf16.safetensors")
        let (tensors, _) = try SafetensorsHeader.readDescriptors(from: url)
        #expect(tensors.count == 379)
        #expect(try MossDiTConfig(descriptors: tensors) == .release)
    }
}

// MARK: - RoPE

@Suite("MossRoPEIndex — 4D (r, t, h, w) coordinates, image first")
struct MossRoPEIndexTests {
    @Test func bandsMatchTrainingConstruction() {
        let b = MossRoPEIndex.bands(headDim: 128, base: 10000)
        #expect(b.count == 16)
        #expect(b[0] == 1)
        #expect(abs(b[1] - Float(1 / pow(10000.0, 1.0 / 16.0))) < 1e-7)
    }

    @Test func imageTokensAreCentredAndFollowText() {
        let p = MossRoPEIndex.positions(gridHeight: 2, gridWidth: 2, textLength: 5)
        #expect(p.count == 4 + 5)
        // image (0,0): r=0, t=L, h=-(H-1)/2, w=-(W-1)/2
        #expect(p[0] == [0, 5, -0.5, -0.5])
        #expect(p[3] == [0, 5, 0.5, 0.5])
        // text tokens follow, t = 0…L-1, h = w = 0
        #expect(p[4] == [0, 0, 0, 0])
        #expect(p[8] == [0, 4, 0, 0])
    }

    @Test func rowMajorImageOrder() {
        let p = MossRoPEIndex.positions(gridHeight: 2, gridWidth: 3, textLength: 1)
        #expect(p[1] == [0, 1, -0.5, 0])  // (row 0, col 1)
        #expect(p[3] == [0, 1, 0.5, -1])  // (row 1, col 0)
    }
}

// MARK: - Quant metadata

@Suite("ComfyQuantBlob — per-tensor comfy_quant bytes")
struct ComfyQuantBlobTests {
    @Test func parsesInt8ConvRot() throws {
        let raw = Data(#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#.utf8)
        let blob = try #require(ComfyQuantBlob(bytes: raw))
        #expect(blob.format == "int8_tensorwise")
        #expect(blob.convRot && blob.groupSize == 256)
        #expect(blob.precision == .int8ConvRot)
    }

    @Test func parsesInt4ConvRot() throws {
        let blob = try #require(ComfyQuantBlob(bytes: Data(#"{"format": "convrot_w4a4", "convrot_groupsize": 256}"#.utf8)))
        #expect(blob.precision == .int4ConvRot)
    }

    @Test func garbageIsNil() {
        #expect(ComfyQuantBlob(bytes: Data([0xff, 0x00])) == nil)
    }
}

// MARK: - Inspector

@Suite("MossInspector — classify checkpoints from their header")
struct MossInspectorTests {
    @Test func bf16DiT() throws {
        let kind = MossInspector.classify(ditDescriptors())
        guard case .dit(let config, let precision) = kind else { Issue.record("not a DiT: \(kind)"); return }
        #expect(config == .release && precision == .bfloat16)
    }

    @Test func int8ConvRotDiT() {
        var ds = ditDescriptors(dtype: "BF16")
        ds.append(TensorDescriptor(name: "layers.0.proj.o.weight", dtype: "I8", shape: [3072, 3072], byteCount: 0))
        ds.append(TensorDescriptor(name: "layers.0.proj.o.comfy_quant", dtype: "U8", shape: [72], byteCount: 72))
        guard case .dit(_, let precision) = MossInspector.classify(ds) else { Issue.record("not a DiT"); return }
        #expect(precision == .int8ConvRot)
    }

    @Test func fp32TurboDiT() {
        guard case .dit(_, let precision) = MossInspector.classify(ditDescriptors(fusedQKV: false, dtype: "F32"))
        else { Issue.record("not a DiT"); return }
        #expect(precision == .float32)
    }

    @Test func dataSpaceSampler() {
        let ds = [
            TensorDescriptor(name: "latent_mix.Q", dtype: "F32", shape: [128, 128], byteCount: 0),
            TensorDescriptor(name: "step_codes", dtype: "F32", shape: [8], byteCount: 0),
        ]
        #expect(MossInspector.classify(ds) == .dmdSampler(steps: 8))
    }

    @Test func kohyaLoRA() {
        let ds = [
            TensorDescriptor(name: "diffusion_model.layers.0.proj.q.lora_down.weight", dtype: "F32", shape: [128, 3072], byteCount: 0),
            TensorDescriptor(name: "diffusion_model.layers.0.proj.q.lora_up.weight", dtype: "F32", shape: [3072, 128], byteCount: 0),
        ]
        #expect(MossInspector.classify(ds) == .lora(rank: 128))
    }

    @Test func qwen35TextEncoder() {
        var ds: [TensorDescriptor] = []
        for i in 0..<32 {
            let attn = i % 4 == 3 ? "self_attn.q_proj" : "linear_attn.in_proj_qkv"
            ds.append(TensorDescriptor(name: "model.language_model.layers.\(i).\(attn).weight", dtype: "I8", shape: [8192, 4096], byteCount: 0))
        }
        #expect(MossInspector.classify(ds) == .textEncoder(layers: 32, fullAttentionLayers: [3, 7, 11, 15, 19, 23, 27, 31]))
    }

    @Test func flux2VAE() {
        let ds = [
            TensorDescriptor(name: "decoder.post_quant_conv.weight", dtype: "F32", shape: [32, 32, 1, 1], byteCount: 0),
            TensorDescriptor(name: "bn.running_mean", dtype: "F32", shape: [128], byteCount: 0),
        ]
        #expect(MossInspector.classify(ds) == .vae(latentChannels: 32))
    }

    @Test func unknown() {
        #expect(MossInspector.classify([]) == .unknown)
    }
}

// MARK: - Engine seam

@Suite("MossEngine + StubBackend")
struct MossEngineTests {
    @Test func stubIsTheDefaultBackend() {
        let engine = MossEngine()
        #expect(engine.backendName == "stub")
        #expect(engine.capabilities.contains(.imageGeneration))
    }

    @Test func stubGeneratesAndReportsProgress() async throws {
        let engine = MossEngine()
        let steps = Counter()
        let result = try await engine.generate(GenerationRequest(prompt: "a red fox", variant: .turbo)) { _ in
            steps.increment()
        }
        #expect(result.width == 1024 && result.height == 1024)
        #expect(!result.imagePNG.isEmpty)
        #expect(steps.value == 4)
    }

    @Test func invalidRequestIsRejectedBeforeTheBackend() async {
        await #expect(throws: EngineError.self) {
            _ = try await MossEngine().generate(GenerationRequest(prompt: "", variant: .sft))
        }
    }

    @Test func stubEnhancesIntoALayoutPrompt() async throws {
        let text = try await MossEngine().enhancePrompt(PromptEnhanceRequest(prompt: "a sale poster", layout: true))
        #expect(text.contains("abs_centralized"))
    }
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = 0
    var value: Int { lock.withLock { _value } }
    func increment() { lock.withLock { _value += 1 } }
}
