import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// Qwen3.5 text-stack geometry, derived from the (prefix-stripped) header the way ComfyUI's
/// `qwen35.py` configures `qwen35_9b`, cross-checked against Qwen/Qwen3.5-9B `config.json`.
@Suite("Qwen35TextConfig — hybrid DeltaNet / attention geometry from the header")
struct Qwen35TextConfigTests {
    /// A synthetic language-model header (keys as `MossWeightPlan` emits them under
    /// `prefix: "model.language_model."`), shaped like the real 9B.
    private func lmDescriptors(layers: Int = 8) -> [TensorDescriptor] {
        func d(_ n: String, _ s: [Int], _ t: String = "BF16") -> TensorDescriptor {
            TensorDescriptor(name: n, dtype: t, shape: s, byteCount: 0)
        }
        var out = [d("embed_tokens.weight", [248320, 4096]), d("norm.weight", [4096])]
        for i in 0..<layers {
            let p = "layers.\(i)."
            out += [
                d(p + "input_layernorm.weight", [4096]), d(p + "post_attention_layernorm.weight", [4096]),
                d(p + "mlp.gate_proj.weight", [12288, 4096]), d(p + "mlp.up_proj.weight", [12288, 4096]),
                d(p + "mlp.down_proj.weight", [4096, 12288]),
            ]
            if (i + 1) % 4 == 0 {
                out += [
                    d(p + "self_attn.q_proj.weight", [8192, 4096]), d(p + "self_attn.k_proj.weight", [1024, 4096]),
                    d(p + "self_attn.v_proj.weight", [1024, 4096]), d(p + "self_attn.o_proj.weight", [4096, 4096]),
                    d(p + "self_attn.q_norm.weight", [256]), d(p + "self_attn.k_norm.weight", [256]),
                ]
            } else {
                out += [
                    d(p + "linear_attn.in_proj_qkv.weight", [8192, 4096]), d(p + "linear_attn.in_proj_z.weight", [4096, 4096]),
                    d(p + "linear_attn.in_proj_a.weight", [32, 4096]), d(p + "linear_attn.in_proj_b.weight", [32, 4096]),
                    d(p + "linear_attn.conv1d.weight", [8192, 1, 4]), d(p + "linear_attn.A_log", [32], "F32"),
                    d(p + "linear_attn.dt_bias", [32]), d(p + "linear_attn.norm.weight", [128], "F32"),
                    d(p + "linear_attn.out_proj.weight", [4096, 4096]),
                ]
            }
        }
        return out
    }

    @Test func derives9BGeometryAndLayerPattern() throws {
        let c = try Qwen35TextConfig(descriptors: lmDescriptors(layers: 32))
        #expect(c == .qwen35_9b)
        #expect(c.fullAttentionLayers == [3, 7, 11, 15, 19, 23, 27, 31])
        #expect(c.heads == 16 && c.kvHeads == 4 && c.headDim == 256 && c.rotaryDim == 64)
        #expect(c.linearKeyHeads == 16 && c.linearValueHeads == 32)
        #expect(c.linearKeyHeadDim == 128 && c.linearValueHeadDim == 128 && c.convKernel == 4)
        #expect(c.linearKeyDim == 2048 && c.linearValueDim == 4096 && c.linearConvDim == 8192)
        #expect(c.isFullAttention(3) && !c.isFullAttention(2))
    }

    @Test func layerCountFollowsTheHeader() throws {
        let c = try Qwen35TextConfig(descriptors: lmDescriptors(layers: 8))
        #expect(c.layers == 8 && c.fullAttentionLayers == [3, 7])
    }

    @Test func rejectsAHeaderWithoutAnEmbedding() {
        #expect(throws: EngineError.self) { try Qwen35TextConfig(descriptors: ditDescriptors(layers: 1)) }
    }

    @Test func ropeIsPartialSplitHalfOverTheFirst64DimsWithTheta1e7() {
        // precompute_freqs_cis(rotary_dim=64, θ=1e7): inv_freq_i = θ^(−2i/64), i = 0..31.
        let f = Qwen35TextConfig.qwen35_9b.ropeInverseFrequencies
        #expect(f.count == 32)
        #expect(f[0] == 1)
        #expect(abs(f[1] - Float(pow(1e7, -2.0 / 64))) < 1e-9)
        #expect(abs(f[31] - Float(pow(1e7, -62.0 / 64))) < 1e-12)
    }

    @Test func realHeaderMatchesConfigJSON() throws {
        let url = modelsDir.appendingPathComponent("text_encoders/qwen3.5_9b_int8_convrot.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let plan = try MossWeightPlan.build(url: url, prefix: "model.language_model.")
        let descriptors = plan.entries.map { TensorDescriptor(name: $0.key, dtype: "BF16", shape: $0.shape, byteCount: 0) }
        #expect(try Qwen35TextConfig(descriptors: descriptors) == .qwen35_9b)
    }
}
