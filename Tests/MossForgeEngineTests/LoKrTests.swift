import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// LoKr (LyCORIS Kronecker) adapters: the factor split, the
/// trainer's output keys and the inference key map.
@Suite struct LoKrTests {
    @Test func loKrFactorizationSplitsCorrectly() {
        let (m, n) = LoRAFactorization.split(3072, factor: -1)
        #expect(m * n == 3072)
        #expect(m == 48 && n == 64)
        #expect(LoRAFactorization.split(3072, factor: 8) == (8, 384))
        #expect(LoRAFactorization.split(3072, factor: 16) == (16, 192))
        for dim in [16, 320, 1024, 3072, 9216, 16384] {
            for factor in [-1, 4, 8, 16] {
                let s = LoRAFactorization.split(dim, factor: factor)
                #expect(s.m * s.n == dim)
                #expect(s.m <= s.n)
            }
        }
        #expect(LoRAFactorization.split(3079, factor: -1) == (1, 3079))
        #expect(LoRAFactorization.allCases == [.lora, .lokr])
    }

    /// `w1 = [out.m, in.m]`, `w2 = [out.n, in.n]` so `w1 ⊗ w2` is `[out, in]`.
    @Test func factorShapesMultiplyBackToTheWeight() {
        let s = MossLoRAOutput.lokrShapes(out: 9216, in: 3072, factor: 8)
        #expect(s.w1 == [8, 8] && s.w2 == [1152, 384])
        #expect(s.w1[0] * s.w2[0] == 9216 && s.w1[1] * s.w2[1] == 3072)
    }

    @Test func trainedLoKrKeysLoadThroughTheKeyMap() throws {
        let config = MossDiTConfig.release
        var descriptors: [TensorDescriptor] = []
        for (module, shape) in MossLoRAOutput.targetShapes(config: config, targets: .attention) {
            let k = MossLoRAOutput.lokrKeys(module: module)
            let s = MossLoRAOutput.lokrShapes(out: shape[0], in: shape[1], factor: -1)
            descriptors.append(TensorDescriptor(name: k.w1, dtype: "F32", shape: s.w1, byteCount: 0))
            descriptors.append(TensorDescriptor(name: k.w2, dtype: "F32", shape: s.w2, byteCount: 0))
        }
        let plan = try MossLoRAKeyMap.plan(descriptors: descriptors, alphas: [:], config: config)
        #expect(plan.unmapped.isEmpty && plan.parts.count == 42 * 2)
        #expect(plan.parts.allSatisfy { $0.kind == .kronecker && $0.scale == 1 && $0.rows == nil })
        #expect(plan.parts.first { $0.target == "layers.0.proj.qkv.weight" }?.down.hasSuffix(".lokr_w1") == true)
    }

    /// LyCORIS files from other trainers name split q/k/v (`lora_unet_…` is not MOSS's; the
    /// diffusion_model prefix is) — they land on the fused qkv's row slice.
    @Test func splitQKVLoKrLandsOnItsRows() throws {
        let config = MossDiTConfig.release
        let d = [
            TensorDescriptor(name: "diffusion_model.layers.3.proj.k.lokr_w1", dtype: "F32", shape: [48, 48], byteCount: 0),
            TensorDescriptor(name: "diffusion_model.layers.3.proj.k.lokr_w2", dtype: "F32", shape: [64, 64], byteCount: 0),
        ]
        let plan = try MossLoRAKeyMap.plan(descriptors: d, alphas: [:], config: config)
        #expect(plan.parts.count == 1)
        #expect(plan.parts[0].rows == config.qkvRowRanges.k)
        #expect(plan.parts[0].block == 3)
    }

    @Test func mismatchedFactorsAreRejected() {
        let d = [
            TensorDescriptor(name: "diffusion_model.layers.0.proj.o.lokr_w1", dtype: "F32", shape: [48, 48], byteCount: 0),
            TensorDescriptor(name: "diffusion_model.layers.0.proj.o.lokr_w2", dtype: "F32", shape: [64, 60], byteCount: 0),
        ]
        #expect(throws: EngineError.self) { try MossLoRAKeyMap.plan(descriptors: d, alphas: [:], config: .release) }
    }

    @Test func trainingRequestCarriesTheFactorization() throws {
        var r = TrainingRequest(baseModel: URL(fileURLWithPath: "/m"), dataset: URL(fileURLWithPath: "/d"), output: URL(fileURLWithPath: "/o"))
        #expect(r.factorization == .lora && r.lokrFactor == -1)
        r.factorization = .lokr
        r.lokrFactor = 8
        try r.validate(sampleCount: 1)
        #expect(r.captionDropout == 0)
        r.captionDropout = 0.9
        #expect(throws: EngineError.self) { try r.validate(sampleCount: 1) }
    }
}
