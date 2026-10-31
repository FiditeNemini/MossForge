import Foundation
import Testing

@testable import MossForgeCore

/// LoRA training on Base: the paper's Stage-3 σ distribution (App. A.2, eq. 4), aspect buckets, and the
/// training request.
@Suite("Training — σ sampling, buckets, request")
struct TrainingTests {
    @Test func stage3SigmaDistributionIsNormalisedAndShapedLikeThePaper() {
        let p = MossNoiseSchedule.stage3.probabilities
        #expect(p.count == 1000)
        #expect(abs(p.reduce(0, +) - 1) < 1e-9)
        #expect(p.allSatisfy { $0 > 0 })  // 1% uniform keeps every σ reachable
        // Truncated logit-normal on S = 1 − sigmoid(Z), μ −1.09 → mass sits at high σ (the noisy end).
        let mean = zip(p.indices, p).reduce(0.0) { $0 + Double($1.0 + 1) / 1000 * $1.1 }
        #expect(mean > 0.6 && mean < 0.8)
        let argmax = p.indices.max { p[$0] < p[$1] }!
        #expect(argmax > 650 && argmax < 900)
    }

    @Test func samplingIsTheInverseCDF() {
        let s = MossNoiseSchedule.stage3
        #expect(s.sigma(u: 0) == 0.001)
        #expect(s.sigma(u: 0.999_999_9) == 1.0)
        let median = s.sigma(u: 0.5)
        #expect(median > 0.6 && median < 0.85)
        #expect(s.sigma(u: 0.25) <= median && median <= s.sigma(u: 0.75))
    }

    @Test func bucketsKeepTheAspectAtTheBudgetInMultiplesOf16() {
        let square = TrainingBuckets.size(forWidth: 3000, height: 3000, pixels: 512 * 512)
        #expect(square == Dimensions(width: 512, height: 512))
        let wide = TrainingBuckets.size(forWidth: 1920, height: 1080, pixels: 1024 * 1024)
        #expect(wide.width % 16 == 0 && wide.height % 16 == 0)
        #expect(wide.width * wide.height <= 1024 * 1024)
        #expect(abs(Double(wide.width) / Double(wide.height) - 16.0 / 9.0) < 0.05)
        // Never upscale a small image past its own size.
        #expect(TrainingBuckets.size(forWidth: 300, height: 200, pixels: 1024 * 1024) == Dimensions(width: 288, height: 192))
    }

    @Test func requestDefaultsAndValidation() throws {
        var r = TrainingRequest(
            baseModel: URL(fileURLWithPath: "/m/diffusion_models/moss_image_base_bf16.safetensors"),
            dataset: URL(fileURLWithPath: "/data/foxes"), output: URL(fileURLWithPath: "/m/lora/trained/foxes.safetensors"))
        #expect(r.rank == 16 && r.alpha == 16 && r.learningRate == 1e-4 && r.steps == 1000 && r.pixels == 1024 * 1024)
        #expect(r.targets == .attention && r.saveEvery == 250)
        try r.validate(sampleCount: 10)
        #expect(throws: EngineError.self) { try r.validate(sampleCount: 0) }
        r.rank = 0
        #expect(throws: EngineError.self) { try r.validate(sampleCount: 10) }
    }

    @Test func targetSetsNameTheFusedLinears() {
        #expect(LoRATargetSet.attention.modules == ["proj.qkv", "proj.o"])
        #expect(LoRATargetSet.attentionAndFFN.modules == ["proj.qkv", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"])
    }

    @Test func captionsGetTheTriggerWord() {
        #expect(TrainingRequest.caption("a fox in snow", trigger: "mossfox") == "mossfox, a fox in snow")
        #expect(TrainingRequest.caption("mossfox, a fox", trigger: "mossfox") == "mossfox, a fox")
        #expect(TrainingRequest.caption("a fox", trigger: "") == "a fox")
    }
}
