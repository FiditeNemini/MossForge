import Foundation
import Testing

@testable import MossForgeCore

/// The official sampler, grounded in OpenMOSS-Team/MOSS-Image-SFT `pipeline_moss_image.py`
/// (`MOSSImageScheduler.set_timesteps` + the `__call__` loop) — the pipeline `gradio_app.py` runs.
@Suite("MossSchedule.pipelineSigmas — the official snapped grid")
struct PipelineSigmaTests {
    @Test func fiftyStepsShiftFiveMatchesTheTorchScheduler() {
        // Expected values computed with torch from the official scheduler code (float32).
        let s = MossSchedule.pipelineSigmas(steps: 50, shift: 5)
        #expect(s.count == 51)  // 50 evaluated + terminal 0
        let expected: [(Int, Double)] = [(0, 1.0), (1, 0.996), (2, 0.992), (25, 0.829), (26, 0.817), (47, 0.193), (48, 0.115), (49, 0.024), (50, 0)]
        for (i, v) in expected { #expect(abs(s[i] - v) < 1e-9, "σ[\(i)] = \(s[i]), expected \(v)") }
        #expect(zip(s, s.dropFirst()).allSatisfy { $0 > $1 })
        // Every evaluated σ sits on the 1000-step training grid.
        #expect(s.dropLast().allSatisfy { abs($0 * 1000 - ($0 * 1000).rounded()) < 1e-6 })
    }

    @Test func otherShiftsAndDedupe() {
        let eight = MossSchedule.pipelineSigmas(steps: 8, shift: 3)
        #expect(eight.count == 9 && abs(eight[1] - 0.948) < 1e-9 && abs(eight[7] - 0.009) < 1e-9)
        // Above ~200 steps neighbouring σ snap to the same grid point and are merged.
        #expect(MossSchedule.pipelineSigmas(steps: 300, shift: 5).count == 288)
    }
}

@Suite("SkipLayerGuidance + MossSamplerPlan — CFG / uncond-skip / SLG per step")
struct MossSamplerPlanTests {
    @Test func officialDefaultsSkipBlock22InTheNegativeBranchEveryStep() {
        let r = GenerationRequest(prompt: "a cat")
        #expect(r.slg == .official)
        #expect(r.slg.uncondSkipLayers == [22] && r.slg.layers.isEmpty)
        let plan = MossSamplerPlan(request: r)
        #expect(plan.steps.count == 50)
        #expect(plan.steps.allSatisfy { $0.uncondSkipLayers == [22] && $0.slgLayers.isEmpty && $0.forwards == 2 })
        #expect(plan.steps[0].timestep == 1000 && plan.steps[0].sigma == 1 && abs(plan.steps[49].nextSigma) < 1e-12)
    }

    @Test func cfgOneDropsTheNegativeBranch() {
        let plan = MossSamplerPlan(request: GenerationRequest(prompt: "a", cfg: 1))
        #expect(plan.steps.allSatisfy { $0.forwards == 1 && !$0.usesNegative })
    }

    @Test func standardSLGWindowIsHalfOpenOnStepFraction() {
        // slg_start ≤ i/N < slg_stop; outside it plain CFG without any skipping.
        let slg = SkipLayerGuidance.standard(layers: [24], scale: 3, start: 0.2, stop: 0.6)
        let plan = MossSamplerPlan(request: GenerationRequest(prompt: "a", steps: 10, slg: slg))
        #expect(plan.steps.map { !$0.slgLayers.isEmpty } == [false, false, true, true, true, true, false, false, false, false])
        #expect(plan.steps[3].forwards == 3 && plan.steps[0].forwards == 2)
        #expect(plan.steps[3].uncondSkipLayers.isEmpty)
    }

    @Test func guidanceCombinationMatchesThePipeline() {
        // prediction = u + cfg·(c − u); with SLG active: + scale·(c − c_skip); cfg 1 + SLG: c_skip + scale·(c − c_skip).
        #expect(MossGuidance.combine(cond: 2, uncond: 1, condSkipped: nil, cfg: 3, slgScale: 0) == 4)
        #expect(MossGuidance.combine(cond: 2, uncond: 1, condSkipped: 1.5, cfg: 3, slgScale: 2) == 5)
        #expect(MossGuidance.combine(cond: 2, uncond: nil, condSkipped: 1.5, cfg: 1, slgScale: 3) == 3)
        #expect(MossGuidance.combine(cond: 2, uncond: nil, condSkipped: nil, cfg: 1, slgScale: 0) == 2)
    }

    @Test func eulerStepMovesAlongTheVelocity() {
        // x_{k+1} = x_k + (σ_{k+1} − σ_k)·v; one step from σ 1 → 0 with v = ε − x₀ lands on x₀.
        #expect(MossGuidance.eulerStep(x: 3.0, velocity: 2.0, sigma: 1.0, nextSigma: 0.0) == 1.0)
    }

    @Test func validationRejectsBadGuidance() {
        #expect(throws: EngineError.self) {
            try GenerationRequest(prompt: "a", slg: .init(uncondSkipLayers: [42], layers: [], scale: 0, start: 0, stop: 1)).validate()
        }
        #expect(throws: EngineError.self) {
            try GenerationRequest(prompt: "a", slg: .standard(layers: [3], scale: 3, start: 0.8, stop: 0.2)).validate()
        }
    }
}
