import Foundation
import Testing

@testable import MossForgeCore

// Sizes are MOSS ratios.
@Suite("Character Studio dataset plan")
struct CharacterDatasetPlanTests {
    @Test("default seeds span close-up to full body, one per shot")
    func defaultSeedsCoverFraming() {
        let shots = CharacterDatasetPlan.defaultSeeds.map(\.shot)
        #expect(shots == CharacterShot.allCases)
        #expect(shots.first == .closeUp)
        #expect(shots.contains(.fullBody))
        #expect(CharacterDatasetPlan.defaultSeeds.allSatisfy { !$0.prompt.isEmpty })
    }

    @Test("default pool is larger than the recommended keep range, so there's room to cull")
    func defaultPoolExceedsKeep() {
        let total = CharacterDatasetPlan.defaultSeeds.count * CharacterDatasetPlan.defaultVariations
        #expect(total > CharacterDatasetPlan.recommendedKeep.upperBound)
    }

    @Test("jobs = non-empty seeds × variations, every seed distinct")
    func jobsCountAndSeeds() {
        var seeds = CharacterDatasetPlan.defaultSeeds
        seeds[2].prompt = "   "  // blank rows are skipped
        let jobs = CharacterDatasetPlan.jobs(
            character: "a red-haired woman with freckles", seeds: seeds, variations: 5, baseSeed: 42)
        #expect(jobs.count == (seeds.count - 1) * 5)
        #expect(Set(jobs.map(\.seed)).count == jobs.count)
        #expect(!jobs.contains { $0.shot == seeds[2].shot })
    }

    @Test("each variation of a seed asks the enhancer for a different view")
    func enhanceInputsDistinct() {
        let jobs = CharacterDatasetPlan.jobs(
            character: "a red-haired woman with freckles", seeds: CharacterDatasetPlan.defaultSeeds,
            variations: 5, baseSeed: 1)
        for shot in CharacterShot.allCases {
            let inputs = jobs.filter { $0.shot == shot }.map(\.enhanceInput)
            #expect(Set(inputs).count == inputs.count)
        }
        for job in jobs {
            #expect(job.enhanceInput.contains("a red-haired woman with freckles"))
            #expect(job.enhanceInput.contains(job.shot.framing))
        }
    }

    @Test("variations clamp to the supported range")
    func variationsClamp() {
        let seeds = [CharacterSeed(shot: .closeUp, prompt: "smiling")]
        #expect(CharacterDatasetPlan.jobs(character: "x", seeds: seeds, variations: 0, baseSeed: 1).count == 1)
        #expect(
            CharacterDatasetPlan.jobs(character: "x", seeds: seeds, variations: 99, baseSeed: 1).count
                == CharacterDatasetPlan.variationRange.upperBound)
    }

    @Test("the final prompt always carries the shot framing, even if the enhancer drops it")
    func finalPromptKeepsFraming() {
        let p = CharacterDatasetPlan.finalPrompt(enhanced: "A woman on a beach at sunset.", shot: .fullBody)
        #expect(p.contains(CharacterShot.fullBody.framing))
        let already = CharacterDatasetPlan.finalPrompt(
            enhanced: "A \(CharacterShot.fullBody.framing) of a woman.", shot: .fullBody)
        #expect(already == "A \(CharacterShot.fullBody.framing) of a woman.")
    }

    @Test("the angle is re-added when the enhancer drops it, so left/right variations differ")
    func finalPromptKeepsAngle() {
        // Seen in a real run: both three-quarter variations came back as the same text.
        let same = "A close-up portrait of a woman, captured in three-quarter view under window light."
        let left = CharacterDatasetPlan.finalPrompt(enhanced: same, shot: .closeUp, angle: CharacterDatasetPlan.angles[1])
        let right = CharacterDatasetPlan.finalPrompt(enhanced: same, shot: .closeUp, angle: CharacterDatasetPlan.angles[2])
        #expect(left != right)
        #expect(left.contains(CharacterDatasetPlan.angles[1]))
    }

    @Test("seeds stay in the 32-bit range other tools reproduce, and are never 0 (= random)")
    func seedsAre32Bit() {
        let jobs = CharacterDatasetPlan.jobs(
            character: "x", seeds: CharacterDatasetPlan.defaultSeeds, variations: 8, baseSeed: UInt64(UInt32.max) - 3)
        #expect(jobs.allSatisfy { $0.seed > 0 && $0.seed <= UInt64(UInt32.max) })
        #expect(Set(jobs.map(\.seed)).count == jobs.count)
    }

    @Test("captions lead with the trigger word")
    func captionTrigger() {
        #expect(CharacterDatasetPlan.caption(trigger: "zxq woman", prompt: "standing in rain") == "zxq woman, standing in rain")
        #expect(CharacterDatasetPlan.caption(trigger: "  ", prompt: "standing in rain") == "standing in rain")
    }

    @Test("close shots are square-ish, full-body shots are tall")
    func shotAspect() {
        #expect(CharacterShot.closeUp.ratio == "1:1")
        #expect(CharacterShot.fullBody.ratio == "2:3")
        let d = CharacterShot.fullBody.dimensions(pixels: 1_048_576)
        #expect(d.height > d.width)
    }
}

/// Character Studio's reference photo: each row restages picture 1 (an edit LoRA's reference).
@Suite struct CharacterReferencePromptTests {
    @Test func wrapsTheRowPromptAroundPicture1() {
        let p = CharacterDatasetPlan.referencePrompt(" full-body shot, standing in a park. ")
        #expect(p.hasPrefix("Show the person from picture 1 in a new photograph: full-body shot, standing in a park. Keep their face"))
        #expect(!p.contains(".."))
    }
}
