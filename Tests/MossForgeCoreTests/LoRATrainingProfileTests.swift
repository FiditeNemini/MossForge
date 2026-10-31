import Testing

@testable import MossForgeCore

// MOSS layer
// sets (attention / attention + FFN) and the run-length scale the Quick / Standard / Detailed presets apply.
@Suite("Train page LoRA types")
struct LoRATrainingProfileTests {
    @Test("character learns detail (attention + FFN) at a higher rank than a concept")
    func characterVersusConcept() throws {
        let character = try #require(LoRATrainingProfile.character.settings(imageCount: 25))
        let concept = try #require(LoRATrainingProfile.concept.settings(imageCount: 25))
        #expect(character.targets == .attentionAndFFN)
        #expect(concept.targets == .attention)
        #expect(character.rank > concept.rank)
        #expect(character.factorization == .lora)
    }

    @Test("style trains longest and learns texture")
    func styleSettings() throws {
        let style = try #require(LoRATrainingProfile.style.settings(imageCount: 30))
        let character = try #require(LoRATrainingProfile.character.settings(imageCount: 30))
        #expect(style.targets == .attentionAndFFN)
        #expect(style.steps >= character.steps)
    }

    @Test("steps scale with the dataset, within each type's band, on the 50-step grid")
    func stepsScale() throws {
        for profile in LoRATrainingProfile.allCases where profile != .custom {
            let small = try #require(profile.settings(imageCount: 5)).steps
            let mid = try #require(profile.settings(imageCount: 25)).steps
            let huge = try #require(profile.settings(imageCount: 500)).steps
            #expect(small == profile.stepBand.lowerBound)
            #expect(huge == profile.stepBand.upperBound)
            #expect(small <= mid && mid <= huge)
            #expect(mid % 50 == 0)
        }
        // No dataset picked yet: the type's typical run.
        #expect(LoRATrainingProfile.character.settings(imageCount: 0)?.steps == LoRATrainingProfile.character.typicalSteps)
    }

    @Test("the run-length scale shortens or lengthens the run after the band, still on the grid")
    func runScale() throws {
        let standard = try #require(LoRATrainingProfile.character.settings(imageCount: 25)).steps
        let quick = try #require(LoRATrainingProfile.character.settings(imageCount: 25, runScale: 0.1)).steps
        let detailed = try #require(LoRATrainingProfile.character.settings(imageCount: 25, runScale: 1.5)).steps
        #expect(quick < LoRATrainingProfile.character.stepBand.lowerBound)  // a quick check isn't clamped back up
        #expect(quick >= 50 && quick % 50 == 0)
        #expect(detailed > standard && detailed % 50 == 0)
        #expect(try #require(LoRATrainingProfile.concept.settings(imageCount: 1, runScale: 0.01)).steps == 50)
    }

    @Test("checkpoints land about four times per run")
    func saveEvery() throws {
        let s = try #require(LoRATrainingProfile.style.settings(imageCount: 30))
        #expect(s.saveEvery % 50 == 0)
        #expect(s.steps / s.saveEvery >= 3 && s.steps / s.saveEvery <= 5)
    }

    @Test("custom changes nothing, and every type says how to caption")
    func customAndTips() {
        #expect(LoRATrainingProfile.custom.settings(imageCount: 20) == nil)
        for profile in LoRATrainingProfile.allCases where profile != .custom {
            #expect(!profile.captionTip.isEmpty)
            #expect(!profile.summary.isEmpty)
            #expect(profile.recommendedImages.lowerBound >= 10)
        }
    }
}
