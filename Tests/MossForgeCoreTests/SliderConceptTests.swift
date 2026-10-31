import Testing

@testable import MossForgeCore

// MOSS's TrainingRequest (slider fields are vars) and the chat-based wizard.

/// Concept-slider range (the apply-weight span a slider is trained for) and the wizard that
/// turns "a digital to analog slider" into name / target / positive / negative.
@Suite struct SliderRangeTests {
    @Test func defaultIsSymmetricUnit() {
        #expect(SliderRange.default == SliderRange(min: -1, max: 1))
        #expect(SliderRange.default.isValid)
    }

    @Test func rangeMustStraddleZero() {
        // Weight 0 is always "off", so a slider range has to contain it.
        #expect(SliderRange(min: 0, max: 3).isValid)
        #expect(SliderRange(min: -2, max: 0).isValid)
        #expect(!SliderRange(min: 1, max: 3).isValid)
        #expect(!SliderRange(min: -3, max: -1).isValid)
        #expect(!SliderRange(min: 0, max: 0).isValid)
        #expect(!SliderRange(min: 2, max: -2).isValid)
    }

    @Test func farEndIsFullStrength() {
        // Per-unit strength is scaled so the range's far end hits the full concept strength.
        #expect(SliderRange(min: -1, max: 1).unitStrength(full: 2) == 2)
        #expect(SliderRange(min: -5, max: 5).unitStrength(full: 2) == 0.4)
        #expect(SliderRange(min: -2, max: 4).unitStrength(full: 2) == 0.5)
    }

    @Test func multipliersCoverBothSidesAndSkipZero() {
        let r = SliderRange(min: -2, max: 4)
        let ms = stride(from: 0.0, to: 1.0, by: 0.01).map { r.multiplier(at: $0) }
        #expect(ms.allSatisfy { $0 >= -2 && $0 <= 4 })
        #expect(ms.allSatisfy { abs($0) >= 0.4 - 1e-9 })  // dead zone = 10% of the far end
        #expect(ms.contains { $0 < 0 })
        #expect(ms.contains { $0 > 0 })
        #expect(r.multiplier(at: 0) == -2)
    }

    @Test func oneSidedRangeOnlyTrainsThatSide() {
        let r = SliderRange(min: 0, max: 3)
        let ms = stride(from: 0.0, to: 1.0, by: 0.05).map { r.multiplier(at: $0) }
        #expect(ms.allSatisfy { $0 > 0 && $0 <= 3 })
    }

    @Test func trainingRequestDefaultsToUnitRange() {
        let req = TrainingRequest.sliderFixture()
        #expect(req.sliderRange == .default)
    }

    /// Slider round 2 (2026-10-17): η 2 drifted the scene and over-cooked skin at +1 on all
    /// three seeds; η 1 stayed clean but needed +1.5…+2 to age clearly.
    @Test func sliderStrengthDefaultsToOne() {
        let req = TrainingRequest.sliderFixture()
        #expect(req.sliderEta == 1)
        #expect(TrainingRequest.defaultSliderEta == 1)
    }

    /// The saved weight band keeps the trained negative end and doubles the positive one.
    @Test func recommendedBandExtendsThePositiveEndTwofold() {
        #expect(SliderRange.default.recommendedBand == SliderRange(min: -1, max: 2))
        #expect(SliderRange(min: 0, max: 3).recommendedBand == SliderRange(min: 0, max: 6))
        #expect(SliderRange(min: -2, max: 0).recommendedBand == SliderRange(min: -2, max: 0))
    }

    @Test func sliderPromptsAnchorTheConceptToTheTarget() {
        // Concept Sliders contrasts "target + positive" against "target + negative" (e.g. "old
        // person" vs "young person"), so the direction isolates the concept. Bare descriptors
        // ("old" vs "young") would also carry the whole subject difference.
        var req = TrainingRequest.sliderFixture(target: "a person", positive: "old, elderly", negative: "young, youthful")
        #expect(req.sliderPositivePrompt == "a person, old, elderly")
        #expect(req.sliderNegativePrompt == "a person, young, youthful")

        // Already written as a full prompt containing the target: used verbatim.
        req.sliderPositive = "a person, very old"
        #expect(req.sliderPositivePrompt == "a person, very old")

        // No target: the descriptors alone.
        req.sliderTarget = "  "
        #expect(req.sliderNegativePrompt == "young, youthful")
    }

    @Test func sliderGuidanceDefaultsMatchTheTrainedBaseline() {
        // 6 latent-sampling steps is the trained baseline. η moved from 2 to 1 after slider
        // round 2 (see sliderStrengthDefaultsToOne).
        let req = TrainingRequest.sliderFixture()
        #expect(req.sliderEta == 1.0)
        #expect(req.sliderSampleSteps == 10)  // MOSS: Base/SFT flow, not a 6-step Turbo
        #expect(req.sliderSettingsAreValid)
    }

    @Test func sliderGuidanceMustBePositiveAndSampleAtLeastTwoSteps() {
        // η scales the target direction (0 would train nothing); the trainer draws
        // k ∈ 1..<sampleSteps, so fewer than 2 steps leaves no partially-denoised latent.
        var req = TrainingRequest.sliderFixture()
        req.sliderEta = 0
        #expect(!req.sliderSettingsAreValid)
        req.sliderEta = 4
        req.sliderSampleSteps = 1
        #expect(!req.sliderSettingsAreValid)
        req.sliderSampleSteps = 2
        #expect(req.sliderSettingsAreValid)
    }

    @Test func preservePromptsAddOneAttributeVariantPerTrainingPrompt() {
        // Concept Sliders' disentanglement: each preserved attribute is appended to all three
        // prompts, so the learned direction is the same with or without it (age shouldn't
        // drag gender/ethnicity along). The unmodified triple always comes first.
        var req = TrainingRequest.sliderFixture(target: "a person", positive: "old", negative: "young")
        #expect(req.sliderPromptSets == [
            SliderPromptSet(target: "a person", positive: "a person, old", negative: "a person, young"),
        ])

        req.sliderPreserve = [" male ", "female", "", "Male"]  // trimmed, blanks + dupes dropped
        #expect(req.sliderPreservedAttributes == ["male", "female"])
        #expect(req.sliderPromptSets == [
            SliderPromptSet(target: "a person", positive: "a person, old", negative: "a person, young"),
            SliderPromptSet(
                target: "a person, male", positive: "a person, old, male", negative: "a person, young, male"),
            SliderPromptSet(
                target: "a person, female", positive: "a person, old, female",
                negative: "a person, young, female"),
        ])
    }

    @Test func preservePromptsParseFromCommaOrNewlineText() {
        #expect(SliderPromptSet.attributes(from: "male, female\nasian;  ") == ["male", "female", "asian"])
        #expect(SliderPromptSet.attributes(from: "   ").isEmpty)
    }
}

@Suite struct SliderConceptDraftTests {
    @Test func parsesLabelledReply() throws {
        let reply = """
            Name: digital_to_analog
            Target: a photograph
            Positive: analog film photo, film grain, soft halation, warm faded colours
            Negative: crisp digital photo, clean sensor, oversharpened, clinical colours
            """
        let d = try #require(SliderConceptDraft.parse(reply))
        #expect(d.name == "digital_to_analog")
        #expect(d.target == "a photograph")
        #expect(d.positive.hasPrefix("analog film photo"))
        #expect(d.negative.hasPrefix("crisp digital photo"))
    }

    @Test func toleratesMarkdownThinkBlocksAndCase() throws {
        let reply = """
            <think>the user wants…</think>
            **name:** Old Age
            - **TARGET:** "a person"
            * positive (+): elderly, wrinkled skin
            * negative (−): young, smooth skin
            """
        let d = try #require(SliderConceptDraft.parse(reply))
        #expect(d.name == "old_age")
        #expect(d.target == "a person")
        #expect(d.positive == "elderly, wrinkled skin")
        #expect(d.negative == "young, smooth skin")
    }

    @Test func missingFieldIsNoDraft() {
        #expect(SliderConceptDraft.parse("Name: x\nTarget: y\nPositive: z") == nil)
        #expect(SliderConceptDraft.parse("a red fox in snow, highly detailed") == nil)
    }

    @Test func heuristicReadsFromToPhrasing() throws {
        let d = try #require(SliderConceptDraft.heuristic("I'd like a digital to analog slider"))
        #expect(d.negative == "digital")
        #expect(d.positive == "analog")
        #expect(d.name == "digital_to_analog")
        #expect(!d.target.isEmpty)
        #expect(SliderConceptDraft.heuristic("more cinematic") == nil)
    }

    @Test func slugIsFileSafe() {
        #expect(SliderConceptDraft.slug("Digital → Analog / v2!") == "digital_analog_v2")
        #expect(SliderConceptDraft.slug("   ") == "slider")
    }

    @Test func wizardRequestUsesItsOwnInstructions() {
        let messages = SliderConceptDraft.messages(for: "a digital to analog slider")
        #expect(messages.first == ChatMessage(.system, SliderConceptDraft.systemPrompt))
        #expect(messages.last == ChatMessage(.user, "a digital to analog slider"))
    }
}

extension TrainingRequest {
    static func sliderFixture(target: String = "", positive: String = "", negative: String = "") -> TrainingRequest {
        var r = TrainingRequest(
            baseModel: .init(fileURLWithPath: "/m"), dataset: .init(fileURLWithPath: "/d"), output: .init(fileURLWithPath: "/o.safetensors"))
        r.mode = .slider
        r.sliderTarget = target
        r.sliderPositive = positive
        r.sliderNegative = negative
        return r
    }
}

@Suite struct MossSliderRequestTests {
    @Test func sliderNeedsNoImagesButValidPrompts() throws {
        var r = TrainingRequest.sliderFixture(target: "a person", positive: "old", negative: "young")
        try r.validate(sampleCount: 0)
        r.sliderPositive = " "
        #expect(throws: EngineError.self) { try r.validate(sampleCount: 0) }
        r.sliderPositive = "old"
        r.sliderRange = SliderRange(min: 1, max: 2)
        #expect(throws: EngineError.self) { try r.validate(sampleCount: 0) }
    }

    /// Slider test images: one prompt at the range's ends and at 0 (off), same seed.
    @Test func sliderTestImagesSweepTheRange() {
        var r = TrainingRequest.sliderFixture(target: "a person", positive: "old", negative: "young")
        #expect(r.sliderTestWeights == [-1, 0, 1])
        r.sliderRange = SliderRange(min: 0, max: 3)
        #expect(r.sliderTestWeights == [0, 1.5, 3])
        #expect(r.preparedSamplePrompts == [])
        #expect(r.sliderTestPrompt == "a person")
        r.samplePrompts = ["an old sailor"]
        #expect(r.sliderTestPrompt == "an old sailor")
    }
}
