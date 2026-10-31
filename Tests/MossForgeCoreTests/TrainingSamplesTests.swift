import Foundation
import Testing

@testable import MossForgeCore

/// Train → test prompts: sample images before training, at every checkpoint and at the end, so the
/// LoRA's effect can be watched step by step.
@Suite struct TrainingSamplesTests {
    let output = URL(fileURLWithPath: "/m/lora/trained/fox.safetensors")

    @Test func samplesRunBeforeTrainingAtEachCheckpointAndAtTheEnd() {
        #expect(TrainingSamples.steps(total: 1000, every: 250) == [0, 250, 500, 750, 1000])
        #expect(TrainingSamples.steps(total: 1000, every: 300) == [0, 300, 600, 900, 1000])
        #expect(TrainingSamples.steps(total: 100, every: 0) == [0, 100])
        #expect(TrainingSamples.steps(total: 100, every: 500) == [0, 100])
    }

    @Test func filesLiveBesideTheCheckpointsAndParseBack() {
        let url = TrainingSamples.url(output: output, step: 250, index: 1)
        #expect(url.path == "/m/lora/trained/fox-checkpoints/samples/step00250-p2.png")
        #expect(TrainingSamples.parse(url) == .init(step: 250, index: 1))
        #expect(TrainingSamples.parse(URL(fileURLWithPath: "/x/fox-step00250.safetensors")) == nil)
    }

    @Test func groupsSamplesByStepInOrder() {
        let urls = [(500, 0), (0, 2), (0, 0), (500, 1), (0, 1)].map { TrainingSamples.url(output: output, step: $0.0, index: $0.1) }
        let rows = TrainingSamples.rows(urls)
        #expect(rows.map(\.step) == [0, 500])
        #expect(rows[0].images.map(\.lastPathComponent) == ["step00000-p1.png", "step00000-p2.png", "step00000-p3.png"])
        #expect(rows[1].images.count == 2)
    }

    @Test func requestKeepsUpToThreeNonEmptyPromptsWithTheTrigger() throws {
        var r = TrainingRequest(baseModel: output, dataset: output, output: output, trigger: "mossfox")
        r.samplePrompts = ["a fox in snow", "  ", "mossfox on a beach", "x", "y"]
        #expect(r.preparedSamplePrompts == ["mossfox, a fox in snow", "mossfox on a beach", "mossfox, x"])
        r.sampleSteps = 0
        #expect(throws: EngineError.self) { try r.validate(sampleCount: 1) }
    }

    @Test func suggestionPromptAsksForThreeVariedPromptsFromTheCaptions() {
        let messages = TrainingPromptSuggester.messages(captions: ["a red fox", "a fox asleep"], trigger: "mossfox")
        #expect(messages.count == 2)
        #expect(messages[0].content.contains("exactly 3"))
        #expect(messages[1].content.contains("- a red fox"))
        #expect(messages[1].content.contains("mossfox"))
    }

    @Test func suggestionAnswersParseToThreeCleanPrompts() {
        let answer = """
            Here are three prompts:
            1. mossfox curled up on a mossy log, soft morning light
            2) **mossfox** in a neon city street at night
            - "mossfox as a watercolour illustration"
            4. extra one
            """
        #expect(TrainingPromptSuggester.parse(answer) == [
            "mossfox curled up on a mossy log, soft morning light",
            "mossfox in a neon city street at night",
            "mossfox as a watercolour illustration",
        ])
    }

    @Test func withoutAModelSuggestionsComeFromTheCaptions() {
        let picks = TrainingPromptSuggester.fallback(captions: ["a", "b", "c", "d", "e"], trigger: "t")
        #expect(picks == ["t, a", "t, c", "t, e"])
        #expect(TrainingPromptSuggester.fallback(captions: [], trigger: "t") == ["t, a photo", "t, an illustration", "t, close-up portrait"])
    }
}
