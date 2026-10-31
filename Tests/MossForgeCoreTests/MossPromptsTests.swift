import Foundation
import Testing
@testable import MossForgeCore

/// Settings → Prompts: the instructions MossForge sends to Qwen3.5, editable without a rebuild.
@Suite("MossPrompts — user-editable model prompts")
struct MossPromptsTests {
    @Test func everyPromptHasATitleAndDefaultsToTheBuiltIn() {
        let p = MossPrompts()
        for kind in MossPrompts.Kind.allCases {
            #expect(!kind.title.isEmpty)
            #expect(p[kind] == kind.builtIn)
            #expect(p.isDefault(kind))
        }
        #expect(p.isDefault)
    }

    @Test func editsChangeTheEnhancerChat() {
        var p = MossPrompts()
        p[.enhancerRules] = "Be brief."
        p[.enhancerCaption] = "Answer {\"caption\": \"…\"}."
        let chat = MossEnhancerPrompt.chat(prompt: "x", layout: false, canvas: Dimensions(width: 512, height: 512), prompts: p)
        #expect(chat.hasPrefix("<|im_start|>system\nBe brief.\n\nAnswer {\"caption\": \"…\"}.<|im_end|>\n"))
        #expect(!p.isDefault(.enhancerRules) && p.isDefault(.enhancerLayout))
    }

    @Test func layoutTemplateFillsTheCanvasPlaceholders() {
        var p = MossPrompts()
        p[.enhancerLayout] = "Canvas {width} by {height}; x in 0..{width}."
        let system = p.enhancerSystem(layout: true, canvas: Dimensions(width: 1536, height: 1024))
        #expect(system.hasSuffix("\n\nCanvas 1536 by 1024; x in 0..1536."))
    }

    @Test func captionTemplateFillsTheReference() {
        var p = MossPrompts()
        p[.caption] = "Caption it. Reference: {{reference}}"
        #expect(MossCaptionPrompt.instruction(reference: "a cat", prompts: p) == "Caption it. Reference: a cat")
        #expect(MossCaptionPrompt.instruction(reference: nil, prompts: p) == "Caption it. Reference: (none)")
    }

    @Test func sliderMessagesUseTheEditedPrompt() {
        var p = MossPrompts()
        p[.slider] = "Design a slider."
        #expect(SliderConceptDraft.messages(for: "warm to cool", prompts: p).first == ChatMessage(.system, "Design a slider."))
    }

    @Test func resetRestoresOneOrAll() {
        var p = MossPrompts()
        p[.caption] = "x"
        p[.slider] = "y"
        p.reset(.caption)
        #expect(p.isDefault(.caption) && !p.isDefault(.slider))
        p.resetAll()
        #expect(p.isDefault)
    }

    /// Only edited prompts are stored, so a default improved in a later release reaches users who never touched it.
    @Test func codableStoresOnlyEditsAndToleratesUnknownKeys() throws {
        var p = MossPrompts()
        p[.slider] = "edited"
        let json = String(decoding: try JSONEncoder().encode(p), as: UTF8.self)
        #expect(json.contains("edited") && !json.contains("recaptioner"))
        #expect(try JSONDecoder().decode(MossPrompts.self, from: Data(json.utf8)) == p)
        let future = #"{"slider":"s","someFuturePrompt":"z"}"#
        #expect(try JSONDecoder().decode(MossPrompts.self, from: Data(future.utf8))[.slider] == "s")
    }

    /// The editor keeps what was typed (clearing a field doesn't snap back); the model gets the built-in.
    @Test func blankEditFallsBackToTheBuiltInWhenUsed() {
        var p = MossPrompts()
        p[.enhancerRules] = "  \n"
        #expect(p[.enhancerRules] == "  \n")
        #expect(p.resolved(.enhancerRules) == MossPrompts.Kind.enhancerRules.builtIn)
        #expect(p.enhancerSystem(layout: false, canvas: Dimensions(width: 64, height: 64)).hasPrefix(MossPrompts.Kind.enhancerRules.builtIn))
    }

    @Test func requestsDefaultToBuiltInPrompts() {
        #expect(PromptEnhanceRequest(prompt: "x").prompts.isDefault)
        #expect(CaptionRequest(image: URL(fileURLWithPath: "/a.png")).prompts.isDefault)
    }
}
