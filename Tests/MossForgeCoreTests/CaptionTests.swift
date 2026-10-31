import Foundation
import Testing

@testable import MossForgeCore

/// The captioner speaks the official USERGUIDE recaption prompt and parses its JSON answer.
@Suite struct CaptionTests {
    @Test func instructionIsTheOfficialRecaptionPromptWithTheReferenceFilledIn() {
        let text = MossCaptionPrompt.instruction(reference: "a lighthouse poster")
        #expect(text.hasPrefix("You are an image recaptioner. Return exactly one JSON object with this schema:"))
        #expect(text.contains("Use normalized 0–1000 [y_min, x_min, y_max, x_max] coordinates"))
        #expect(text.hasSuffix("Here is the original caption reference:\na lighthouse poster"))
        #expect(!text.contains("{{reference}}"))
        // No reference: say so rather than leave the placeholder.
        #expect(MossCaptionPrompt.instruction(reference: nil).hasSuffix("reference:\n(none)"))
        #expect(MossCaptionPrompt.instruction(reference: "  ").hasSuffix("reference:\n(none)"))
    }

    /// Text then image (the USERGUIDE request order), one `<|image_pad|>` per merged image token,
    /// non-thinking generation prompt.
    @Test func chatPlacesOneImagePadPerTokenAfterTheInstruction() {
        let chat = MossCaptionPrompt.chat(imageTokens: 3, reference: "x")
        #expect(chat.hasPrefix("<|im_start|>user\nYou are an image recaptioner."))
        #expect(chat.hasSuffix(
            "reference:\nx<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        #expect(chat.components(separatedBy: MossCaptionPrompt.imagePad).count - 1 == 3)
    }

    @Test func parsesTheAnswerAndPicksTheRequestedDetail() throws {
        let answer = """
            Sure! ```json
            {"caption": {"long": "A tall white lighthouse on a cliff.", "medium": "A lighthouse.", "short": "Lighthouse."},
             "watermark": [{"box_2d": [900, 10, 980, 200], "label": "stock"}], "borders": ["none"]}
            ```
            """
        let c = try MossCaption.parse(answer)
        #expect(c.long == "A tall white lighthouse on a cliff.")
        #expect(c.medium == "A lighthouse." && c.short == "Lighthouse.")
        #expect(c.watermarks == [.init(box: [900, 10, 980, 200], label: "stock")])
        #expect(c.borders == ["none"])
        #expect(c.text(.long) == c.long && c.text(.medium) == c.medium && c.text(.short) == c.short)
    }

    @Test func rejectsAnswersWithoutAllThreeCaptions() {
        for bad in [
            "no json here", #"{"caption": {"long": "x", "medium": "y"}}"#,
            #"{"caption": {"long": "", "medium": "y", "short": "z"}}"#, #"{"watermark": []}"#,
        ] {
            #expect(throws: EngineError.self, "\(bad)") { try MossCaption.parse(bad) }
        }
    }

    @Test func requestDefaultsToTheLongCaption() {
        let r = CaptionRequest(image: URL(fileURLWithPath: "/a.png"))
        #expect(r.detail == .long && r.reference == nil && r.maxPixels == CaptionRequest.defaultMaxPixels)
        #expect(CaptionRequest(image: URL(fileURLWithPath: "/a.png"), maxPixels: 1).maxPixels == CaptionRequest.minPixels)
        #expect(CaptionDetail.allCases.map(\.rawValue) == ["long", "medium", "short"])
    }
}
