import Foundation
import Testing

@testable import MossForgeCore

/// Port of the official enhancer's compiler (`skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`,
/// Apache-2.0). Golden outputs below were produced by running the Python reference on the same drafts.
@Suite("MossPromptCompiler — draft JSON → MOSS prompt (reference parity)")
struct MossPromptCompilerTests {
    let landscape = #"""
        {"width":1536,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy",
         "summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},
         "items":[{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\n\"MOSS\" 🪴","bbox":[928,272,1408,448]},
                  {"label":"object","caption":"A cup\twith a back\\slash.","bbox":[176.5,224.2,800.7,864]}]}
        """#

    func compile(_ json: String) throws -> MossPromptCompiler.Output {
        try MossPromptCompiler.compile(Data(json.utf8))
    }

    @Test func layoutMatchesThePythonReferenceByteForByte() throws {
        let out = try compile(landscape)
        let golden = #"{"summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},"items":[{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\n\"MOSS\" 🪴","bbox":[-240,160,-64,640]},{"label":"object","caption":"A cup\twith a back\\slash.","bbox":[-288,-592,352,33]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#
        #expect(out.prompt == golden)
        #expect(out.size == Dimensions(width: 1536, height: 1024))
        #expect(!out.prompt.contains("\n"))
    }

    @Test func portraitWithPaletteAndTagsAndFractionalEnclosure() throws {
        let out = try compile(#"""
            {"width":768,"height":1344,"bbox_coord_space":"abs","bbox_format":"xyxy",
             "summary":{"caption":"Portrait.","palette":["teal"]},
             "items":[{"label":"character","caption":"A hiker.","bbox":[10.9,20.1,100.01,200.99],"tags":["outdoor"]}]}
            """#)
        #expect(out.prompt == #"{"summary":{"caption":"Portrait.","palette":["teal"]},"items":[{"label":"character","caption":"A hiker.","bbox":[-652,-374,-471,-283],"tags":["outdoor"]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#)
    }

    @Test func centredBoxEdgeCasesFromTheReferenceTests() throws {
        #expect(try MossPromptCompiler.centredBox([0, 0, 768, 1344], width: 768, height: 1344) == [-672, -384, 672, 384])
        #expect(try MossPromptCompiler.centredBox([10.9, 20.1, 100.01, 200.99], width: 768, height: 1344) == [-652, -374, -471, -283])
        #expect(try MossPromptCompiler.centredBox([1.00000000001, 2, 100, 200], width: 768, height: 1344) == [-670, -383, -472, -284])
        for bad: [Double] in [[-1, 0, 100, 100], [0, 0, 1537, 100], [0, 0, 100, 1025], [100, 0, 50, 100], [0, 0, 0, 100], [0, 0, 0.00000000001, 100]] {
            #expect(throws: EngineError.self, "\(bad)") { try MossPromptCompiler.centredBox(bad, width: 1536, height: 1024) }
        }
    }

    @Test func rejectsBadDimensionsConventionsAndUnknownFields() {
        let cases = [
            landscape.replacingOccurrences(of: #""width":1536"#, with: #""width":1023"#),
            landscape.replacingOccurrences(of: #""width":1536"#, with: #""width":1536.0"#),
            landscape.replacingOccurrences(of: #""width":1536"#, with: #""width":true"#),
            landscape.replacingOccurrences(of: #""abs","bbox_format""#, with: #""abs_centralized","bbox_format""#),
            landscape.replacingOccurrences(of: #""xyxy""#, with: #""yxyx""#),
            landscape.replacingOccurrences(of: #""label":"object","#, with: #""label":"object","font_size":32,"#),
            landscape.replacingOccurrences(of: #""text":"把日常，过成诗。\n\"MOSS\" 🪴","#, with: ""),  // text item without text
            landscape.replacingOccurrences(of: "[928,272,1408,448]", with: "[928,272,1408]"),
            landscape.replacingOccurrences(of: "[928,272,1408,448]", with: #"[928,272,"1408",448]"#),
            landscape.replacingOccurrences(of: "[928,272,1408,448]", with: "[true,272,1408,448]"),
            #"{"caption":""}"#, #"{"caption":"x","items":[]}"#, "[]", "not json",
        ]
        for json in cases {
            #expect(throws: EngineError.self, "\(json.prefix(60))") { try compile(json) }
        }
    }

    @Test func captionDraftIsPlainTextVerbatim() throws {
        let out = try compile(#"{"caption":"An ivory cup marked \"茶\".\nPreserve this second line."}"#)
        #expect(out.prompt == "An ivory cup marked \"茶\".\nPreserve this second line." && out.size == nil)
    }

    @Test func layoutPromptJSONUsesTheSameCanonicalWriter() throws {
        // The canvas → prompt path (LayoutDraft / LayoutPrompt) emits the same key order and escaping.
        let p = LayoutPrompt(
            summary: .init(meta: ["poster"], canvas: "single", caption: "A ceramic campaign."),
            items: [.init(label: .text, caption: "Dark-green lettering.", text: "把日常，过成诗。\n\"MOSS\" 🪴", bbox: [-240, 160, -64, 640])])
        #expect(try p.jsonString() == #"{"summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},"items":[{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\n\"MOSS\" 🪴","bbox":[-240,160,-64,640]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#)
    }

    @Test func extractsTheDraftFromAModelReply() throws {
        let reply = "Sure! Here is the draft:\n```json\n{\"caption\": \"A fox {in} snow\"}\n```\nDone."
        #expect(MossPromptCompiler.extractJSON(from: reply) == #"{"caption": "A fox {in} snow"}"#)
        #expect(MossPromptCompiler.extractJSON(from: "no json here") == nil)
        #expect(MossPromptCompiler.extractJSON(from: #"{"a": "unbalanced"#) == nil)
    }
}

@Suite("MossEnhancerPrompt — Qwen3.5 non-thinking chat for the on-device enhancer")
struct MossEnhancerPromptTests {
    @Test func rendersTheNonThinkingChatTemplate() {
        let chat = MossEnhancerPrompt.chat(prompt: "a summer sale poster", layout: false, canvas: Dimensions(width: 1024, height: 1024))
        #expect(chat.hasPrefix("<|im_start|>system\n"))
        #expect(chat.contains("<|im_end|>\n<|im_start|>user\na summer sale poster<|im_end|>\n"))
        #expect(chat.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        #expect(chat.contains(#"{"caption": "...""#) && !chat.contains("bbox"))
    }

    @Test func layoutModeStatesTheCanvasAndPixelConvention() {
        let chat = MossEnhancerPrompt.chat(prompt: "x", layout: true, canvas: Dimensions(width: 1536, height: 1024))
        #expect(chat.contains(#""width": 1536, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy""#))
        #expect(chat.contains("x in 0..1536, y in 0..1024"))
    }

    @Test func retryCarriesThePreviousAnswerAndError() {
        let chat = MossEnhancerPrompt.chat(
            prompt: "x", layout: true, canvas: Dimensions(width: 512, height: 512), retry: (answer: "{bad}", error: "items must be a nonempty flat array"))
        #expect(chat.contains("<|im_start|>assistant\n<think>\n\n</think>\n\n{bad}<|im_end|>\n"))
        #expect(chat.contains("That draft is invalid: items must be a nonempty flat array."))
        #expect(chat.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }
}

/// The enhancer LLM often fills optional fields with blanks (`"text": ""` on an object, `"tags": [""]`).
/// `tidyDraft` drops them before the strict compile, so one blank no longer fails "Layout from prompt".
@Suite("MossPromptCompiler.tidyDraft — blank optional fields from the enhancer")
struct MossPromptCompilerTidyTests {
    func compileTidied(_ json: String) throws -> MossPromptCompiler.Output {
        try MossPromptCompiler.compile(MossPromptCompiler.tidyDraft(Data(json.utf8)))
    }

    @Test func blankTextOnAnObjectIsDropped() throws {
        let out = try compileTidied(#"""
            {"width":1024,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"A dog."},
             "items":[{"label":"object","caption":"An ice cream cone.","text":" ","bbox":[0,0,512,512]}]}
            """#)
        #expect(!out.prompt.contains(#""text""#))
    }

    @Test func textItemWithBlankTextBecomesAnObject() throws {
        let out = try compileTidied(#"""
            {"width":1024,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"A dog."},
             "items":[{"label":"text","caption":"A silver tag.","text":"","bbox":[0,0,512,512]}]}
            """#)
        #expect(out.prompt.contains(#"{"label":"object","caption":"A silver tag.","bbox""#))
    }

    @Test func blankTagsAndPaletteEntriesAreDropped() throws {
        let out = try compileTidied(#"""
            {"width":1024,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy",
             "summary":{"caption":"A dog.","meta":[""],"palette":["red",""]},
             "items":[{"label":"object","caption":"A cone.","tags":[""],"palette":[" "],"bbox":[0,0,512,512]}]}
            """#)
        #expect(out.prompt.hasPrefix(#"{"summary":{"caption":"A dog.","palette":["red"]},"items":[{"label":"object","caption":"A cone.","bbox":[-512,-512,0,0]}]"#))
    }

    @Test func offListLabelsMapToTheNearestAllowedOne() throws {
        let out = try compileTidied(#"""
            {"width":1024,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"A dog."},
             "items":[{"label":"Animal","caption":"A black dog.","bbox":[0,0,512,512]},
                      {"label":"background","caption":"A garden.","bbox":[0,512,1024,1024]},
                      {"label":"sign","caption":"A shop sign.","text":"OPEN","bbox":[512,0,1024,256]},
                      {"label":"logo","caption":"A blank badge.","bbox":[512,256,768,512]},
                      {"label":"Character","caption":"A cat.","bbox":[768,256,1024,512]}]}
            """#)
        let labels = out.prompt.components(separatedBy: #""label":""#).dropFirst().map { $0.prefix { $0 != "\"" } }
        #expect(labels == ["character", "object", "text", "object", "character"])
    }

    @Test func validDraftsAndGarbagePassThroughUnchanged() throws {
        let caption = Data(#"{"caption":"A dog."}"#.utf8)
        #expect(try MossPromptCompiler.compile(MossPromptCompiler.tidyDraft(caption)).prompt == "A dog.")
        #expect(MossPromptCompiler.tidyDraft(Data("not json".utf8)) == Data("not json".utf8))
        #expect(throws: EngineError.self) { try compileTidied(#"{"width":1024}"#) }
    }
}
