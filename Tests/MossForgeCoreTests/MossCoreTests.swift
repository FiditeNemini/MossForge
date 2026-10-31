import Foundation
import Testing

@testable import MossForgeCore

@Suite("MossSchedule — shifted flow-match sigmas")
struct MossScheduleTests {
    /// Paper App. J: N=30, s=3 — step 1 is labelled σ = 0.9886.
    @Test func matchesPaperFigureNine() {
        let sigmas = MossSchedule.sigmas(steps: 30, shift: 3)
        #expect(sigmas.count == 31)
        #expect(sigmas.first == 1.0)
        #expect(sigmas.last == 0.0)
        #expect(abs(sigmas[1] - 0.9886) < 5e-5)
    }

    @Test func shiftOneIsLinear() {
        let sigmas = MossSchedule.sigmas(steps: 4, shift: 1)
        #expect(sigmas == [1.0, 0.75, 0.5, 0.25, 0.0])
    }

    @Test func modelTimestepIsThousandSigma() {
        let t = MossSchedule.timesteps(steps: 2, shift: 1)
        #expect(t == [1000, 500])  // one per denoising update (σ = 0 never hits the model)
    }

    @Test func sigmasAreMonotoneDecreasing() {
        let s = MossSchedule.sigmas(steps: 50, shift: 5)
        #expect(zip(s, s.dropFirst()).allSatisfy { $0 > $1 })
    }
}

@Suite("MossDimensions — pixel budget × ratio, multiples of 16")
struct MossDimensionsTests {
    @Test func oneMegapixelSquare() {
        #expect(MossDimensions.fit(pixels: 1_048_576, ratio: "1:1") == Dimensions(width: 1024, height: 1024))
    }

    /// Mirrors gradio_presets.fit_size: round(edge/16) * 16 with floor(x + 0.5).
    @Test func sixteenNineAtOneMegapixel() {
        // w = sqrt(1048576 * 16/9) = 1365.33 → 1360 ; h = 768
        #expect(MossDimensions.fit(pixels: 1_048_576, ratio: "16:9") == Dimensions(width: 1360, height: 768))
    }

    @Test func validatesMultiplesOfSixteen() {
        #expect(MossDimensions.isValid(width: 1024, height: 1360))
        #expect(!MossDimensions.isValid(width: 1000, height: 1024))
        #expect(!MossDimensions.isValid(width: 0, height: 1024))
    }

    @Test func imageTokenCount() {
        #expect(MossDimensions.imageTokens(width: 1024, height: 1024) == 4096)
        #expect(MossDimensions.imageTokens(width: 512, height: 512) == 1024)
    }
}

@Suite("Prompt template + variants")
struct PromptAndVariantTests {
    @Test func chatTemplateMatchesOfficialTokenizer() {
        #expect(
            MossPromptTemplate.wrap("a red fox")
                == "<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n<think>\n")
        #expect(MossPromptTemplate.maxTokens == 4096)
    }

    @Test func releaseDefaultsFromUserGuide() {
        let d = MossVariant.base.defaults
        #expect(d.steps == 50)
        #expect(d.cfg == 3.0)
        #expect(d.shift == 5.0)
        #expect(d.sampler == .flowEuler)
    }

    /// Turbo = akhaliq 4-step: stock flow Euler, CFG 1, shift 5 (chosen by an A/B in 0.2.12).
    @Test func turboIsTheAkhaliq4StepRecipe() {
        let d = MossVariant.turbo.defaults
        #expect(d.steps == 4)
        #expect(d.cfg == 1.0)
        #expect(d.shift == 5.0)
        #expect(d.sampler == .flowEuler)
        #expect(MossVariant.turboLoRAPath == "lora/akhaliq/akhaliq-4step-sft.safetensors")
    }

    @Test func requestValidationRejectsOffGridSizes() {
        var r = GenerationRequest(prompt: "x", variant: .sft, width: 1000, height: 1024)
        #expect(throws: EngineError.self) { try r.validate() }
        r.width = 1024
        #expect(throws: Never.self) { try r.validate() }
    }

    @Test func requestDefaultsComeFromVariant() {
        let r = GenerationRequest(prompt: "x", variant: .turbo)
        #expect(r.steps == 4)
        #expect(r.cfg == 1.0)
    }
}

@Suite("LayoutPrompt — abs_centralized yxyx JSON")
struct LayoutPromptTests {
    /// USERGUIDE: 1024² canvas spans −512…+512; subtract H/2, W/2 from top-left coords.
    @Test func convertsTopLeftPixelBoxToCentralizedYXYX() {
        let box = LayoutBox.fromTopLeft(x: 0, y: 0, width: 512, height: 256, canvas: Dimensions(width: 1024, height: 1024))
        #expect(box.yxyx == [-512, -512, -256, 0])
    }

    @Test func encodesTheGuideSchema() throws {
        let layout = LayoutPrompt(
            summary: .init(meta: ["poster"], canvas: "single", caption: "A travel poster."),
            items: [.init(label: .text, caption: "Big title", text: "EXPLORE", bbox: [-416, -384, -288, 384])])
        let json = try layout.jsonString()
        let decoded = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
        #expect(decoded["bbox_coord_space"] as? String == "abs_centralized")
        #expect(decoded["bbox_format"] as? String == "yxyx")
        let items = decoded["items"] as! [[String: Any]]
        #expect(items[0]["text"] as? String == "EXPLORE")
        #expect(items[0]["label"] as? String == "text")
        #expect(items[0]["bbox"] as? [Int] == [-416, -384, -288, 384])
    }

    @Test func keepsLineBreaksAndNonASCIIText() throws {
        let layout = LayoutPrompt(
            summary: .init(caption: "note"),
            items: [.init(label: .text, caption: "handwriting", text: "记得买牛奶！\n鸡蛋一打。", bbox: [0, 0, 10, 10])])
        let json = try layout.jsonString()
        #expect(json.contains("记得买牛奶"))  // not \u-escaped
        #expect(json.contains("\\n"))  // line break preserved as an escaped newline
    }
}
