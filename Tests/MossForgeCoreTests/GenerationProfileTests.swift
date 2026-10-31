import Foundation
import Testing

@testable import MossForgeCore

/// Generate → Profile: named model + Turbo LoRA + steps combinations (built-in and user-saved),
/// resolved against what's in the library.
@Suite("GenerationProfile — one-pick model + Turbo combinations")
struct GenerationProfileTests {
    let library = [
        "diffusion_models/moss_image_sft_bf16.safetensors", "diffusion_models/moss_image_base_bf16.safetensors",
        "lora/akhaliq/akhaliq-2step-sft.safetensors", "lora/akhaliq/akhaliq-4step-sft.safetensors",
        "lora/lodestones/lodestones-8step-sft-latest.safetensors",
    ]

    @Test func builtInsCoverDraftToFullQuality() {
        let ids = GenerationProfile.builtIn.map(\.id)
        #expect(ids == ["draft", "fast", "sft-8step", "full", "base-turbo", "base-full"])
        // lodestones' SFT data-space Turbo is retired (0.2.12); lora_latest is the 8-step choice.
        #expect(GenerationProfile.builtIn.first { $0.id == "sft-8step" }?.turbo == "lora/lodestones/lodestones-8step-sft-latest.safetensors")
        let fast = GenerationProfile.builtIn[1]
        #expect(fast.turbo == MossVariant.turboLoRAPath)
        #expect(fast.model == "diffusion_models/moss_image_sft_bf16.safetensors" && fast.turbo == "lora/akhaliq/akhaliq-4step-sft.safetensors")
        #expect(GenerationProfile.builtIn.first { $0.id == "full" }?.turbo == nil)
    }

    @Test func missingFilesAreReported() {
        let baseTurbo = GenerationProfile.builtIn.first { $0.id == "base-turbo" }!
        #expect(baseTurbo.missing(in: Set(library)) == ["lora/lodestones/lodestones-8step-base.safetensors"])
        #expect(GenerationProfile.builtIn[1].missing(in: Set(library)).isEmpty)
    }

    @Test func theCurrentSelectionIsRecognised() {
        let profiles = GenerationProfile.builtIn
        let fast = GenerationProfile.matching(
            model: "diffusion_models/moss_image_sft_bf16.safetensors", turbo: "lora/akhaliq/akhaliq-4step-sft.safetensors", steps: nil,
            in: profiles)
        #expect(fast?.id == "fast")
        // Overriding the steps makes it a custom combination.
        #expect(GenerationProfile.matching(
            model: "diffusion_models/moss_image_sft_bf16.safetensors", turbo: "lora/akhaliq/akhaliq-4step-sft.safetensors", steps: 6,
            in: profiles) == nil)
    }

    @Test func userProfilesRoundTrip() throws {
        let mine = GenerationProfile(
            id: "user-1", name: "My finetune fast", detail: "", model: "diffusion_models/ft.safetensors",
            turbo: "lora/akhaliq/akhaliq-4step-sft.safetensors", steps: 6)
        let data = try JSONEncoder().encode([mine])
        #expect(try JSONDecoder().decode([GenerationProfile].self, from: data) == [mine])
        #expect(!mine.isBuiltIn && GenerationProfile.builtIn[0].isBuiltIn)
    }
}
