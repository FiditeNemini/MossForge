import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// Where each MOSS-Image component lives under `models/` — one table shared by the MLX backend's
/// resolver and the app's Models screen.
@Suite("MossModelLayout — models folder layout")
struct MossModelLayoutTests {
    /// A throwaway models folder holding empty files at `paths`.
    private func folder(_ paths: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("layout-\(UUID())")
        for p in paths {
            let url = root.appendingPathComponent(p)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: url.path, contents: Data())
        }
        return root
    }

    /// Since 0.2.12: akhaliq 4-step is the Turbo; lodestones' SFT data-space adapter is gone.
    @Test func turboIsAkhaliq4StepOnly() throws {
        let m = try folder(["lora/akhaliq/akhaliq-4step-sft.safetensors", "lora/lodestones/lodestones-8step-sft.safetensors"])
        defer { try? FileManager.default.removeItem(at: m) }
        #expect(MossModelLayout.find(.turboLoRA, in: m)?.lastPathComponent == "akhaliq-4step-sft.safetensors")
        let old = try folder(["lora/lodestones/lodestones-8step-sft.safetensors", "lora/moss_8step_dmd_turbo.safetensors"])
        defer { try? FileManager.default.removeItem(at: old) }
        #expect(MossModelLayout.find(.turboLoRA, in: old) == nil)
        #expect(MossModelLayout.Component.turboLoRA.candidates == [MossVariant.turboLoRAPath])
    }

    @Test func ditPrefersBF16OverInt8ConvRot() throws {
        let m = try folder(["diffusion_models/moss_image_sft_int8_convrot.safetensors", "diffusion_models/moss_image_sft_bf16.safetensors",
            "diffusion_models/moss_image_base_int8_convrot.safetensors"])
        defer { try? FileManager.default.removeItem(at: m) }
        #expect(MossModelLayout.find(.sftDiT, in: m)?.lastPathComponent == "moss_image_sft_bf16.safetensors")
        #expect(MossModelLayout.find(.baseDiT, in: m)?.lastPathComponent == "moss_image_base_int8_convrot.safetensors")
        #expect(MossModelLayout.dit(for: .turbo) == .sftDiT)  // Turbo distils SFT
    }

    @Test func variantNeedsListsWhatIsMissing() throws {
        let m = try folder([
            "diffusion_models/moss_image_sft_bf16.safetensors", "vae/flux2-vae.safetensors",
            "text_encoders/qwen3.5_9b_int8_convrot.safetensors", "text_encoders/qwen3.5-9b-tokenizer/tokenizer.json",
        ])
        defer { try? FileManager.default.removeItem(at: m) }
        #expect(MossModelLayout.missing(for: .sft, in: m).isEmpty)
        #expect(MossModelLayout.missing(for: .base, in: m) == [.baseDiT])
        #expect(MossModelLayout.missing(for: .turbo, in: m) == [.turboLoRA])
    }

    @Test func settingsOverridesWinOverTheFolderLayout() throws {
        let m = try folder(["vae/flux2-vae.safetensors", "elsewhere/my-vae.safetensors"])
        defer { try? FileManager.default.removeItem(at: m) }
        let custom = m.appendingPathComponent("elsewhere/my-vae.safetensors")
        let o = MossModelLayout.Overrides(vae: custom)
        #expect(MossModelLayout.find(.vae, in: m, overrides: o) == custom)
        #expect(MossModelLayout.find(.vae, in: m, overrides: .init())?.lastPathComponent == "flux2-vae.safetensors")
        // A missing override falls back to the layout rather than failing.
        let gone = MossModelLayout.Overrides(vae: m.appendingPathComponent("nope.safetensors"))
        #expect(MossModelLayout.find(.vae, in: m, overrides: gone)?.lastPathComponent == "flux2-vae.safetensors")
    }

    @Test func labelsFilesTheInspectorDoesNotKnow() {
        #expect(MossModelLayout.label(for: "vosr/VOSR2/checkpoints/ema_model.safetensors") == "VOSR2 upscaler")
        #expect(MossModelLayout.label(for: "esrgan/4x-UltraSharp.safetensors") == "ESRGAN / SPAN upscaler")
        #expect(MossModelLayout.label(for: "misc/thing.safetensors") == nil)
    }

    @Test func turboAdaptersAreNotUserLoRAs() {
        #expect(MossModelLayout.isTurboAdapter("lora/lodestones-dmd/LodeStones-4Step-DMD-SFT.safetensors"))
        #expect(MossModelLayout.isTurboAdapter("lora/moss_8step_dmd_turbo.safetensors"))
        #expect(!MossModelLayout.isTurboAdapter("lora/kijai/moss_image_sft_lora_blocks_21_to_49_rank_512_bf16.safetensors"))
    }
}

@Suite("MossModelKind — one-line summary for the Models / LoRAs screens")
struct MossModelKindSummaryTests {
    @Test func summaries() {
        #expect(MossModelKind.lora(rank: 128).summary == "LoRA · rank 128")
        #expect(MossModelKind.vae(latentChannels: 32).summary == "VAE · 32 ch")
        #expect(MossModelKind.textEncoder(layers: 32, fullAttentionLayers: [3, 7]).summary == "Qwen3.5 encoder · 32 layers")
        #expect(MossModelKind.dmdSampler(steps: 8).summary == "DMD sampler · 8 steps")
        #expect(MossModelKind.unknown.summary == "unrecognised")
    }
}
