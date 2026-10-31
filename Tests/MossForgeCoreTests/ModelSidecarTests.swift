import Foundation
import Testing

@testable import MossForgeCore

/// Per-file properties the safetensors header can't tell us — which MOSS family a DiT is or a LoRA
/// works with, and whether a LoRA is a few-step distillation (and how to sample it) — kept in a
/// `<file>.mfmeta.json` sidecar, with filename-based defaults until the user edits them.
@Suite("ModelSidecar — family + distillation properties")
struct ModelSidecarTests {
    @Test func defaultsFromTheFilename() {
        let base = ModelSidecar.inferred(fileName: "lodestones-8step-base.safetensors", isLoRA: true, siblings: [])
        #expect(base.families == [.base])
        #expect(base.distill == DistillInfo(steps: 8, route: .flow))

        let sft = ModelSidecar.inferred(fileName: "akhaliq-4step-sft.safetensors", isLoRA: true, siblings: [])
        #expect(sft.families == [.sft] && sft.distill?.steps == 4)

        let style = ModelSidecar.inferred(fileName: "watercolour.safetensors", isLoRA: true, siblings: [])
        #expect(style.families == [.sft, .base] && style.distill == nil)

        let dit = ModelSidecar.inferred(fileName: "moss_image_base_bf16.safetensors", isLoRA: false, siblings: [])
        #expect(dit.families == [.base] && dit.distill == nil)
        #expect(ModelSidecar.inferred(fileName: "my_finetune.safetensors", isLoRA: false, siblings: []).families == [.sft])
    }

    @Test func siblingAdaptersMeanTheDataSpaceRoute() {
        let s = ModelSidecar.inferred(
            fileName: "lodestones-8step-sft.safetensors", isLoRA: true,
            siblings: ["lodestones-8step-sft-dataspace.safetensors", "lodestones-8step-sft-time_embedding.safetensors"])
        #expect(s.distill == DistillInfo(
            steps: 8, route: .dataSpace, dataSpace: "lodestones-8step-sft-dataspace.safetensors",
            timeEmbedding: "lodestones-8step-sft-time_embedding.safetensors"))
    }

    @Test func roundTripsBesideTheModel() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sidecar-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let lora = dir.appendingPathComponent("x.safetensors")
        #expect(ModelSidecar.url(for: lora).lastPathComponent == "x.safetensors.mfmeta.json")
        #expect(ModelSidecar.read(for: lora) == nil)
        var s = ModelSidecar(families: [.base, .sft], distill: DistillInfo(steps: 2, route: .flow), source: "https://hf.co/akhaliq")
        s.notes = "draft"
        try s.write(for: lora)
        #expect(ModelSidecar.read(for: lora) == s)
        #expect(ModelSidecar.read(for: lora)?.families == [.sft, .base])  // stored in a stable order
    }

    @Test func aLoRAsStrengthRangeIsSavedAndDefaultsToPlusMinusTwo() throws {
        // Generate's strength slider spans the LoRA's range; ±2 unless the Models screen sets one.
        var s = ModelSidecar(families: [.sft])
        #expect(s.strengthRange == nil && s.sliderRange == SliderRange(min: -2, max: 2))
        s.strengthRange = SliderRange(min: -10, max: 10)
        #expect(s.sliderRange == SliderRange(min: -10, max: 10))
        // An invalid range (empty, or not spanning 0) falls back to ±2 rather than breaking the slider.
        s.strengthRange = SliderRange(min: 1, max: 3)
        #expect(s.sliderRange == SliderRange(min: -2, max: 2))

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sidecar-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let lora = dir.appendingPathComponent("age.safetensors")
        let saved = ModelSidecar(families: [.sft], strength: -3, strengthRange: SliderRange(min: -10, max: 10))
        try saved.write(for: lora)
        #expect(ModelSidecar.read(for: lora) == saved)
    }

    @Test func aTrainedSlidersMetadataRangeFillsAMissingOne() throws {
        // MossForge's slider trainer writes `mossforge_slider_range` ("min,max") into the LoRA header.
        #expect(SliderRange(parsing: "-1.0,2.0") == SliderRange(min: -1, max: 2))
        #expect(SliderRange(parsing: " -10 , 10 ") == SliderRange(min: -10, max: 10))
        #expect(SliderRange(parsing: "2,1") == nil && SliderRange(parsing: "x") == nil)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sidecar-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let lora = dir.appendingPathComponent("age-slider.safetensors")
        let metadata = ["mossforge_slider_range": "-1.0,2.0"]
        #expect(ModelSidecar.load(for: lora, isLoRA: true, metadata: metadata).strengthRange == SliderRange(min: -1, max: 2))
        // A range saved from the Models screen wins over the header.
        try ModelSidecar(families: [.sft], strengthRange: SliderRange(min: -4, max: 4)).write(for: lora)
        #expect(ModelSidecar.load(for: lora, isLoRA: true, metadata: metadata).strengthRange == SliderRange(min: -4, max: 4))
        // A saved sidecar without a range still picks up the header's.
        try ModelSidecar(families: [.sft], strength: 0.5).write(for: lora)
        #expect(ModelSidecar.load(for: lora, isLoRA: true, metadata: metadata).strengthRange == SliderRange(min: -1, max: 2))
        #expect(ModelSidecar.load(for: lora, isLoRA: false, metadata: metadata).strengthRange == nil)
    }

    @Test func distillSpecSetsTheSamplingAndResolvesAdapters() throws {
        let lora = URL(fileURLWithPath: "/m/lora/lodestones/lodestones-8step-sft.safetensors")
        let info = DistillInfo(steps: 8, route: .dataSpace, dataSpace: "a-dataspace.safetensors", timeEmbedding: "a-te.safetensors")
        let spec = try #require(DistillSpec(lora: lora, info: info))
        #expect(spec.dataSpace?.path == "/m/lora/lodestones/a-dataspace.safetensors")
        var r = GenerationRequest(prompt: "p", negativePrompt: "bad", variant: .sft)
        r.apply(spec)
        #expect(r.steps == 8 && r.cfg == 1 && r.sampler == .dataSpaceDMD && r.negativePrompt.isEmpty && r.distill == spec)

        let flow = try #require(DistillSpec(lora: lora, info: DistillInfo(steps: 4, route: .flow)))
        var f = GenerationRequest(prompt: "p", variant: .base)
        f.apply(flow)
        #expect(f.steps == 4 && f.cfg == 1 && f.shift == 5 && f.sampler == .flowEuler)
        // The data-space route is unusable without its adapter files.
        #expect(DistillSpec(lora: lora, info: DistillInfo(steps: 8, route: .dataSpace)) == nil)
    }

    @Test func anAlreadyDistilledModelRunsAtEightStepsWithoutALoRA() throws {
        // Generate → Turbo → "None — model is already 8-step": a few-step DiT with no sidecar recipe.
        let info = DistillInfo.alreadyDistilled
        #expect(info.steps == 8 && info.route == .flow && info.cfg == 1)
        let spec = try #require(DistillSpec(builtInto: URL(fileURLWithPath: "/m/diffusion_models/fast.safetensors"), info: info))
        #expect(spec.lora == nil && spec.dataSpace == nil && spec.timeEmbedding == nil)
        var r = GenerationRequest(prompt: "p", negativePrompt: "bad", variant: .sft)
        r.apply(spec)
        #expect(r.steps == 8 && r.cfg == 1 && r.sampler == .flowEuler && r.negativePrompt.isEmpty)
    }
}
