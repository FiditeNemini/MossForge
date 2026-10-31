import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// The Models screen's library: DiTs under `diffusion_models/`, LoRAs under `lora/` (any depth), their
/// sidecars, and the file operations — import (incl. extracting the DiT from an all-in-one checkpoint
/// or merging a diffusers `transformer/` folder), move, rename, trash.
@Suite("ModelLibrary — scan, import, extract, move")
struct ModelLibraryTests {
    private func tmp() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("lib-\(UUID())")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    private func f32(_ name: String, _ values: [Float]) -> SafetensorsWriter.Tensor {
        .init(name: name, dtype: .float32, shape: [values.count], data: values.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    private func lora(_ url: URL) throws {
        try SafetensorsWriter.write(
            [f32("diffusion_model.layers.0.proj.q.lora_down.weight", [1, 2]), f32("diffusion_model.layers.0.proj.q.lora_up.weight", [3, 4])],
            to: url)
    }

    @Test func repackStreamsRenamedTensorsFromSeveralFiles() throws {
        let d = try tmp()
        defer { try? FileManager.default.removeItem(at: d) }
        let a = d.appendingPathComponent("a.safetensors"), b = d.appendingPathComponent("b.safetensors")
        try SafetensorsWriter.write([f32("model.diffusion_model.x", [1, 2, 3]), f32("vae.y", [9])], to: a)
        try SafetensorsWriter.write([f32("z", [4, 5])], to: b)
        let out = d.appendingPathComponent("out.safetensors")
        try SafetensorsRepack.write([.init(source: a, name: "model.diffusion_model.x", as: "x"), .init(source: b, name: "z", as: "z")], to: out)
        let (tensors, _) = try SafetensorsHeader.readDescriptors(from: out)
        #expect(tensors.map(\.name) == ["x", "z"])
        let x = try SafetensorsHeader.readTensor(name: "x", from: out).bytes
        #expect(x == f32("x", [1, 2, 3]).data)
        #expect(try SafetensorsHeader.readTensor(name: "z", from: out).bytes == f32("z", [4, 5]).data)
    }

    @Test func findsTheDiTInsideAnAllInOneCheckpoint() {
        let names = [
            "model.diffusion_model.layers.0.proj.qkv.weight", "model.diffusion_model.time_embedding.0.weight",
            "vae.decoder.conv_in.weight", "text_encoders.qwen35.transformer.model.layers.0.mlp.up_proj.weight",
        ]
        let plan = CheckpointExtract.plan(tensorNames: names)
        #expect(plan?.prefix == "model.diffusion_model.")
        #expect(plan?.dit == ["model.diffusion_model.layers.0.proj.qkv.weight", "model.diffusion_model.time_embedding.0.weight"])
        #expect(plan?.hasVAE == true && plan?.hasTextEncoder == true)
        // A bare DiT (no prefix, nothing else baked in) needs no extraction.
        #expect(CheckpointExtract.plan(tensorNames: ["layers.0.proj.qkv.weight", "time_embedding.0.weight"]) == nil)
    }

    @Test func scanGivesATrainedSliderItsHeaderStrengthRange() throws {
        let m = try tmp()
        defer { try? FileManager.default.removeItem(at: m) }
        let dir = m.appendingPathComponent("lora")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try SafetensorsWriter.write(
            [f32("diffusion_model.layers.0.proj.q.lora_down.weight", [1, 2]), f32("diffusion_model.layers.0.proj.q.lora_up.weight", [3, 4])],
            metadata: [ModelSidecar.sliderRangeMetadataKey: "-1.0,2.0"], to: dir.appendingPathComponent("age-slider.safetensors"))
        let lib = ModelLibrary.scan(models: m)
        #expect(lib.loras.first?.sidecar.strengthRange == SliderRange(min: -1, max: 2))
        #expect(lib.loras.first?.hasSavedSidecar == false)
    }

    @Test func scanSplitsDiTsLoRAsAndSupportingFiles() throws {
        let m = try tmp()
        defer { try? FileManager.default.removeItem(at: m) }
        let dits = m.appendingPathComponent("diffusion_models"), lod = m.appendingPathComponent("lora/lodestones")
        try FileManager.default.createDirectory(at: dits, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: lod, withIntermediateDirectories: true)
        try SafetensorsWriter.write([f32("layers.0.x", [1])], to: dits.appendingPathComponent("moss_image_base_bf16.safetensors"))
        try lora(lod.appendingPathComponent("lodestones-8step-sft.safetensors"))
        try SafetensorsWriter.write([f32("latent_mix.Q", [1]), f32("step_codes", [1])], to: lod.appendingPathComponent("lodestones-8step-sft-dataspace.safetensors"))
        try SafetensorsWriter.write([f32("time_embedding.0.weight", [1])], to: lod.appendingPathComponent("lodestones-8step-sft-time_embedding.safetensors"))

        let lib = ModelLibrary.scan(models: m)
        #expect(lib.dits.map(\.relativePath) == ["diffusion_models/moss_image_base_bf16.safetensors"])
        #expect(lib.dits[0].sidecar.families == [.base])
        #expect(lib.loras.map(\.relativePath) == ["lora/lodestones/lodestones-8step-sft.safetensors"])
        let turbo = lib.loras[0]
        #expect(turbo.sidecar.distill?.route == .dataSpace && turbo.folder == "lodestones")
        #expect(lib.supportingFiles(of: turbo).map(\.lastPathComponent).sorted() == [
            "lodestones-8step-sft-dataspace.safetensors", "lodestones-8step-sft-time_embedding.safetensors",
        ])
        #expect(lib.distills(for: .sft).count == 1 && lib.distills(for: .base).isEmpty)
        #expect(lib.plainLoRAs(for: .sft).isEmpty)
    }

    @Test func aFusedTurbosDataspaceFileIsNotListedAsADiT() throws {
        // Fuse copies the data-space adapter beside the fused DiT; it's a supporting file, not a model.
        let m = try tmp()
        defer { try? FileManager.default.removeItem(at: m) }
        let dits = m.appendingPathComponent("diffusion_models")
        try FileManager.default.createDirectory(at: dits, withIntermediateDirectories: true)
        let fused = dits.appendingPathComponent("moss_image_sft_lodestones-8step-sft_int8_convrot.safetensors")
        try SafetensorsWriter.write([f32("layers.0.x", [1])], to: fused)
        try SafetensorsWriter.write(
            [f32("latent_mix.Q", [1]), f32("step_codes", [1])],
            to: dits.appendingPathComponent("moss_image_sft_lodestones-8step-sft_int8_convrot-dataspace.safetensors"))
        try ModelSidecar(
            families: [.sft],
            distill: DistillInfo(steps: 8, route: .dataSpace, dataSpace: "moss_image_sft_lodestones-8step-sft_int8_convrot-dataspace.safetensors")
        ).write(for: fused)

        let lib = ModelLibrary.scan(models: m)
        #expect(lib.dits.map(\.url.lastPathComponent) == [fused.lastPathComponent])
        #expect(lib.supportingFiles(of: lib.dits[0]).count == 1)
    }

    @Test func importMoveRenameAndTrashCarryTheSidecarAndAdapters() throws {
        let m = try tmp(), outside = try tmp()
        defer { try? FileManager.default.removeItem(at: m); try? FileManager.default.removeItem(at: outside) }
        let src = outside.appendingPathComponent("akhaliq-4step-sft.safetensors")
        try lora(src)
        let imported = try ModelLibrary.importFile(src, as: .lora, subfolder: "akhaliq", models: m, move: true)
        #expect(imported.path.hasSuffix("lora/akhaliq/akhaliq-4step-sft.safetensors"))
        #expect(!FileManager.default.fileExists(atPath: src.path))
        // Not a LoRA → refused.
        let notLoRA = outside.appendingPathComponent("vae.safetensors")
        try SafetensorsWriter.write([f32("decoder.post_quant_conv.weight", [1])], to: notLoRA)
        #expect(throws: ModelLibraryError.self) { try ModelLibrary.importFile(notLoRA, as: .lora, subfolder: "", models: m, move: false) }

        try ModelSidecar(families: [.sft], distill: DistillInfo(steps: 4, route: .flow)).write(for: imported)
        let renamed = try ModelLibrary.rename(imported, to: "akhaliq-4step-v2", models: m)
        #expect(renamed.lastPathComponent == "akhaliq-4step-v2.safetensors" && ModelSidecar.read(for: renamed)?.distill?.steps == 4)
        let moved = try ModelLibrary.move(renamed, toSubfolder: "turbo/akhaliq", models: m)
        #expect(moved.path.hasSuffix("lora/turbo/akhaliq/akhaliq-4step-v2.safetensors") && ModelSidecar.read(for: moved) != nil)
        // Outside the models folder → refused.
        #expect(throws: ModelLibraryError.self) { try ModelLibrary.move(moved, toSubfolder: "../../escape", models: m) }
    }
}
