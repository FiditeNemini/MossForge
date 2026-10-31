import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// Model tooling through the engine seam: convert / fuse / merge requests, and the sidecar an output
/// DiT gets — its family, and a fused Turbo LoRA as a built-in recipe.
@Suite("ModelTool — requests, seam, output sidecars")
struct ModelToolTests {
    private func tmp() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("mt-\(UUID())")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    @Test func theStubDoesNotDoTooling() async throws {
        let engine = MossEngine()
        #expect(!engine.capabilities.contains(.modelTooling))
        let r = ModelToolRequest(input: URL(fileURLWithPath: "/m/a.safetensors"), operation: .convert, precision: .int8ConvRot,
            output: URL(fileURLWithPath: "/m/b.safetensors"))
        await #expect(throws: EngineError.self) { try await engine.runModelTool(r) }
        #expect(EngineCapabilities.all.names.contains("tools"))
    }

    @Test func convertAndMergeKeepTheFamily() throws {
        let source = ModelSidecar(families: [.base], notes: "my base finetune")
        let out = ModelToolRequest.outputSidecar(source: source, operation: .convert, precision: .int8ConvRot, sourceName: "base_ft")
        #expect(out.sidecar.families == [.base] && out.sidecar.distill == nil && out.copies.isEmpty)
        #expect(out.sidecar.notes?.contains("int8-convrot") == true && out.sidecar.notes?.contains("base_ft") == true)
    }

    @Test func fusingAFlowTurboMakesItBuiltIn() throws {
        let d = try tmp()
        defer { try? FileManager.default.removeItem(at: d) }
        let lora = d.appendingPathComponent("akhaliq-4step-sft.safetensors")
        try ModelSidecar(families: [.sft], distill: DistillInfo(steps: 4, route: .flow)).write(for: lora)
        let op = ModelToolOperation.fuse(loras: [LoRASpec(url: lora)], overrides: nil)
        let out = ModelToolRequest.outputSidecar(source: ModelSidecar(families: [.sft]), operation: op, precision: .float16, sourceName: "sft")
        #expect(out.sidecar.distill == DistillInfo(steps: 4, route: .flow) && out.copies.isEmpty)
    }

    @Test func fusingADataSpaceTurboBakesTheEmbedderAndCopiesTheDataSpace() throws {
        let d = try tmp()
        defer { try? FileManager.default.removeItem(at: d) }
        let lora = d.appendingPathComponent("lodestones-8step-sft.safetensors")
        try ModelSidecar(
            families: [.sft],
            distill: DistillInfo(steps: 8, route: .dataSpace, dataSpace: "lodestones-8step-sft-dataspace.safetensors",
                timeEmbedding: "lodestones-8step-sft-time_embedding.safetensors")
        ).write(for: lora)
        // The request picks the LoRA's time embedding as the fp32 overrides.
        let req = ModelToolRequest.fuse(
            dit: URL(fileURLWithPath: "/m/diffusion_models/sft.safetensors"), loras: [LoRASpec(url: lora)], precision: .int8ConvRot,
            output: URL(fileURLWithPath: "/m/diffusion_models/sft_lodestones_int8_convrot.safetensors"))
        guard case .fuse(_, let overrides) = req.operation else { Issue.record("not a fuse"); return }
        #expect(overrides?.lastPathComponent == "lodestones-8step-sft-time_embedding.safetensors")

        let out = ModelToolRequest.outputSidecar(source: ModelSidecar(families: [.sft]), operation: req.operation, precision: .int8ConvRot,
            sourceName: "sft", output: req.output)
        #expect(out.sidecar.distill == DistillInfo(steps: 8, route: .dataSpace, dataSpace: "sft_lodestones_int8_convrot-dataspace.safetensors"))
        #expect(out.copies.map(\.from.lastPathComponent) == ["lodestones-8step-sft-dataspace.safetensors"])
        #expect(out.copies.map(\.to.path) == ["/m/diffusion_models/sft_lodestones_int8_convrot-dataspace.safetensors"])
    }

    @Test func aBuiltInRecipeRunsWithoutALoRA() throws {
        let model = URL(fileURLWithPath: "/m/diffusion_models/x.safetensors")
        let spec = try #require(DistillSpec(builtInto: model, info: DistillInfo(steps: 8, route: .dataSpace, dataSpace: "x-dataspace.safetensors")))
        #expect(spec.lora == nil && spec.timeEmbedding == nil && spec.dataSpace?.path == "/m/diffusion_models/x-dataspace.safetensors")
        #expect(DistillSpec(builtInto: model, info: DistillInfo(steps: 8, route: .dataSpace)) == nil)  // no data space → unusable
        var r = GenerationRequest(prompt: "p")
        r.apply(try #require(DistillSpec(builtInto: model, info: DistillInfo(steps: 4, route: .flow))))
        #expect(r.steps == 4 && r.cfg == 1 && r.sampler == .flowEuler)
    }
}

@Suite("LoRATraining — through the engine seam")
struct LoRATrainingSeamTests {
    @Test func theStubDoesNotTrain() async {
        let engine = MossEngine()
        #expect(!engine.capabilities.contains(.loRATraining))
        #expect(EngineCapabilities.all.names.contains("train"))
        let r = TrainingRequest(
            baseModel: URL(fileURLWithPath: "/m/base.safetensors"), dataset: URL(fileURLWithPath: "/d"),
            output: URL(fileURLWithPath: "/m/lora/x.safetensors"))
        await #expect(throws: EngineError.self) {
            try await engine.trainLoRA(r, samples: [TrainingSample(image: URL(fileURLWithPath: "/d/a.png"), caption: "a")])
        }
    }
}
