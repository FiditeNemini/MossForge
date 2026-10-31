import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// The rules of a MOSS DiT export, matched to the Comfy-Org int8·convrot files: only the 168 block
/// linears are quantized, each with `weight_scale` + a per-tensor `comfy_quant` blob; everything else
/// stays bf16 — except Turbo's step embedder / modulation biases, which stay fp32.
@Suite("MossExport — what a DiT export quantizes and how it's labelled")
struct MossExportTests {
    @Test func onlyAlignedBlockLinearsAreQuantized() {
        #expect(MossExport.quantizes("layers.7.proj.qkv.weight", shape: [9216, 3072], as: .int8ConvRot))
        #expect(MossExport.quantizes("layers.41.proj.ffn_down.weight", shape: [3072, 8192], as: .nvfp4))
        #expect(!MossExport.quantizes("layers.7.proj.qkv.weight", shape: [9216, 3072], as: .bfloat16))
        #expect(!MossExport.quantizes("layers.7.norm1.weight", shape: [3072], as: .int8ConvRot))
        #expect(!MossExport.quantizes("text_adapter.0.proj.qkv.weight", shape: [9216, 3072], as: .int8ConvRot))
        #expect(!MossExport.quantizes("layers.7.proj.o.weight", shape: [3072, 3000], as: .int8ConvRot))  // not a multiple of 256
    }

    @Test func comfyQuantBlobsRoundTripThroughTheReader() throws {
        for (p, format) in [
            (WeightPrecision.int8ConvRot, "int8_tensorwise"), (.int4ConvRot, "convrot_w4a4"), (.float8, "float8_e4m3fn"),
            (.float8E5M2, "float8_e5m2"), (.nvfp4, "nvfp4"),
        ] {
            let blob = try #require(MossExport.comfyQuantBlob(for: p))
            let parsed = try #require(ComfyQuantBlob(bytes: blob))
            #expect(parsed.format == format && parsed.precision == p)
            if p == .int8ConvRot || p == .int4ConvRot { #expect(parsed.convRot && parsed.groupSize == 256) }
        }
        #expect(MossExport.comfyQuantBlob(for: .bfloat16) == nil)
    }

    @Test func turboOverridesStayFP32() {
        #expect(MossExport.keepsFP32("time_embedding.0.weight") && MossExport.keepsFP32("shared_block_modulation.bias"))
        #expect(MossExport.keepsFP32("video_head_modulation.bias") && !MossExport.keepsFP32("layers.0.proj.qkv.weight"))
    }

    @Test func fusingIntoAQuantizedBaseIsRefused() {
        // A distillation delta is below int8 / fp8 resolution: fuse into bf16 / fp32, then quantize.
        #expect(throws: EngineError.self) { try MossExport.checkFuseBase(.int8ConvRot) }
        #expect(throws: EngineError.self) { try MossExport.checkFuseBase(.nvfp4) }
        #expect(throws: Never.self) { try MossExport.checkFuseBase(.bfloat16) }
        #expect(throws: Never.self) { try MossExport.checkFuseBase(.float32) }
    }

    @Test func outputNamesSayWhatHappened() {
        #expect(MossExport.outputStem("moss_image_sft_bf16", precision: .int8ConvRot) == "moss_image_sft_int8_convrot")
        #expect(MossExport.outputStem("moss_image_sft_bf16", precision: .float16, suffix: "akhaliq-4step") == "moss_image_sft_akhaliq-4step_fp16")
        #expect(MossExport.outputStem("my-merge", precision: .nvfp4) == "my-merge_nvfp4")
    }

    @Test func streamingWriterMatchesTheFormat() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("w-\(UUID()).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let a = [Float](repeating: 1.5, count: 5)
        try SafetensorsWriter.write(
            [
                .init(name: "b", dtype: .float32, shape: [5], data: a.withUnsafeBufferPointer { Data(buffer: $0) }),
                .init(name: "a.comfy_quant", dtype: .uint8, shape: [3], data: Data([1, 2, 3])),
            ], metadata: ["produced_by": "MossForge"], to: url)
        let (tensors, meta) = try SafetensorsHeader.readDescriptors(from: url)
        #expect(tensors.map(\.name) == ["a.comfy_quant", "b"] && meta["produced_by"] == "MossForge")
        #expect(try SafetensorsHeader.readTensor(name: "a.comfy_quant", from: url).bytes == Data([1, 2, 3]))
    }
}
