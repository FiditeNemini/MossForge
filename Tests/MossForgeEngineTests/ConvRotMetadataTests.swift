import Foundation
import Testing

@testable import MossForgeEngine

/// The ConvRot metadata parser pulls per-layer group sizes out of the `_quantization_metadata`
/// blob so the loader knows which int8 weights to un-rotate.
@Suite struct ConvRotMetadataTests {
    @Test func parsesConvRotLayersWithGroupSize() {
        let blob = """
            {"format_version":"1.0","layers":{
              "blocks.0.attn.wq":{"format":"int8_tensorwise","per_channel":true,"convrot":true,"convrot_groupsize":256},
              "blocks.0.mlp.down":{"format":"int8_tensorwise","per_channel":true,"convrot":true,"convrot_groupsize":128}
            }}
            """
        let m = ConvRotMetadata.groupSizes(from: ["_quantization_metadata": blob])
        #expect(m == ["blocks.0.attn.wq": 256, "blocks.0.mlp.down": 128])
    }

    @Test func skipsNonConvRotLayersAndDefaultsGroupSize() {
        let blob = """
            {"layers":{
              "a":{"convrot":false,"convrot_groupsize":256},
              "b":{"convrot":true}
            }}
            """
        let m = ConvRotMetadata.groupSizes(from: ["_quantization_metadata": blob])
        #expect(m == ["b": 256])  // false skipped; missing groupsize defaults to 256
    }

    @Test func absentOrMalformedMetadataYieldsEmpty() {
        #expect(ConvRotMetadata.groupSizes(from: [:]).isEmpty)
        #expect(ConvRotMetadata.groupSizes(from: ["_quantization_metadata": "not json"]).isEmpty)
        #expect(ConvRotMetadata.layers(from: [:]).isEmpty)
    }

    @Test func layersFlagsInt4ByConvRotW4A4Format() {
        // INT8 ConvRot and INT4 ConvRot both store I8 bytes; only the `format` tag tells them
        // apart. A `convrot_w4a4` layer is flagged isInt4 even without an explicit convrot bool.
        let blob = """
            {"format_version":"1.0","layers":{
              "blocks.0.attn.wq":{"format":"int8_tensorwise","per_channel":true,"convrot":true,"convrot_groupsize":256},
              "blocks.0.mlp.down":{"format":"convrot_w4a4","convrot_groupsize":128}
            }}
            """
        let m = ConvRotMetadata.layers(from: ["_quantization_metadata": blob])
        #expect(m["blocks.0.attn.wq"] == .init(groupSize: 256, isInt4: false))
        #expect(m["blocks.0.mlp.down"] == .init(groupSize: 128, isInt4: true))
        // groupSizes projects both, regardless of bit width.
        #expect(ConvRotMetadata.groupSizes(from: ["_quantization_metadata": blob])
            == ["blocks.0.attn.wq": 256, "blocks.0.mlp.down": 128])
    }
}
