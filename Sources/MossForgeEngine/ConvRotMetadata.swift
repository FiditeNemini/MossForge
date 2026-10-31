import Foundation

/// Decodes the `_quantization_metadata` blob that comfy-kitchen ConvRot checkpoints carry in
/// the safetensors `__metadata__`. The blob is a JSON string:
///
/// ```json
/// {"format_version":"1.0",
///  "layers":{"blocks.0.attn.wq":{"format":"int8_tensorwise","per_channel":true,
///                                "convrot":true,"convrot_groupsize":256}, ...}}
/// ```
///
/// Per layer we need the ConvRot group size (so the loader can un-rotate the folded group-wise
/// Hadamard back to the original weight basis) and whether it is **INT4** (`format` ==
/// `convrot_w4a4`), which packs two signed nibbles per byte and so must be unpacked before the
/// shared un-rotation. Without the INT4 flag a `convrot_w4a4` weight — stored as `I8` bytes just
/// like INT8 ConvRot — would be read as full INT8 and silently corrupt (see `WeightLoadPlan`).
public enum ConvRotMetadata {
    /// One ConvRot layer's decode parameters.
    public struct ConvRotLayer: Sendable, Equatable {
        /// Hadamard rotation group size (a power of four; comfy-kitchen default 256).
        public var groupSize: Int
        /// `true` for `convrot_w4a4` (packed signed INT4); `false` for INT8 ConvRot.
        public var isInt4: Bool

        public init(groupSize: Int, isInt4: Bool) {
            self.groupSize = groupSize
            self.isInt4 = isInt4
        }
    }

    private struct Blob: Decodable {
        struct Layer: Decodable {
            var convrot: Bool?
            var convrot_groupsize: Int?
            var format: String?
        }
        var layers: [String: Layer]
    }

    /// The `convrot_w4a4` format tag marking a packed-INT4 ConvRot layer.
    static let int4Format = "convrot_w4a4"

    /// Full per-layer ConvRot decode map: native layer key → `(groupSize, isInt4)`. A layer is
    /// included when it is rotated — either `convrot == true` (INT8) or tagged `convrot_w4a4`
    /// (INT4, which need not repeat the `convrot` flag). Absent / malformed metadata yields an
    /// empty map (⇒ nothing is treated as ConvRot).
    public static func layers(from metadata: [String: String]) -> [String: ConvRotLayer] {
        guard let raw = metadata["_quantization_metadata"],
            let data = raw.data(using: .utf8),
            let blob = try? JSONDecoder().decode(Blob.self, from: data)
        else { return [:] }

        var result: [String: ConvRotLayer] = [:]
        for (key, layer) in blob.layers {
            let isInt4 = layer.format == int4Format
            guard layer.convrot == true || isInt4 else { continue }
            result[key] = ConvRotLayer(groupSize: layer.convrot_groupsize ?? 256, isInt4: isInt4)
        }
        return result
    }

    /// Map of native layer key → ConvRot group size, for every rotated layer (INT8 and INT4).
    /// A thin projection of `layers(from:)` kept for callers that only need the group size.
    public static func groupSizes(from metadata: [String: String]) -> [String: Int] {
        layers(from: metadata).mapValues(\.groupSize)
    }
}
