import Foundation
import MossForgeCore

/// The per-tensor `<layer>.comfy_quant` U8 blob that ComfyUI-native quantized checkpoints carry
/// next to each quantized weight, e.g.
/// `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`.
///
/// The Comfy-Org MOSS-Image int8·convrot files have **no** `__metadata__._quantization_metadata`
/// (which `ConvRotMetadata` reads) — the per-layer decode parameters live only here.
public struct ComfyQuantBlob: Sendable, Equatable {
    public var format: String
    public var convRot: Bool
    public var groupSize: Int

    private struct Raw: Decodable {
        var format: String?
        var convrot: Bool?
        var convrot_groupsize: Int?
    }

    public init?(bytes: Data) {
        guard let raw = try? JSONDecoder().decode(Raw.self, from: bytes), let format = raw.format else { return nil }
        self.format = format
        self.convRot = raw.convrot ?? (format == ConvRotMetadata.int4Format)
        self.groupSize = raw.convrot_groupsize ?? 256
    }

    /// The ConvRot decode map for a file's blobs (`<layer>.comfy_quant` name → bytes): layer key
    /// (`.comfy_quant` dropped, prefix kept) → group size + INT4 flag, for every rotated layer.
    /// Unrotated, non-ConvRot and unparseable blobs are left out (⇒ no un-rotation).
    public static func convRotLayers(blobs: [String: Data]) -> [String: ConvRotMetadata.ConvRotLayer] {
        var layers: [String: ConvRotMetadata.ConvRotLayer] = [:]
        for (name, bytes) in blobs where name.hasSuffix(".comfy_quant") {
            guard let blob = ComfyQuantBlob(bytes: bytes), blob.convRot else { continue }
            let isInt4 = blob.format == ConvRotMetadata.int4Format
            guard isInt4 || blob.format == "int8_tensorwise" else { continue }
            layers[String(name.dropLast(".comfy_quant".count))] = .init(groupSize: blob.groupSize, isInt4: isInt4)
        }
        return layers
    }

    /// The weight precision this blob describes, if MossForge knows it.
    public var precision: WeightPrecision? {
        switch format {
        case ConvRotMetadata.int4Format: .int4ConvRot
        case "int8_tensorwise": convRot ? .int8ConvRot : .int8
        case "float8_e4m3fn": .float8
        case "float8_e5m2": .float8E5M2
        case "nvfp4": .nvfp4
        default: nil
        }
    }
}
