import Foundation
import MossForgeCore

/// The rules of a MOSS DiT export (convert / quantize / fuse / merge), matched to the Comfy-Org
/// int8·convrot release files so MossForge, ComfyUI and Forge all read the output: only the 168 block
/// linears (`layers.N.proj.{qkv,o,ffn_up_gate,ffn_down}`) are quantized, each with `<weight>_scale`
/// and a per-tensor `<layer>.comfy_quant` JSON blob; norms, embedders, modulation, text adapter and
/// head stay bf16 — and Turbo's fp32 step embedder + modulation biases stay fp32.
public enum MossExport {
    /// Precisions a DiT can be written in, best quality first.
    public static let precisions: [WeightPrecision] = [.float32, .bfloat16, .float16, .int8ConvRot, .float8, .float8E5M2, .int4ConvRot, .nvfp4]

    /// ConvRot rotation group (comfy-kitchen default; the Comfy-Org MOSS files use 256).
    public static let convRotGroupSize = 256

    public static func isQuantized(_ p: WeightPrecision) -> Bool {
        ![.float32, .bfloat16, .float16].contains(p)
    }

    public static func displayName(_ p: WeightPrecision) -> String {
        switch p {
        case .float32: "fp32 (lossless, 20 GB)"
        case .bfloat16: "bf16 (10 GB)"
        case .float16: "fp16 (10 GB, closest to fp32 at half size)"
        case .int8ConvRot: "int8·convrot (5.6 GB, near-lossless)"
        case .float8: "fp8 E4M3 (5.6 GB)"
        case .float8E5M2: "fp8 E5M2 (5.6 GB)"
        case .int4ConvRot: "int4·convrot (3.2 GB)"
        case .nvfp4: "nvfp4 (3.2 GB)"
        default: p.rawValue
        }
    }

    /// Whether `key` (canonical, fused-qkv name) is written quantized at `precision`.
    public static func quantizes(_ key: String, shape: [Int], as precision: WeightPrecision) -> Bool {
        guard isQuantized(precision), shape.count == 2, isBlockLinear(key) else { return false }
        switch precision {
        case .int8ConvRot, .int4ConvRot: return shape[1] % convRotGroupSize == 0
        case .nvfp4: return shape[0] % 128 == 0 && shape[1] % 64 == 0
        case .float8, .float8E5M2: return true
        default: return false
        }
    }

    static func isBlockLinear(_ key: String) -> Bool {
        let parts = key.split(separator: ".")
        return parts.count == 5 && parts[0] == "layers" && Int(parts[1]) != nil && parts[2] == "proj"
            && ["qkv", "q", "k", "v", "o", "ffn_up_gate", "ffn_down"].contains(parts[3]) && parts[4] == "weight"
    }

    /// The `<layer>.comfy_quant` U8 blob for a quantized weight (`nil` for float precisions).
    public static func comfyQuantBlob(for precision: WeightPrecision) -> Data? {
        let json: String? = switch precision {
        case .int8ConvRot: #"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": \#(convRotGroupSize)}"#
        case .int4ConvRot: #"{"format": "convrot_w4a4", "convrot": true, "convrot_groupsize": \#(convRotGroupSize)}"#
        case .float8: #"{"format": "float8_e4m3fn"}"#
        case .float8E5M2: #"{"format": "float8_e5m2"}"#
        case .nvfp4: #"{"format": "nvfp4"}"#
        default: nil
        }
        return json.map { Data($0.utf8) }
    }

    /// lodestones' Turbo overrides: the step embedder and the biases the t=1 time vector is folded into.
    public static func keepsFP32(_ key: String) -> Bool {
        key.hasPrefix("time_embedding.") || key == "shared_block_modulation.bias" || key == "video_head_modulation.bias"
    }

    /// Fusing a LoRA — above all a distillation LoRA, whose delta is below int8 / fp8 resolution —
    /// must start from bf16 / fp32 weights; quantize afterwards.
    public static func checkFuseBase(_ sourcePrecision: WeightPrecision) throws {
        guard !isQuantized(sourcePrecision) else {
            throw EngineError.invalidRequest(
                "can't fuse into a \(sourcePrecision.rawValue) DiT — the LoRA's delta is lost in quantized weights. "
                    + "Fuse into the bf16 / fp32 model, then pick the quantized output precision.")
        }
    }

    /// `<source stem minus its precision tag>[_suffix]_<precision tag>`.
    public static func outputStem(_ source: String, precision: WeightPrecision, suffix: String? = nil) -> String {
        var base = source
        for tag in ["_bf16", "_fp16", "_fp32", "_int8_convrot", "_int4_convrot", "_fp8", "_fp8_e5m2", "_nvfp4"] where base.hasSuffix(tag) {
            base = String(base.dropLast(tag.count))
            break
        }
        if let suffix, !suffix.isEmpty { base += "_" + suffix }
        return base + "_" + tag(precision)
    }

    public static func tag(_ p: WeightPrecision) -> String {
        switch p {
        case .int8ConvRot: "int8_convrot"
        case .int4ConvRot: "int4_convrot"
        case .float8E5M2: "fp8_e5m2"
        default: p.rawValue
        }
    }
}
