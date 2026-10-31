/// Numeric precision of model weights.
///
/// Covers the precisions handled by the model tooling so the converter and
/// quantizer layers can describe inputs and outputs uniformly. The shipped base
/// weights are `bf16`; the "runs on a typical Mac" target profile is `int8`.
public enum WeightPrecision: String, Sendable, CaseIterable, Codable {
    case float32 = "fp32"
    case float16 = "fp16"
    case bfloat16 = "bf16"
    /// 8-bit float, E4M3 (the ComfyUI "fp8 scaled" checkpoints: `F8_E4M3` weight +
    /// scalar `weight_scale` + `comfy_quant` blob).
    case float8 = "fp8"
    /// 8-bit float, E5M2 (5 exponent, 2 mantissa; IEEE-like, with inf/NaN). The other fp8
    /// export the community tooling offers alongside E4M3 — wider range, coarser mantissa.
    /// Stored exactly like `fp8`: `F8_E5M2` weight + scalar `weight_scale` + `comfy_quant`.
    case float8E5M2 = "fp8-e5m2"
    /// 4-bit NVFP4: U8-packed weights + E4M3 block scales (block 16) + an F32
    /// global `weight_scale_2`.
    case nvfp4
    case int8 = "q8"
    case int4 = "q4"
    /// INT8 ConvRot (Comfy-Org "comfy-kitchen" W8A8): per-output-channel `I8` weight +
    /// F32 `weight_scale` `[out,1]`, with a group-wise **regular-Hadamard** rotation folded
    /// into the weights (`_quantization_metadata` records `convrot`/`convrot_groupsize`). The
    /// rotation spreads activation outliers so INT8 stays near-lossless. On load we dequant
    /// and un-rotate back to the original bf16 basis.
    case int8ConvRot = "int8-convrot"
    /// INT4 ConvRot (Comfy-Org "comfy-kitchen" W4A4, `convrot_w4a4`): two signed 4-bit codes
    /// packed per `I8` byte (so the on-disk inner dim is halved), one F32 `weight_scale` per
    /// output row, and the same group-wise **regular-Hadamard** rotation folded in as INT8
    /// ConvRot. On load we unpack the nibbles, scale per row, and un-rotate to the bf16 basis.
    case int4ConvRot = "int4-convrot"

    /// Whether this precision is a low-bit format the engine treats as quantized
    /// (needs a dequant path on load; a valid quantize/convert target).
    public var isQuantized: Bool {
        switch self {
        case .int8, .int4, .nvfp4, .int8ConvRot, .int4ConvRot: true
        case .float32, .float16, .bfloat16, .float8, .float8E5M2: false
        }
    }

    /// Whether a checkpoint at this precision can actually be **loaded and run** — i.e. the
    /// planner emits a `DequantOp` for it and the loader can materialize it. Independent of
    /// `isQuantized`: a quantized format is runnable as soon as its dequant path exists.
    ///
    /// Float formats are passthrough; `fp8`/`fp8-e5m2` decode via their E4M3/E5M2 LUTs; `nvfp4`
    /// unpacks its nibbles + swizzled block scales; `q8` and `int8-convrot` share the int8 path,
    /// and `int4-convrot` unpacks two signed nibbles per byte before the shared per-row-scale +
    /// Hadamard un-rotation. Only plain `q4` (no ConvRot metadata) has no materialization path.
    public var isRunnable: Bool {
        switch self {
        case .float32, .float16, .bfloat16, .float8, .float8E5M2, .nvfp4, .int8, .int8ConvRot,
            .int4ConvRot:
            true
        case .int4: false
        }
    }
}
