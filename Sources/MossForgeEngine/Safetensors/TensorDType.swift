import MossForgeCore

/// A tensor element type, named exactly as in the safetensors header (a public
/// Hugging Face specification). Includes the fp8 / packed-int types quantized
/// diffusion checkpoints use.
public enum TensorDType: String, Sendable, CaseIterable {
    case float64 = "F64"
    case float32 = "F32"
    case float16 = "F16"
    case bfloat16 = "BF16"
    /// 8-bit float, E4M3 (fp8-scaled checkpoints; also NVFP4 block scales).
    case float8E4M3 = "F8_E4M3"
    /// 8-bit float, E5M2.
    case float8E5M2 = "F8_E5M2"
    case int64 = "I64"
    case int32 = "I32"
    case int16 = "I16"
    case int8 = "I8"
    /// Unsigned 8-bit — packed NVFP4 weights + ComfyUI `comfy_quant` blobs.
    case uint8 = "U8"
    case bool = "BOOL"

    /// Size of a single element in bytes.
    public var byteWidth: Int {
        switch self {
        case .float64, .int64: 8
        case .float32, .int32: 4
        case .float16, .bfloat16, .int16: 2
        case .float8E4M3, .float8E5M2, .int8, .uint8, .bool: 1
        }
    }

    /// Whether this is a floating-point type.
    public var isFloatingPoint: Bool {
        switch self {
        case .float64, .float32, .float16, .bfloat16, .float8E4M3, .float8E5M2: true
        case .int64, .int32, .int16, .int8, .uint8, .bool: false
        }
    }
}
