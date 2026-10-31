import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

// Plans via `MossWeightPlan` (per-tensor `comfy_quant` ConvRot params, prefix scoping),
// reads through one `SafetensorsReader` instead of reopening the file per tensor, and applies the
// plan's fusions (split `proj.{q,k,v}` → fused `proj.qkv`). See NOTES.md.

/// The MLX half of the weight loader: it consumes the pure-logic `WeightLoadPlan` (built in
/// `MossForgeEngine`) and materializes native-keyed `MLXArray`s. Passthrough files go through
/// `MLX.loadArrays`; quantized ones are dequantized on-device per tensor.
public enum WeightLoader {
    /// Load a safetensors file into native module keys.
    ///
    /// - Parameters:
    ///   - prefix: only tensors under it are loaded, keyed with it stripped (`""` = whole file;
    ///     `"model.language_model."` = the Qwen3.5 text stack without vision / heads).
    ///   - working: dtype to cast every tensor to. `nil` keeps passthrough tensors' on-disk dtype
    ///     and lands dequantized weights as bf16.
    public static func load(url: URL, prefix: String = "", working: DType? = nil) throws -> LoadedWeights {
        let plan = try MossWeightPlan.build(url: url, prefix: prefix)
        // Cheap geometry check before a byte is read: MLX matmul shape errors `fatalError`.
        try MossWeightPlan.validate(plan)

        var arrays: [String: MLXArray] = [:]
        arrays.reserveCapacity(plan.entries.count)
        if plan.entries.allSatisfy(\.isPassthrough), prefix.isEmpty {
            let raw = try MLX.loadArrays(url: url)
            var missing: [String] = []
            for entry in plan.entries {
                guard case let .passthrough(rawKey) = entry.dequant, let tensor = raw[rawKey] else {
                    missing.append(entry.key)
                    continue
                }
                arrays[entry.key] = working.map { tensor.asType($0) } ?? tensor
            }
            guard missing.isEmpty else { throw WeightLoaderError.missingTensors(Array(missing.prefix(8))) }
        } else {
            let source = try TensorSource(url: url)
            defer { source.close() }
            for entry in plan.entries {
                let tensor = try materialize(entry, from: source)
                arrays[entry.key] = tensor.asType(working ?? (entry.isPassthrough ? tensor.dtype : .bfloat16))
            }
        }
        fuse(plan.fusions, into: &arrays)
        return LoadedWeights(arrays: arrays, tensorCount: arrays.count)
    }

    /// Concatenate each fusion's parts along axis 0 and drop the parts.
    static func fuse(_ fusions: [WeightFusion], into arrays: inout [String: MLXArray]) {
        for fusion in fusions {
            let parts = fusion.parts.compactMap { arrays[$0] }
            guard parts.count == fusion.parts.count else { continue }
            arrays[fusion.key] = concatenated(parts, axis: 0)
            for p in fusion.parts { arrays[p] = nil }
        }
    }

    /// Materialize a single logical key of a plan (fused keys concatenate their parts) — the
    /// streaming primitive `load-parity` and the convert/merge tools build on.
    public static func tensor(_ key: String, plan: WeightLoadPlan, from source: TensorSource) throws -> MLXArray? {
        if let fusion = plan.fusions.first(where: { $0.key == key }) {
            let parts = try fusion.parts.compactMap { try tensor($0, plan: plan, from: source) }
            return parts.count == fusion.parts.count ? concatenated(parts, axis: 0) : nil
        }
        guard let entry = plan.entries.first(where: { $0.key == key }) else { return nil }
        return try materialize(entry, from: source)
    }

    /// Materialize one plan entry from raw file bytes. Dequantized weights come back F32;
    /// passthrough tensors keep their on-disk dtype.
    public static func materialize(_ entry: WeightLoadEntry, from source: TensorSource) throws -> MLXArray {
        switch entry.dequant {
        case let .passthrough(rawKey):
            let (descriptor, bytes) = try source.read(rawKey)
            return try reinterpret(bytes: bytes, dtype: descriptor.dtype, shape: descriptor.shape)

        case let .fp8Scaled(weightKey, scaleKey):
            let (descriptor, weightBytes) = try source.read(weightKey)
            // Empty scaleKey ⇒ raw fp8 cast with an implicit 1.0, as ComfyUI does.
            let scale: Float = scaleKey.isEmpty ? 1.0 : try source.read(scaleKey).1.withUnsafeBytes { $0.load(as: Float.self) }
            // GPU dequant: gather the uint8 codes through a 256-entry LUT.
            let lut = descriptor.dtype == TensorDType.float8E5M2.rawValue ? fp8E5M2LUT() : fp8LUT()
            let codes = MLXArray(weightBytes, [weightBytes.count], dtype: .uint8).asType(.int32)
            return lut.take(codes, axis: 0).reshaped(descriptor.shape) * scale

        case let .nvfp4(weightKey, blockScaleKey, globalScaleKey, blockSize):
            return try nvfp4(weightKey, blockScaleKey, globalScaleKey, blockSize: blockSize, from: source)

        case let .int8ConvRot(weightKey, scaleKey, groupSize):
            let (wDesc, wBytes) = try source.read(weightKey)
            let (_, sBytes) = try source.read(scaleKey)
            let out = wDesc.shape[0], inD = wDesc.shape[1]
            let codes = MLXArray(wBytes, [out, inD], dtype: .int8).asType(.float32)
            return unRotate(codes * MLXArray(sBytes, [out, 1], dtype: .float32), groupSize: groupSize)

        case let .int4ConvRot(weightKey, scaleKey, groupSize):
            let (wDesc, wBytes) = try source.read(weightKey)
            let (_, sBytes) = try source.read(scaleKey)
            let out = wDesc.shape[0], inD = wDesc.shape[1] * 2  // two nibbles per packed byte
            // Two signed nibbles per byte (low then high) → interleaved values `[out, in]`.
            let codes = MLXArray(wBytes, [wBytes.count], dtype: .uint8).asType(.int32)
            let values = stacked([int4LowLUT().take(codes, axis: 0), int4HighLUT().take(codes, axis: 0)], axis: -1)
                .reshaped([out, inD])
            return unRotate(values * MLXArray(sBytes, [out, 1], dtype: .float32), groupSize: groupSize)
        }
    }

    /// Undo the folded group-wise regular-Hadamard rotation: `W = W_rot @ H` block-diagonally
    /// over `groupSize` input columns (H symmetric orthogonal ⇒ rotate and un-rotate coincide).
    /// Group size ≤ 1 or a non-dividing one is plain per-channel int (no rotation).
    static func unRotate(_ wRot: MLXArray, groupSize g: Int) -> MLXArray {
        let out = wRot.dim(0), inD = wRot.dim(1)
        guard g > 1, inD % g == 0 else { return wRot }
        return matmul(wRot.reshaped([out, inD / g, g]), regularHadamard(g)).reshaped([out, inD])
    }

    /// The normalized **regular** Hadamard matrix ConvRot uses (comfy-kitchen), not Sylvester:
    /// base `H4` (no all-ones column) Kronecker-powered to `size`, then `/√size`. Size must be a
    /// power of four (MOSS files use 16, 64 and 256).
    static func regularHadamard(_ size: Int) -> MLXArray {
        let h4: [[Float]] = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
        var m = h4
        var cur = 4
        while cur < size {
            let r = m.count, c = m[0].count
            var next = [[Float]](repeating: [Float](repeating: 0, count: c * 4), count: r * 4)
            for i in 0..<r {
                for j in 0..<c where m[i][j] != 0 {
                    let v = m[i][j]
                    for a in 0..<4 {
                        for b in 0..<4 { next[i * 4 + a][j * 4 + b] = v * h4[a][b] }
                    }
                }
            }
            m = next
            cur *= 4
        }
        let norm = Float(1.0 / Double(size).squareRoot())
        return MLXArray(m.flatMap { $0.map { $0 * norm } }, [size, size])
    }

    private static func fp8LUT() -> MLXArray { MLXArray((0..<256).map { Float8E4M3.decode(UInt8($0)) }) }
    private static func fp8E5M2LUT() -> MLXArray { MLXArray((0..<256).map { Float8E5M2.decode(UInt8($0)) }) }

    /// Signed value of a 4-bit two's-complement nibble (`n ≥ 8 ⇒ n − 16`).
    private static let signedNibble: [Float] = (0..<16).map { Float($0 >= 8 ? $0 - 16 : $0) }
    private static func int4LowLUT() -> MLXArray { MLXArray((0..<256).map { signedNibble[$0 & 0xF] }) }
    private static func int4HighLUT() -> MLXArray { MLXArray((0..<256).map { signedNibble[($0 >> 4) & 0xF] }) }

    /// The 16 e2m1 (fp4) code → value entries; the sign is the top bit.
    private static let e2m1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    private static func nvfp4HighLUT() -> MLXArray { MLXArray((0..<256).map { e2m1[($0 >> 4) & 0xF] }) }
    private static func nvfp4LowLUT() -> MLXArray { MLXArray((0..<256).map { e2m1[$0 & 0xF] }) }

    /// nvfp4 → float: U8 `[out, in/2]` high-nibble-first e2m1 codes × cutlass-128×4-swizzled
    /// E4M3 block scales `[out, in/16]` × an F32 global scale. (Verified bit-exact against a reference.)
    private static func nvfp4(
        _ weightKey: String, _ blockScaleKey: String, _ globalScaleKey: String, blockSize: Int, from source: TensorSource
    ) throws -> MLXArray {
        let (wDesc, wBytes) = try source.read(weightKey)
        let (sDesc, sBytes) = try source.read(blockScaleKey)
        let (_, gBytes) = try source.read(globalScaleKey)
        let out = wDesc.shape[0], inD = wDesc.shape[1] * 2, blk = sDesc.shape[1]
        guard out % 128 == 0, blk % 4 == 0 else { throw WeightLoaderError.dequantNotImplemented(count: 1) }
        let global = gBytes.withUnsafeBytes { $0.load(as: Float.self) }

        let codes = MLXArray(wBytes, [wBytes.count], dtype: .uint8).asType(.int32)
        let fp4 = stacked([nvfp4HighLUT().take(codes, axis: 0), nvfp4LowLUT().take(codes, axis: 0)], axis: -1)
            .reshaped([out, inD])
        let sCodes = MLXArray(sBytes, [sBytes.count], dtype: .uint8).asType(.int32)
        let sf = fp8LUT().take(sCodes, axis: 0)
            .reshaped([out / 128, blk / 4, 32, 4, 4])
            .transposed(0, 3, 2, 1, 4)
            .reshaped([out, blk])
        let scale = broadcast(sf.reshaped([out, blk, 1]), to: [out, blk, blockSize]).reshaped([out, inD])
        return fp4 * scale * global
    }

    /// Wrap raw little-endian bytes as an `MLXArray` of the on-disk dtype.
    private static func reinterpret(bytes: Data, dtype: String, shape: [Int]) throws -> MLXArray {
        switch TensorDType(rawValue: dtype) {
        case .float32: return MLXArray(bytes, shape, dtype: .float32)
        case .bfloat16: return MLXArray(bytes, shape, dtype: .bfloat16)
        case .float16: return MLXArray(bytes, shape, dtype: .float16)
        default: throw WeightLoaderError.unsupportedPassthrough(dtype)
        }
    }
}

/// One open safetensors file: header parsed once, tensors read by name.
public final class TensorSource {
    private let reader: SafetensorsReader
    private let byName: [String: TensorDescriptor]

    public init(url: URL) throws {
        reader = try SafetensorsReader(url: url)
        byName = Dictionary(reader.descriptors.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
    }

    public func read(_ name: String) throws -> (TensorDescriptor, Data) {
        guard let d = byName[name] else { throw WeightLoaderError.missingTensors([name]) }
        return (d, try reader.bytes(name))
    }

    public func close() { reader.close() }
}

extension WeightLoadEntry {
    var isPassthrough: Bool {
        if case .passthrough = dequant { return true }
        return false
    }
}

/// Result of a successful load: native-keyed arrays plus a count for reporting.
public struct LoadedWeights {
    public var arrays: [String: MLXArray]
    public var tensorCount: Int

    public init(arrays: [String: MLXArray], tensorCount: Int) {
        self.arrays = arrays
        self.tensorCount = tensorCount
    }

    public subscript(_ key: String) -> MLXArray? { arrays[key] }
}

public enum WeightLoaderError: Error, CustomStringConvertible {
    /// An nvfp4 tensor whose scales aren't 128×4-tiled (the cutlass swizzle assumption).
    case dequantNotImplemented(count: Int)
    /// The header promised tensors the buffer didn't yield (truncated/corrupt file).
    case missingTensors([String])
    /// A passthrough tensor used a dtype the byte reinterpreter doesn't handle.
    case unsupportedPassthrough(String)

    public var description: String {
        switch self {
        case let .dequantNotImplemented(count):
            "weight loader: \(count) nvfp4 weight(s) have non-128×4-tiled block scales"
        case let .missingTensors(keys):
            "weight loader: tensors absent from the file: \(keys.joined(separator: ", "))"
        case let .unsupportedPassthrough(dtype):
            "weight loader: can't reinterpret passthrough dtype '\(dtype)'"
        }
    }
}
