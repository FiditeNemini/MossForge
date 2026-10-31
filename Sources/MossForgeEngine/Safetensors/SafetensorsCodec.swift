import Foundation
import MossForgeCore

/// Pure-Swift float codec + writer for the safetensors layout, so the engine can
/// **produce** checkpoints (merge / quantize / convert), not just read them. The
/// decode side reuses `Float8E4M3` for fp8; the encode side does correct
/// round-to-nearest-even bf16. All little-endian, matching the published format.
public enum SafetensorsCodec {
    /// Decode a tensor's raw bytes to `Float`. Supports F32 / F16 / BF16 / F8_E4M3.
    /// Contiguous unsafe loops (and a bulk memcpy for F32) — these run over billions
    /// of DiT params, so allocation/closure overhead per element is not affordable.
    public static func decodeFloats(_ bytes: Data, dtype: TensorDType) throws -> [Float] {
        switch dtype {
        case .float32:
            // Both little-endian IEEE-754 — a straight copy into the float buffer.
            let n = bytes.count / 4
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBytes { dst in
                bytes.withUnsafeBytes { src in
                    dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.prefix(n * 4)))
                }
            }
            return out
        case .float16:
            let n = bytes.count / 2
            var out = [Float](repeating: 0, count: n)
            bytes.withUnsafeBytes { src in
                out.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<n {
                        let u = src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                        dst[i] = Float(Float16(bitPattern: u))
                    }
                }
            }
            return out
        case .bfloat16:
            let n = bytes.count / 2
            var out = [Float](repeating: 0, count: n)
            bytes.withUnsafeBytes { src in
                out.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<n {
                        let u = src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                        dst[i] = Float(bitPattern: UInt32(u) << 16)
                    }
                }
            }
            return out
        case .float8E4M3:
            return QuantizedDecode.fp8Scaled(weightBytes: bytes, scale: 1.0)
        case .float8E5M2:
            return QuantizedDecode.fp8E5M2Scaled(weightBytes: bytes, scale: 1.0)
        default:
            throw EngineError.backendFailure("decode: unsupported dtype \(dtype.rawValue)")
        }
    }

    /// Encode `Float`s to raw bytes in the target dtype. Supports F32 / F16 / BF16.
    /// Preallocated buffers + tight loops (bulk copy for F32) — no per-element append.
    public static func encodeFloats(_ vals: [Float], as dtype: TensorDType) throws -> Data {
        switch dtype {
        case .float32:
            return vals.withUnsafeBytes { Data($0) }  // little-endian IEEE-754 already
        case .float16:
            var data = Data(count: vals.count * 2)
            data.withUnsafeMutableBytes { dst in
                vals.withUnsafeBufferPointer { src in
                    for i in 0..<src.count {
                        dst.storeBytes(of: Float16(src[i]).bitPattern, toByteOffset: i * 2, as: UInt16.self)
                    }
                }
            }
            return data
        case .bfloat16:
            var data = Data(count: vals.count * 2)
            data.withUnsafeMutableBytes { dst in
                vals.withUnsafeBufferPointer { src in
                    for i in 0..<src.count {
                        dst.storeBytes(of: bf16Bits(src[i]), toByteOffset: i * 2, as: UInt16.self)
                    }
                }
            }
            return data
        default:
            throw EngineError.backendFailure("encode: unsupported target dtype \(dtype.rawValue)")
        }
    }

    /// Round-to-nearest-even truncation of an F32 to BF16 (top 16 bits).
    static func bf16Bits(_ v: Float) -> UInt16 {
        if v.isNaN { return 0x7FC0 }
        let bits = v.bitPattern
        let rounding = UInt32(0x7FFF) + ((bits >> 16) & 1)
        return UInt16((bits &+ rounding) >> 16)
    }
}

/// Writes the `<u64 header-length><JSON header><tensor buffer>` safetensors layout.
/// The header is padded with spaces so the data buffer starts on an 8-byte boundary
/// (spec recommendation; round-trips with `SafetensorsHeader`).
public enum SafetensorsWriter {
    public struct Tensor: Sendable {
        public var name: String
        public var dtype: TensorDType
        public var shape: [Int]
        public var data: Data
        public init(name: String, dtype: TensorDType, shape: [Int], data: Data) {
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.data = data
        }
    }

    /// Returns the total bytes written. Streams header + tensors to disk (via a temporary file moved
    /// into place) instead of concatenating a multi-GB buffer in memory.
    @discardableResult
    public static func write(
        _ tensors: [Tensor], metadata: [String: String] = [:], to url: URL
    ) throws -> Int {
        var header: [String: Any] = [:]
        var offset = 0
        for t in tensors {
            header[t.name] = ["dtype": t.dtype.rawValue, "shape": t.shape, "data_offsets": [offset, offset + t.data.count]]
            offset += t.data.count
        }
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerData.count % 8 != 0 { headerData.append(0x20) }  // pad with spaces

        let fm = FileManager.default
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let temp = url.deletingLastPathComponent().appendingPathComponent(".\(url.lastPathComponent).partial")
        fm.createFile(atPath: temp.path, contents: nil)
        do {
            let out = try FileHandle(forWritingTo: temp)
            defer { try? out.close() }
            var len = UInt64(headerData.count).littleEndian
            try out.write(contentsOf: withUnsafeBytes(of: &len) { Data($0) })
            try out.write(contentsOf: headerData)
            for t in tensors { try out.write(contentsOf: t.data) }
        } catch {
            try? fm.removeItem(at: temp)
            throw error
        }
        if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
        try fm.moveItem(at: temp, to: url)
        return 8 + headerData.count + offset
    }
}
