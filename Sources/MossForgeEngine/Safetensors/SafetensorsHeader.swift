import Foundation
import MossForgeCore

/// A tensor's header entry: everything known without reading its bytes. The dtype
/// is the raw header string, so formats the codec can't decode (`F8_E4M3`, …) are
/// still describable — multi-GB files inspect instantly.
public struct TensorDescriptor: Sendable, Equatable {
    public var name: String
    public var dtype: String
    public var shape: [Int]
    public var byteCount: Int

    public init(name: String, dtype: String, shape: [Int], byteCount: Int) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.byteCount = byteCount
    }
}

/// Header-only reader for the public Hugging Face safetensors layout:
/// `<u64 little-endian header length><JSON header><tensor buffer>`. Independent
/// implementation of the published format.
public enum SafetensorsHeader {
    /// Read tensor descriptors + string metadata without loading the tensor
    /// buffer. Throws `EngineError.backendFailure` on a malformed file.
    public static func readDescriptors(
        from url: URL
    ) throws -> (tensors: [TensorDescriptor], metadata: [String: String]) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let (header, _) = try parseHeader(handle)

        var metadata: [String: String] = [:]
        var tensors: [TensorDescriptor] = []
        for (key, value) in header {
            if key == "__metadata__" {
                metadata = (value as? [String: String]) ?? [:]
                continue
            }
            guard let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = (entry["shape"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
                let offsets = (entry["data_offsets"] as? [Any])?.compactMap({
                    ($0 as? NSNumber)?.intValue
                }),
                offsets.count == 2
            else {
                throw EngineError.backendFailure("safetensors: malformed entry for tensor '\(key)'")
            }
            tensors.append(
                TensorDescriptor(
                    name: key,
                    dtype: dtype,
                    shape: shape,
                    byteCount: offsets[1] - offsets[0]
                )
            )
        }
        tensors.sort { $0.name < $1.name }
        return (tensors, metadata)
    }

    /// Read one tensor's raw bytes by name, seeking to its slice of the buffer. Lets
    /// the loader pull fp8 / packed tensors that `MLX.loadArrays` can't open (no fp8
    /// DType in mlx-swift). Throws if the tensor is absent or the slice is truncated.
    public static func readTensor(
        name: String, from url: URL
    ) throws -> (descriptor: TensorDescriptor, bytes: Data) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let (header, headerLength) = try parseHeader(handle)
        guard let entry = header[name] as? [String: Any],
            let dtype = entry["dtype"] as? String,
            let shape = (entry["shape"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
            let offsets = (entry["data_offsets"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
            offsets.count == 2
        else {
            throw EngineError.backendFailure("safetensors: no tensor named '\(name)'")
        }
        let dataStart = 8 + headerLength
        try handle.seek(toOffset: UInt64(dataStart + offsets[0]))
        let count = offsets[1] - offsets[0]
        guard let bytes = try handle.read(upToCount: count), bytes.count == count else {
            throw EngineError.backendFailure("safetensors: truncated tensor '\(name)'")
        }
        let descriptor = TensorDescriptor(name: name, dtype: dtype, shape: shape, byteCount: count)
        return (descriptor, bytes)
    }

    /// One tensor's place in the file: dtype, shape and absolute byte range.
    public struct Entry: Sendable, Equatable {
        public var dtype: String
        public var shape: [Int]
        public var start: Int
        public var end: Int
    }

    /// Every tensor's absolute byte range (data start + `data_offsets`) and the metadata — what a
    /// streaming copy needs.
    public static func readEntries(from url: URL) throws -> (entries: [String: Entry], metadata: [String: String]) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let (header, headerLength) = try parseHeader(handle)
        let dataStart = 8 + headerLength
        var entries: [String: Entry] = [:]
        var metadata: [String: String] = [:]
        for (key, value) in header {
            if key == "__metadata__" { metadata = (value as? [String: String]) ?? [:]; continue }
            guard let e = value as? [String: Any], let dtype = e["dtype"] as? String,
                let shape = (e["shape"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
                let o = (e["data_offsets"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }), o.count == 2
            else { throw EngineError.backendFailure("safetensors: malformed entry for tensor '\(key)'") }
            entries[key] = Entry(dtype: dtype, shape: shape, start: dataStart + o[0], end: dataStart + o[1])
        }
        return (entries, metadata)
    }

    /// Shared header parse: `<u64 length><JSON>`. Returns the decoded object and the
    /// header length (so callers can compute the data-buffer start = 8 + length).
    fileprivate static func parseHeader(_ handle: FileHandle) throws -> ([String: Any], Int) {
        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else {
            throw EngineError.backendFailure("safetensors: file shorter than the 8-byte header length")
        }
        let headerLength = Int(readUInt64LE([UInt8](lengthBytes)))
        guard headerLength > 0, headerLength < 500_000_000 else {
            throw EngineError.backendFailure("safetensors: implausible header length \(headerLength)")
        }
        guard let headerData = try handle.read(upToCount: headerLength),
            headerData.count == headerLength
        else {
            throw EngineError.backendFailure("safetensors: truncated header")
        }
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw EngineError.backendFailure("safetensors: header is not a JSON object")
        }
        return (header, headerLength)
    }
}

func readUInt64LE(_ bytes: [UInt8]) -> UInt64 {
    var value: UInt64 = 0
    for i in 0..<8 { value |= UInt64(bytes[i]) << (8 * i) }
    return value
}

/// Streaming reader that opens the file and parses the header **once**, then reads
/// many tensors from the single handle by seeking. Essential for multi-GB
/// checkpoints — the per-call `SafetensorsHeader.readTensor` re-parses the whole
/// header on every tensor (hundreds of times for a DiT), which is prohibitively slow.
public final class SafetensorsReader {
    public let descriptors: [TensorDescriptor]
    private let handle: FileHandle
    private let offsets: [String: (start: Int, end: Int)]
    private let dataStart: Int

    public init(url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        let (header, headerLength) = try SafetensorsHeader.parseHeader(handle)
        self.handle = handle
        self.dataStart = 8 + headerLength
        var offsets: [String: (Int, Int)] = [:]
        var descs: [TensorDescriptor] = []
        for (key, value) in header where key != "__metadata__" {
            guard let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = (entry["shape"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
                let off = (entry["data_offsets"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.intValue }),
                off.count == 2
            else { continue }
            offsets[key] = (off[0], off[1])
            descs.append(
                TensorDescriptor(name: key, dtype: dtype, shape: shape, byteCount: off[1] - off[0]))
        }
        self.offsets = offsets
        self.descriptors = descs.sorted { $0.name < $1.name }
    }

    public func contains(_ name: String) -> Bool { offsets[name] != nil }

    /// Raw bytes for a named tensor (seeks within the already-open handle).
    public func bytes(_ name: String) throws -> Data {
        guard let o = offsets[name] else {
            throw EngineError.backendFailure("safetensors: no tensor named '\(name)'")
        }
        try handle.seek(toOffset: UInt64(dataStart + o.start))
        let count = o.end - o.start
        guard let data = try handle.read(upToCount: count), data.count == count else {
            throw EngineError.backendFailure("safetensors: truncated tensor '\(name)'")
        }
        return data
    }

    public func close() { try? handle.close() }
    deinit { try? handle.close() }
}
