/// Resume state for one file fetched in fixed-size byte ranges: saved as `<file>.part.json` beside the
/// preallocated `<file>.part`, so a cancelled or dropped download keeps every finished piece.
public struct ChunkedDownload: Codable, Sendable, Equatable {
    /// 32 MiB: small enough that a cancel loses little, large enough that a 10 GB file is ~300 requests.
    public static let defaultChunkSize: Int64 = 32 << 20

    public var size: Int64
    /// The manifest's checksum: a changed file upstream invalidates the saved pieces.
    public var sha256: String
    public var chunkSize: Int64
    public var done: Set<Int>

    public init(size: Int64, sha256: String, chunkSize: Int64 = ChunkedDownload.defaultChunkSize, done: Set<Int> = []) {
        self.size = size
        self.sha256 = sha256
        self.chunkSize = chunkSize
        self.done = done
    }

    public var chunkCount: Int { size <= 0 ? 0 : Int((size + chunkSize - 1) / chunkSize) }

    /// Byte range of piece `i` (inclusive, as in an HTTP `Range` header).
    public func range(_ i: Int) -> ClosedRange<Int64> {
        let start = Int64(i) * chunkSize
        return start...(min(start + chunkSize, size) - 1)
    }

    public var pending: [Int] { (0..<chunkCount).filter { !done.contains($0) } }
    public var bytesDone: Int64 { done.reduce(0) { $0 + range($1).upperBound - range($1).lowerBound + 1 } }
    public var isComplete: Bool { pending.isEmpty }

    /// This saved state belongs to the manifest entry (same size and checksum).
    public func resumes(size: Int64, sha256: String) -> Bool {
        self.size == size && self.sha256.lowercased() == sha256.lowercased()
    }
}
