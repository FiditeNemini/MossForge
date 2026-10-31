import CryptoKit
import Foundation
import MossForgeCore

/// Setup's download progress: the file in flight and the running total over every pending file.
public struct ModelBundleProgress: Sendable, Equatable {
    public var file: String
    public var fileIndex: Int
    public var fileCount: Int
    public var bytes: Int64
    public var totalBytes: Int64
    /// Over the last few seconds (0 until measured).
    public var bytesPerSecond: Double = 0
    public var fraction: Double { totalBytes > 0 ? Double(bytes) / Double(totalBytes) : 0 }
    public var secondsLeft: Double? { bytesPerSecond > 0 ? Double(totalBytes - bytes) / bytesPerSecond : nil }
}

/// Fetches a model bundle (`ModelBundleSource`) into a `models/` folder: the manifest, then each
/// pending file of the chosen groups. A big file comes down as 32 MiB byte ranges, several at once on
/// separate connections, written into a preallocated `<path>.part`; finished pieces are listed in
/// `<path>.part.json` (`ChunkedDownload`), so a cancelled or dropped download resumes where it was.
/// Each piece is retried with backoff. Then the file is SHA-256-checked and moved into place. A server
/// that ignores `Range` gets one stream. Outbound requests need `NetworkPolicy` egress (Settings →
/// Privacy), like every other network use.
public struct ModelBundleDownloader: Sendable {
    public var source: ModelBundleSource
    public var models: URL
    public var policy: NetworkPolicy
    public var transport: any RangeTransport
    /// Parallel ranges per file.
    public var connections = 8
    public var chunkSize = ChunkedDownload.defaultChunkSize
    /// Tries per piece; the wait doubles from `retryDelay`.
    public var attempts = 5
    public var retryDelay: Duration = .seconds(1)

    public init(source: ModelBundleSource, models: URL, policy: NetworkPolicy, transport: (any RangeTransport)? = nil) {
        self.source = source
        self.models = models
        self.policy = policy
        self.transport = transport ?? (source.endpoint.isFileURL ? FileRangeTransport() : URLSessionRangeTransport())
    }

    public func fetchManifest() async throws -> ModelBundleManifest {
        try checkPolicy()
        if source.endpoint.isFileURL { return try ModelBundleManifest.decode(Data(contentsOf: source.manifestURL)) }
        var request = URLRequest(url: source.manifestURL)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw EngineError.invalidRequest("\(source.repo) has no \(ModelBundleManifest.fileName) (HTTP \(http.statusCode))")
        }
        return try ModelBundleManifest.decode(data)
    }

    /// On-disk sizes of the manifest's files (absent ones left out).
    public func existingSizes(_ manifest: ModelBundleManifest) -> [String: Int64] {
        var out: [String: Int64] = [:]
        for f in manifest.files {
            let attrs = try? FileManager.default.attributesOfItem(atPath: models.appendingPathComponent(f.path).path)
            if let size = (attrs?[.size] as? NSNumber)?.int64Value { out[f.path] = size }
        }
        return out
    }

    /// Bytes already fetched toward the groups' pending files (finished pieces of interrupted downloads).
    public func resumableBytes(groups: Set<String>, manifest: ModelBundleManifest) -> Int64 {
        manifest.pending(groups: groups, existingSizes: existingSizes(manifest)).reduce(0) { $0 + (savedState(for: $1)?.bytesDone ?? 0) }
    }

    /// Download the groups' pending files; returns the paths written.
    @discardableResult
    public func download(
        groups: Set<String>, manifest: ModelBundleManifest, progress: @escaping @Sendable (ModelBundleProgress) -> Void = { _ in }
    ) async throws -> [String] {
        try checkPolicy()
        let pending = manifest.pending(groups: groups, existingSizes: existingSizes(manifest))
        let meter = ProgressMeter(total: pending.reduce(0) { $0 + $1.size }, emit: progress)
        meter.preload(pending.reduce(0) { $0 + (savedState(for: $1)?.bytesDone ?? 0) })
        for (i, file) in pending.enumerated() {
            try Task.checkCancellation()
            meter.start(file: file.path, index: i, count: pending.count)
            let target = models.appendingPathComponent(file.path)
            let part = target.appendingPathExtension("part")
            let state = part.appendingPathExtension("json")
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try await fetch(file, to: part, state: state, meter: meter)
            let digest = try Self.sha256(of: part)
            guard digest == file.sha256.lowercased() else {
                try? FileManager.default.removeItem(at: part)
                try? FileManager.default.removeItem(at: state)
                throw EngineError.invalidRequest("\(file.path) failed its checksum (got \(digest.prefix(12))…, expected \(file.sha256.prefix(12))…)")
            }
            _ = try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: part, to: target)
            try? FileManager.default.removeItem(at: state)
            meter.finishFile()
        }
        return pending.map(\.path)
    }

    /// The saved pieces of an interrupted download of `file`, if they still belong to it.
    func savedState(for file: ModelBundleManifest.File) -> ChunkedDownload? {
        let part = models.appendingPathComponent(file.path).appendingPathExtension("part")
        guard let data = try? Data(contentsOf: part.appendingPathExtension("json")),
            let s = try? JSONDecoder().decode(ChunkedDownload.self, from: data), s.resumes(size: file.size, sha256: file.sha256),
            let size = (try? FileManager.default.attributesOfItem(atPath: part.path))?[.size] as? NSNumber, size.int64Value == file.size
        else { return nil }
        return s
    }

    /// One file into `part`: byte ranges in parallel when the server allows (resuming `state`), else one stream.
    private func fetch(_ file: ModelBundleManifest.File, to part: URL, state stateURL: URL, meter: ProgressMeter) async throws {
        let url = source.url(for: file.path)
        let probe = try await transport.probe(url)
        guard probe.size == file.size else {
            throw EngineError.invalidRequest("\(file.path) is \(probe.size) bytes on the server, \(file.size) in the manifest")
        }
        let saved = savedState(for: file)
        guard probe.ranges, file.size > chunkSize else {
            if let saved { meter.add(-saved.bytesDone) }
            try? FileManager.default.removeItem(at: stateURL)
            try await single(url, to: part, meter: meter)
            return
        }
        var state = saved ?? ChunkedDownload(size: file.size, sha256: file.sha256, chunkSize: chunkSize)
        if saved == nil {
            FileManager.default.createFile(atPath: part.path, contents: nil)
            let h = try FileHandle(forWritingTo: part)
            try h.truncate(atOffset: UInt64(file.size))  // sparse until written
            try h.close()
            try JSONEncoder().encode(state).write(to: stateURL, options: .atomic)
        }
        let writer = try PartWriter(part)
        defer { writer.close() }
        let plan = state
        let queue = PieceQueue(plan.pending)
        let book = StateBook(plan, url: stateURL)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<min(connections, plan.pending.count) {
                group.addTask {
                    while let i = queue.next() {
                        try await piece(i, of: plan, from: url, writer: writer, meter: meter)
                        try book.markDone(i)
                    }
                }
            }
            try await group.waitForAll()
        }
        state = book.state
        guard state.isComplete else { throw EngineError.backendFailure("\(file.path): \(state.pending.count) pieces missing") }
    }

    /// One byte range, retried with doubling waits; bytes from a failed try are taken back off the meter.
    private func piece(_ i: Int, of state: ChunkedDownload, from url: URL, writer: PartWriter, meter: ProgressMeter) async throws {
        let range = state.range(i)
        let expected = range.upperBound - range.lowerBound + 1
        for attempt in 1...attempts {
            let written = Tally()
            do {
                try await transport.fetch(url, range: range) { data in
                    try writer.write(data, at: range.lowerBound + written.value)
                    written.add(Int64(data.count))
                    meter.add(Int64(data.count))
                }
                guard written.value == expected else { throw URLError(.networkConnectionLost) }
                return
            } catch {
                meter.add(-written.value)
                if error is CancellationError || Task.isCancelled || attempt == attempts { throw error }
                try await Task.sleep(for: retryDelay * (1 << (attempt - 1)))
            }
        }
    }

    /// The whole file as one stream (small files, servers without ranges), restarted on failure.
    private func single(_ url: URL, to part: URL, meter: ProgressMeter) async throws {
        for attempt in 1...attempts {
            FileManager.default.createFile(atPath: part.path, contents: nil)
            let writer = try PartWriter(part)
            let written = Tally()
            do {
                try await transport.fetch(url, range: nil) { data in
                    try writer.write(data, at: written.value)
                    written.add(Int64(data.count))
                    meter.add(Int64(data.count))
                }
                writer.close()
                return
            } catch {
                writer.close()
                meter.add(-written.value)
                if error is CancellationError || Task.isCancelled || attempt == attempts { throw error }
                try await Task.sleep(for: retryDelay * (1 << (attempt - 1)))
            }
        }
    }

    func checkPolicy() throws {
        guard source.endpoint.isFileURL || policy.allowsOutbound(to: source.endpoint.host ?? "") else {
            throw EngineError.invalidRequest("downloading models needs network access — turn it on in Settings → Privacy")
        }
    }

    /// Streamed hex SHA-256 (multi-GB files, 16 MB at a time).
    public static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 24), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// Positioned writes into a `.part` file from several pieces at once.
final class PartWriter: @unchecked Sendable {
    private let handle: FileHandle
    private let lock = NSLock()
    private var closed = false

    init(_ url: URL) throws { handle = try FileHandle(forWritingTo: url) }

    func write(_ data: Data, at offset: Int64) throws {
        try lock.withLock {
            try handle.seek(toOffset: UInt64(offset))
            try handle.write(contentsOf: data)
        }
    }

    func close() {
        lock.withLock {
            guard !closed else { return }
            closed = true
            try? handle.close()
        }
    }
}

/// Hands out the pieces still to fetch.
final class PieceQueue: @unchecked Sendable {
    private var items: [Int]
    private let lock = NSLock()
    init(_ items: [Int]) { self.items = items.reversed() }
    func next() -> Int? { lock.withLock { items.popLast() } }
}

/// The resume state, saved after every finished piece.
final class StateBook: @unchecked Sendable {
    private(set) var state: ChunkedDownload
    private let url: URL
    private let lock = NSLock()
    init(_ state: ChunkedDownload, url: URL) {
        self.state = state
        self.url = url
    }
    func markDone(_ i: Int) throws {
        try lock.withLock {
            state.done.insert(i)
            try JSONEncoder().encode(state).write(to: url, options: .atomic)
        }
    }
}

final class Tally: @unchecked Sendable {
    private var n: Int64 = 0
    private let lock = NSLock()
    var value: Int64 { lock.withLock { n } }
    func add(_ d: Int64) { lock.withLock { n += d } }
}

/// Running total for the UI: at most five updates a second, with speed over the last five seconds.
final class ProgressMeter: @unchecked Sendable {
    private let total: Int64
    private let emit: @Sendable (ModelBundleProgress) -> Void
    private let lock = NSLock()
    private var bytes: Int64 = 0
    private var file = ""
    private var index = 0
    private var count = 0
    private var lastEmit = Date.distantPast
    private var samples: [(Date, Int64)] = []

    init(total: Int64, emit: @escaping @Sendable (ModelBundleProgress) -> Void) {
        self.total = total
        self.emit = emit
    }

    func add(_ n: Int64) { report(force: false) { bytes += n } }
    /// Bytes already on disk from an interrupted download: counted, not reported until a file starts.
    func preload(_ n: Int64) { lock.withLock { bytes += n } }
    func start(file: String, index: Int, count: Int) {
        report(force: true) {
            self.file = file
            self.index = index
            self.count = count
        }
    }
    func finishFile() { report(force: true) {} }

    private func report(force: Bool, _ change: () -> Void) {
        let snapshot: ModelBundleProgress? = lock.withLock {
            change()
            let now = Date()
            guard force || now.timeIntervalSince(lastEmit) >= 0.2 else { return nil }
            lastEmit = now
            samples.append((now, bytes))
            samples.removeAll { now.timeIntervalSince($0.0) > 5 }
            var rate = 0.0
            if let first = samples.first, let dt = Optional(now.timeIntervalSince(first.0)), dt > 0.5 {
                rate = max(0, Double(bytes - first.1) / dt)
            }
            return ModelBundleProgress(
                file: file, fileIndex: index, fileCount: count, bytes: bytes, totalBytes: total, bytesPerSecond: rate)
        }
        if let snapshot { emit(snapshot) }
    }
}
