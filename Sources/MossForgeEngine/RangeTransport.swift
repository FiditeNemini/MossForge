import Foundation

/// What a server says about a file: its size and whether it serves byte ranges.
public struct RangeProbe: Sendable, Equatable {
    public var size: Int64
    public var ranges: Bool
    public init(size: Int64, ranges: Bool) {
        self.size = size
        self.ranges = ranges
    }
}

/// Byte-range access to a remote file — URLSession for `https`, plain file reads for `file://` bundles
/// (and tests). `ModelBundleDownloader` fetches big files as several ranges at once through it.
public protocol RangeTransport: Sendable {
    func probe(_ url: URL) async throws -> RangeProbe
    /// Stream `range` (the whole file when `nil`) to `sink`, in order. Throwing from `sink` aborts.
    func fetch(_ url: URL, range: ClosedRange<Int64>?, sink: @escaping @Sendable (Data) throws -> Void) async throws
}

/// A non-success HTTP status.
public struct HTTPStatusError: Error, Sendable, Equatable, CustomStringConvertible {
    public var status: Int
    public var file: String
    public var description: String { "HTTP \(status) for \(file)" }
}

/// `file://` sources: always range-capable.
public struct FileRangeTransport: RangeTransport {
    public init() {}

    public func probe(_ url: URL) async throws -> RangeProbe {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        return RangeProbe(size: (attrs[.size] as? NSNumber)?.int64Value ?? 0, ranges: true)
    }

    public func fetch(_ url: URL, range: ClosedRange<Int64>?, sink: @escaping @Sendable (Data) throws -> Void) async throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var left = Int64.max
        if let range {
            try handle.seek(toOffset: UInt64(range.lowerBound))
            left = range.upperBound - range.lowerBound + 1
        }
        while left > 0, let chunk = try handle.read(upToCount: Int(min(left, 1 << 20))), !chunk.isEmpty {
            try Task.checkCancellation()
            try sink(chunk)
            left -= Int64(chunk.count)
        }
    }
}

/// `https` sources. Each fetch gets its own `URLSession`, so parallel ranges travel on separate TCP
/// connections (one session would multiplex them onto a single HTTP/2 connection, which is what made
/// one-stream downloads slow). Hugging Face redirects `resolve/` URLs to a signed CDN URL: the probe
/// remembers it, and an expired one (403/410) is resolved again.
public final class URLSessionRangeTransport: RangeTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var resolved: [URL: URL] = [:]

    public init() {}

    public func probe(_ url: URL) async throws -> RangeProbe {
        var request = URLRequest(url: url)
        request.setValue("bytes=0-0", forHTTPHeaderField: "Range")
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        bytes.task.cancel()  // a server that ignores Range would otherwise start sending the whole file
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        if http.statusCode == 206, let range = http.value(forHTTPHeaderField: "Content-Range"),
            let total = range.split(separator: "/").last.flatMap({ Int64($0) })
        {
            lock.withLock { resolved[url] = http.url ?? url }
            return RangeProbe(size: total, ranges: true)
        }
        if http.statusCode == 200 { return RangeProbe(size: http.expectedContentLength, ranges: false) }
        throw HTTPStatusError(status: http.statusCode, file: url.lastPathComponent)
    }

    public func fetch(_ url: URL, range: ClosedRange<Int64>?, sink: @escaping @Sendable (Data) throws -> Void) async throws {
        let target = lock.withLock { resolved[url] } ?? url
        do {
            try await stream(target, range: range, name: url.lastPathComponent, sink: sink)
        } catch let e as HTTPStatusError where [401, 403, 410].contains(e.status) && target != url {
            lock.withLock { _ = resolved.removeValue(forKey: url) }  // the signed CDN link expired
            _ = try await probe(url)
            try await stream(lock.withLock { resolved[url] } ?? url, range: range, name: url.lastPathComponent, sink: sink)
        }
    }

    private func stream(_ url: URL, range: ClosedRange<Int64>?, name: String, sink: @escaping @Sendable (Data) throws -> Void) async throws {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        let delegate = StreamDelegate(expected: range == nil ? 200 : 206, name: name, sink: sink)
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        var request = URLRequest(url: url)
        if let range { request.setValue("bytes=\(range.lowerBound)-\(range.upperBound)", forHTTPHeaderField: "Range") }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                delegate.continuation = c
                session.dataTask(with: request).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
    }
}

/// Streams a data task's bytes to a sink; checks the status before the first byte.
private final class StreamDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let expected: Int
    let name: String
    let sink: @Sendable (Data) throws -> Void
    var continuation: CheckedContinuation<Void, Error>?
    private var failure: Error?

    init(expected: Int, name: String, sink: @escaping @Sendable (Data) throws -> Void) {
        self.expected = expected
        self.name = name
        self.sink = sink
    }

    func urlSession(
        _ s: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? expected
        guard status == expected else {
            failure = HTTPStatusError(status: status, file: name)
            return completionHandler(.cancel)
        }
        completionHandler(.allow)
    }

    func urlSession(_ s: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard failure == nil else { return }
        do { try sink(data) } catch {
            failure = error
            dataTask.cancel()
        }
    }

    func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let c = continuation
        continuation = nil
        if let error = failure ?? error { c?.resume(throwing: error) } else { c?.resume() }
    }
}
