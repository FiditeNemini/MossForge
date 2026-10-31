import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// The setup downloader against a `file://` "repo" laid out as `<repo>/resolve/main/…`.
@Suite struct ModelBundleDownloaderTests {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("mf-bundle-\(UUID().uuidString)")
    var repo: URL { root.appendingPathComponent("hub/me/Bundle/resolve/main") }
    var models: URL { root.appendingPathComponent("models") }

    func publish(_ files: [String: String], checksums: [String: String] = [:]) throws -> ModelBundleManifest {
        var entries: [ModelBundleManifest.File] = []
        for (path, text) in files.sorted(by: { $0.key < $1.key }) {
            let url = repo.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(text.utf8).write(to: url)
            let sha = try checksums[path] ?? ModelBundleDownloader.sha256(of: url)
            entries.append(.init(path: path, group: "essentials", size: Int64(text.utf8.count), sha256: sha))
        }
        let m = ModelBundleManifest(
            format: ModelBundleManifest.format, version: 1,
            groups: [.init(id: "essentials", title: "E", summary: "", default: true, size: entries.reduce(0) { $0 + $1.size })],
            files: entries)
        try JSONEncoder().encode(m).write(to: repo.appendingPathComponent(ModelBundleManifest.fileName))
        return m
    }

    var downloader: ModelBundleDownloader {
        ModelBundleDownloader(
            source: ModelBundleSource(repo: "me/Bundle", endpoint: root.appendingPathComponent("hub")), models: models,
            policy: NetworkPolicy(egressAllowed: false))
    }

    @Test func downloadsVerifiesAndSkipsWhatIsThere() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try publish(["vae/v.safetensors": "vae bytes", "text_encoders/tok/tokenizer.json": "{}"])
        let manifest = try await downloader.fetchManifest()
        let written = try await downloader.download(groups: ["essentials"], manifest: manifest)
        #expect(Set(written) == ["vae/v.safetensors", "text_encoders/tok/tokenizer.json"])
        #expect(try String(contentsOf: models.appendingPathComponent("vae/v.safetensors"), encoding: .utf8) == "vae bytes")
        #expect(!FileManager.default.fileExists(atPath: models.appendingPathComponent("vae/v.safetensors.part").path))
        #expect(try await downloader.download(groups: ["essentials"], manifest: manifest).isEmpty)  // all present
    }

    @Test func aBadChecksumLeavesNothingBehind() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = try publish(["vae/v.safetensors": "vae bytes"], checksums: ["vae/v.safetensors": String(repeating: "0", count: 64)])
        await #expect(throws: EngineError.self) { try await downloader.download(groups: ["essentials"], manifest: manifest) }
        let dir = models.appendingPathComponent("vae")
        #expect(((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).isEmpty)
    }

    /// A file transport that drops the connection after `failAfter` range requests, or fails each
    /// range's first attempt (`flaky`), or ignores `Range` (`ranges: false`); it counts what it fetched.
    final class FakeTransport: RangeTransport, @unchecked Sendable {
        let inner = FileRangeTransport()
        var failAfter = Int.max
        var flaky = false
        var ranges = true
        private let lock = NSLock()
        private(set) var fetched: [ClosedRange<Int64>?] = []
        private var attempted: Set<Int64> = []

        func probe(_ url: URL) async throws -> RangeProbe {
            var p = try await inner.probe(url)
            p.ranges = ranges && p.ranges
            return p
        }

        func fetch(_ url: URL, range: ClosedRange<Int64>?, sink: @escaping @Sendable (Data) throws -> Void) async throws {
            let fail: Bool = lock.withLock {
                if fetched.count >= failAfter { return true }
                if flaky, let r = range, !attempted.contains(r.lowerBound) {
                    attempted.insert(r.lowerBound)
                    return true
                }
                fetched.append(range)
                return false
            }
            if fail { throw URLError(.networkConnectionLost) }
            try await inner.fetch(url, range: range, sink: sink)
        }
    }

    func chunked(_ t: FakeTransport) -> ModelBundleDownloader {
        var d = downloader
        d.transport = t
        d.chunkSize = 4
        d.retryDelay = .zero
        return d
    }

    @Test func aDroppedDownloadResumesFromItsFinishedPieces() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let text = "0123456789abcdefghij"  // 5 pieces of 4 bytes
        let manifest = try publish(["dit/big.safetensors": text])
        let dropping = FakeTransport()
        dropping.failAfter = 2  // two pieces land, then the network is gone for good
        await #expect(throws: (any Error).self) { try await chunked(dropping).download(groups: ["essentials"], manifest: manifest) }
        let part = models.appendingPathComponent("dit/big.safetensors.part")
        let state = try JSONDecoder().decode(
            ChunkedDownload.self, from: Data(contentsOf: part.appendingPathExtension("json")))
        #expect(state.done.count == 2)

        let good = FakeTransport()
        let written = try await chunked(good).download(groups: ["essentials"], manifest: manifest)
        #expect(written == ["dit/big.safetensors"])
        #expect(good.fetched.count == 3)  // only the missing pieces
        #expect(try String(contentsOf: models.appendingPathComponent("dit/big.safetensors"), encoding: .utf8) == text)
        #expect(!FileManager.default.fileExists(atPath: part.path))
        #expect(!FileManager.default.fileExists(atPath: part.appendingPathExtension("json").path))
    }

    @Test func aFlakyNetworkIsRetriedPieceByPiece() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = try publish(["dit/big.safetensors": "0123456789abcdefghij"])
        let flaky = FakeTransport()
        flaky.flaky = true
        _ = try await chunked(flaky).download(groups: ["essentials"], manifest: manifest)
        #expect(try String(contentsOf: models.appendingPathComponent("dit/big.safetensors"), encoding: .utf8) == "0123456789abcdefghij")
    }

    @Test func aServerWithoutRangesGetsOneStream() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = try publish(["dit/big.safetensors": "0123456789abcdefghij"])
        let plain = FakeTransport()
        plain.ranges = false
        _ = try await chunked(plain).download(groups: ["essentials"], manifest: manifest)
        #expect(plain.fetched == [nil])
        #expect(try String(contentsOf: models.appendingPathComponent("dit/big.safetensors"), encoding: .utf8) == "0123456789abcdefghij")
    }

    @Test func progressCountsResumedBytesAndReportsSpeed() async throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = try publish(["dit/big.safetensors": "0123456789abcdefghij"])
        let dropping = FakeTransport()
        dropping.failAfter = 3
        _ = try? await chunked(dropping).download(groups: ["essentials"], manifest: manifest)
        let seen = Box()
        _ = try await chunked(FakeTransport()).download(groups: ["essentials"], manifest: manifest) { seen.append($0) }
        #expect(seen.first?.bytes ?? 0 >= 12)  // starts from the 3 pieces already here
        #expect(seen.last?.bytes == 20 && seen.last?.totalBytes == 20)
    }

    final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [ModelBundleProgress] = []
        func append(_ p: ModelBundleProgress) { lock.withLock { items.append(p) } }
        var first: ModelBundleProgress? { lock.withLock { items.first } }
        var last: ModelBundleProgress? { lock.withLock { items.last } }
    }

    @Test func theNetworkNeedsEgress() async {
        let remote = ModelBundleDownloader(source: ModelBundleSource(), models: models, policy: NetworkPolicy(egressAllowed: false))
        await #expect(throws: EngineError.self) { try await remote.fetchManifest() }
    }
}
