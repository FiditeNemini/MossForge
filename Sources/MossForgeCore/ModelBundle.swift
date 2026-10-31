import Foundation

/// `mossforge-manifest.json` at the root of a model-bundle repo laid out like `models/`: download
/// groups (Essentials, SFT, Base, Turbo LoRAs, …) and every file's size and SHA-256.
public struct ModelBundleManifest: Codable, Sendable, Equatable {
    public struct Group: Codable, Sendable, Equatable, Identifiable {
        public var id: String
        public var title: String
        public var summary: String
        /// Selected by default in setup.
        public var `default`: Bool
        public var size: Int64

        public init(id: String, title: String, summary: String, default isDefault: Bool, size: Int64) {
            self.id = id
            self.title = title
            self.summary = summary
            self.default = isDefault
            self.size = size
        }
    }

    public struct File: Codable, Sendable, Equatable {
        /// Relative to `models/`.
        public var path: String
        public var group: String
        public var size: Int64
        public var sha256: String

        public init(path: String, group: String, size: Int64, sha256: String) {
            self.path = path
            self.group = group
            self.size = size
            self.sha256 = sha256
        }
    }

    public static let fileName = "mossforge-manifest.json"
    public static let format = "mossforge-models"

    public var format: String
    public var version: Int
    public var groups: [Group]
    public var files: [File]

    public init(format: String = ModelBundleManifest.format, version: Int = 1, groups: [Group], files: [File]) {
        self.format = format
        self.version = version
        self.groups = groups
        self.files = files
    }

    /// Decode and check: the right format, and only plain relative paths (it comes off the network).
    public static func decode(_ data: Data) throws -> ModelBundleManifest {
        let m: ModelBundleManifest
        do { m = try JSONDecoder().decode(ModelBundleManifest.self, from: data) } catch {
            throw EngineError.invalidRequest("the model bundle's manifest isn't readable: \(error)")
        }
        guard m.format == format else { throw EngineError.invalidRequest("not a MossForge model bundle (format '\(m.format)')") }
        for f in m.files where !isSafe(f.path) {
            throw EngineError.invalidRequest("the model bundle lists an unsafe path '\(f.path)'")
        }
        return m
    }

    /// `a/b/c.ext`: no leading `/` or `~`, no empty, `.` or `..` components.
    static func isSafe(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.hasPrefix("~") else { return false }
        return path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    public var defaultGroups: Set<String> { Set(groups.filter(\.default).map(\.id)) }

    /// The groups' files that are missing, or on disk at a different size (a partial or older copy).
    public func pending(groups wanted: Set<String>, existingSizes: [String: Int64]) -> [File] {
        files.filter { wanted.contains($0.group) && existingSizes[$0.path] != $0.size }
    }
}

/// Where the bundle lives: a Hugging Face model repo (`owner/name`, Settings → Models) and revision.
public struct ModelBundleSource: Codable, Sendable, Equatable {
    public static let defaultRepo = "FiditeNemini/MossForge"

    public var repo: String
    public var revision: String
    /// `https://huggingface.co` — a `file://` folder in tests.
    public var endpoint: URL

    /// Accepts `owner/name` or a pasted `https://huggingface.co/owner/name` link.
    public init(repo: String = ModelBundleSource.defaultRepo, revision: String = "main", endpoint: URL = URL(string: "https://huggingface.co")!) {
        var r = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"] where r.hasPrefix(prefix) {
            r = String(r.dropFirst(prefix.count))
        }
        while r.hasSuffix("/") { r.removeLast() }
        self.repo = r
        self.revision = revision
        self.endpoint = endpoint
    }

    /// Exactly `owner/name`, both non-empty.
    public var isValid: Bool {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts.allSatisfy { !$0.isEmpty && !$0.contains(" ") }
    }

    public var pageURL: URL { endpoint.appendingPathComponent(repo) }
    public var manifestURL: URL { url(for: ModelBundleManifest.fileName) }

    /// `<endpoint>/<repo>/resolve/<revision>/<path>`, each segment percent-encoded.
    public func url(for path: String) -> URL {
        path.split(separator: "/").reduce(endpoint.appendingPathComponent(repo).appendingPathComponent("resolve").appendingPathComponent(revision)) {
            $0.appendingPathComponent(String($1))
        }
    }
}
