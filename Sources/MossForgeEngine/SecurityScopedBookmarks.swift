import Foundation

/// Folders and files the user picked (models, output, component overrides), kept as security-scoped
/// bookmarks so a sandboxed build (the Mac App Store app) can open them again after a relaunch. Outside the
/// sandbox the bookmarks still resolve and starting access is a no-op, so the same code serves both builds.
/// Keyed by the standardised path; a bookmark that no longer resolves is skipped.
public struct SecurityScopedBookmarks: Sendable, Equatable, Codable {
    private var entries: [String: Data] = [:]

    public init() {}

    public var count: Int { entries.count }

    private static func key(_ url: URL) -> String { url.standardizedFileURL.path }

    /// Remember `url` (call it with the URL an open panel or drop returned, while access is granted).
    public mutating func remember(_ url: URL) {
        #if os(macOS)
        let options: URL.BookmarkCreationOptions = .withSecurityScope
        #else
        let options: URL.BookmarkCreationOptions = []
        #endif
        if let data = try? url.bookmarkData(options: options, includingResourceValuesForKeys: nil, relativeTo: nil) {
            entries[Self.key(url)] = data
        }
    }

    public mutating func forget(_ url: URL) { entries[Self.key(url)] = nil }

    /// Drop bookmarks for anything no longer in use.
    public mutating func keepOnly(_ urls: [URL]) {
        let keep = Set(urls.map(Self.key))
        entries = entries.filter { keep.contains($0.key) }
    }

    /// Resolve every bookmark and start accessing it; returns the URLs now open (keep them for `stopAccessing`).
    /// Stale bookmarks are refreshed in place.
    public mutating func startAccessing() -> [URL] {
        #if os(macOS)
        let options: URL.BookmarkResolutionOptions = [.withSecurityScope, .withoutUI]
        #else
        let options: URL.BookmarkResolutionOptions = [.withoutUI]
        #endif
        var opened: [URL] = []
        for (key, data) in entries.sorted(by: { $0.key < $1.key }) {
            var stale = false
            guard let url = try? URL(resolvingBookmarkData: data, options: options, relativeTo: nil, bookmarkDataIsStale: &stale),
                FileManager.default.fileExists(atPath: url.path)
            else { continue }
            _ = url.startAccessingSecurityScopedResource()
            if stale {
                entries[key] = nil
                remember(url)
            }
            opened.append(url)
        }
        return opened
    }

    /// Stop the access `startAccessing` began.
    public func stopAccessing(_ urls: [URL]) {
        for url in urls { url.stopAccessingSecurityScopedResource() }
    }
}
