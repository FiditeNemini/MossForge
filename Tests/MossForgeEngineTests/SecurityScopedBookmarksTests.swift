import Foundation
import Testing
@testable import MossForgeEngine

/// App Sandbox: folders and files the user picks (models, output, component overrides) are remembered as
/// security-scoped bookmarks so the sandboxed app can open them again after a relaunch.
@Suite("SecurityScopedBookmarks — remember picked folders across launches")
struct SecurityScopedBookmarksTests {
    private func tempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bm-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func rememberedFolderResolvesToTheSamePath() throws {
        let dir = try tempDir()
        var bookmarks = SecurityScopedBookmarks()
        bookmarks.remember(dir)
        #expect(bookmarks.count == 1)
        let opened = bookmarks.startAccessing()
        #expect(opened.map { $0.resolvingSymlinksInPath().path } == [dir.resolvingSymlinksInPath().path])
        bookmarks.stopAccessing(opened)
    }

    @Test func rememberingTheSamePathTwiceKeepsOneEntry() throws {
        let dir = try tempDir()
        var bookmarks = SecurityScopedBookmarks()
        bookmarks.remember(dir)
        bookmarks.remember(dir.appendingPathComponent("."))
        #expect(bookmarks.count == 1)
    }

    @Test func forgetAndPruneKeepOnlyWhatIsInUse() throws {
        let (a, b) = (try tempDir(), try tempDir())
        var bookmarks = SecurityScopedBookmarks()
        bookmarks.remember(a)
        bookmarks.remember(b)
        bookmarks.keepOnly([a])
        #expect(bookmarks.count == 1)
        bookmarks.forget(a)
        #expect(bookmarks.count == 0)
    }

    @Test func aDeletedFolderIsSkippedNotFatal() throws {
        let dir = try tempDir()
        var bookmarks = SecurityScopedBookmarks()
        bookmarks.remember(dir)
        try FileManager.default.removeItem(at: dir)
        #expect(bookmarks.startAccessing().isEmpty)
    }

    @Test func codableRoundTrip() throws {
        var bookmarks = SecurityScopedBookmarks()
        bookmarks.remember(try tempDir())
        let back = try JSONDecoder().decode(SecurityScopedBookmarks.self, from: JSONEncoder().encode(bookmarks))
        #expect(back == bookmarks)
    }
}
