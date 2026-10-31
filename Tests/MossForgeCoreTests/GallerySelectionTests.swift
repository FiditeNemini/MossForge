import Foundation
import Testing

@testable import MossForgeCore

/// Finder-style selection over the gallery grid: click selects one, ⌘-click toggles, ⇧-click extends
/// a range from the anchor, ⌘A selects all; deleted files drop out.
@Suite("GallerySelection — Finder-style multi-select")
struct GallerySelectionTests {
    let order = ["a", "b", "c", "d", "e"]

    @Test func clickSelectsOnlyThatItem() {
        var s = GallerySelection<String>()
        s.click("b", in: order)
        s.click("d", in: order)
        #expect(s.selected == ["d"] && s.primary == "d")
    }

    @Test func commandClickToggles() {
        var s = GallerySelection<String>()
        s.click("b", in: order)
        s.click("d", in: order, modifier: .command)
        #expect(s.selected == ["b", "d"])
        s.click("b", in: order, modifier: .command)
        #expect(s.selected == ["d"] && s.primary == "d")
    }

    @Test func shiftClickExtendsFromTheAnchorInEitherDirection() {
        var s = GallerySelection<String>()
        s.click("b", in: order)
        s.click("d", in: order, modifier: .shift)
        #expect(s.selected == ["b", "c", "d"])
        s.click("a", in: order, modifier: .shift)  // anchor stays at b
        #expect(s.selected == ["a", "b"])
    }

    @Test func selectAllAndPrune() {
        var s = GallerySelection<String>()
        s.selectAll(order)
        #expect(s.selected.count == 5)
        s.prune(keeping: ["a", "c"])
        #expect(s.selected == ["a", "c"])
        s.clear()
        #expect(s.selected.isEmpty && s.primary == nil)
    }

    @Test func orderedSelectionFollowsTheGrid() {
        var s = GallerySelection<String>()
        s.click("d", in: order)
        s.click("a", in: order, modifier: .command)
        #expect(s.ordered(in: order) == ["a", "d"])
    }

    @Test func trashTakesTheSidecarAlong() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gallery-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let png = dir.appendingPathComponent("x.png")
        FileManager.default.createFile(atPath: png.path, contents: Data([1]))
        try GallerySidecar.write(["prompt": "p"], for: png)
        #expect(GallerySidecar.files(for: png) == [png, GallerySidecar.url(for: png)])
        let lone = dir.appendingPathComponent("y.png")
        FileManager.default.createFile(atPath: lone.path, contents: Data([1]))
        #expect(GallerySidecar.files(for: lone) == [lone])
    }
}
