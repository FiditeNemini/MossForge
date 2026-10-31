import Foundation
import Testing

@testable import MossForgeCore

/// The layout canvas's model: boxes edited in normalised [0,1] canvas space (resolution-free)
/// and emitted as MOSS's abs_centralized yxyx pixel boxes for the target size.
@Suite("LayoutDraft — canvas boxes ↔ abs_centralized layout JSON")
struct LayoutDraftTests {
    @Test func normalisedBoxesBecomeCentredPixelYXYX() throws {
        var d = LayoutDraft(canvas: Dimensions(width: 1024, height: 768), caption: "a travel poster", meta: ["poster"])
        let title = d.add(label: .text, frame: .init(top: 0.05, left: 0.1, bottom: 0.2, right: 0.9))
        d.update(title) { $0.text = "EXPLORE\nTHE NORTH"; $0.caption = "bold serif title" }
        d.add(label: .object, frame: .init(top: 0.3, left: 0.25, bottom: 0.95, right: 0.75))
        let layout = d.layoutPrompt()
        #expect(layout.summary.caption == "a travel poster" && layout.summary.meta == ["poster"])
        #expect(layout.summary.canvas == "single")
        // y: 0.05·768 − 384 = −345.6 → −346; x: 0.1·1024 − 512 = −409.6 → −410 …
        #expect(layout.items[0].bbox == [-346, -410, -230, 410])
        #expect(layout.items[0].text == "EXPLORE\nTHE NORTH" && layout.items[0].label == .text)
        #expect(layout.items[1].bbox == [-154, -256, 346, 256])
        #expect(layout.items[1].text == nil)  // objects carry no literal text
        let json = try layout.jsonString()
        #expect(json.contains(#""bbox_coord_space":"abs_centralized""#) && json.contains(#"EXPLORE\nTHE NORTH"#))
    }

    @Test func moveAndResizeClampToTheCanvasAndKeepAMinimumSize() throws {
        var d = LayoutDraft(canvas: Dimensions(width: 1024, height: 1024))
        let id = d.add(label: .object, frame: .init(top: 0.1, left: 0.1, bottom: 0.3, right: 0.4))
        d.move(id, dx: 0.9, dy: -0.5)  // pushed past the right / top edges
        let moved = try #require(d.item(id)?.frame)
        let want: [Double] = [0, 0.7, 0.2, 1]  // size kept, clamped inside
        #expect(zip([moved.top, moved.left, moved.bottom, moved.right], want).allSatisfy { abs($0 - $1) < 1e-9 })
        d.resize(id, toBottom: -1, right: 0.05)  // dragged past the top-left corner
        let resized = try #require(d.item(id)?.frame)
        #expect(resized.bottom - resized.top >= LayoutDraft.minimumSide && resized.right - resized.left >= LayoutDraft.minimumSide)
        #expect(resized.top == moved.top && resized.left == moved.left)  // the anchor corner doesn't move
    }

    @Test func framesNormaliseReversedDragsAndHitTestTopmostFirst() throws {
        var d = LayoutDraft(canvas: Dimensions(width: 512, height: 512))
        let a = d.add(label: .object, frame: .init(top: 0.8, left: 0.8, bottom: 0.2, right: 0.2))  // dragged up-left
        #expect(d.item(a)?.frame == .init(top: 0.2, left: 0.2, bottom: 0.8, right: 0.8))
        let b = d.add(label: .text, frame: .init(top: 0.4, left: 0.4, bottom: 0.6, right: 0.6))
        #expect(d.hit(x: 0.5, y: 0.5) == b && d.hit(x: 0.25, y: 0.25) == a && d.hit(x: 0.05, y: 0.05) == nil)
        d.remove(b)
        #expect(d.items.count == 1 && d.selection == nil)
    }

    @Test func roundTripsAnExistingLayoutPromptAndSurvivesAspectChanges() throws {
        let source = LayoutPrompt(
            summary: .init(meta: ["poster"], canvas: "single", caption: "c"),
            items: [.init(label: .text, caption: "t", text: "HI", bbox: [-256, -256, 0, 256])])
        var d = LayoutDraft(source, canvas: Dimensions(width: 1024, height: 1024))
        #expect(d.layoutPrompt() == source)
        d.canvas = Dimensions(width: 2048, height: 1024)  // same relative placement, new pixels
        #expect(d.layoutPrompt().items[0].bbox == [-256, -512, 0, 512])
    }
}
