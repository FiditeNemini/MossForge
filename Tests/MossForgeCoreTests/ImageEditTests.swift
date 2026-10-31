import Foundation
import Testing

@testable import MossForgeCore

/// Edit in Layout: only the boxes you change are redrawn; everything else is pinned to the original.
@Suite struct ImageEditTests {
    let canvas = Dimensions(width: 256, height: 128)  // 16 × 8 tokens

    func item(_ text: String, _ bbox: [Int]) -> LayoutPrompt.Item { .init(label: .text, caption: "title", text: text, bbox: bbox) }

    @Test func changedBoxesAreEditedAddedAndRemovedOnes() {
        let keep = item("SALE", [-64, -128, -32, 0])
        let old = item("Monday", [0, 0, 32, 64])
        let original = LayoutPrompt(summary: .init(caption: "poster"), items: [keep, old])
        var moved = old
        moved.text = "Friday"
        let added = LayoutPrompt.Item(label: .object, caption: "a fox", bbox: [32, -128, 64, -64])
        let edited = LayoutPrompt(summary: .init(caption: "poster"), items: [keep, moved, added])
        let boxes = LayoutEditMask.changedBoxes(original: original, edited: edited)
        #expect(Set(boxes) == Set([[0, 0, 32, 64], [32, -128, 64, -64]]))
        // Deleting a box redraws where it was.
        let removed = LayoutPrompt(summary: .init(caption: "poster"), items: [keep])
        #expect(LayoutEditMask.changedBoxes(original: original, edited: removed) == [[0, 0, 32, 64]])
        #expect(LayoutEditMask.changedBoxes(original: original, edited: original).isEmpty)
    }

    /// abs_centralized pixels → token cells (16 px), with a one-token margin so the edit blends.
    @Test func tokenMaskCoversTheBoxPlusAMargin() {
        let m = LayoutEditMask.tokenMask(boxes: [[0, 0, 32, 64]], canvas: canvas, margin: 1)
        #expect(m.width == 16 && m.height == 8 && m.values.count == 128)
        // Box: rows 4..<6, cols 8..<12 (centre origin) → with margin rows 3..<7, cols 7..<13.
        func at(_ r: Int, _ c: Int) -> Float { m.values[r * 16 + c] }
        #expect(at(4, 8) == 1 && at(5, 11) == 1)
        #expect(at(3, 7) == 1 && at(6, 12) == 1)
        #expect(at(2, 8) == 0 && at(4, 13) == 0 && at(0, 0) == 0)
        #expect(m.coverage > 0 && m.coverage < 0.5)
    }

    @Test func boxesAreClampedToTheCanvas() {
        let m = LayoutEditMask.tokenMask(boxes: [[-200, -400, 200, 400]], canvas: canvas, margin: 0)
        #expect(m.values.allSatisfy { $0 == 1 })
    }

    /// The pixel mask is a smooth bilinear upsample: 1 deep inside, 0 far outside, in between at edges.
    @Test func pixelMaskFeathersTheEdge() {
        let m = LayoutEditMask.tokenMask(boxes: [[0, 0, 32, 64]], canvas: canvas, margin: 0)
        let p = m.pixels(scale: 16)
        #expect(p.count == 256 * 128)
        func px(_ y: Int, _ x: Int) -> Float { p[y * 256 + x] }
        #expect(px(80, 160) == 1)  // inside
        #expect(px(8, 8) == 0)  // far outside
        let edge = px(72, 126)  // between the col-7 centre (0) and col-8 centre (1)
        #expect(edge > 0 && edge < 1)
    }

    @Test func requestsValidateTheMaskAgainstTheSize() throws {
        var r = GenerationRequest(prompt: "x", width: 256, height: 128)
        let m = LayoutEditMask.tokenMask(boxes: [[0, 0, 32, 64]], canvas: canvas, margin: 1)
        r.edit = ImageEditSpec(source: URL(fileURLWithPath: "/a.png"), mask: m)
        try r.validate()
        r.width = 512
        #expect(throws: EngineError.self) { try r.validate() }
        r.width = 256
        r.edit = ImageEditSpec(source: URL(fileURLWithPath: "/a.png"), mask: .init(width: 16, height: 8, values: Array(repeating: 0, count: 128)))
        #expect(throws: EngineError.self) { try r.validate() }
    }

    /// A plain-prompt image opens as a layout with its prompt as the caption and no boxes yet.
    @Test func plainPromptsBecomeABoxlessLayout() {
        let l = LayoutEditMask.layout(fromPrompt: "a red fox in snow")
        #expect(l.summary.caption == "a red fox in snow" && l.items.isEmpty)
        let json = #"{"summary":{"caption":"poster"},"items":[],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#
        #expect(LayoutEditMask.layout(fromPrompt: json).summary.caption == "poster")
    }
}
