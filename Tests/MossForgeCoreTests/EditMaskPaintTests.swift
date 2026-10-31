import Foundation
import Testing

@testable import MossForgeCore

/// Masked reference edits: the mask painted on Picture 1 (or a mask image) on the token grid, so the
/// edit LoRA's change is pinned to it and everything else stays the original (Edit in Layout's pin).
@Suite struct EditMaskPaintTests {
    /// A pixel mask (1 = change) → the token grid by area average.
    @Test func pixelMaskResamplesByArea() {
        // 4×2 pixels, left half set → a 2×1 token grid.
        let m = EditMask.resampled(from: [1, 1, 0, 0, 1, 1, 0, 0], width: 4, height: 2, toWidth: 2, toHeight: 1)
        #expect(m.width == 2 && m.height == 1 && m.values == [1, 0])
        let half = EditMask.resampled(from: [1, 0, 0, 0], width: 2, height: 2, toWidth: 1, toHeight: 1)
        #expect(half.values == [0.25])
    }

    @Test func invertSwapsChangeAndKeep() {
        let m = EditMask(width: 2, height: 1, values: [1, 0.25])
        #expect(m.inverted().values == [0, 0.75])
    }

    /// A round brush in token units; painting adds (max), erasing removes (min).
    @Test func brushPaintsAndErases() {
        var m = EditMask(width: 5, height: 5, values: .init(repeating: 0, count: 25))
        m.paint(atX: 2.5, y: 2.5, radius: 1, value: 1)
        #expect(m.values[2 * 5 + 2] == 1)  // centre
        #expect(m.values[2 * 5 + 3] == 1)  // one cell right (centre distance 1)
        #expect(m.values[0] == 0)  // corner untouched
        m.paint(atX: 2.5, y: 2.5, radius: 0.5, value: 0)
        #expect(m.values[2 * 5 + 2] == 0 && m.values[2 * 5 + 3] == 1)
        #expect(EditMask.empty(width: 3, height: 2).values == [0, 0, 0, 0, 0, 0])
    }
}
