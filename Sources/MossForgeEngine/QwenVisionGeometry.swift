import Foundation
import MossForgeCore

// Qwen3.5 vision geometry: 256² minimum, and positions transcribed from transformers 5.8
// `Qwen3_5Model.get_rope_index` / `get_vision_position_ids` (text after an image resumes at
// `start + max(h, w)/merge`, and the next position is returned for cached decoding). See NOTES.md.

/// Qwen3.5 image geometry: `smart_resize` and the patch grid. Patch 16, temporal patch 2, merge 2,
/// so height and width snap to multiples of 32 and each image token covers a 2×2 patch block.
public enum QwenVisionGeometry {
    public static let patch = 16
    public static let merge = 2
    public static let factor = patch * merge

    public struct Grid: Sendable, Equatable {
        public var t: Int
        public var h: Int
        public var w: Int
        public init(t: Int, h: Int, w: Int) {
            self.t = t
            self.h = h
            self.w = w
        }
        /// Patches the vision tower sees.
        public var patches: Int { t * h * w }
        /// Merged tokens the language model sees (`<|image_pad|>` count).
        public var tokens: Int { patches / (QwenVisionGeometry.merge * QwenVisionGeometry.merge) }
    }

    /// transformers `qwen2_vl.smart_resize`: both sides multiples of `factor`, aspect kept, pixel
    /// count within `[minPixels, maxPixels]`. Python `round` is half-to-even.
    public static func smartResize(
        height: Int, width: Int, factor: Int = QwenVisionGeometry.factor,
        minPixels: Int = 256 * 256, maxPixels: Int = 1024 * 1024
    ) -> (height: Int, width: Int) {
        let h = Double(height), w = Double(width), f = Double(factor)
        func roundMul(_ x: Double) -> Int { Int((x / f).rounded(.toNearestOrEven)) * factor }
        var hBar = roundMul(h)
        var wBar = roundMul(w)
        if hBar * wBar > maxPixels {
            let beta = (h * w / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((h / beta / f).rounded(.down)) * factor)
            wBar = max(factor, Int((w / beta / f).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = Int((h * beta / f).rounded(.up)) * factor
            wBar = Int((w * beta / f).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    /// The text encoder's view of an edit reference (ComfyUI `TextEncodeMOSSImageEdit`): area-scaled
    /// to at most `MossEditPrompt.visionPixels` (sides rounded half-to-even), then `smart_resize` with
    /// ComfyUI's Qwen2-VL bounds (3136 … 12 845 056 px). The pixels are resampled once, to the result.
    public static func editReferenceSize(height: Int, width: Int) -> (height: Int, width: Int) {
        let scale = min(1, (Double(MossEditPrompt.visionPixels) / Double(height * width)).squareRoot())
        let h = Int((Double(height) * scale).rounded(.toNearestOrEven))
        let w = Int((Double(width) * scale).rounded(.toNearestOrEven))
        return smartResize(height: max(h, 1), width: max(w, 1), minPixels: 3136, maxPixels: 12_845_056)
    }

    /// The patch grid of an already-resized image (one frame → t = 1).
    public static func grid(height: Int, width: Int) -> Grid { Grid(t: 1, h: height / patch, w: width / patch) }
}

/// Qwen3.5 multimodal M-RoPE 3-D positions for one sequence. Text tokens advance all three axes
/// together (plain RoPE); an image's tokens get `(start + t, start + row, start + col)` over the
/// merged grid, and text after it resumes at `start + max(h, w)/merge`.
public enum QwenMRoPEIndex {
    /// - Returns: `ids` `[3][seq]` (temporal, height, width) and `next`, the position of the
    ///   token after the sequence (decoding continues from there).
    public static func positions(
        ids: [Int], imageTokenID: Int, grids: [QwenVisionGeometry.Grid], merge: Int = QwenVisionGeometry.merge
    ) -> (ids: [[Int]], next: Int) {
        var t: [Int] = [], h: [Int] = [], w: [Int] = []
        t.reserveCapacity(ids.count)
        var pos = 0
        var i = 0
        var image = 0
        while i < ids.count {
            if ids[i] == imageTokenID, image < grids.count {
                let g = grids[image]
                image += 1
                let (gt, gh, gw) = (g.t, g.h / merge, g.w / merge)
                for ti in 0..<gt {
                    for hi in 0..<gh {
                        for wi in 0..<gw {
                            t.append(pos + ti)
                            h.append(pos + hi)
                            w.append(pos + wi)
                        }
                    }
                }
                i += gt * gh * gw
                pos += max(g.h, g.w) / merge
            } else {
                t.append(pos)
                h.append(pos)
                w.append(pos)
                pos += 1
                i += 1
            }
        }
        return ([t, h, w], pos)
    }
}
