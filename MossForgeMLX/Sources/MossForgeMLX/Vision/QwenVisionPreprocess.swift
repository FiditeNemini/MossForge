import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

// Qwen3.5 preprocessing: mean/std 0.5 (Qwen/Qwen3.5-9B `preprocessor_config.json`), a 256² minimum,
// and the resize is PIL-exact bicubic (`PILResample`) instead of CoreGraphics, so the torch reference
// (`Scripts/vision_ref.py`) sees the same pixels. See NOTES.md.

/// An image file → the `[patches, 1536]` sequence and grid the Qwen3.5 vision tower consumes:
/// `smart_resize` → bicubic → `x/127.5 − 1` → patchify in merge-block order (one frame duplicated
/// over the temporal patch).
public enum QwenVisionPreprocess {
    public static func preprocess(url: URL, maxPixels: Int) throws -> (patches: MLXArray, grid: QwenVisionGeometry.Grid) {
        try preprocess(RGB8.load(url: url), maxPixels: maxPixels)
    }

    public static func preprocess(_ image: RGB8, maxPixels: Int) -> (patches: MLXArray, grid: QwenVisionGeometry.Grid) {
        let (rh, rw) = QwenVisionGeometry.smartResize(
            height: image.height, width: image.width, minPixels: CaptionRequest.minPixels, maxPixels: maxPixels)
        return preprocess(image, height: rh, width: rw)
    }

    /// The same at a size already snapped to `QwenVisionGeometry.factor` (edit references:
    /// `QwenVisionGeometry.editReferenceSize`).
    public static func preprocess(_ image: RGB8, height rh: Int, width rw: Int) -> (patches: MLXArray, grid: QwenVisionGeometry.Grid) {
        let px = PILResample.bicubic(image.pixels, width: image.width, height: image.height, channels: 3, toWidth: rw, toHeight: rh)
        let chw = (MLXArray(px, [rh, rw, 3]).asType(.float32) / 127.5 - 1).transposed(2, 0, 1)  // (x/255 − 0.5)/0.5
        return patchify(chw, height: rh, width: rw)
    }

    /// `[3, H, W]` → `[(H/16)·(W/16), 3·2·16·16]` in (block-row, block-col, row-in-block, col-in-block)
    /// order, each patch flattened as (channel, temporal, py, px) — the Conv3d weight's layout.
    public static func patchify(_ pixels: MLXArray, height rh: Int, width rw: Int) -> (patches: MLXArray, grid: QwenVisionGeometry.Grid) {
        let (ch, patch, temporal, merge) = (3, QwenVisionGeometry.patch, 2, QwenVisionGeometry.merge)
        let grid = QwenVisionGeometry.grid(height: rh, width: rw)
        let frames = stacked(Array(repeating: pixels, count: temporal), axis: 0)  // [t, 3, rh, rw]
        let x = frames.reshaped([1, temporal, ch, grid.h / merge, merge, patch, grid.w / merge, merge, patch])
            .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshaped([grid.patches, ch * temporal * patch * patch])
        return (x, grid)
    }
}
