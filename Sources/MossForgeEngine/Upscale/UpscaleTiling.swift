
import Foundation

/// Tile layout for the VOSR passes, mirroring Forge's `modules/vosr/pipeline.py`: square tiles
/// stepped by `tile − overlap` with the last one flush to the edge, blended with a raised-cosine
/// feather across the whole overlap. (Upstream's variance-0.01 Gaussian made each pixel come
/// from one tile, so texture a single tile invented showed as a hard-edged rectangle.)
public enum UpscaleTiling {
    /// VAE spatial factor (pixels per latent cell).
    public static let aeFactor = 8

    /// Sorted start positions of `tile`-sized windows covering `length`.
    public static func grid(length: Int, tile: Int, overlap: Int) -> [Int] {
        guard length > tile else { return [0] }
        let stride = max(tile - overlap, 1)
        var positions = Array(Swift.stride(from: 0, through: length - tile, by: stride))
        if let last = positions.last, last + tile < length { positions.append(length - tile) }
        return positions
    }

    /// Tile origins in row-major order, grouped into DiT batches of up to `size`.
    public static func batches(rows: [Int], cols: [Int], size: Int) -> [[(Int, Int)]] {
        let all = rows.flatMap { r in cols.map { (r, $0) } }
        let n = max(size, 1)
        return stride(from: 0, to: all.count, by: n).map { Array(all[$0..<min($0 + n, all.count)]) }
    }

    /// Separable feather: 1 inside, easing to ~0 (floored at 1e-3) across `overlap` at each edge.
    public static func feather(tile: Int, overlap: Int) -> [Float] {
        (0..<tile).map { i in
            let x = Float(i) + 0.5
            let ramp = min(min(x, Float(tile) - x) / Float(max(overlap, 1)), 1)
            let s = sin(ramp * .pi / 2)
            return max(s * s, 1e-3)
        }
    }

    /// DiT tile in latent cells: a multiple of the patch size, no larger than the latent's short
    /// side (the model only takes square latents).
    public static func ditTile(
        tilePixels: Int, overlapPixels: Int, latentH: Int, latentW: Int, patch: Int
    ) -> (tile: Int, overlap: Int) {
        var lt = max((tilePixels / aeFactor / patch) * patch, patch)
        lt = min(lt, (min(latentH, latentW) / patch) * patch)
        let lo = min(max(overlapPixels / aeFactor, lt / 8), lt - 1)
        return (lt, lo)
    }

    /// VAE tile in latent cells, or `tiled == false` when the whole latent fits one tile.
    public static func vaeTile(
        tilePixels: Int, overlapPixels: Int, latentH: Int, latentW: Int
    ) -> (tile: Int, overlap: Int, tiled: Bool) {
        let lt0 = max(tilePixels / aeFactor, 1)
        guard tilePixels > 0, latentH > lt0 || latentW > lt0 else { return (lt0, 0, false) }
        let lt = min(lt0, latentH, latentW)
        let lo = min(max(overlapPixels / aeFactor, lt / 8), lt - 1)
        return (lt, lo, true)
    }

    /// Output size for `scale`: a multiple of 8, at least one DiT patch (16 px) per side.
    public static func targetSize(width: Int, height: Int, scale: Double, patch: Int = 2) -> (Int, Int) {
        func side(_ v: Int) -> Int {
            max(Int(Double(v) * scale) / aeFactor * aeFactor, aeFactor * patch)
        }
        return (side(width), side(height))
    }
}
