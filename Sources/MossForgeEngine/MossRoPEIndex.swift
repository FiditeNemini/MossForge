import Foundation
import MossForgeCore

/// MOSS-Image's 4D RoPE coordinates, as built by the reference `_RoPE.forward`
/// (comfyui/moss-image/model.py). Pure logic so it is unit-tested here; the MLX DiT turns these
/// into the sin/cos table.
///
/// - Axes per token: **(r, t, h, w)** — reference index, sequence position, height, width
///   (the code's order; the paper's prose says (r,t,w,h)).
/// - Sequence order is **image first, then text**. Image tokens all share `t = textLength`
///   and get zero-centred `h, w`; text token k gets `t = k`, `h = w = 0`. `r = 0` everywhere, so
///   the r subspace stays unrotated (partial RoPE).
/// - Each axis uses the same `headDim/8` frequencies `1 / base^(j/n)`; the table is laid out
///   `[sin(r,t,h,w) | cos(r,t,h,w)]` and applied rotate-half style.
public enum MossRoPEIndex {
    public static let axes = 4

    /// Per-axis rotary frequencies (fp32, like training's CPU construction).
    public static func bands(headDim: Int, base: Double) -> [Float] {
        let n = headDim / 8
        return (0..<n).map { j in
            let exponent = Float(j) / Float(n)
            return 1 / Float(pow(Float(base), exponent))
        }
    }

    /// `(r, t, h, w)` for every token: `gridHeight·gridWidth` image tokens (row-major), then text.
    /// `sequenceOffset` shifts every t (relative attention is unchanged; isolated edit references need it).
    public static func positions(gridHeight: Int, gridWidth: Int, textLength: Int, sequenceOffset: Float = 0) -> [[Float]] {
        var out: [[Float]] = []
        out.reserveCapacity(gridHeight * gridWidth + textLength)
        let ch = Float(gridHeight - 1) / 2
        let cw = Float(gridWidth - 1) / 2
        for i in 0..<gridHeight {
            for j in 0..<gridWidth {
                out.append([0, Float(textLength) + sequenceOffset, Float(i) - ch, Float(j) - cw])
            }
        }
        for k in 0..<textLength { out.append([0, Float(k) + sequenceOffset, 0, 0]) }
        return out
    }
}

extension MossRoPEIndex {
    /// Edit references (the community edit-LoRA method): reference `index` ≥ 1 on the r axis the
    /// released model leaves at 0, t = `sequencePosition` (`referenceSequence`), and its own
    /// zero-centred h/w grid.
    public static func referencePositions(index: Int, gridHeight: Int, gridWidth: Int, sequencePosition: Int) -> [[Float]] {
        let ch = Float(gridHeight - 1) / 2
        let cw = Float(gridWidth - 1) / 2
        var out: [[Float]] = []
        for i in 0..<gridHeight {
            for j in 0..<gridWidth { out.append([Float(index), Float(sequencePosition), Float(i) - ch, Float(j) - cw]) }
        }
        return out
    }

    /// The references' t and the shift applied to a branch's image + text t for `layout` (ComfyUI's
    /// `f_offset`). Isolated references are cached at t = 0, so the ai-toolkit branch moves its image
    /// there too; Moss-Image-Edit keeps the image after the text with the references at the text's start.
    public static func referenceSequence(layout: MossEditLayout, textLength: Int) -> (reference: Int, offset: Int) {
        if layout.referencesAtTextStart { return (0, 0) }
        return layout.isolatedReferences ? (0, -textLength) : (textLength, 0)
    }

    public static func referenceAngles(
        index: Int, gridHeight: Int, gridWidth: Int, sequencePosition: Int, headDim: Int, base: Double
    ) -> [Float] {
        let b = bands(headDim: headDim, base: base)
        return referencePositions(index: index, gridHeight: gridHeight, gridWidth: gridWidth, sequencePosition: sequencePosition)
            .flatMap { p in (0..<axes).flatMap { axis in b.map { p[axis] * $0 } } }
    }

    /// Rotation angles `[N, 4·headDim/8]` (row-major): column `axis·n + j` = `coord[axis] · bands[j]`,
    /// tokens image-first. The reference table is `[sin(angles) | cos(angles)]` and `_rotate` uses
    /// angle column `c` for both dims `c` and `c + headDim/2` (rotate-half), so the MLX DiT builds
    /// `cos/sin` as `[angles | angles]`. The text slice (rows ≥ H·W) also drives the text adapter.
    public static func angles(
        gridHeight: Int, gridWidth: Int, textLength: Int, headDim: Int, base: Double, sequenceOffset: Float = 0
    ) -> [Float] {
        let b = bands(headDim: headDim, base: base)
        var out: [Float] = []
        out.reserveCapacity((gridHeight * gridWidth + textLength) * axes * b.count)
        for p in positions(gridHeight: gridHeight, gridWidth: gridWidth, textLength: textLength, sequenceOffset: sequenceOffset) {
            for axis in 0..<axes {
                for f in b { out.append(p[axis] * f) }
            }
        }
        return out
    }
}

/// The DiT's timestep input: `timestep = 1000·σ`, embedded as a 256-d sinusoid computed in
/// **float64** (training does this), `[cos | sin]`, `freq_k = 10000^(−k/half)`, then fed to the
/// fp32 time MLP (`comfyui/moss-image/model.py`).
public enum MossTimestepEmbedding {
    public static func timestep(sigma: Double) -> Double { 1000 * sigma }

    public static func sinusoid(timestep t: Double, dim: Int = 256) -> [Float] {
        let half = dim / 2
        let phases = (0..<half).map { t * pow(10000, -Double($0) / Double(half)) }
        return phases.map { Float(cos($0)) } + phases.map { Float(sin($0)) }
    }
}
