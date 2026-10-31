import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// The two pieces of DiT math that are pure functions of the request (no weights): the RoPE angle
/// table and the timestep sinusoid. Grounded in `comfyui/moss-image/model.py` (`_RoPE.forward`,
/// `_rotate`, the float64 time embedding) and cross-checked with ComfyUI core `ldm/moss_image`.
@Suite("MossDiT math — RoPE angles + timestep sinusoid")
struct MossDiTMathTests {
    @Test func ropeAnglesAreCoordinateTimesBandPerAxis() {
        // [N, 64]: column a·16 + j = coord[a] · bands[j]; `_rotate` pairs dim c with c + 64, both
        // using column c, so the MLX side builds cos/sin [N,128] as [angles | angles].
        let (gh, gw, text) = (2, 3, 4)
        let angles = MossRoPEIndex.angles(gridHeight: gh, gridWidth: gw, textLength: text, headDim: 128, base: 10000)
        let positions = MossRoPEIndex.positions(gridHeight: gh, gridWidth: gw, textLength: text)
        let bands = MossRoPEIndex.bands(headDim: 128, base: 10000)
        #expect(angles.count == (gh * gw + text) * 64)
        for n in [0, 4, gh * gw, gh * gw + text - 1] {
            for axis in 0..<4 {
                for j in [0, 7, 15] {
                    #expect(angles[n * 64 + axis * 16 + j] == positions[n][axis] * bands[j])
                }
            }
        }
        // r = 0 ⇒ the first 16 angles of every token are 0 (identity rotation).
        #expect((0..<(gh * gw + text)).allSatisfy { n in (0..<16).allSatisfy { angles[n * 64 + $0] == 0 } })
    }

    @Test func timestepSinusoidIsCosThenSinWithFloat64Phases() {
        // half = 128, freq_k = 10000^(−k/128), phase = t·freq_k in float64, [cos | sin] → fp32.
        let t = 973.5
        let e = MossTimestepEmbedding.sinusoid(timestep: t, dim: 256)
        #expect(e.count == 256)
        #expect(e[0] == Float(cos(t)) && e[128] == Float(sin(t)))
        let k = 77
        let phase = t * pow(10000, -Double(k) / 128)
        #expect(e[k] == Float(cos(phase)) && e[128 + k] == Float(sin(phase)))
    }

    @Test func timestepIsThousandTimesSigma() {
        #expect(MossTimestepEmbedding.timestep(sigma: 0.25) == 250)
    }
}
