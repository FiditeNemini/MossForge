// Qwen3.5-9B's vision tower (transformers 5.8 `Qwen3_5VisionModel`): 27 blocks, hidden 1152, 16 heads × 72,
// MLP 4304, merger → 4096, **no deepstack mergers**; weights from the int8·convrot file through `WeightLoader`.
// See NOTES.md.

import Foundation
import MLX
import MLXFast
import MLXNN

public struct QwenVisionConfig: Sendable {
    public var hidden = 1152
    public var heads = 16
    public var headDim = 72
    public var intermediate = 4304
    public var outHidden = 4096
    public var patch = 16
    public var temporalPatch = 2
    public var merge = 2
    public var inChannels = 3
    public var depth = 27
    public var numPositions = 2304
    public var gridSide = 48
        public var eps: Float = 1e-6
    public var ropeTheta: Float = 10_000
    public var patchInputDim: Int { inChannels * temporalPatch * patch * patch }  // 1536
    public var mergedDim: Int { hidden * merge * merge }  // 4608
    public init() {}
}

/// Rotate the two halves of the last dim (vision rope operates over the full head dim).
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

/// Conv3d(kernel==stride==patch) collapses to a linear over the flattened `[in·t·p·p]`
/// patch — the stored 5-D `proj.weight` is reshaped to `[hidden, patchInputDim]` on load.
final class VisionPatchEmbed: Module {
    @ModuleInfo var proj: Linear
    init(_ c: QwenVisionConfig) { _proj.wrappedValue = Linear(c.patchInputDim, c.hidden, bias: true) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// Full (bidirectional) self-attention — for a single image `cu_seqlens` spans the whole
/// sequence, so every patch attends to every other (no mask). 2-D rope on q,k.
final class VisionAttention: Module {
    let heads, headDim: Int
    let scale: Float
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    init(_ c: QwenVisionConfig) {
        heads = c.heads
        headDim = c.headDim
        scale = 1.0 / Float(c.headDim).squareRoot()
        _qkv.wrappedValue = Linear(c.hidden, c.hidden * 3, bias: true)
        _proj.wrappedValue = Linear(c.hidden, c.hidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let s = x.dim(0)
        // qkv → [s,3,heads,hd] → [3,s,heads,hd] → q,k,v
        let parts = qkv(x).reshaped([s, 3, heads, headDim]).transposed(1, 0, 2, 3)
        var q = parts[0]
        var k = parts[1]
        let v = parts[2]
        let c2 = cos.reshaped([s, 1, headDim])
        let s2 = sin.reshaped([s, 1, headDim])
        q = q * c2 + rotateHalf(q) * s2
        k = k * c2 + rotateHalf(k) * s2
        // → [1, heads, s, hd] for SDPA
        let qh = q.transposed(1, 0, 2).reshaped([1, heads, s, headDim])
        let kh = k.transposed(1, 0, 2).reshaped([1, heads, s, headDim])
        let vh = v.transposed(1, 0, 2).reshaped([1, heads, s, headDim])
        let o = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale, mask: .none)
        let merged = o.transposed(0, 2, 1, 3).reshaped([s, heads * headDim])
        return proj(merged)
    }
}

final class VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear
    init(_ c: QwenVisionConfig) {
        _fc1.wrappedValue = Linear(c.hidden, c.intermediate, bias: true)
        _fc2.wrappedValue = Linear(c.intermediate, c.hidden, bias: true)
    }
    // hidden_act = gelu_pytorch_tanh → the tanh approximation.
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

final class VisionBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var attn: VisionAttention
    @ModuleInfo var mlp: VisionMLP
    init(_ c: QwenVisionConfig) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hidden, eps: c.eps)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hidden, eps: c.eps)
        _attn.wrappedValue = VisionAttention(c)
        _mlp.wrappedValue = VisionMLP(c)
    }
    func callAsFunction(_ h: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let a = h + attn(norm1(h), cos: cos, sin: sin)
        return a + mlp(norm2(a))
    }
}

/// Spatial-merge projection to the LM hidden size (4096). `use_postshuffle_norm`
/// controls whether the LayerNorm is over the pre-shuffle `hidden` (Qwen3.5's merger) or the
/// post-shuffle `hidden·merge²` width (Qwen3-VL's deepstack mergers; unused here).
final class VisionPatchMerger: Module {
    let postshuffle: Bool
    let mergedDim: Int
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(_ c: QwenVisionConfig, postshuffle: Bool) {
        self.postshuffle = postshuffle
        mergedDim = c.mergedDim
        _norm.wrappedValue = LayerNorm(dimensions: postshuffle ? mergedDim : c.hidden, eps: c.eps)
        _fc1.wrappedValue = Linear(mergedDim, mergedDim, bias: true)
        _fc2.wrappedValue = Linear(mergedDim, c.outHidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = postshuffle ? norm(x.reshaped([-1, mergedDim])) : norm(x).reshaped([-1, mergedDim])
        return fc2(gelu(fc1(normed)))  // merger GELU is exact (nn.GELU default)
    }
}

/// Qwen3.5 vision tower: patch-embed → +interpolated pos-embed → 27 rope blocks → merger
/// (image tokens, 4096-d, one per 2×2 patch block).
public final class QwenVisionTower: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: VisionPatchEmbed
    @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
    @ModuleInfo var blocks: [VisionBlock]
    @ModuleInfo var merger: VisionPatchMerger
    let config: QwenVisionConfig

    public init(config: QwenVisionConfig = .init()) {
        self.config = config
        _patchEmbed.wrappedValue = VisionPatchEmbed(config)
        _posEmbed.wrappedValue = Embedding(embeddingCount: config.numPositions, dimensions: config.hidden)
        _blocks.wrappedValue = (0..<config.depth).map { _ in VisionBlock(config) }
        _merger.wrappedValue = VisionPatchMerger(config, postshuffle: false)
    }

    // MARK: Stage-isolated entry points (used by the parity harness)

    public func patchEmbedForward(_ patches: MLXArray) -> MLXArray { patchEmbed(patches) }
    public func blockForward(_ h: MLXArray, cos: MLXArray, sin: MLXArray, index: Int) -> MLXArray {
        blocks[index](h, cos: cos, sin: sin)
    }
    public func mergerForward(_ h: MLXArray) -> MLXArray { merger(h) }

    // MARK: Full forward

    /// - Parameters:
    ///   - patches: `[seq, patchInputDim]` patch features (merge-block order).
    ///   - gridTHW: one `(t, h, w)` per image.
    ///   - lastHidden: receives the final block output (pre-merger) for the parity harness.
    /// - Returns: image tokens `[seq/merge², outHidden]`.
    public func callAsFunction(
        _ patches: MLXArray, gridTHW: [(Int, Int, Int)], lastHidden: ((MLXArray) -> Void)? = nil
    ) -> MLXArray {
        var h = patchEmbed(patches)
        h = h + positionEmbeddings(gridTHW: gridTHW)
        let (cos, sin) = rotaryCosSin(gridTHW: gridTHW)
        for block in blocks { h = block(h, cos: cos, sin: sin) }
        lastHidden?(h)
        return merger(h)
    }

    // MARK: rope + interpolated position embeddings (index math on CPU, gather on GPU)

    /// 2-D rotary cos/sin `[seq, headDim]`. Row/col positions are in merge-block order;
    /// each contributes `headDim/4` frequencies, concatenated then duplicated to headDim.
    func rotaryCosSin(gridTHW: [(Int, Int, Int)]) -> (MLXArray, MLXArray) {
        let rotDim = config.headDim / 2  // 32
        let freqCount = rotDim / 2  // 16
        let invFreq = (0..<freqCount).map { j in
            1.0 / powf(config.ropeTheta, Float(2 * j) / Float(rotDim))
        }
        var rot: [Float] = []
        rot.reserveCapacity(seqLength(gridTHW) * rotDim)
        for (t, gh, gw) in gridTHW {
            let (mh, mw) = (gh / config.merge, gw / config.merge)
            for _ in 0..<t {
                for br in 0..<mh {
                    for bc in 0..<mw {
                        for ir in 0..<config.merge {
                            for ic in 0..<config.merge {
                                let row = Float(br * config.merge + ir)
                                let col = Float(bc * config.merge + ic)
                                for j in 0..<freqCount { rot.append(row * invFreq[j]) }
                                for j in 0..<freqCount { rot.append(col * invFreq[j]) }
                            }
                        }
                    }
                }
            }
        }
        let seq = rot.count / rotDim
        let rotA = MLXArray(rot, [seq, rotDim])
        let emb = concatenated([rotA, rotA], axis: -1)  // [seq, headDim]
        return (cos(emb), sin(emb))
    }

    /// Bilinear-interpolate the 48×48 learned position grid to each image's (h,w), then
    /// reorder into merge-block order to match the patch sequence.
    func positionEmbeddings(gridTHW: [(Int, Int, Int)]) -> MLXArray {
        let side = config.gridSide
        var perImage: [MLXArray] = []
        for (t, gh, gw) in gridTHW {
            let hIdx = linspace(0, side - 1, count: gh)
            let wIdx = linspace(0, side - 1, count: gw)
            let hf = hIdx.map { Int($0) }
            let wf = wIdx.map { Int($0) }
            let hc = hf.map { min($0 + 1, side - 1) }
            let wc = wf.map { min($0 + 1, side - 1) }
            let dh = zip(hIdx, hf).map { $0 - Float($1) }
            let dw = zip(wIdx, wf).map { $0 - Float($1) }

            var idx0: [Int32] = [], idx1: [Int32] = [], idx2: [Int32] = [], idx3: [Int32] = []
            var w0: [Float] = [], w1: [Float] = [], w2: [Float] = [], w3: [Float] = []
            idx0.reserveCapacity(gh * gw)
            for hi in 0..<gh {
                for wi in 0..<gw {
                    idx0.append(Int32(hf[hi] * side + wf[wi]))
                    idx1.append(Int32(hf[hi] * side + wc[wi]))
                    idx2.append(Int32(hc[hi] * side + wf[wi]))
                    idx3.append(Int32(hc[hi] * side + wc[wi]))
                    w0.append((1 - dh[hi]) * (1 - dw[wi]))
                    w1.append((1 - dh[hi]) * dw[wi])
                    w2.append(dh[hi] * (1 - dw[wi]))
                    w3.append(dh[hi] * dw[wi])
                }
            }
            let n = gh * gw
            let weight = posEmbed.weight  // [numPositions, hidden]
            func gather(_ idx: [Int32], _ wt: [Float]) -> MLXArray {
                weight.take(MLXArray(idx), axis: 0) * MLXArray(wt, [n, 1])
            }
            var patch = gather(idx0, w0) + gather(idx1, w1) + gather(idx2, w2) + gather(idx3, w3)

            // Reorder full-res (hi,wi) rows into merge-block (br,bc,ir,ic) order.
            let (mh, mw) = (gh / config.merge, gw / config.merge)
            var perm: [Int32] = []
            perm.reserveCapacity(n)
            for br in 0..<mh {
                for bc in 0..<mw {
                    for ir in 0..<config.merge {
                        for ic in 0..<config.merge {
                            let hi = br * config.merge + ir
                            let wi = bc * config.merge + ic
                            perm.append(Int32(hi * gw + wi))
                        }
                    }
                }
            }
            patch = patch.take(MLXArray(perm), axis: 0)
            if t > 1 { patch = tiled(patch, repetitions: [t, 1]) }
            perImage.append(patch)
        }
        return perImage.count == 1 ? perImage[0] : concatenated(perImage, axis: 0)
    }

    private func seqLength(_ gridTHW: [(Int, Int, Int)]) -> Int {
        gridTHW.reduce(0) { $0 + $1.0 * $1.1 * $1.2 }
    }

    /// `count` evenly spaced points from 0…`hi` inclusive (torch.linspace semantics).
    private func linspace(_ lo: Int, _ hi: Int, count: Int) -> [Float] {
        guard count > 1 else { return [Float(lo)] }
        let step = Float(hi - lo) / Float(count - 1)
        return (0..<count).map { Float(lo) + Float($0) * step }
    }

    // MARK: Load

    /// Load `model.visual.*` from the Qwen3.5 file (int8·convrot dequantised by `WeightLoader`): f32
    /// for parity, bf16 for inference. The 5-D `patch_embed.proj.weight` is folded to a linear.
    public static func load(url: URL, config: QwenVisionConfig = .init(), dtype: DType = .bfloat16) throws -> QwenVisionTower {
        let tower = QwenVisionTower(config: config)
        var weights = try WeightLoader.load(url: url, prefix: "model.visual.", working: dtype).arrays
        guard let pe = weights["patch_embed.proj.weight"] else {
            throw WeightLoaderError.missingTensors(["model.visual.patch_embed.proj.weight"])
        }
        weights["patch_embed.proj.weight"] = pe.reshaped([config.hidden, -1])  // [1152,3,2,16,16] → [1152,1536]
        try tower.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return tower
    }
}
