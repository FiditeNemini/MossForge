
import Foundation
import MossForgeCore
import MossForgeEngine
import MLX
import MLXFast
import MLXNN

/// DINOv2-L/14 (`facebook/dinov2-large`, Apache-2.0) — just enough of it for VOSR's
/// conditioning: the patch tokens after block `outputLayer` (VOSR2 uses 17 of 24, so the last
/// six blocks are never run). Independent MLX port of transformers' `Dinov2Model`, with
/// torch.hub's position-embedding resize (scale factor `(n + 0.1) / 37`), which VOSR was
/// trained with and which changes layer-17 features by ~3% vs transformers' exact resize.
/// Verified against `Scripts/vosr_ref.py` via `moss-mlx vosr-parity`.
public struct DINOv2 {
    public static let hidden = 1024
    public static let heads = 16
    public static let patch = 14
    /// VOSR feeds 448×448 crops: a 32×32 patch grid.
    public static let inputSize = 448
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    private let w: [String: MLXArray]
    private let outputLayer: Int
    /// `[1, 1 + grid², hidden]` position embeddings already resized for `inputSize`.
    public let positions: MLXArray
    private let patchWeight: MLXArray

    public init(weights: [String: MLXArray], outputLayer: Int = 17) {
        w = weights
        self.outputLayer = outputLayer
        patchWeight = weights["embeddings.patch_embeddings.projection.weight"]!.reshaped([Self.hidden, -1])
        positions = Self.resizePositions(weights["embeddings.position_embeddings"]!, grid: Self.inputSize / Self.patch)
    }

    /// Load from a transformers `model.safetensors` (fp32).
    public static func load(url: URL, outputLayer: Int = 17, dtype: DType = .float32) throws -> DINOv2 {
        let raw = try MLX.loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in raw {
            // Blocks past the output layer never run; skip their memory.
            if k.hasPrefix("encoder.layer."), let n = Int(k.split(separator: ".")[2]), n > outputLayer { continue }
            weights[k] = v.asType(dtype)
        }
        for k in ["embeddings.cls_token", "embeddings.position_embeddings",
            "embeddings.patch_embeddings.projection.weight", "encoder.layer.\(outputLayer).mlp.fc2.weight"]
        where weights[k] == nil {
            throw EngineError.backendFailure("DINOv2: \(url.lastPathComponent) has no \(k)")
        }
        return DINOv2(weights: weights, outputLayer: outputLayer)
    }

    /// torch.hub `interpolate_pos_encoding`: bicubic, `scale_factor = (grid + 0.1) / 37`.
    static func resizePositions(_ pos: MLXArray, grid: Int) -> MLXArray {
        let n = pos.dim(1) - 1
        let m = Int(Double(n).squareRoot())
        if m == grid { return pos }
        let mat = MLXArray(TorchBicubic.matrix(inSize: m, outSize: grid, scaleFactor: (Double(grid) + 0.1) / Double(m)), [grid, m])
        let p = pos[0, 1...].asType(.float32).reshaped([m, m, hidden])  // rows, cols, C
        // rows: [grid, m] · [m, m·C] ; cols: same on the transposed grid.
        var r = mat.matmul(p.reshaped([m, m * hidden])).reshaped([grid, m, hidden])
        r = r.transposed(1, 0, 2).reshaped([m, grid * hidden])
        r = mat.matmul(r).reshaped([grid, grid, hidden]).transposed(1, 0, 2)
        return concatenated([pos[0..., ..<1].asType(.float32), r.reshaped([1, grid * grid, hidden])], axis: 1)
            .asType(pos.dtype)
    }

    /// Normalised pixels for one crop: `[1,3,h,w]` in `[-1,1]` → resized to 448 (torch bicubic,
    /// clamped) → ImageNet-normalised `[1,3,448,448]`.
    public static func preprocess(_ crop: MLXArray) -> MLXArray {
        let (h, wd) = (crop.dim(2), crop.dim(3))
        var x = clip(crop.asType(.float32) * 0.5 + 0.5, min: 0, max: 1)
        let wh = MLXArray(TorchBicubic.matrix(inSize: h, outSize: inputSize), [inputSize, h])
        let ww = MLXArray(TorchBicubic.matrix(inSize: wd, outSize: inputSize), [inputSize, wd])
        x = wh.matmul(x[0]).matmul(ww.transposed())  // [3, 448, 448]
        x = clip(x, min: 0, max: 1)
        let m = MLXArray(mean, [3, 1, 1]), s = MLXArray(std, [3, 1, 1])
        return ((x - m) / s).expandedDimensions(axis: 0)
    }

    /// Patch tokens after block `outputLayer` (no CLS): `[1, grid², hidden]`.
    public func features(_ crop: MLXArray) -> MLXArray { features(crops: [crop]) }

    /// The same for several crops in one pass: `[B, grid², hidden]`, row `i` from `crops[i]`.
    public func features(crops: [MLXArray]) -> MLXArray {
        let x = concatenated(crops.map(Self.preprocess), axis: 0).asType(patchWeight.dtype)  // [B,3,448,448]
        let b = crops.count, g = Self.inputSize / Self.patch
        // Conv(14, stride 14) as a linear over (c, kh, kw) patches.
        let patches = x.reshaped([b, 3, g, Self.patch, g, Self.patch]).transposed(0, 2, 4, 1, 3, 5)
            .reshaped([b, g * g, 3 * Self.patch * Self.patch])
        var h = linear(patches, patchWeight, w["embeddings.patch_embeddings.projection.bias"])
        let cls = w["embeddings.cls_token"]!
        h = concatenated([broadcast(cls, to: [b, 1, cls.dim(2)]), h], axis: 1) + positions
        for i in 0...outputLayer { h = block(h, "encoder.layer.\(i)") }
        return h[0..., 1...]
    }

    private func linear(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray?) -> MLXArray {
        let y = x.matmul(weight.transposed())
        return bias.map { y + $0 } ?? y
    }

    private func linear(_ x: MLXArray, _ p: String) -> MLXArray {
        linear(x, w["\(p).weight"]!, w["\(p).bias"])
    }

    private func norm(_ x: MLXArray, _ p: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: w["\(p).weight"], bias: w["\(p).bias"], eps: 1e-6)
    }

    private func block(_ x: MLXArray, _ p: String) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), hd = Self.hidden / Self.heads
        let y = norm(x, "\(p).norm1")
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped([b, n, Self.heads, hd]).transposed(0, 2, 1, 3) }
        let q = heads(linear(y, "\(p).attention.attention.query"))
        let k = heads(linear(y, "\(p).attention.attention.key"))
        let v = heads(linear(y, "\(p).attention.attention.value"))
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(hd).squareRoot(), mask: .none)
            .transposed(0, 2, 1, 3).reshaped([b, n, Self.hidden])
        var h = x + w["\(p).layer_scale1.lambda1"]! * linear(o, "\(p).attention.output.dense")
        let m = linear(gelu(linear(norm(h, "\(p).norm2"), "\(p).mlp.fc1")), "\(p).mlp.fc2")
        h = h + w["\(p).layer_scale2.lambda1"]! * m
        return h
    }
}
