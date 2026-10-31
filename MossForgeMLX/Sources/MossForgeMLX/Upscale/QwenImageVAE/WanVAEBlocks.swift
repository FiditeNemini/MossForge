
import Foundation
import MLX
import MLXNN

/// Wan VAE channel RMS norm. Normalizes each location over the **channel** dim then
/// scales by `sqrt(C)` and the learned `gamma` — equivalent to `x / RMS_C(x) * gamma`.
/// Operates on `NCDHW` (channel axis 1); `gamma` is loaded as `[C,1,1,1]`.
public struct WanRMSNorm {
    public let gamma: MLXArray  // reshaped to [1,C,1,1,1]
    public let scale: Float
    private let eps: Float = 1e-12

    public init(gamma: MLXArray) {
        let c = gamma.shape[0]
        self.gamma = gamma.reshaped([1, c, 1, 1, 1])
        self.scale = Float(c).squareRoot()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // F.normalize(x, dim=1): x / max(||x||_2 over C, eps)
        let norm = sqrt((x * x).sum(axis: 1, keepDims: true))
        return x / maximum(norm, MLXArray(eps)) * scale * gamma
    }
}

/// A Wan VAE residual block: `conv6(silu(norm3(conv2(silu(norm0(x)))))) + shortcut(x)`.
/// Child index map (from the weights): 0 RMSNorm · 1 SiLU · 2 CausalConv3d · 3 RMSNorm
/// · 4 SiLU · 5 Dropout(identity) · 6 CausalConv3d. `shortcut` is a 1×1×1 CausalConv3d
/// when in≠out channels, else identity.
public struct ResidualBlock3D {
    public let norm0: WanRMSNorm
    public let conv2: CausalConv3d
    public let norm3: WanRMSNorm
    public let conv6: CausalConv3d
    public let shortcut: CausalConv3d?

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv2(silu(norm0(x)))
        h = conv6(silu(norm3(h)))
        return h + (shortcut?(x) ?? x)
    }
}
