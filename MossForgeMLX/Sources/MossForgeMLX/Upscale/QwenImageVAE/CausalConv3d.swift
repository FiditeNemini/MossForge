
import Foundation
import MLX

/// A 3D causal convolution, the core primitive of the Wan-style Qwen Image VAE.
///
/// Temporal padding is **causal** — the front is padded by `kD-1`, the back not at
/// all — so a frame never sees the future; spatial padding is symmetric. MLX convs
/// are channels-last (`NDHWC`, weight `[O,kD,kH,kW,I]`), while the on-disk PyTorch
/// weight is `[O,I,kD,kH,kW]`, so the weight is transposed once at init.
public struct CausalConv3d {
    public let weight: MLXArray  // MLX layout [O, kD, kH, kW, I]
    public let bias: MLXArray?
    public let kD: Int
    public let kH: Int
    public let kW: Int

    /// - Parameter weightPyTorch: weight in PyTorch layout `[O, I, kD, kH, kW]`.
    public init(weightPyTorch: MLXArray, bias: MLXArray? = nil) {
        let s = weightPyTorch.shape
        precondition(s.count == 5, "CausalConv3d expects a 5-D weight [O,I,kD,kH,kW]")
        kD = s[2]
        kH = s[3]
        kW = s[4]
        weight = weightPyTorch.transposed(0, 2, 3, 4, 1)  // → [O,kD,kH,kW,I]
        self.bias = bias
    }

    /// - Parameter x: input in `NCDHW` (channels-first, matching PyTorch latents).
    /// - Returns: output in `NCDHW`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xc = x.transposed(0, 2, 3, 4, 1)  // N C D H W → N D H W C
        let widths: [IntOrPair] = [
            IntOrPair((0, 0)),  // N
            IntOrPair((kD - 1, 0)),  // D — causal: front only
            IntOrPair((kH / 2, kH / 2)),  // H — symmetric
            IntOrPair((kW / 2, kW / 2)),  // W — symmetric
            IntOrPair((0, 0)),  // C
        ]
        let padded = MLX.padded(xc, widths: widths)
        var y = conv3d(padded, weight, stride: 1, padding: 0)
        if let bias { y = y + bias }  // [O] broadcasts over the trailing channel dim
        return y.transposed(0, 4, 1, 2, 3)  // N D H W C → N C D H W
    }
}
