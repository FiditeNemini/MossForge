
import MossForgeCore
import Foundation

/// The OpenCV filters VOSR's input softening uses (`cv2.bilateralFilter`, `cv2.GaussianBlur`),
/// on interleaved 8-bit RGB. Few-step generations carry fine sampler/VAE grain that VOSR misreads
/// as fabric-weave or scale texture on skin; a light edge-preserving pre-filter removes it.
public enum ImageFilters {
    /// `BORDER_REFLECT_101`: …2 1 | 0 1 2 … n-1 | n-2 …
    @inline(__always) public static func reflect101(_ i: Int, _ n: Int) -> Int {
        guard n > 1 else { return 0 }
        var i = i
        while i < 0 || i >= n { i = i < 0 ? -i : 2 * n - 2 - i }
        return i
    }

    /// `cv2.bilateralFilter(src, 0, sigmaColor, sigmaSpace)` for 3-channel 8-bit images: a
    /// circular window of radius `round(1.5·σspace)`, colour distance = summed |ΔR|+|ΔG|+|ΔB|.
    public static func bilateral(
        _ src: [UInt8], width: Int, height: Int, sigmaColor: Double, sigmaSpace: Double
    ) -> [UInt8] {
        let cn = 3
        let colorCoeff = -0.5 / (sigmaColor * sigmaColor)
        let spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace)
        let radius = max(Int((sigmaSpace * 1.5).rounded(.toNearestOrEven)), 1)
        let colorWeight = (0..<(256 * cn)).map { Float(exp(Double($0 * $0) * colorCoeff)) }
        var offsets: [(dy: Int, dx: Int, w: Float)] = []
        for i in -radius...radius {
            for j in -radius...radius {
                let r = (Double(i * i + j * j)).squareRoot()
                if r > Double(radius) { continue }
                offsets.append((i, j, Float(exp(r * r * spaceCoeff))))
            }
        }
        var out = [UInt8](repeating: 0, count: src.count)
        for y in 0..<height {
            for x in 0..<width {
                let c = (y * width + x) * cn
                let (b0, g0, r0) = (Int(src[c]), Int(src[c + 1]), Int(src[c + 2]))
                var sum: (Float, Float, Float) = (0, 0, 0)
                var wsum: Float = 0
                for o in offsets {
                    let p = (reflect101(y + o.dy, height) * width + reflect101(x + o.dx, width)) * cn
                    let (b, g, r) = (Int(src[p]), Int(src[p + 1]), Int(src[p + 2]))
                    let w = o.w * colorWeight[abs(b - b0) + abs(g - g0) + abs(r - r0)]
                    sum.0 += Float(b) * w
                    sum.1 += Float(g) * w
                    sum.2 += Float(r) * w
                    wsum += w
                }
                let inv = 1 / wsum
                out[c] = UInt8(clamping: Int((sum.0 * inv).rounded(.toNearestOrEven)))
                out[c + 1] = UInt8(clamping: Int((sum.1 * inv).rounded(.toNearestOrEven)))
                out[c + 2] = UInt8(clamping: Int((sum.2 * inv).rounded(.toNearestOrEven)))
            }
        }
        return out
    }

    /// `cv2.getGaussianKernel` with OpenCV's automatic size for `GaussianBlur(…, (0,0), σ)`:
    /// `round(σ·(8-bit ? 3 : 4)·2 + 1) | 1` taps.
    public static func gaussianKernel(sigma: Double, eightBit: Bool) -> [Double] {
        let n = Int((sigma * (eightBit ? 3 : 4) * 2 + 1).rounded(.toNearestOrEven)) | 1
        let scale = -0.5 / (sigma * sigma)
        let k = (0..<n).map { i -> Double in
            let x = Double(i) - Double(n - 1) * 0.5
            return exp(scale * x * x)
        }
        let sum = k.reduce(0, +)
        return k.map { $0 / sum }
    }

    /// `cv2.GaussianBlur(src, (0,0), σ)` on interleaved 8-bit RGB (separable, reflect-101).
    /// OpenCV's 8-bit path is fixed-point, so results can differ from it by one level.
    public static func gaussian(_ src: [UInt8], width: Int, height: Int, sigma: Double) -> [UInt8] {
        let cn = 3
        let k = gaussianKernel(sigma: sigma, eightBit: true)
        let r = k.count / 2
        var tmp = [Double](repeating: 0, count: src.count)
        for y in 0..<height {
            for x in 0..<width {
                for ch in 0..<cn {
                    var s = 0.0
                    for (t, w) in k.enumerated() {
                        s += w * Double(src[(y * width + reflect101(x + t - r, width)) * cn + ch])
                    }
                    tmp[(y * width + x) * cn + ch] = s
                }
            }
        }
        var out = [UInt8](repeating: 0, count: src.count)
        for y in 0..<height {
            for x in 0..<width {
                for ch in 0..<cn {
                    var s = 0.0
                    for (t, w) in k.enumerated() {
                        s += w * tmp[(reflect101(y + t - r, height) * width + x) * cn + ch]
                    }
                    out[(y * width + x) * cn + ch] = UInt8(clamping: Int(s.rounded(.toNearestOrEven)))
                }
            }
        }
        return out
    }
}

extension UpscaleSoftening {
    /// Forge's presets: light = bilateral(σc 12, σs 1.5), medium = bilateral(20, 1.5),
    /// strong = Gaussian σ 1. Bilateral keeps hair strands and lip lines a blur would soften.
    public func apply(_ rgb: [UInt8], width: Int, height: Int) -> [UInt8] {
        switch self {
        case .off: rgb
        case .light: ImageFilters.bilateral(rgb, width: width, height: height, sigmaColor: 12, sigmaSpace: 1.5)
        case .medium: ImageFilters.bilateral(rgb, width: width, height: height, sigmaColor: 20, sigmaSpace: 1.5)
        case .strong: ImageFilters.gaussian(rgb, width: width, height: height, sigma: 1.0)
        }
    }
}
