
import Foundation

/// Pillow's `Image.resize(..., BICUBIC)` for 8-bit images, reproduced exactly: the a = −0.5
/// cubic, antialiasing support when shrinking, 22-bit fixed-point coefficients, and a horizontal
/// pass (rounded to 8 bits) before the vertical one. VOSR's input resize goes through PIL, so
/// matching it bit-for-bit keeps the MLX port on the reference's exact pixels.
public enum PILResample {
    static let precisionBits = 32 - 8 - 2

    /// Bicubic resize of an interleaved `width×height×channels` 8-bit image.
    public static func bicubic(
        _ pixels: [UInt8], width: Int, height: Int, channels: Int, toWidth: Int, toHeight: Int
    ) -> [UInt8] {
        var current = pixels
        var w = width
        if toWidth != width {
            current = horizontal(current, width: width, height: height, channels: channels, toWidth: toWidth)
            w = toWidth
        }
        if toHeight != height {
            current = vertical(current, width: w, height: height, channels: channels, toHeight: toHeight)
        }
        return current
    }

    /// Per output index: first input index, tap count, and fixed-point weights.
    struct Coefficients {
        var bounds: [(min: Int, count: Int)]
        var kernels: [[Int]]
    }

    static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    static func coefficients(inSize: Int, outSize: Int) -> Coefficients {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 2.0 * filterScale
        let ss = 1 / filterScale
        var bounds: [(Int, Int)] = []
        var kernels: [[Int]] = []
        let one = Double(1 << precisionBits)
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var k = (0..<xmax).map { cubic((Double($0 + xmin) - center + 0.5) * ss) }
            let ww = k.reduce(0, +)
            if ww != 0 { k = k.map { $0 / ww } }
            kernels.append(k.map { $0 < 0 ? Int(-0.5 + $0 * one) : Int(0.5 + $0 * one) })
            bounds.append((xmin, xmax))
        }
        return Coefficients(bounds: bounds, kernels: kernels)
    }

    @inline(__always) static func clip8(_ v: Int) -> UInt8 {
        let s = v >> precisionBits
        return UInt8(clamping: s)
    }

    static func horizontal(_ src: [UInt8], width: Int, height: Int, channels: Int, toWidth: Int) -> [UInt8] {
        let c = coefficients(inSize: width, outSize: toWidth)
        var out = [UInt8](repeating: 0, count: toWidth * height * channels)
        let half = 1 << (precisionBits - 1)
        for y in 0..<height {
            let row = y * width * channels
            for xx in 0..<toWidth {
                let (xmin, count) = c.bounds[xx]
                let k = c.kernels[xx]
                for ch in 0..<channels {
                    var ss = half
                    for x in 0..<count { ss += Int(src[row + (xmin + x) * channels + ch]) * k[x] }
                    out[(y * toWidth + xx) * channels + ch] = clip8(ss)
                }
            }
        }
        return out
    }

    static func vertical(_ src: [UInt8], width: Int, height: Int, channels: Int, toHeight: Int) -> [UInt8] {
        let c = coefficients(inSize: height, outSize: toHeight)
        var out = [UInt8](repeating: 0, count: width * toHeight * channels)
        let half = 1 << (precisionBits - 1)
        let stride = width * channels
        for yy in 0..<toHeight {
            let (ymin, count) = c.bounds[yy]
            let k = c.kernels[yy]
            for i in 0..<stride {
                var ss = half
                for y in 0..<count { ss += Int(src[(ymin + y) * stride + i]) * k[y] }
                out[yy * stride + i] = clip8(ss)
            }
        }
        return out
    }
}

/// `torch.nn.functional.interpolate(mode="bicubic", align_corners=False, antialias=False)` as a
/// dense `[outSize, inSize]` matrix (row-major), so a 2D resize is `Wh · X · Wwᵀ`. A = −0.75,
/// border taps clamped. With `scaleFactor` the source coordinate uses `1/scaleFactor` (as torch
/// does when it's given), which is how DINOv2's hub position-embedding resize differs.
public enum TorchBicubic {
    public static func matrix(inSize: Int, outSize: Int, scaleFactor: Double? = nil) -> [Float] {
        let scale: Float = scaleFactor.map { Float(1 / $0) } ?? Float(inSize) / Float(outSize)
        let a: Float = -0.75
        func near(_ x: Float) -> Float { ((a + 2) * x - (a + 3)) * x * x + 1 }
        func far(_ x: Float) -> Float { ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        var m = [Float](repeating: 0, count: outSize * inSize)
        for o in 0..<outSize {
            let real = scale * (Float(o) + 0.5) - 0.5
            let index = Int(real.rounded(.down))
            let t = real - Float(index)
            let weights = [far(t + 1), near(t), near(1 - t), far(2 - t)]
            for (k, w) in weights.enumerated() {
                let i = min(max(index - 1 + k, 0), inSize - 1)
                m[o * inSize + i] += w
            }
        }
        return m
    }
}
