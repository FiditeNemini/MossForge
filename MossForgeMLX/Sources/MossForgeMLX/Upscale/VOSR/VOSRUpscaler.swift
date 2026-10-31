
import AppKit
import Foundation
import ImageIO
import MossForgeCore
import MossForgeEngine
import MLX
import MLXRandom

/// An interleaved 8-bit RGB image, read and written without colour management so the bytes
/// match what PIL sees (`Image.open(...).convert("RGB")`).
public struct RGB8: Sendable {
    public var width: Int
    public var height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public static func load(url: URL) throws -> RGB8 {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw EngineError.backendFailure("cannot decode image: \(url.lastPathComponent)") }
        let (w, h) = (img.width, img.height)
        // Fast path: 8-bit RGB/RGBA straight from the decoder, alpha dropped like PIL's convert.
        let alpha = img.alphaInfo
        let rgbaOrder = img.bitmapInfo.intersection(.byteOrderMask) == [] || img.bitmapInfo.contains(.byteOrder32Big)
        if img.bitsPerComponent == 8, img.colorSpace?.model == .rgb, rgbaOrder,
            [.none, .last, .noneSkipLast, .premultipliedLast].contains(alpha),
            let data = img.dataProvider?.data, let base = CFDataGetBytePtr(data)
        {
            let bpp = img.bitsPerPixel / 8
            if bpp == 3 || bpp == 4 {
                var out = [UInt8](repeating: 0, count: w * h * 3)
                for y in 0..<h {
                    let row = base + y * img.bytesPerRow
                    for x in 0..<w {
                        let p = row + x * bpp, o = (y * w + x) * 3
                        out[o] = p[0]; out[o + 1] = p[1]; out[o + 2] = p[2]
                    }
                }
                return RGB8(width: w, height: h, pixels: out)
            }
        }
        // Anything else (16-bit, palette, grey, BGRA): let CoreGraphics convert.
        var buffer = [UInt8](repeating: 0, count: w * h * 4)
        buffer.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: img.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) { out[i * 3] = buffer[i * 4]; out[i * 3 + 1] = buffer[i * 4 + 1]; out[i * 3 + 2] = buffer[i * 4 + 2] }
        return RGB8(width: w, height: h, pixels: out)
    }

    public func savePNG(to url: URL) throws {
        var bytes = pixels
        guard let rep = bytes.withUnsafeMutableBufferPointer({ ptr -> NSBitmapImageRep? in
            var plane: UnsafeMutablePointer<UInt8>? = ptr.baseAddress
            return NSBitmapImageRep(
                bitmapDataPlanes: &plane, pixelsWide: width, pixelsHigh: height,
                bitsPerSample: 8, samplesPerPixel: 3, hasAlpha: false, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: width * 3, bitsPerPixel: 24)
        }), let png = rep.representation(using: .png, properties: [:]) else {
            throw EngineError.backendFailure("PNG encode failed")
        }
        try png.write(to: url)
    }

    /// `[1, 3, H, W]` float in `[-1, 1]` (`x / 127.5 − 1`).
    func nchw() -> MLXArray {
        MLXArray(pixels, [height, width, 3]).asType(.float32).transposed(2, 0, 1).expandedDimensions(axis: 0) / 127.5 - 1
    }

    /// From `[1, 3, H, W]` in `[-1, 1]`: `round(clip(x·0.5 + 0.5, 0, 1)·255)` — the reference's quantisation.
    static func from(_ x: MLXArray) -> RGB8 {
        let v = (clip(x[0] * 0.5 + 0.5, min: 0, max: 1) * 255).round().transposed(1, 2, 0).asType(.uint8)
        eval(v)
        return RGB8(width: x.dim(3), height: x.dim(2), pixels: v.asArray(UInt8.self))
    }
}

/// VOSR2 one-step generative super-resolution (pixel in, pixel out), the MLX port of Forge's
/// `VOSRPipeline.upscale`:
///  1. soften the input's sampler grain (bilateral), PIL-bicubic both copies to the output size;
///  2. Qwen-Image VAE encode → normalised latent (tiled, feathered) — VOSR's
///     "2D" VAE is the same weights at the causal conv's last temporal slice;
///  3. per square latent tile: DINOv2-L layer-17 tokens of the matching pixel crop, one DiT step
///     from noise (t = 1 → 0), velocities feathered across tiles;
///  4. VAE decode (tiled) → wavelet colour fix against the unsoftened bicubic input.
public final class VOSRUpscaler {
    public let dit: LightningDiT
    public let dinov2: DINOv2
    public let vaeEncoder: WanVAEEncoder
    public let vaeDecoder: WanVAEDecoder
    static var mean: MLXArray { MLXArray(QwenImageLatent.mean).reshaped([1, 16, 1, 1]) }
    static var std: MLXArray { MLXArray(QwenImageLatent.std).reshaped([1, 16, 1, 1]) }

    public init(dit: LightningDiT, dinov2: DINOv2, vaeEncoder: WanVAEEncoder, vaeDecoder: WanVAEDecoder) {
        self.dit = dit
        self.dinov2 = dinov2
        self.vaeEncoder = vaeEncoder
        self.vaeDecoder = vaeDecoder
    }

    /// Load from a models folder (`UpscaleModels.vosrFiles`): the VOSR2 DiT, DINOv2-L and the
    /// Qwen-Image VAE, all under `vosr/`.
    public static func load(modelsDirectory dir: URL, ditDType: DType = .float32) throws -> VOSRUpscaler {
        let vae = dir.appendingPathComponent(UpscaleModels.vosrVAE)
        return VOSRUpscaler(
            dit: try LightningDiT.load(url: dir.appendingPathComponent(UpscaleModels.vosrDiT), dtype: ditDType),
            dinov2: try DINOv2.load(url: dir.appendingPathComponent(UpscaleModels.vosrDINOv2)),
            vaeEncoder: try WanVAEEncoder.load(url: vae),
            vaeDecoder: try WanVAEDecoder.load(url: vae))
    }

    // MARK: stages

    /// Normalised latent of `x` `[1,3,H,W]` in `[-1,1]` (H, W multiples of 8).
    func encode(_ x: MLXArray, tilePixels: Int) -> MLXArray {
        let (lh, lw) = (x.dim(2) / 8, x.dim(3) / 8)
        func enc(_ t: MLXArray) -> MLXArray { Self.latent(t, encoder: vaeEncoder) }
        let p = UpscaleTiling.vaeTile(tilePixels: tilePixels, overlapPixels: tilePixels / 8, latentH: lh, latentW: lw)
        guard p.tiled else { return enc(x) }
        return blend(height: lh, width: lw, channels: 16, tile: p.tile, overlap: p.overlap, scale: 1) { hi, wi in
            enc(x[0..., 0..., (hi * 8)..<((hi + p.tile) * 8), (wi * 8)..<((wi + p.tile) * 8)])
        }
    }

    /// Image `[1,3,8h,8w]` in `[-1,1]` from a normalised latent.
    func decode(_ z: MLXArray, tilePixels: Int) -> MLXArray {
        let (lh, lw) = (z.dim(2), z.dim(3))
        func dec(_ t: MLXArray) -> MLXArray {
            vaeDecoder((t * Self.std + Self.mean).expandedDimensions(axis: 2)).squeezed(axis: 2)
        }
        let p = UpscaleTiling.vaeTile(tilePixels: tilePixels, overlapPixels: tilePixels / 8, latentH: lh, latentW: lw)
        guard p.tiled else { return dec(z) }
        let out = blend(height: lh * 8, width: lw * 8, channels: 3, tile: p.tile * 8, overlap: p.overlap * 8, scale: 8, gridLength: (lh, lw), gridTile: p.tile, gridOverlap: p.overlap) { hi, wi in
            dec(z[0..., 0..., hi..<(hi + p.tile), wi..<(wi + p.tile)])
        }
        return clip(out, min: -1, max: 1)
    }

    /// One-step flow from noise to the SR latent over square latent tiles, `batch` tiles per
    /// DiT/DINOv2 call. Batching changes nothing numerically (rows are independent); it only
    /// keeps the GPU busier, since a 512 px tile is ~1k tokens.
    func restore(
        lqLatent: MLXArray, lqPixels: MLXArray, noise: MLXArray, tilePixels: Int, overlapPixels: Int,
        batch: Int = 1, progress: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        let (lh, lw) = (lqLatent.dim(2), lqLatent.dim(3))
        let p = UpscaleTiling.ditTile(
            tilePixels: tilePixels, overlapPixels: overlapPixels, latentH: lh, latentW: lw, patch: dit.config.patch)
        let rows = UpscaleTiling.grid(length: lh, tile: p.tile, overlap: p.overlap)
        let cols = UpscaleTiling.grid(length: lw, tile: p.tile, overlap: p.overlap)
        let total = rows.count * cols.count
        var done = 0
        let f = UpscaleTiling.feather(tile: p.tile, overlap: p.overlap)
        let g = MLXArray(f, [p.tile, 1]) * MLXArray(f, [1, p.tile])
        var acc = MLXArray.zeros([1, 16, lh, lw])
        var wacc = MLXArray.zeros([1, 1, lh, lw])
        for group in UpscaleTiling.batches(rows: rows, cols: cols, size: batch) {
            if Task.isCancelled { throw EngineError.cancelled }
            let crops = group.map { hi, wi in
                lqPixels[0..., 0..., (hi * 8)..<((hi + p.tile) * 8), (wi * 8)..<((wi + p.tile) * 8)]
            }
            let inputs = group.map { hi, wi in
                concatenated([
                    lqLatent[0..., 0..., hi..<(hi + p.tile), wi..<(wi + p.tile)],
                    noise[0..., 0..., hi..<(hi + p.tile), wi..<(wi + p.tile)],
                ], axis: 1)
            }
            let v = dit(concatenated(inputs, axis: 0), t: 1, features: dinov2.features(crops: crops))
            for (i, (hi, wi)) in group.enumerated() {
                let pad = [(0, 0), (0, 0), (hi, lh - hi - p.tile), (wi, lw - wi - p.tile)].map { IntOrPair($0) }
                acc = acc + padded(v[i ..< (i + 1)].asType(.float32) * g, widths: pad)
                wacc = wacc + padded(g.reshaped([1, 1, p.tile, p.tile]), widths: pad)
            }
            eval(acc, wacc)
            done += group.count
            progress?(done, total)
        }
        return noise - acc / wacc  // z0 = z1 − (1 − 0)·u
    }

    /// Normalised latent `(μ − mean) / std` of one image or tile `[1,3,H,W]` in `[-1,1]`.
    public static func latent(_ x: MLXArray, encoder: WanVAEEncoder) -> MLXArray {
        (encoder(x.expandedDimensions(axis: 2)).squeezed(axis: 2) - mean) / std
    }

    // MARK: tiled blending

    private func blend(
        height: Int, width: Int, channels: Int, tile: Int, overlap: Int, scale: Int,
        gridLength: (Int, Int)? = nil, gridTile: Int? = nil, gridOverlap: Int? = nil,
        _ body: (Int, Int) -> MLXArray
    ) -> MLXArray {
        let gl = gridLength ?? (height, width)
        let gt = gridTile ?? tile, go = gridOverlap ?? overlap
        let rows = UpscaleTiling.grid(length: gl.0, tile: gt, overlap: go)
        let cols = UpscaleTiling.grid(length: gl.1, tile: gt, overlap: go)
        return try! blendThrowing(
            height: height, width: width, channels: channels, tile: tile, overlap: overlap,
            rows: rows.map { $0 * scale }, cols: cols.map { $0 * scale }) { h, w in body(h / scale, w / scale) }
    }

    /// Accumulate `tile·feather` and `feather` on the CPU-side grid; each tile result is
    /// evaluated before the next so peak memory stays one tile.
    private func blendThrowing(
        height: Int, width: Int, channels: Int, tile: Int, overlap: Int, rows: [Int], cols: [Int],
        _ body: (Int, Int) throws -> MLXArray
    ) throws -> MLXArray {
        let f = UpscaleTiling.feather(tile: tile, overlap: overlap)
        let g = MLXArray(f, [tile, 1]) * MLXArray(f, [1, tile])  // [tile, tile]
        var acc = MLXArray.zeros([1, channels, height, width])
        var wacc = MLXArray.zeros([1, 1, height, width])
        for hi in rows {
            for wi in cols {
                let t = try body(hi, wi).asType(.float32) * g
                let pad = [(0, 0), (0, 0), (hi, height - hi - tile), (wi, width - wi - tile)].map { IntOrPair($0) }
                acc = acc + padded(t, widths: pad)
                wacc = wacc + padded(g.reshaped([1, 1, tile, tile]), widths: pad)
                eval(acc, wacc)
            }
        }
        return acc / wacc
    }

    // MARK: colour fix

    /// Separable Gaussian (OpenCV float path: 4σ·2+1 taps, reflect-101) over `[C, H, W]`.
    static func gaussianBlur(_ x: MLXArray, sigma: Double) -> MLXArray {
        let k = ImageFilters.gaussianKernel(sigma: sigma, eightBit: false).map(Float.init)
        let r = k.count / 2
        func pass(_ x: MLXArray, axis: Int) -> MLXArray {
            let n = x.dim(axis)
            let idx = MLXArray((-r..<(n + r)).map { Int32(ImageFilters.reflect101($0, n)) })
            let p = x.take(idx, axis: axis)
            var out = MLXArray.zeros(x.shape)
            for (t, w) in k.enumerated() {
                let s = axis == 1 ? p[0..., t..<(t + n), 0...] : p[0..., 0..., t..<(t + n)]
                out = out + w * s
            }
            return out
        }
        return pass(pass(x, axis: 2), axis: 1)
    }

    /// Keep `target`'s high frequencies, take `source`'s low ones (σ = 5 Gaussian).
    static func waveletColorFix(target: RGB8, source: RGB8) -> RGB8 {
        func chw(_ i: RGB8) -> MLXArray { MLXArray(i.pixels, [i.height, i.width, 3]).asType(.float32).transposed(2, 0, 1) / 255 }
        let t = chw(target), s = chw(source)
        let out = clip(gaussianBlur(s, sigma: 5) + (t - gaussianBlur(t, sigma: 5)), min: 0, max: 1) * 255
        let v = out.round().transposed(1, 2, 0).asType(.uint8)
        eval(v)
        return RGB8(width: target.width, height: target.height, pixels: v.asArray(UInt8.self))
    }

    /// Match per-channel mean/std to `source` (AdaIN).
    static func adainColorFix(target: RGB8, source: RGB8) -> RGB8 {
        func hwc(_ i: RGB8) -> MLXArray { MLXArray(i.pixels, [i.height * i.width, 3]).asType(.float32) / 255 }
        let t = hwc(target), s = hwc(source)
        let tm = t.mean(axis: 0), sm = s.mean(axis: 0)
        let ts = t.variance(axis: 0, ddof: 1).sqrt() + 1e-5, ss = s.variance(axis: 0, ddof: 1).sqrt() + 1e-5
        let v = (clip((t - tm) / ts * ss + sm, min: 0, max: 1) * 255).round().asType(.uint8)
        eval(v)
        return RGB8(width: target.width, height: target.height, pixels: v.asArray(UInt8.self))
    }

    // MARK: public

    public struct Intermediates {
        public var lq: RGB8
        public var lqLatent: MLXArray
        public var srLatent: MLXArray
        public var sr: RGB8
    }

    /// Upscale `image` to `targetWidth×targetHeight` (rounded down to /8).
    /// - Parameter noise: override for parity runs; otherwise drawn from `seed`.
    public func upscale(
        _ image: RGB8, targetWidth: Int, targetHeight: Int, settings: UpscaleSettings, noise: MLXArray? = nil,
        intermediates: ((Intermediates) -> Void)? = nil, progress: ((Int, Int) -> Void)? = nil
    ) throws -> RGB8 {
        let tw = max(targetWidth / 8 * 8, 8 * dit.config.patch)
        let th = max(targetHeight / 8 * 8, 8 * dit.config.patch)
        func resized(_ px: [UInt8]) -> RGB8 {
            RGB8(width: tw, height: th, pixels: PILResample.bicubic(
                px, width: image.width, height: image.height, channels: 3, toWidth: tw, toHeight: th))
        }
        let ref = resized(image.pixels)
        let lq = resized(settings.soften.apply(image.pixels, width: image.width, height: image.height))
        let lqPixels = lq.nchw()
        let lqLatent = encode(lqPixels, tilePixels: settings.vaeTile)
        eval(lqLatent)
        let z1 = noise ?? MLXRandom.normal(lqLatent.shape, key: MLXRandom.key(settings.seed))
        let sr = try restore(
            lqLatent: lqLatent, lqPixels: lqPixels, noise: z1, tilePixels: settings.tile,
            overlapPixels: settings.overlap, batch: settings.batch, progress: progress)
        let decoded = RGB8.from(decode(sr, tilePixels: settings.vaeTile))
        intermediates?(Intermediates(lq: lq, lqLatent: lqLatent, srLatent: sr, sr: decoded))
        switch settings.colorFix {
        case .wavelet: return Self.waveletColorFix(target: decoded, source: ref)
        case .adain: return Self.adainColorFix(target: decoded, source: ref)
        case .none: return decoded
        }
    }
}

/// The Qwen-Image VAE's per-channel latent statistics (diffusers `AutoencoderKLQwenImage`
/// `latents_mean` / `latents_std`): VOSR2 works on `(μ − mean) / std`.
public enum QwenImageLatent {
    public static let mean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    public static let std: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ]
}
