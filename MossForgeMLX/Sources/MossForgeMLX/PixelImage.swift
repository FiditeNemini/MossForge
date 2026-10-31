import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// PNG/JPEG ↔ `MLXArray` `[1,3,H,W]` F32 in [0,1] (sRGB, alpha dropped) — the VAE's pixel layout.
public enum PixelImage {
    public enum Failure: Error, CustomStringConvertible {
        case unreadable(URL), unwritable(URL)
        public var description: String {
            switch self {
            case let .unreadable(u): "can't read image \(u.path)"
            case let .unwritable(u): "can't write image \(u.path)"
            }
        }
    }

    /// Load an image, centre-cropped so both sides are multiples of `multiple` (16 = one DiT token).
    public static func load(_ url: URL, multiple: Int = 16) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw Failure.unreadable(url) }
        let (w, h) = (image.width / multiple * multiple, image.height / multiple * multiple)
        guard w > 0, h > 0 else { throw Failure.unreadable(url) }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard
                let ctx = CGContext(
                    data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            // Centre crop: draw at a negative offset so the excess falls off both edges.
            let dx = -CGFloat(image.width - w) / 2, dy = -CGFloat(image.height - h) / 2
            ctx.draw(image, in: CGRect(x: dx, y: dy, width: CGFloat(image.width), height: CGFloat(image.height)))
            return true
        }
        guard drawn else { throw Failure.unreadable(url) }
        let hwc = MLXArray(rgba, [h, w, 4]).asType(.float32) / 255
        return hwc[0..., 0..., ..<3].transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    /// Load an image scaled to cover `width`×`height` (high-quality interpolation) and centre-cropped to
    /// exactly that size — a training bucket.
    public static func load(_ url: URL, width w: Int, height h: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(src, 0, nil), w > 0, h > 0
        else { throw Failure.unreadable(url) }
        let scale = max(CGFloat(w) / CGFloat(image.width), CGFloat(h) / CGFloat(image.height))
        let (dw, dh) = (CGFloat(image.width) * scale, CGFloat(image.height) * scale)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard
                let ctx = CGContext(
                    data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: (CGFloat(w) - dw) / 2, y: (CGFloat(h) - dh) / 2, width: dw, height: dh))
            return true
        }
        guard drawn else { throw Failure.unreadable(url) }
        let hwc = MLXArray(rgba, [h, w, 4]).asType(.float32) / 255
        return hwc[0..., 0..., ..<3].transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    /// Image size in pixels without decoding it.
    public static func size(of url: URL) -> (width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
            let w = props[kCGImagePropertyPixelWidth] as? Int, let h = props[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        let o = props[kCGImagePropertyOrientation] as? Int ?? 1
        return o >= 5 ? (h, w) : (w, h)
    }

    /// Save `[1,3,H,W]` (or `[3,H,W]`) in [0,1] as a PNG.
    public static func save(_ pixels: MLXArray, to url: URL) throws {
        do { try pngData(pixels).write(to: url) } catch { throw Failure.unwritable(url) }
    }

    /// Encode `[1,3,H,W]` (or `[3,H,W]`) in [0,1] as PNG bytes (sRGB, opaque).
    public static func pngData(_ pixels: MLXArray) throws -> Data {
        let chw = pixels.ndim == 4 ? pixels[0] : pixels
        let (h, w) = (chw.dim(1), chw.dim(2))
        let rgb = (clip(chw.asType(.float32), min: 0, max: 1) * 255).round().asType(.uint8).transposed(1, 2, 0)
        let alpha = MLXArray.full([h, w, 1], values: MLXArray(UInt8(255)), type: UInt8.self)
        var bytes = concatenated([rgb, alpha], axis: 2).asArray(UInt8.self)
        let image: CGImage? = bytes.withUnsafeMutableBytes { buf in
            CGContext(
                data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )?.makeImage()
        }
        let data = NSMutableData()
        guard let image,
            let dest = CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil)
        else { throw Failure.unwritable(URL(fileURLWithPath: "<memory>")) }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw Failure.unwritable(URL(fileURLWithPath: "<memory>")) }
        return data as Data
    }

    /// Peak signal-to-noise ratio (dB) between two [0,1] images.
    public static func psnr(_ a: MLXArray, _ b: MLXArray) -> Float {
        let mse = ((a.asType(.float32) - b.asType(.float32)).square()).mean().item(Float.self)
        return -10 * log10(max(mse, 1e-12))
    }
}
