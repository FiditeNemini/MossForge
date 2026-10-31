import AppKit
import SwiftUI

/// Shows a generated image from disk when available, else a seeded placeholder swatch.
/// Used everywhere a gallery item is rendered (canvas, recent strip, grid, detail).
struct ResultImage: View {
    var url: URL?
    var seed: UInt64
    var cornerRadius: CGFloat = 8
    /// `true` (default) fills + crops the frame (square thumbnails); `false` fits the
    /// whole image at its true aspect ratio (the main canvas).
    var fill: Bool = true

    var body: some View {
        if let url, let img = NSImage(contentsOf: url) {
            Image(nsImage: img)
                .resizable()
                .aspectRatio(contentMode: fill ? .fill : .fit)
                .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
        } else {
            SeedSwatch(seed: seed, cornerRadius: cornerRadius)
        }
    }
}

/// Deterministic seeded placeholder visuals. Until the MLX backend produces real
/// pixels, the app renders a stable per-seed gradient so Generate + Gallery show
/// "images" and the end-to-end flow is visible. Same seed ⇒ same swatch.
enum PlaceholderImage {
    /// Two gradient stops derived from the seed.
    static func colors(seed: UInt64) -> [Color] {
        let hue = Double(seed % 360) / 360.0
        let hue2 = (hue + 0.12).truncatingRemainder(dividingBy: 1.0)
        return [
            Color(hue: hue, saturation: 0.45, brightness: 0.80),
            Color(hue: hue2, saturation: 0.55, brightness: 0.55),
        ]
    }
}

/// A seeded gradient swatch standing in for a generated image.
struct SeedSwatch: View {
    var seed: UInt64
    var cornerRadius: CGFloat = 8

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius)
            .fill(
                LinearGradient(
                    colors: PlaceholderImage.colors(seed: seed),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .overlay(
                Image(systemName: "mountain.2")
                    .font(.system(size: 22))
                    .foregroundStyle(.white.opacity(0.55))
            )
    }
}
