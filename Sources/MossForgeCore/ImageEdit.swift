import Foundation

/// Which part of an image to redraw, on the DiT's token grid (16 px cells): 1 = redraw, 0 = keep.
public struct EditMask: Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var values: [Float]

    public init(width: Int, height: Int, values: [Float]) {
        self.width = width
        self.height = height
        self.values = values
    }

    /// Share of the image that is redrawn.
    public var coverage: Double { values.isEmpty ? 0 : Double(values.reduce(0, +)) / Double(values.count) }

    /// Bilinear upsample to pixels (cell centres at `(i + ½)·scale`), so the pasted-back original
    /// fades into the redrawn region over about one cell.
    public func pixels(scale: Int = 16) -> [Float] {
        let (pw, ph) = (width * scale, height * scale)
        var out = [Float](repeating: 0, count: pw * ph)
        func v(_ r: Int, _ c: Int) -> Float { values[min(max(r, 0), height - 1) * width + min(max(c, 0), width - 1)] }
        for y in 0..<ph {
            let fy = (Float(y) + 0.5) / Float(scale) - 0.5
            let r0 = Int(fy.rounded(.down))
            let ty = fy - Float(r0)
            for x in 0..<pw {
                let fx = (Float(x) + 0.5) / Float(scale) - 0.5
                let c0 = Int(fx.rounded(.down))
                let tx = fx - Float(c0)
                let top = v(r0, c0) * (1 - tx) + v(r0, c0 + 1) * tx
                let bottom = v(r0 + 1, c0) * (1 - tx) + v(r0 + 1, c0 + 1) * tx
                out[y * pw + x] = top * (1 - ty) + bottom * ty
            }
        }
        return out
    }
}

extension EditMask {
    public static func empty(width: Int, height: Int) -> EditMask {
        EditMask(width: width, height: height, values: .init(repeating: 0, count: width * height))
    }

    /// A pixel mask (`width`×`height`, row-major, 1 = change) area-averaged onto a token grid.
    public static func resampled(from pixels: [Float], width: Int, height: Int, toWidth tw: Int, toHeight th: Int) -> EditMask {
        var values = [Float](repeating: 0, count: tw * th)
        for r in 0..<th {
            let y0 = r * height / th, y1 = max(y0 + 1, (r + 1) * height / th)
            for c in 0..<tw {
                let x0 = c * width / tw, x1 = max(x0 + 1, (c + 1) * width / tw)
                var sum: Float = 0
                for y in y0..<y1 { for x in x0..<x1 { sum += pixels[y * width + x] } }
                values[r * tw + c] = min(1, max(0, sum / Float((y1 - y0) * (x1 - x0))))
            }
        }
        return EditMask(width: tw, height: th, values: values)
    }

    /// Change ↔ keep (paint what to protect, then invert).
    public func inverted() -> EditMask { EditMask(width: width, height: height, values: values.map { 1 - $0 }) }

    /// A round brush centred at `(x, y)` in token units (cell `c` spans `c…c+1`): cells whose centre is
    /// within `radius` take `value` — painting (1) keeps the max, erasing (0) the min.
    public mutating func paint(atX x: Double, y: Double, radius: Double, value: Float) {
        let r0 = max(0, Int((y - radius).rounded(.down))), r1 = min(height - 1, Int((y + radius).rounded(.up)))
        let c0 = max(0, Int((x - radius).rounded(.down))), c1 = min(width - 1, Int((x + radius).rounded(.up)))
        guard r0 <= r1, c0 <= c1 else { return }
        for r in r0...r1 {
            for c in c0...c1 {
                let (dx, dy) = (Double(c) + 0.5 - x, Double(r) + 0.5 - y)
                guard dx * dx + dy * dy <= radius * radius + 1e-9 else { continue }
                let i = r * width + c
                values[i] = value > 0 ? max(values[i], value) : min(values[i], value)
            }
        }
    }
}

/// Edit in Layout: the original image and the region to redraw.
public struct ImageEditSpec: Sendable, Equatable {
    public var source: URL
    public var mask: EditMask
    public init(source: URL, mask: EditMask) {
        self.source = source
        self.mask = mask
    }
}

/// Turns "what changed on the layout canvas" into the region to redraw.
public enum LayoutEditMask {
    /// Boxes (abs_centralized yxyx) of items that were edited, added or removed — an item that is
    /// identical in both layouts is kept.
    public static func changedBoxes(original: LayoutPrompt, edited: LayoutPrompt) -> [[Int]] {
        var unmatched = original.items
        var boxes: [[Int]] = []
        for item in edited.items {
            if let i = unmatched.firstIndex(of: item) {
                unmatched.remove(at: i)
            } else {
                boxes.append(item.bbox)
            }
        }
        boxes += unmatched.map(\.bbox)  // removed or changed originals: redraw where they were
        var seen = Set<[Int]>()
        return boxes.filter { seen.insert($0).inserted }
    }

    /// Token mask for `boxes` on `canvas`, grown by `margin` cells so the edit blends with its surroundings.
    public static func tokenMask(boxes: [[Int]], canvas: Dimensions, margin: Int = 1) -> EditMask {
        let (w, h) = (canvas.width / 16, canvas.height / 16)
        var values = [Float](repeating: 0, count: w * h)
        for b in boxes where b.count == 4 {
            // yxyx centred on the canvas → pixel rows/cols from the top-left.
            let (y0, x0) = (Double(b[0] + canvas.height / 2), Double(b[1] + canvas.width / 2))
            let (y1, x1) = (Double(b[2] + canvas.height / 2), Double(b[3] + canvas.width / 2))
            let r0 = max(0, Int((min(y0, y1) / 16).rounded(.down)) - margin)
            let r1 = min(h, Int((max(y0, y1) / 16).rounded(.up)) + margin)
            let c0 = max(0, Int((min(x0, x1) / 16).rounded(.down)) - margin)
            let c1 = min(w, Int((max(x0, x1) / 16).rounded(.up)) + margin)
            guard r0 < r1, c0 < c1 else { continue }
            for r in r0..<r1 { for c in c0..<c1 { values[r * w + c] = 1 } }
        }
        return EditMask(width: w, height: h, values: values)
    }

    /// An image's prompt as the layout to edit: its layout JSON, or a box-less layout whose caption is
    /// the plain prompt (draw boxes where things should change).
    public static func layout(fromPrompt prompt: String) -> LayoutPrompt {
        if let l = try? JSONDecoder().decode(LayoutPrompt.self, from: Data(prompt.utf8)) { return l }
        return LayoutPrompt(summary: .init(caption: prompt), items: [])
    }
}
