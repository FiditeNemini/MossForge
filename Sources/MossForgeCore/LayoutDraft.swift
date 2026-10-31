import Foundation

/// The editable model behind the layout canvas. Boxes live in normalised canvas space
/// (`0…1`, top-left origin, so the canvas is resolution-free and a
/// size change keeps placement), and `layoutPrompt()` emits MOSS's JSON layout with
/// **abs_centralized yxyx pixel** boxes for the current `canvas` (USERGUIDE "Layout Prompts").
public struct LayoutDraft: Sendable, Equatable {
    /// A box in normalised canvas coordinates (always `top ≤ bottom`, `left ≤ right` after `normalised`).
    public struct Frame: Sendable, Equatable, Hashable {
        public var top: Double
        public var left: Double
        public var bottom: Double
        public var right: Double

        public init(top: Double, left: Double, bottom: Double, right: Double) {
            self.top = top
            self.left = left
            self.bottom = bottom
            self.right = right
        }

        /// Ordered corners, clamped to the canvas.
        public var normalised: Frame {
            func c(_ v: Double) -> Double { min(1, max(0, v)) }
            return Frame(top: c(min(top, bottom)), left: c(min(left, right)), bottom: c(max(top, bottom)), right: c(max(left, right)))
        }

        public func contains(x: Double, y: Double) -> Bool { (left...right).contains(x) && (top...bottom).contains(y) }
    }

    public struct Item: Sendable, Equatable, Identifiable {
        public let id: UUID
        public var label: LayoutPrompt.Label
        public var caption: String
        /// Literal visible text (text items); explicit `\n` line breaks help rendering.
        public var text: String
        public var frame: Frame
    }

    /// Smallest box side, as a canvas fraction (keeps a resize from collapsing a box).
    public static let minimumSide = 0.02

    public var canvas: Dimensions
    public var caption: String
    public var meta: [String]
    public var items: [Item] = []
    public var selection: UUID?

    public init(canvas: Dimensions, caption: String = "", meta: [String] = []) {
        self.canvas = canvas
        self.caption = caption
        self.meta = meta
    }

    /// Start from an existing layout prompt (e.g. one the enhancer wrote) for a given canvas.
    public init(_ prompt: LayoutPrompt, canvas: Dimensions) {
        self.init(canvas: canvas, caption: prompt.summary.caption ?? "", meta: prompt.summary.meta ?? [])
        let (w, h) = (Double(canvas.width), Double(canvas.height))
        for item in prompt.items where item.bbox.count == 4 {
            let b = item.bbox.map(Double.init)
            items.append(Item(
                id: UUID(), label: item.label, caption: item.caption ?? "", text: item.text ?? "",
                frame: Frame(top: (b[0] + h / 2) / h, left: (b[1] + w / 2) / w, bottom: (b[2] + h / 2) / h, right: (b[3] + w / 2) / w)))
        }
    }

    public func item(_ id: UUID) -> Item? { items.first { $0.id == id } }

    @discardableResult
    public mutating func add(label: LayoutPrompt.Label, frame: Frame, caption: String = "", text: String = "") -> UUID {
        let item = Item(id: UUID(), label: label, caption: caption, text: text, frame: frame.normalised)
        items.append(item)
        selection = item.id
        return item.id
    }

    public mutating func update(_ id: UUID, _ change: (inout Item) -> Void) {
        guard let i = items.firstIndex(where: { $0.id == id }) else { return }
        change(&items[i])
        items[i].frame = items[i].frame.normalised
    }

    public mutating func remove(_ id: UUID) {
        items.removeAll { $0.id == id }
        if selection == id { selection = nil }
    }

    /// Translate a box, keeping its size and clamping it fully inside the canvas.
    public mutating func move(_ id: UUID, dx: Double, dy: Double) {
        update(id) { item in
            let f = item.frame
            let (w, h) = (f.right - f.left, f.bottom - f.top)
            let left = min(1 - w, max(0, f.left + dx)), top = min(1 - h, max(0, f.top + dy))
            item.frame = Frame(top: top, left: left, bottom: top + h, right: left + w)
        }
    }

    /// Drag the bottom-right corner; the top-left stays put and the box keeps `minimumSide`.
    public mutating func resize(_ id: UUID, toBottom bottom: Double, right: Double) {
        update(id) { item in
            let f = item.frame
            item.frame = Frame(
                top: f.top, left: f.left,
                bottom: min(1, max(f.top + Self.minimumSide, bottom)), right: min(1, max(f.left + Self.minimumSide, right)))
        }
    }

    /// Topmost box containing a normalised point.
    public func hit(x: Double, y: Double) -> UUID? { items.last { $0.frame.contains(x: x, y: y) }?.id }

    /// The MOSS layout prompt for `canvas`: boxes in abs_centralized pixels, `[y1, x1, y2, x2]`.
    public func layoutPrompt() -> LayoutPrompt {
        let (w, h) = (Double(canvas.width), Double(canvas.height))
        func px(_ v: Double, _ size: Double) -> Int { Int((v * size - size / 2).rounded()) }
        return LayoutPrompt(
            summary: .init(meta: meta.isEmpty ? nil : meta, canvas: "single", caption: caption.isEmpty ? nil : caption),
            items: items.map { item in
                let f = item.frame
                return .init(
                    label: item.label, caption: item.caption.isEmpty ? nil : item.caption,
                    text: item.label == .text && !item.text.isEmpty ? item.text : nil,
                    bbox: [px(f.top, h), px(f.left, w), px(f.bottom, h), px(f.right, w)])
            })
    }
}
