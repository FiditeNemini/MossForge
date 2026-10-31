import Foundation

/// A MOSS-Image JSON layout prompt (USERGUIDE "Layout Prompts"): a global summary plus
/// localized items with `[y_min, x_min, y_max, x_max]` boxes in **absolute pixels centred on
/// the canvas** (`abs_centralized`). It is fed through the ordinary text pathway — the DiT
/// learned this schema during pretraining.
public struct LayoutPrompt: Sendable, Equatable, Codable {
    public struct Summary: Sendable, Equatable, Codable {
        public var meta: [String]?
        public var canvas: String?
        public var caption: String?

        public init(meta: [String]? = nil, canvas: String? = nil, caption: String? = nil) {
            self.meta = meta
            self.canvas = canvas
            self.caption = caption
        }
    }

    public enum Label: String, Sendable, Codable, CaseIterable {
        case text, object, character, panel, watermark
    }

    public struct Item: Sendable, Equatable, Codable {
        public var label: Label
        public var caption: String?
        /// Literal visible text; keep intended line breaks (`\n`) — they measurably help rendering.
        public var text: String?
        /// `[y_min, x_min, y_max, x_max]`, abs_centralized pixels.
        public var bbox: [Int]

        public init(label: Label, caption: String? = nil, text: String? = nil, bbox: [Int]) {
            self.label = label
            self.caption = caption
            self.text = text
            self.bbox = bbox
        }
    }

    public var summary: Summary
    public var items: [Item]
    public var bboxCoordSpace = "abs_centralized"
    public var bboxFormat = "yxyx"

    enum CodingKeys: String, CodingKey {
        case summary, items
        case bboxCoordSpace = "bbox_coord_space"
        case bboxFormat = "bbox_format"
    }

    public init(summary: Summary, items: [Item]) {
        self.summary = summary
        self.items = items
    }

    /// Compact JSON for the text encoder, in MOSS-Image's canonical key order with the reference
    /// enhancer's escaping (`CanonicalJSON`) — deterministic, unlike `JSONEncoder`.
    public func jsonString() throws -> String {
        var s: [String: Any] = [:]
        if let meta = summary.meta { s["meta"] = meta }
        if let canvas = summary.canvas { s["canvas"] = canvas }
        if let caption = summary.caption { s["caption"] = caption }
        let items: [[String: Any]] = items.map { item in
            var d: [String: Any] = ["label": item.label.rawValue, "bbox": item.bbox]
            if let caption = item.caption { d["caption"] = caption }
            if let text = item.text { d["text"] = text }
            return d
        }
        return CanonicalJSON.layout(summary: s, items: items)
    }
}

/// A box in the layout's coordinate space.
public struct LayoutBox: Sendable, Equatable {
    public var yxyx: [Int]

    /// Convert a conventional top-left-origin pixel rect to abs_centralized yxyx.
    public static func fromTopLeft(x: Int, y: Int, width: Int, height: Int, canvas: Dimensions) -> LayoutBox {
        let cx = canvas.width / 2
        let cy = canvas.height / 2
        return LayoutBox(yxyx: [y - cy, x - cx, y + height - cy, x + width - cx])
    }
}
