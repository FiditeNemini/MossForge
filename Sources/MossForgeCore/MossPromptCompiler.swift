import Foundation

/// Port of the official MOSS-Image prompt enhancer's compiler
/// (`skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`, Apache-2.0, OpenMOSS).
///
/// An enhancer (an LLM) writes a **draft**: either `{"caption": "…"}` or a Layout draft with concrete
/// `width`/`height` and boxes as `[x1, y1, x2, y2]` pixels from the top-left (`abs` / `xyxy`). This
/// validates it strictly (unknown fields, bad geometry and missing text are errors, never silently
/// dropped) and emits the model-facing prompt: the caption verbatim, or one-line compact JSON with
/// `abs_centralized` / `yxyx` integer boxes (fractional edges enclosed: floor top/left, ceil
/// bottom/right, then minus the centre). Byte-for-byte the reference's output for drafts in canonical
/// key order (`MossPromptCompilerTests` golden strings).
public enum MossPromptCompiler {
    public struct Output: Sendable, Equatable {
        /// What goes into the positive prompt.
        public var prompt: String
        /// The canvas a Layout was drafted for (generate at exactly this size); nil for a caption.
        public var size: Dimensions?
    }

    static let rootFields: Set<String> = ["width", "height", "bbox_coord_space", "bbox_format", "summary", "items"]
    static let summaryFields: Set<String> = ["caption", "meta", "canvas", "palette"]
    static let itemFields: Set<String> = ["label", "caption", "bbox", "text", "tags", "palette"]

    public static func compile(_ data: Data) throws -> Output {
        guard let draft = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw fail("Draft must be a JSON object")
        }
        if draft["caption"] != nil {
            try fields(draft, allowed: ["caption"], required: ["caption"], "Caption draft")
            return Output(prompt: try text(draft["caption"], "caption"), size: nil)
        }
        try fields(draft, allowed: rootFields, required: rootFields, "Layout draft")
        let width = try dimension(draft["width"], "width"), height = try dimension(draft["height"], "height")
        guard draft["bbox_coord_space"] as? String == "abs" else { throw fail("Draft bbox_coord_space must be abs") }
        guard draft["bbox_format"] as? String == "xyxy" else { throw fail("Draft bbox_format must be xyxy") }

        guard let summary = draft["summary"] as? [String: Any] else { throw fail("summary must be an object") }
        try fields(summary, allowed: summaryFields, required: ["caption"], "summary")
        _ = try text(summary["caption"], "summary.caption")
        if let canvas = summary["canvas"] {
            guard let c = canvas as? String, ["single", "multi_panel", "collage"].contains(c) else {
                throw fail("summary.canvas must be single, multi_panel or collage")
            }
        }
        for key in ["meta", "palette"] where summary[key] != nil { _ = try strings(summary[key], "summary.\(key)") }

        guard let items = draft["items"] as? [Any], !items.isEmpty else { throw fail("items must be a nonempty flat array") }
        var compiled: [[String: Any]] = []
        for (index, raw) in items.enumerated() {
            let location = "items[\(index)]"
            guard let item = raw as? [String: Any] else { throw fail("\(location) must be an object") }
            try fields(item, allowed: itemFields, required: ["label", "caption", "bbox"], location)
            guard let label = item["label"] as? String, ["object", "character", "text"].contains(label) else {
                throw fail("\(location).label must be object, character or text")
            }
            _ = try text(item["caption"], "\(location).caption")
            if label == "text", item["text"] == nil { throw fail("\(location): text items require literal text") }
            if item["text"] != nil { _ = try text(item["text"], "\(location).text") }
            for key in ["tags", "palette"] where item[key] != nil { _ = try strings(item[key], "\(location).\(key)") }
            var out = item
            out["bbox"] = try centredBox(numbers(item["bbox"], "\(location).bbox"), width: width, height: height, "\(location).bbox")
            compiled.append(out)
        }
        let prompt = CanonicalJSON.layout(summary: summary, items: compiled)
        return Output(prompt: prompt, size: Dimensions(width: width, height: height))
    }

    /// Drops the blanks an enhancer LLM puts in optional fields before the strict `compile`: blank
    /// entries in `meta`/`palette`/`tags` (and the arrays they empty), off-list labels (`allowedLabel`), blank `text` (a `text` item left
    /// without literal text becomes an `object`). Anything else — including non-JSON — is returned as is,
    /// so required-field and geometry errors still reach the retry.
    public static func tidyDraft(_ data: Data) -> Data {
        guard var draft = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            draft["caption"] == nil
        else { return data }
        func blank(_ v: Any?) -> Bool { (v as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true }
        func tidyLists(_ dict: inout [String: Any], _ keys: [String]) {
            for key in keys {
                guard let list = dict[key] as? [Any] else { continue }
                let kept = list.filter { !blank($0) }
                dict[key] = kept.isEmpty ? nil : kept
            }
        }
        if var summary = draft["summary"] as? [String: Any] {
            tidyLists(&summary, ["meta", "palette"])
            draft["summary"] = summary
        }
        if let items = draft["items"] as? [Any] {
            draft["items"] = items.map { raw -> Any in
                guard var item = raw as? [String: Any] else { return raw }
                tidyLists(&item, ["tags", "palette"])
                if let label = item["label"] as? String { item["label"] = allowedLabel(label, hasText: !blank(item["text"]) && item["text"] is String) }
                if item["text"] is NSNull || blank(item["text"]) {
                    item["text"] = nil
                    if item["label"] as? String == "text" { item["label"] = "object" }
                }
                return item
            }
        }
        return (try? JSONSerialization.data(withJSONObject: draft)) ?? data
    }

    static let characterLabels: Set<String> = [
        "character", "person", "people", "human", "man", "woman", "child", "boy", "girl", "figure", "animal", "creature", "pet",
    ]

    /// An off-list item label → `object` / `character` / `text`: case-folded; people and animals are
    /// characters; anything else carrying literal text is `text`; the rest are objects.
    static func allowedLabel(_ label: String, hasText: Bool) -> String {
        let l = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["object", "character", "text"].contains(l) { return l }
        if characterLabels.contains(l) { return "character" }
        return hasText ? "text" : "object"
    }

    /// `[x1,y1,x2,y2]` (abs, top-left) → `[y1,x1,y2,x2]` centred: floor top/left, ceil bottom/right,
    /// minus `H/2`, `W/2` (integer halves, as the reference).
    public static func centredBox(_ box: [Double], width: Int, height: Int, _ location: String = "box") throws -> [Int] {
        guard box.count == 4 else { throw fail("\(location) must be [x1,y1,x2,y2]") }
        guard box.allSatisfy(\.isFinite) else { throw fail("\(location) must contain four finite numbers") }
        let (x1, y1, x2, y2) = (box[0], box[1], box[2], box[3])
        guard x1 < x2, y1 < y2 else { throw fail("\(location) must have positive width and height") }
        guard 0 <= x1, x2 <= Double(width), 0 <= y1, y2 <= Double(height) else {
            throw fail("\(location) exceeds the \(width)x\(height) canvas")
        }
        func r9(_ v: Double) -> Double { (v * 1e9).rounded() / 1e9 }  // Python round(v, 9)
        let result = [
            Int(r9(y1).rounded(.down)) - height / 2, Int(r9(x1).rounded(.down)) - width / 2,
            Int(r9(y2).rounded(.up)) - height / 2, Int(r9(x2).rounded(.up)) - width / 2,
        ]
        guard result[0] < result[2], result[1] < result[3] else { throw fail("\(location) is too small at export precision") }
        return result
    }

    /// The first balanced `{…}` object in a model reply (code fences and chatter around it ignored;
    /// braces inside JSON strings respected).
    public static func extractJSON(from reply: String) -> String? {
        let chars = Array(reply)
        guard let start = chars.firstIndex(of: "{") else { return nil }
        var depth = 0, inString = false, escaped = false
        for i in start..<chars.count {
            let c = chars[i]
            if inString {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == "\"" { inString = false }
                continue
            }
            switch c {
            case "\"": inString = true
            case "{": depth += 1
            case "}":
                depth -= 1
                if depth == 0 { return String(chars[start...i]) }
            default: break
            }
        }
        return nil
    }

    // MARK: - Validation helpers (mirroring the reference's `require`/`fields`/`text`/`strings`)

    static func fail(_ message: String) -> EngineError { .invalidRequest(message) }

    static func fields(_ value: [String: Any], allowed: Set<String>, required: Set<String>, _ location: String) throws {
        let unknown = Set(value.keys).subtracting(allowed), missing = required.subtracting(value.keys)
        guard unknown.isEmpty else { throw fail("\(location): unknown fields \(unknown.sorted())") }
        guard missing.isEmpty else { throw fail("\(location): missing fields \(missing.sorted())") }
    }

    static func text(_ value: Any?, _ location: String) throws -> String {
        guard let s = value as? String, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw fail("\(location) must be nonempty text")
        }
        return s
    }

    static func strings(_ value: Any?, _ location: String) throws -> [String] {
        guard let a = value as? [Any] else { throw fail("\(location) must be an array") }
        return try a.enumerated().map { try text($1, "\(location)[\($0)]") }
    }

    static func isBool(_ n: NSNumber) -> Bool { CFGetTypeID(n) == CFBooleanGetTypeID() }

    static func dimension(_ value: Any?, _ name: String) throws -> Int {
        guard let n = value as? NSNumber, !isBool(n), String(cString: n.objCType) != "d", String(cString: n.objCType) != "f",
            n.intValue > 0, n.intValue % 16 == 0
        else { throw fail("\(name) must be a positive integer multiple of 16") }
        return n.intValue
    }

    static func numbers(_ value: Any?, _ location: String) throws -> [Double] {
        guard let a = value as? [Any] else { throw fail("\(location) must be [x1,y1,x2,y2]") }
        return try a.map { v in
            guard let n = v as? NSNumber, !isBool(n) else { throw fail("\(location) must contain four finite numbers") }
            return n.doubleValue
        }
    }
}

/// Compact JSON in MOSS-Image's canonical key order (the USERGUIDE/enhancer order): top
/// `summary, items, bbox_coord_space, bbox_format`; summary `meta, canvas, caption, palette`; item
/// `label, caption, text, bbox, tags, palette`. Escaping matches Python `json.dumps(ensure_ascii=False)`
/// (non-ASCII literal, slashes unescaped, `separators=(",", ":")`). Deterministic, unlike JSONEncoder.
enum CanonicalJSON {
    static let summaryOrder = ["meta", "canvas", "caption", "palette"]
    static let itemOrder = ["label", "caption", "text", "bbox", "tags", "palette"]

    static func layout(summary: [String: Any], items: [[String: Any]]) -> String {
        let s = object(summary, order: summaryOrder)
        let i = "[" + items.map { object($0, order: itemOrder) }.joined(separator: ",") + "]"
        return #"{"summary":\#(s),"items":\#(i),"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#
    }

    static func object(_ dict: [String: Any], order: [String]) -> String {
        let keys = order.filter { dict[$0] != nil }
        return "{" + keys.map { quote($0) + ":" + value(dict[$0]!) }.joined(separator: ",") + "}"
    }

    static func value(_ v: Any) -> String {
        switch v {
        case let s as String: return quote(s)
        case let ints as [Int]: return "[" + ints.map(String.init).joined(separator: ",") + "]"
        case let a as [Any]: return "[" + a.map(value).joined(separator: ",") + "]"
        case let n as NSNumber: return n.stringValue
        default: return "null"
        }
    }

    static func quote(_ s: String) -> String {
        var out = "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case let c where c.value < 0x20: out += String(format: "\\u%04x", c.value)
            default: out.unicodeScalars.append(u)
            }
        }
        return out + "\""
    }
}
