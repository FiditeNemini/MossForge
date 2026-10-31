import Foundation

/// The gallery's JSON sidecar (`<image>.json` beside each PNG): the generation request as written at
/// save time, plus what the app adds later — a `caption` (Gallery → Caption) and, for an upscaled copy,
/// an `upscale` record pointing at its source. Kept as a loose dictionary so older or hand-edited
/// sidecars survive a round trip untouched.
public enum GallerySidecar {
    public static func url(for image: URL) -> URL {
        image.deletingPathExtension().appendingPathExtension("json")
    }

    /// The files that make up one gallery entry — the image, plus its sidecar when there is one.
    public static func files(for image: URL) -> [URL] {
        let sidecar = url(for: image)
        return [image] + (FileManager.default.fileExists(atPath: sidecar.path) ? [sidecar] : [])
    }

    /// The sidecar's object, or empty when there is none (or it isn't a JSON object).
    public static func read(for image: URL) -> [String: Any] {
        guard let data = try? Data(contentsOf: url(for: image)),
            let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return [:] }
        return obj
    }

    public static func write(_ meta: [String: Any], for image: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url(for: image), options: .atomic)
    }

    /// `meta` with `caption: {long, medium, short, watermarks, borders}` set (replacing any earlier one).
    public static func adding(_ caption: MossCaption, to meta: [String: Any]) -> [String: Any] {
        var out = meta
        out["caption"] = [
            "long": caption.long, "medium": caption.medium, "short": caption.short,
            "watermarks": caption.watermarks.map { ["box": $0.box, "label": $0.label] },
            "borders": caption.borders,
        ] as [String: Any]
        return out
    }

    /// The caption recorded by `adding(_:to:)`, if any.
    public static func caption(in meta: [String: Any]) -> MossCaption? {
        guard let c = meta["caption"] as? [String: Any], let long = c["long"] as? String,
            let medium = c["medium"] as? String, let short = c["short"] as? String
        else { return nil }
        let marks = (c["watermarks"] as? [[String: Any]] ?? []).compactMap { m -> MossCaption.Watermark? in
            guard let box = m["box"] as? [NSNumber] else { return nil }
            return .init(box: box.map(\.doubleValue), label: m["label"] as? String ?? "")
        }
        return MossCaption(long: long, medium: medium, short: short, watermarks: marks, borders: c["borders"] as? [String] ?? ["none"])
    }

    /// The collections (tags) an image belongs to.
    public static func collections(in meta: [String: Any]) -> Set<String> {
        Set(meta["collections"] as? [String] ?? [])
    }

    /// `meta` with its collections replaced (sorted; the key is dropped when empty).
    public static func setting(_ collections: Set<String>, in meta: [String: Any]) -> [String: Any] {
        var out = meta
        out["collections"] = collections.isEmpty ? nil : collections.sorted()
        return out
    }

    /// The sidecar for an upscaled copy: the source's request, the new size, and where it came from.
    public static func upscaled(_ meta: [String: Any], request: UpscaleRequest, result: UpscaleResult) -> [String: Any] {
        var out = meta
        out["width"] = result.width
        out["height"] = result.height
        var record: [String: Any] = [
            "from": request.image.lastPathComponent, "method": request.method.rawValue, "scale": request.scale,
        ]
        if request.method == .esrgan, let model = request.esrganModel { record["model"] = model }
        if request.method == .vosr {
            record["seed"] = request.settings.seed
            record["soften"] = request.settings.soften.rawValue
            record["color_fix"] = request.settings.colorFix.rawValue
        }
        out["upscale"] = record
        return out
    }
}
