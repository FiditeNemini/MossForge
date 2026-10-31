import Foundation

/// Captioning a folder of images for training, kohya / ai-toolkit style: `name.png` gets `name.txt`.
public enum FolderCaptioning {
    public static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "tif", "tiff", "bmp"]

    /// Images in `folder` (top level, sorted) that still need a caption — all of them with `overwrite`.
    public static func pending(in folder: URL, overwrite: Bool) -> [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        return files.filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .filter { overwrite || !FileManager.default.fileExists(atPath: captionURL(for: $0).path) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    public static func captionURL(for image: URL) -> URL {
        image.deletingPathExtension().appendingPathExtension("txt")
    }

    /// Write `name.txt`, prefixed `"trigger, caption"` when a trigger word is given (not doubled).
    public static func write(_ caption: String, trigger: String = "", for image: URL) throws {
        try prefixed(caption, trigger: trigger).write(to: captionURL(for: image), atomically: true, encoding: .utf8)
    }

    public static func prefixed(_ caption: String, trigger: String) -> String {
        let body = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        let word = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !word.isEmpty else { return body }
        if body.isEmpty { return word }
        return body.lowercased().hasPrefix(word.lowercased()) ? body : "\(word), \(body)"
    }
}
