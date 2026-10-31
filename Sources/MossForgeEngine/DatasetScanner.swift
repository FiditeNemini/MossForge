
import Foundation
import MossForgeCore

/// Builds traditional-mode training pairs from a **directory** of images with caption
/// sidecars — the layout external tools (kohya, diffusers, ComfyUI dataset dumps, other
/// base models like Z-Image Turbo) ship. This is the directory-based alternative to the
/// tag-based path (a Gallery collection), so a dataset produced anywhere can train a LoRA
/// here without first importing it into the gallery.
///
/// Convention: each image `<stem>.<img-ext>` is paired with the caption in a sibling
/// `<stem>.txt` (then `<stem>.caption`); with no sidecar the humanised filename stem is a
/// weak fallback. Non-recursive; sorted by filename so runs are deterministic.
public enum DatasetScanner {
    /// Image extensions treated as training samples.
    public static let imageExtensions: Set<String> = [
        "png", "jpg", "jpeg", "webp", "heic", "heif", "tiff", "tif", "bmp",
    ]

    /// Caption sidecar extensions, tried in order.
    public static let captionExtensions = ["txt", "caption"]

    /// Scan a directory into `(image, caption)` pairs. Returns `[]` if the directory can't be
    /// read or holds no images.
    public static func scan(directory: URL, fileManager: FileManager = .default) -> [TrainingSample] {
        let contents =
            (try? fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        let images =
            contents
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted {
                $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
            }
        return images.map {
            TrainingSample(image: $0, caption: caption(for: $0, fileManager: fileManager))
        }
    }

    /// Resolve an image's caption: sidecar `<stem>.txt` / `<stem>.caption` (first non-empty),
    /// else the filename stem with `_`/`-` turned into spaces.
    static func caption(for image: URL, fileManager: FileManager = .default) -> String {
        let base = image.deletingPathExtension()
        for ext in captionExtensions {
            let sidecar = base.appendingPathExtension(ext)
            if fileManager.fileExists(atPath: sidecar.path),
                let text = try? String(contentsOf: sidecar, encoding: .utf8)
            {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        return base.lastPathComponent
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
    }
}
