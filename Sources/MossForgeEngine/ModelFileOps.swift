import Foundation
import MossForgeCore

/// Why a model file may not be deleted.
public enum ModelFileError: Error, LocalizedError, Equatable {
    /// The file lives outside the configured models directory.
    case outsideModelsDirectory(URL)
    /// Not a weight file — the app only ever removes checkpoints/adapters it scanned.
    case unsupportedFile(URL)
    case notFound(URL)

    public var errorDescription: String? {
        switch self {
        case .outsideModelsDirectory(let url):
            "\(url.lastPathComponent) is outside the models directory, so MossForge won't delete it."
        case .unsupportedFile(let url):
            "\(url.lastPathComponent) isn't a weight file."
        case .notFound(let url):
            "\(url.lastPathComponent) is no longer there — rescan."
        }
    }
}

/// The destructive half of model management, kept out of the views so its guard rails are
/// testable. Two rules, both deliberate: only `.safetensors` **inside the configured models
/// directory** can go, and going means the **Trash** — never an unlink — so a mis-click costs
/// a drag back out of the Trash rather than a re-download of 26 GB.
public enum ModelFileOps {
    /// Extensions the app is willing to remove.
    public static let deletableExtensions: Set<String> = ["safetensors"]

    /// The per-model metadata sidecar: the file path with `.mfmeta.json` appended.
    public static func sidecarURL(for model: URL) -> URL {
        URL(fileURLWithPath: model.path + ".mfmeta.json")
    }

    /// Files that belong to `model` and should be removed with it — today the
    /// `.mfmeta.json` sidecar (trigger words / weight band).
    public static func companions(of model: URL) -> [URL] {
        [sidecarURL(for: model)]
    }

    /// Throws unless `url` is a weight file that exists inside `root`.
    public static func validateDeletable(_ url: URL, root: URL) throws {
        let file = url.resolvingSymlinksInPath().standardizedFileURL
        let base = root.resolvingSymlinksInPath().standardizedFileURL
        guard deletableExtensions.contains(url.pathExtension.lowercased()) else {
            throw ModelFileError.unsupportedFile(url)
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory),
            !isDirectory.boolValue
        else { throw ModelFileError.notFound(url) }
        // Containment, on normalised paths, with the trailing separator so `models-old/` can't
        // pass as a prefix match for `models/`.
        let rootPath = base.path.hasSuffix("/") ? base.path : base.path + "/"
        guard file.path.hasPrefix(rootPath) else {
            throw ModelFileError.outsideModelsDirectory(url)
        }
    }

    /// Move a model — and its companions — to the Trash. Returns what was moved.
    /// The model itself must succeed; a companion that won't budge is skipped rather than
    /// leaving the delete half-done.
    @discardableResult
    public static func moveToTrash(_ url: URL, root: URL) throws -> [URL] {
        try validateDeletable(url, root: root)
        let fm = FileManager.default
        try fm.trashItem(at: url, resultingItemURL: nil)
        var moved = [url]
        for companion in companions(of: url) where fm.fileExists(atPath: companion.path) {
            if (try? fm.trashItem(at: companion, resultingItemURL: nil)) != nil {
                moved.append(companion)
            }
        }
        return moved
    }
}
