
import Foundation
import MossForgeCore

/// One image to export: its file on disk, a name hint (usually the prompt) and the text that
/// becomes its caption sidecar.
public struct DatasetExportItem: Sendable, Equatable {
    public var source: URL
    public var nameHint: String
    public var caption: String
    public var prompt: String

    public init(source: URL, nameHint: String, caption: String, prompt: String) {
        self.source = source
        self.nameHint = nameHint
        self.caption = caption
        self.prompt = prompt
    }
}

/// Where one item lands: `<stem>.<ext>` plus, for a dataset, `<stem>.txt`.
public struct DatasetExportFile: Sendable, Equatable {
    public var item: DatasetExportItem
    public var stem: String
    public var imageName: String
    public var captionName: String { stem + ".txt" }
}

/// Writes Gallery images to a folder in the kohya / ai-toolkit layout (`name.png` +
/// `name.txt`), the inverse of `DatasetScanner`. Names are filesystem-safe slugs, unique
/// case-insensitively (APFS default), and never reuse a stem already in the folder, so an
/// export never overwrites anything.
public enum DatasetExporter {
    public struct Result: Sendable, Equatable {
        public var written: [DatasetExportFile]
        /// Sources that were missing on disk (e.g. stub swatches with no PNG).
        public var skipped: [URL]
    }

    static let maxSlugLength = 60

    /// Lowercase ASCII letters and digits, other runs collapsed to `-`, capped in length.
    static func slug(_ text: String) -> String {
        var out = ""
        var pendingDash = false
        for ch in text.lowercased() {
            if ch.isASCII, ch.isLetter || ch.isNumber {
                if pendingDash, !out.isEmpty { out.append("-") }
                pendingDash = false
                out.append(ch)
                if out.count >= maxSlugLength { break }
            } else {
                pendingDash = true
            }
        }
        while out.hasSuffix("-") { out.removeLast() }
        return out.isEmpty ? "image" : out
    }

    /// The caption sidecar text: caption, else prompt, with an optional trigger-word prefix
    /// (`"trigger, caption"`) that is not doubled if the caption already starts with it.
    static func captionText(caption: String, prompt: String, trigger: String) -> String {
        let trimmedCaption = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        let body =
            trimmedCaption.isEmpty
            ? prompt.trimmingCharacters(in: .whitespacesAndNewlines) : trimmedCaption
        let word = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !word.isEmpty else { return body }
        if body.isEmpty { return word }
        if body.lowercased().hasPrefix(word.lowercased()) { return body }
        return "\(word), \(body)"
    }

    /// Assign each item a unique stem, avoiding every stem in `existing` (file names already
    /// in the destination folder, any extension).
    static func plan(_ items: [DatasetExportItem], existing: [String]) -> [DatasetExportFile] {
        var taken = Set(existing.map { ($0 as NSString).deletingPathExtension.lowercased() })
        return items.map { item in
            let base = slug(item.nameHint)
            var stem = base
            var n = 2
            while taken.contains(stem) {
                stem = "\(base)-\(n)"
                n += 1
            }
            taken.insert(stem)
            let ext = item.source.pathExtension.lowercased()
            return DatasetExportFile(
                item: item,
                stem: stem,
                imageName: stem + "." + (ext.isEmpty ? "png" : ext)
            )
        }
    }

    /// Copy the images (and, with `captions`, write the `.txt` sidecars) into `directory`.
    public static func export(
        _ items: [DatasetExportItem],
        to directory: URL,
        captions: Bool,
        trigger: String = "",
        fileManager: FileManager = .default
    ) throws -> Result {
        let present = items.filter { fileManager.fileExists(atPath: $0.source.path) }
        let skipped = items.filter { !fileManager.fileExists(atPath: $0.source.path) }.map(\.source)
        let existing = try fileManager.contentsOfDirectory(atPath: directory.path)
        let files = plan(present, existing: existing)
        for file in files {
            try fileManager.copyItem(
                at: file.item.source,
                to: directory.appendingPathComponent(file.imageName)
            )
            if captions {
                let text = captionText(caption: file.item.caption, prompt: file.item.prompt, trigger: trigger)
                try text.write(
                    to: directory.appendingPathComponent(file.captionName),
                    atomically: true,
                    encoding: .utf8
                )
            }
        }
        return Result(written: files, skipped: skipped)
    }
}

/// Gallery images as dataset export items: the sidecar caption at the chosen length, else the prompt.
public enum GalleryDataset {
    public static func item(image: URL, meta: [String: Any], detail: CaptionDetail) -> DatasetExportItem {
        let prompt = meta["prompt"] as? String ?? ""
        let isJSON = prompt.trimmingCharacters(in: .whitespaces).hasPrefix("{")
        return DatasetExportItem(
            source: image, nameHint: isJSON || prompt.isEmpty ? image.deletingPathExtension().lastPathComponent : prompt,
            caption: GallerySidecar.caption(in: meta)?.text(detail) ?? "", prompt: prompt)
    }
}
