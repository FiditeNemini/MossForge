import Foundation

/// Save a Copy (Generate preview, Gallery): where copies land and what they're called. Never
/// overwrites — clashes get Finder-style " 2", " 3" suffixes, compared case-insensitively (APFS).
public enum ImageExport {
    /// One destination in `folder` for `source`, avoiding `taken` (file names, any case).
    public static func destination(for source: URL, in folder: URL, taken: Set<String>) -> URL {
        let lowered = Set(taken.map { $0.lowercased() })
        let ext = source.pathExtension
        let stem = source.deletingPathExtension().lastPathComponent
        var name = source.lastPathComponent
        var n = 2
        while lowered.contains(name.lowercased()) {
            name = "\(stem) \(n)" + (ext.isEmpty ? "" : ".\(ext)")
            n += 1
        }
        return folder.appendingPathComponent(name)
    }

    /// Destinations for several sources, avoiding the folder's `existing` names and each other.
    public static func destinations(for sources: [URL], in folder: URL, existing: Set<String>) -> [URL] {
        var taken = existing
        return sources.map { src in
            let url = destination(for: src, in: folder, taken: taken)
            taken.insert(url.lastPathComponent)
            return url
        }
    }

    /// A readable default name for a save panel: the prompt's first words + seed. Layout JSON gets
    /// `moss-layout-<seed>`; no prompt gets `moss-<seed>`.
    public static func suggestedName(prompt: String, seed: UInt64) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{") { return "moss-layout-\(seed).png" }
        let words = trimmed.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
        var slug = ""
        for w in words {
            let next = slug.isEmpty ? w : slug + "-" + w
            if next.count > 44 { break }
            slug = next
        }
        return slug.isEmpty ? "moss-\(seed).png" : "\(slug)-\(seed).png"
    }
}
