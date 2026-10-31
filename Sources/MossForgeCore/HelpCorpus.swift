import Foundation

/// One bundled documentation file.
public struct HelpDocument: Sendable, Equatable {
    /// Repo-relative path, e.g. `docs/user/index.md`.
    public var path: String
    public var text: String

    public init(path: String, text: String) {
        self.path = path
        self.text = text
    }

    /// The first `# ` heading, else the file name.
    public var title: String {
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) where line.hasPrefix("# ") {
            return line.dropFirst(2).trimmingCharacters(in: .whitespaces)
        }
        return (path as NSString).lastPathComponent
    }

    public var isDeveloperDoc: Bool { path.hasPrefix("docs/dev/") }
}

/// A retrievable piece of a document: one heading's section (or part of a long one).
public struct HelpChunk: Sendable, Equatable, Identifiable {
    public var id: String
    /// The document's title.
    public var document: String
    /// "Section" or "Section › Subsection".
    public var heading: String
    public var text: String

    public init(id: String, document: String, heading: String, text: String) {
        self.id = id
        self.document = document
        self.heading = heading
        self.text = text
    }

    /// "Document › Heading" (the heading alone when it *is* the document title).
    public var title: String { heading == document ? document : "\(document) › \(heading)" }
}

/// The documentation the app ships for Help → Ask and for grounding feedback drafts. Curated: user
/// docs plus the shareable developer docs (architecture, licensing) — not release/signing notes or
/// internal plans. `HelpCorpusDocuments.swift` is generated from these files by
/// `swift run mossforge help-corpus` (`HelpCorpus.regenerate`); a test keeps it in sync and checks it for personal details.
public enum HelpCorpus {
    public static let manifest = [
        "docs/user/index.md",
        "docs/user/settings-guide.md",
        "docs/user/help-and-feedback.md",
        "docs/user/training.md",
        "docs/user/editing.md",
        "docs/user/upscaling.md",
        "docs/user/captioning.md",
        "docs/dev/architecture.md",
        "docs/dev/licensing.md",
        "docs/dev/technical-reference.md",
    ]

    public static var documents: [HelpDocument] { embedded.map { HelpDocument(path: $0.path, text: $0.text) } }

    public static let chunks: [HelpChunk] = documents.flatMap { chunks(of: $0) }

    /// Split at `##`/`###` headings (not inside code fences), drop images and the title line, and
    /// cut sections longer than `maxChars` at paragraph breaks (a paragraph longer than that is cut hard).
    /// A paragraph that opens with a bold title (`**Train screen.** …`, the user guide's style) starts
    /// a subsection of the current `##`/`###` heading, named up to its first " (" or ".".
    public static func chunks(of doc: HelpDocument, maxChars: Int = 1600) -> [HelpChunk] {
        let title = doc.title
        var sections: [(heading: String, lines: [String])] = [(title, [])]
        var h2 = ""
        /// The latest `##` / `###` heading — what a bold-lead paragraph nests under.
        var base = ""
        var inFence = false
        var previousBlank = true
        for raw in doc.text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            if line.hasPrefix("```") { inFence.toggle() }
            if !inFence {
                if line.hasPrefix("# ") { continue }
                if line.hasPrefix("## ") {
                    h2 = line.dropFirst(3).trimmingCharacters(in: .whitespaces)
                    base = h2
                    sections.append((h2, []))
                    previousBlank = true
                    continue
                }
                if line.hasPrefix("### ") {
                    let h3 = line.dropFirst(4).trimmingCharacters(in: .whitespaces)
                    base = h2.isEmpty ? h3 : "\(h2) › \(h3)"
                    sections.append((base, []))
                    previousBlank = true
                    continue
                }
                if line.trimmingCharacters(in: .whitespaces).hasPrefix("![") { continue }
                if previousBlank, let lead = boldLead(line) {
                    sections.append((base.isEmpty ? lead : "\(base) › \(lead)", [line]))
                    previousBlank = false
                    continue
                }
            }
            previousBlank = line.trimmingCharacters(in: .whitespaces).isEmpty
            sections[sections.count - 1].lines.append(line)
        }
        var out: [HelpChunk] = []
        for section in sections {
            let body = section.lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !body.isEmpty else { continue }
            for piece in split(body, maxChars: maxChars) {
                out.append(HelpChunk(id: "\(doc.path)#\(out.count)", document: title, heading: section.heading, text: piece))
            }
        }
        return out
    }

    /// "Train screen" from `**Train screen.** Train an adapter…` / `**Upscale (Gallery → …).** …`.
    static func boldLead(_ line: String) -> String? {
        guard line.hasPrefix("**"), let close = line.range(of: "**", range: line.index(line.startIndex, offsetBy: 2)..<line.endIndex)
        else { return nil }
        var title = String(line[line.index(line.startIndex, offsetBy: 2)..<close.lowerBound])
        if let paren = title.range(of: " (") { title = String(title[..<paren.lowerBound]) }
        title = title.trimmingCharacters(in: CharacterSet(charactersIn: ".:").union(.whitespaces))
        return title.isEmpty || title.count > 60 ? nil : title
    }

    /// Paragraphs (blank-line separated, code fences kept whole where they fit) packed up to `maxChars`.
    static func split(_ body: String, maxChars: Int) -> [String] {
        guard body.count > maxChars else { return [body] }
        var paragraphs: [String] = []
        var current: [String] = []
        var inFence = false
        for line in body.split(separator: "\n", omittingEmptySubsequences: false).map(String.init) {
            if line.hasPrefix("```") { inFence.toggle() }
            if !inFence, line.trimmingCharacters(in: .whitespaces).isEmpty {
                if !current.isEmpty { paragraphs.append(current.joined(separator: "\n")) }
                current = []
            } else {
                current.append(line)
            }
        }
        if !current.isEmpty { paragraphs.append(current.joined(separator: "\n")) }
        var pieces: [String] = []
        var piece = ""
        for var p in paragraphs {
            while p.count > maxChars {  // one huge paragraph: hard cut
                if !piece.isEmpty { pieces.append(piece); piece = "" }
                pieces.append(String(p.prefix(maxChars)))
                p = String(p.dropFirst(maxChars))
            }
            if piece.isEmpty {
                piece = p
            } else if piece.count + 2 + p.count <= maxChars {
                piece += "\n\n" + p
            } else {
                pieces.append(piece)
                piece = p
            }
        }
        if !piece.isEmpty { pieces.append(piece) }
        return pieces
    }

    /// Regenerate (or, with `check`, verify) the embedded docs from `root`. Wire it to a CLI
    /// subcommand (`<cli> help-corpus [--check]`) and run it from the repo root after editing docs.
    /// Returns a one-line summary; throws `HelpCorpusError.stale` when `check` finds a mismatch.
    @discardableResult
    public static func regenerate(root: URL, output: URL, check: Bool = false) throws -> String {
        let docs = try manifest.map {
            HelpDocument(path: $0, text: try String(contentsOf: root.appendingPathComponent($0), encoding: .utf8))
        }
        let source = swiftSource(docs)
        if check {
            guard (try? String(contentsOf: output, encoding: .utf8)) == source else { throw HelpCorpusError.stale }
            return "help corpus up to date (\(docs.count) documents)"
        }
        try source.write(to: output, atomically: true, encoding: .utf8)
        return "wrote \(output.path) (\(docs.count) documents, \(docs.map(\.text.count).reduce(0, +)) characters)"
    }

    /// The generated Swift source embedding `documents` (raw string literals).
    public static func swiftSource(_ documents: [HelpDocument]) -> String {
        var s = """
            // GENERATED by `swift run mossforge help-corpus` (HelpCorpus.regenerate) from HelpCorpus.manifest — do not edit.
            // Edit the docs, then regenerate (HelpAssistantTests checks they match).

            extension HelpCorpus {
                static let embedded: [(path: String, text: String)] = [

            """
        for doc in documents {
            var hashes = "#"
            while doc.text.contains("\"\"\"" + hashes) || doc.text.contains("\"" + hashes) { hashes += "#" }
            s += "        (\n            \"\(doc.path)\",\n            \(hashes)\"\"\"\n"
            for line in doc.text.split(separator: "\n", omittingEmptySubsequences: false) {
                s += line.isEmpty ? "\n" : "            \(line)\n"
            }
            s += "            \"\"\"\(hashes)\n        ),\n"
        }
        return s + "    ]\n}\n"
    }
}

public enum HelpCorpusError: Error, LocalizedError {
    case stale

    public var errorDescription: String? { "help corpus is stale — run `swift run mossforge help-corpus` from the repo root" }
}
