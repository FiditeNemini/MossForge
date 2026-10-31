import Foundation

/// What the user is reporting.
public enum FeedbackKind: String, Sendable, Codable, CaseIterable, Identifiable {
    case bug, performance, idea, question

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .bug: "Bug report"
        case .performance: "Performance problem"
        case .idea: "Idea / feature request"
        case .question: "Question"
        }
    }
}

/// The app's recent activity for diagnostics: job outcomes and errors, never prompts or images.
/// Kept in memory only (gone when the app quits), only while diagnostics are on, and bounded.
public struct DiagnosticsLog: Sendable, Equatable {
    public enum Level: String, Sendable, Codable { case info, warning, error }

    public struct Entry: Sendable, Equatable, Codable {
        public var date: Date
        public var level: Level
        /// "generate", "caption", "upscale", …
        public var area: String
        public var message: String
    }

    public let capacity: Int
    public private(set) var entries: [Entry] = []
    /// Entries pushed out by newer ones.
    public private(set) var dropped = 0

    public init(capacity: Int = 200) { self.capacity = max(1, capacity) }

    public mutating func append(_ level: Level, area: String, _ message: String, at date: Date = Date()) {
        entries.append(Entry(date: date, level: level, area: area, message: message))
        if entries.count > capacity {
            dropped += entries.count - capacity
            entries.removeFirst(entries.count - capacity)
        }
    }

    public mutating func clear() {
        entries = []
        dropped = 0
    }
}

/// One group of findings (App, System, Models, …) the user can include or leave out.
public struct DiagnosticSection: Sendable, Equatable, Identifiable {
    public enum Status: String, Sendable, Codable { case info, ok, warn, fail }

    public struct Item: Sendable, Equatable {
        public var key: String
        public var value: String
        public var status: Status

        public init(_ key: String, _ value: String, status: Status = .info) {
            self.key = key
            self.value = value
            self.status = status
        }
    }

    public var id: String
    public var title: String
    public var items: [Item]
    /// Whether this section goes into the report (the user's choice; personal-leaning sections start off).
    public var included: Bool
    /// One line on what the section holds, shown beside its toggle.
    public var note: String

    public init(id: String, title: String, items: [Item], included: Bool = true, note: String = "") {
        self.id = id
        self.title = title
        self.items = items
        self.included = included
        self.note = note
    }

    public var problems: [Item] { items.filter { $0.status == .warn || $0.status == .fail } }
}

/// A feedback report: what the user says plus the sections they chose — always rendered through the
/// redactor, so what they preview is exactly what they can share.
public struct DiagnosticsReport: Sendable, Equatable {
    public var kind: FeedbackKind
    public var description: String
    public var sections: [DiagnosticSection]
    public var createdAt: Date

    public init(kind: FeedbackKind, description: String, sections: [DiagnosticSection], createdAt: Date = Date()) {
        self.kind = kind
        self.description = description
        self.sections = sections
        self.createdAt = createdAt
    }

    public var includedSections: [DiagnosticSection] { sections.filter(\.included) }

    /// The shareable Markdown: header, the user's words, the (edited) AI draft, the chosen diagnostics.
    public func markdown(product: HelpProduct, redactor: PrivacyRedactor, assistantDraft: String? = nil) -> String {
        var s = "# \(product.name) feedback — \(kind.title)\n\n"
        s += "_Generated on this Mac by \(product.name), \(Self.stamp(createdAt)). Personal details (folders, names, "
        s += "addresses) are replaced with placeholders such as `~`, `<models>` and `<user>`. Nothing was sent automatically._\n\n"
        let description = self.description.trimmingCharacters(in: .whitespacesAndNewlines)
        if !description.isEmpty { s += "## In my words\n\n\(description)\n\n" }
        if let draft = assistantDraft?.trimmingCharacters(in: .whitespacesAndNewlines), !draft.isEmpty { s += draft + "\n\n" }
        let included = includedSections
        if !included.isEmpty {
            s += "## Diagnostics\n"
            for section in included {
                s += "\n### \(section.title)\n"
                for item in section.items { s += "- \(Self.mark(item.status))**\(item.key)**: \(item.value)\n" }
            }
        }
        return redactor.redact(s)
    }

    /// Compact, problems-first text for the language model (included sections only).
    public func digest(redactor: PrivacyRedactor) -> String {
        let included = includedSections
        let problems = included.flatMap { section in section.problems.map { "- \(section.title) › \($0.key): \($0.value)" } }
        var s = "Problems found:" + (problems.isEmpty ? " none\n" : "\n" + problems.joined(separator: "\n") + "\n")
        s += "\nDiagnostics:\n"
        for section in included {
            s += "\(section.title): " + section.items.map { "\($0.key) \($0.value)" }.joined(separator: "; ") + "\n"
        }
        return redactor.redact(s)
    }

    static func mark(_ status: DiagnosticSection.Status) -> String {
        switch status {
        case .info: ""
        case .ok: "✓ "
        case .warn: "⚠︎ "
        case .fail: "✗ "
        }
    }

    public static func stamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH:mm"
        return f.string(from: date)
    }
}
