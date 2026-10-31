import Foundation

/// Replaces personal details in diagnostics and feedback with placeholders before anything is shown
/// to the language model or leaves the Mac: known folders → `<models>` / `<output>`, the home folder →
/// `~`, other users' home folders, the account and host names, e-mail, IPv4 (not loopback), MAC
/// addresses and UUIDs. Applied again to the final text, so details typed into the description are
/// caught too.
public struct PrivacyRedactor: Sendable {
    public struct Context: Sendable {
        public var homeDirectory: String?
        public var userName: String?
        public var hostNames: [String]
        /// Exact path prefixes and their labels, e.g. the models folder → `<models>`.
        public var placeholders: [(path: String, label: String)]

        public init(homeDirectory: String? = nil, userName: String? = nil, hostNames: [String] = [], placeholders: [(String, String)] = []) {
            self.homeDirectory = homeDirectory
            self.userName = userName
            self.hostNames = hostNames
            self.placeholders = placeholders.map { (path: $0.0, label: $0.1) }
        }

        /// This Mac's home folder, account and host names, plus the given folders.
        public static func current(placeholders: [(String, String)] = []) -> Context {
            let host = ProcessInfo.processInfo.hostName
            let short = host.components(separatedBy: ".").first ?? host
            return Context(
                homeDirectory: FileManager.default.homeDirectoryForCurrentUser.path, userName: NSUserName(),
                hostNames: [host, short].filter { $0.count >= 3 && $0 != "localhost" }, placeholders: placeholders)
        }
    }

    public let context: Context
    private let rules: [(NSRegularExpression, String)]

    public init(context: Context) {
        self.context = context
        var rules: [(NSRegularExpression, String)] = []
        func add(_ pattern: String, _ template: String, _ options: NSRegularExpression.Options = []) {
            if let re = try? NSRegularExpression(pattern: pattern, options: options) { rules.append((re, template)) }
        }
        // Longest first, so a folder inside the home folder keeps its own label.
        for p in context.placeholders.filter({ $0.path.count > 1 }).sorted(by: { $0.path.count > $1.path.count }) {
            add(NSRegularExpression.escapedPattern(for: Self.trimSlash(p.path)) + "(?=/|\\b|$)", NSRegularExpression.escapedTemplate(for: p.label))
        }
        if let home = context.homeDirectory.map(Self.trimSlash), home.count > 1 {
            add(NSRegularExpression.escapedPattern(for: home) + "(?![\\w.-])", "~")
        }
        add("/Users/(?!Shared\\b)[^/\\s]+", "/Users/<user>")
        for host in context.hostNames.filter({ !$0.isEmpty }).sorted(by: { $0.count > $1.count }) {
            add("(?<![\\w-])" + NSRegularExpression.escapedPattern(for: host) + "(?![\\w-])", "<host>", .caseInsensitive)
        }
        add("[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}", "<email>", .caseInsensitive)
        add("\\b(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}\\b", "<mac>", .caseInsensitive)
        add("\\b[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\\b", "<id>", .caseInsensitive)
        add("\\b(?!127\\.0\\.0\\.1\\b)(?!0\\.0\\.0\\.0\\b)(?:(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)\\b", "<ip>")
        if let user = context.userName, user.count >= 3 {
            add("(?<![\\w<])" + NSRegularExpression.escapedPattern(for: user) + "(?![\\w>])", "<user>", .caseInsensitive)
        }
        self.rules = rules
    }

    public func redact(_ text: String) -> String { apply(text).text }

    /// How many replacements `redact` makes (shown as "N details replaced").
    public func redactionCount(in text: String) -> Int { apply(text).count }

    private func apply(_ text: String) -> (text: String, count: Int) {
        var t = text
        var count = 0
        for (re, template) in rules {
            let range = NSRange(t.startIndex..., in: t)
            let n = re.numberOfMatches(in: t, range: range)
            guard n > 0 else { continue }
            count += n
            t = re.stringByReplacingMatches(in: t, range: range, withTemplate: template)
        }
        return (t, count)
    }

    static func trimSlash(_ p: String) -> String { p.count > 1 && p.hasSuffix("/") ? String(p.dropLast()) : p }
}
