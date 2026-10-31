import Foundation

/// Help → Ask: the system prompt that grounds the app's own language model in the retrieved
/// documentation excerpts.
public enum HelpAssistantPrompt {
    /// Earlier messages kept for follow-ups (two question/answer pairs).
    public static let historyTurns = 4
    /// Each kept message is cut to this many characters.
    static let historyChars = 1200

    public static func system(product: HelpProduct, excerpts: [HelpChunk], appState: String?) -> String {
        var s = """
            You are \(product.name) Help, the built-in assistant of \(product.name) — \(product.summary).

            Rules:
            - Answer using only the documentation excerpts below. If they don't cover the question, say so plainly \
            and suggest the closest thing they do cover — never guess at features, settings or numbers.
            - Be concise and practical: steps the user can follow in the app, with the exact names of screens, \
            buttons and settings. For settings, give the recommended values *and why*.
            - Cite the excerpts you used as [1], [2] after the sentence they support.
            - Use short Markdown (bold, lists, `code`). No headings for short answers.

            """
        if let appState, !appState.isEmpty { s += "\nThe user's app right now: \(appState)\n" }
        s += "\nDocumentation excerpts:\n"
        if excerpts.isEmpty {
            s += "(No documentation matched this question.)\n"
        } else {
            for (i, c) in excerpts.enumerated() { s += "\n[\(i + 1)] \(c.title)\n\(c.text)\n" }
        }
        return s
    }

    /// System prompt, the last few turns (starting on a user turn), the question.
    public static func messages(
        product: HelpProduct, question: String, history: [ChatMessage], excerpts: [HelpChunk], appState: String?
    ) -> [ChatMessage] {
        var kept = Array(history.filter { $0.role != .system }.suffix(historyTurns))
        while kept.first?.role == .assistant { kept.removeFirst() }
        let trimmed = kept.map {
            ChatMessage($0.role, $0.content.count > historyChars ? String($0.content.prefix(historyChars)) + " …" : $0.content)
        }
        return [ChatMessage(.system, system(product: product, excerpts: excerpts, appState: appState))] + trimmed
            + [ChatMessage(.user, question)]
    }

    /// What to search the docs for: the question, plus the previous question when this one is a short
    /// follow-up ("and for Base?") with fewer than four content words.
    public static func retrievalQuery(question: String, history: [ChatMessage]) -> String {
        guard HelpRetriever.terms(question).count < 4, let previous = history.last(where: { $0.role == .user }) else {
            return question
        }
        return question + " " + previous.content
    }
}

/// Feedback → Draft with AI: turns the user's description + the redacted diagnostics into a clear
/// report, grounded in the docs (so it can point at a documented fix before anything is sent).
public enum FeedbackPrompt {
    public static func messages(
        product: HelpProduct, report: DiagnosticsReport, redactor: PrivacyRedactor, excerpts: [HelpChunk]
    ) -> [ChatMessage] {
        var system = """
            You help a \(product.name) user write a clear feedback report for the developer. \(product.name) is \
            \(product.summary).

            Write Markdown with exactly these sections:
            ## Summary
            (one or two sentences)
            ## What happened
            (the user's account, tidied; keep their facts)
            ## Likely cause
            (only what the diagnostics or documentation support; say "unclear" if they don't)
            ## Suggested next steps
            (things the user can try first, citing documentation excerpts as [1], [2]; then what the developer should look at)

            Never invent versions, numbers, errors or settings that are not in the input. Don't repeat the full \
            diagnostics — they are attached separately. Keep placeholders such as <models>, <user> or ~ exactly as written.

            """
        if !excerpts.isEmpty {
            system += "\nDocumentation excerpts:\n"
            for (i, c) in excerpts.enumerated() { system += "\n[\(i + 1)] \(c.title)\n\(c.text)\n" }
        }
        let description = redactor.redact(report.description.trimmingCharacters(in: .whitespacesAndNewlines))
        let user = """
            Feedback type: \(report.kind.title)

            The user's description:
            \(description.isEmpty ? "(none given)" : description)

            \(report.digest(redactor: redactor))
            """
        return [ChatMessage(.system, system), ChatMessage(.user, user)]
    }
}
