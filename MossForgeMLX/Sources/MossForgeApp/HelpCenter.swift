import AppKit
import Foundation
import MossForgeCore
import Observation

/// What Help needs from the app. INSTANTIATE: make the app's store conform and set
/// `store.help.host = store` once at launch.
@MainActor
public protocol HelpHost: AnyObject {
    /// The language model is installed and loadable (otherwise Ask shows matching sections only).
    var canChat: Bool { get }
    /// Settings → Diagnostics & feedback. Off by default; turning it off calls `forgetDiagnostics()`.
    var diagnosticsEnabled: Bool { get }
    /// One line of non-personal state for the assistant: backend, model, key settings. Never prompts.
    func helpAppState() -> String
    /// Run one chat completion on the app's own model (through its GPU queue). Report progress text
    /// ("Loading the language model…") through `status` and the growing answer through `onText`.
    func helpChat(
        _ messages: [ChatMessage], maxTokens: Int, temperature: Float,
        status: @escaping @MainActor (String) -> Void, onText: @escaping @MainActor (String) -> Void
    ) async throws -> String
    /// The app's state for Run diagnostics (folders, settings, extra sections such as models / GPU).
    func diagnosticsContext(log: DiagnosticsLog) async -> DiagnosticsContext
    /// Folders → placeholders for the redactor, e.g. (modelsDir.path, "<models>").
    func redactorPlaceholders() -> [(String, String)]
    /// Navigate to Settings → Diagnostics & feedback.
    func openDiagnosticsSettings()
}

/// Help → Ask / Feedback tabs.
public enum HelpTab: String, CaseIterable, Identifiable, Sendable {
    case ask = "Ask", feedback = "Feedback"
    public var id: String { rawValue }
}

/// One question and its answer, with the documentation sections the answer was grounded in.
public struct HelpTurn: Identifiable, Equatable, Sendable {
    public let id = UUID()
    public var question: String
    public var answer = ""
    public var sources: [HelpChunk] = []
    public var error: String?
    /// No language model: the sources are the answer.
    public var offline = false
    public var done = false
}

/// Help's state and actions: Ask (docs-grounded answers from the app's own model) and Feedback
/// (private, redacted diagnostics the user reviews and shares themselves). Owned by the app's store.
@MainActor @Observable
public final class HelpCenter {
    public let product: HelpProduct
    public weak var host: (any HelpHost)?
    let retriever: HelpRetriever

    public var tab: HelpTab = .ask
    public var turns: [HelpTurn] = []
    public var status = ""
    var askTask: Task<Void, Never>?
    public var busy: Bool { askTask != nil }

    public var diagnosticsLog = DiagnosticsLog()
    public var feedbackKind: FeedbackKind = .bug
    public var feedbackDescription = ""
    public var feedbackSections: [DiagnosticSection] = []
    public var feedbackDraft = ""
    public var feedbackStatus = ""
    public var isCollectingDiagnostics = false
    var feedbackTask: Task<Void, Never>?
    public var feedbackBusy: Bool { feedbackTask != nil }

    public init(product: HelpProduct, chunks: [HelpChunk] = HelpCorpus.chunks) {
        self.product = product
        retriever = HelpRetriever(chunks: chunks)
    }

    public var canChat: Bool { host?.canChat ?? false }
    public var diagnosticsEnabled: Bool { host?.diagnosticsEnabled ?? false }

    // MARK: Ask

    public func ask(_ raw: String) {
        let question = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, !busy else { return }
        let history = turns.filter { $0.done && $0.error == nil && !$0.offline }.flatMap {
            [ChatMessage(.user, $0.question), ChatMessage(.assistant, $0.answer)]
        }
        let sources = retriever.search(HelpAssistantPrompt.retrievalQuery(question: question, history: history))
        var turn = HelpTurn(question: question, sources: sources)
        guard let host, host.canChat else {
            turn.offline = true
            turn.done = true
            turns.append(turn)
            return
        }
        turns.append(turn)
        let id = turn.id
        let messages = HelpAssistantPrompt.messages(
            product: product, question: question, history: history, excerpts: sources, appState: host.helpAppState())
        status = "Thinking…"
        let t0 = Date()
        askTask = Task {
            do {
                let answer = try await host.helpChat(messages, maxTokens: 768, temperature: 0.3) { s in
                    self.status = s
                } onText: { text in
                    self.updateTurn(id) { $0.answer = text }
                }
                updateTurn(id) { $0.answer = answer }
                note(.info, "help", "answered in " + String(format: "%.1f s", Date().timeIntervalSince(t0)))
            } catch is CancellationError {
                updateTurn(id) { $0.error = "Stopped." }
            } catch {
                updateTurn(id) { $0.error = "\(error)" }
                note(.error, "help", "\(error)")
            }
            updateTurn(id) { $0.done = true }
            status = ""
            askTask = nil
        }
    }

    public func stopAsk() { askTask?.cancel() }

    public func newChat() {
        stopAsk()
        turns = []
    }

    private func updateTurn(_ id: UUID, _ change: (inout HelpTurn) -> Void) {
        guard let i = turns.firstIndex(where: { $0.id == id }) else { return }
        change(&turns[i])
    }

    // MARK: diagnostics + feedback

    /// Record a job outcome — only while diagnostics are on. Never pass prompts or file names.
    public func note(_ level: DiagnosticsLog.Level, _ area: String, _ message: String) {
        guard diagnosticsEnabled else { return }
        diagnosticsLog.append(level, area: area, message)
    }

    /// Call when the user turns diagnostics off: forgets everything collected.
    public func forgetDiagnostics() {
        feedbackTask?.cancel()
        diagnosticsLog.clear()
        feedbackSections = []
        feedbackDraft = ""
        feedbackStatus = ""
    }

    /// "1.6.1 (146)" from the bundle, or "(dev build)" when run outside one.
    public static var appVersion: String {
        let info = Bundle.main.infoDictionary ?? [:]
        guard let v = info["CFBundleShortVersionString"] as? String else { return "dev build" }
        return (info["CFBundleVersion"] as? String).map { "\(v) (\($0))" } ?? v
    }

    /// This Mac's names and folders → placeholders.
    public var redactor: PrivacyRedactor {
        PrivacyRedactor(context: .current(placeholders: host?.redactorPlaceholders() ?? []))
    }

    public var report: DiagnosticsReport {
        DiagnosticsReport(kind: feedbackKind, description: feedbackDescription, sections: feedbackSections)
    }

    /// Exactly what Copy / Save / Share hand over.
    public var feedbackMarkdown: String { report.markdown(product: product, redactor: redactor, assistantDraft: feedbackDraft) }

    /// Details the redactor replaced in what the user wrote, the draft and the included diagnostics.
    public var redactionCount: Int {
        let r = redactor
        let raw = [feedbackDescription, feedbackDraft] + report.includedSections.flatMap { $0.items.map { $0.key + " " + $0.value } }
        return raw.reduce(0) { $0 + r.redactionCount(in: $1) }
    }

    public func runDiagnostics() {
        guard let host, diagnosticsEnabled, !isCollectingDiagnostics else { return }
        isCollectingDiagnostics = true
        feedbackStatus = "Running checks…"
        let previous = Dictionary(uniqueKeysWithValues: feedbackSections.map { ($0.id, $0.included) })
        let log = diagnosticsLog
        feedbackTask = Task {
            let context = await host.diagnosticsContext(log: log)
            let sections = await Task.detached { DiagnosticsCollector.collect(context) }.value
            // Keep the user's include / leave-out choices across re-runs.
            feedbackSections = sections.map { section in
                var x = section
                x.included = previous[x.id] ?? x.included
                return x
            }
            let problems = feedbackSections.filter(\.included).flatMap(\.problems).count
            feedbackStatus = problems == 0
                ? "No problems found." : "\(problems) thing\(problems == 1 ? "" : "s") worth a look (⚠︎ / ✗)."
            isCollectingDiagnostics = false
            feedbackTask = nil
        }
    }

    /// The model drafts Summary / What happened / Likely cause / Next steps from the redacted digest.
    public func draftFeedback() {
        guard let host, diagnosticsEnabled, host.canChat, feedbackTask == nil else { return }
        let redactor = redactor
        let report = report
        let problems = report.includedSections.flatMap(\.problems).map { "\($0.key) \($0.value)" }.joined(separator: " ")
        let excerpts = retriever.search(report.description + " " + problems, limit: 3, budget: 4000)
        let messages = FeedbackPrompt.messages(product: product, report: report, redactor: redactor, excerpts: excerpts)
        feedbackStatus = "Drafting on device…"
        feedbackDraft = ""
        feedbackTask = Task {
            do {
                let text = try await host.helpChat(messages, maxTokens: 900, temperature: 0.2) { s in
                    self.feedbackStatus = s
                } onText: { text in
                    self.feedbackDraft = text
                }
                // Whatever the model wrote goes through the redactor too.
                feedbackDraft = redactor.redact(text)
                feedbackStatus = "Draft ready — read and edit it before sharing."
            } catch is CancellationError {
                feedbackStatus = "Stopped."
            } catch {
                feedbackStatus = "\(error)"
            }
            feedbackTask = nil
        }
    }

    public func stopFeedback() { feedbackTask?.cancel() }

    public func copyFeedback() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(feedbackMarkdown, forType: .string)
        feedbackStatus = "Copied — paste it wherever you want to send it."
    }

    public func saveFeedback() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.init(filenameExtension: "md") ?? .plainText]
        let stamp = DiagnosticsReport.stamp(Date()).replacingOccurrences(of: " ", with: "-")
            .replacingOccurrences(of: ":", with: "")
        panel.nameFieldStringValue = "\(product.name)-feedback-\(stamp).md"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try feedbackMarkdown.write(to: url, atomically: true, encoding: .utf8)
            feedbackStatus = "Saved \(url.lastPathComponent)."
        } catch {
            feedbackStatus = "\(error)"
        }
    }
}
