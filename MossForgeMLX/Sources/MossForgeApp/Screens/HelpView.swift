import AppKit
import MossForgeCore
import SwiftUI

/// Help: **Ask** — questions answered on device from the bundled docs (the app's own model),
/// with sources — and **Feedback** — a private, redacted report the user reviews and shares
/// themselves (only once diagnostics are enabled in Settings).
///
/// INSTANTIATE: `HelpView().environment(store.help)` as the Help screen; add `.tweakable` tags.
public struct HelpView: View {
    @Environment(HelpCenter.self) private var store

    public init() {}

    public var body: some View {
        @Bindable var store = store
        VStack(spacing: 0) {
            Picker("", selection: $store.tab) {
                ForEach(HelpTab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().fixedSize()
            .helpTag("help.tabs")
            .padding(.vertical, 10)
            Divider()
            switch store.tab {
            case .ask: AskView().helpTag("help.ask")
            case .feedback: FeedbackView().helpTag("help.feedback")
            }
        }
    }
}

// MARK: - Ask

private struct AskView: View {
    @Environment(HelpCenter.self) private var store
    @State private var question = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        if store.turns.isEmpty { intro.helpTag("help.ask.intro") }
                        ForEach(store.turns) { turn in
                            TurnView(turn: turn).helpTag("help.ask.turn").id(turn.id)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(20)
                    .frame(maxWidth: 820)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: store.turns.last?.answer) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                .onChange(of: store.turns.count) { _, _ in withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
            }
            inputBar.helpTag("help.ask.composer")
        }
        .onAppear { focused = true }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Ask about \(store.product.name)", systemImage: "questionmark.bubble").font(.title3.weight(.semibold))
            Text(store.canChat
                ? "Answers come from the documentation built into the app, written on this Mac by the built-in \(store.product.modelName) — your questions never leave it. Each answer lists the sections it used."
                : "The language model isn't available, so Ask shows the documentation sections that match your question.")
                .foregroundStyle(.secondary)
            FlowLayout(spacing: 8) {
                ForEach(store.product.starters, id: \.self) { s in
                    Button(s) { store.ask(s) }.buttonStyle(.bordered).controlSize(.small)
                }
            }
        }
    }

    /// A roomy composer card, centred to the answer column: multi-line field, then the buttons
    /// on one baseline at the trailing edge (same size, vertically centred with each other).
    private var inputBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !store.status.isEmpty {
                Text(store.status).font(.caption).foregroundStyle(.secondary)
            }
            HStack(alignment: .center, spacing: 10) {
                TextField("Ask how to do something, how it works, or which settings to use…", text: $question, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.body)
                    .lineLimit(2...6)
                    .focused($focused)
                    .onSubmit(send)
                    .frame(minHeight: 44, alignment: .topLeading)
                    .helpTag("help.ask.field")
                // One container for the tag: `.tweakable` on a bare Group wraps its children in a single
                // view, which stacked the two buttons vertically.
                HStack(spacing: 10) {
                    Group {
                        Button { store.newChat() } label: { Image(systemName: "square.and.pencil") }
                            .help("New chat")
                            .disabled(store.turns.isEmpty)
                        if store.busy {
                            Button { store.stopAsk() } label: { Image(systemName: "stop.circle.fill") }
                                .help("Stop")
                        } else {
                            Button(action: send) { Image(systemName: "arrow.up.circle.fill") }
                                .disabled(question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                                .keyboardShortcut(.return, modifiers: .command)
                                .help("Ask (↩)")
                        }
                    }
                    .buttonStyle(.borderless)
                    .font(.system(size: 22))
                    .frame(width: 28, height: 28)
                }
                .helpTag("help.ask.buttons")
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(RoundedRectangle(cornerRadius: 12).fill(.background.opacity(0.55)))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.secondary.opacity(0.3)))
        }
        .frame(maxWidth: 820)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 20).padding(.top, 12).padding(.bottom, 24)
    }

    private func send() {
        let q = question
        question = ""
        store.ask(q)
    }
}

private struct TurnView: View {
    let turn: HelpTurn
    @State private var open: HelpChunk?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Spacer(minLength: 80)
                Text(turn.question)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Color.accentColor.opacity(0.18)))
            }
            if turn.offline {
                Text(turn.sources.isEmpty ? "No documentation matched — try other words." : "The closest documentation:")
                    .foregroundStyle(.secondary)
                ForEach(turn.sources) { c in
                    GroupBox(c.title) { MarkdownText(c.text).frame(maxWidth: .infinity, alignment: .leading) }
                }
            } else {
                if turn.answer.isEmpty && !turn.done {
                    ProgressView().controlSize(.small)
                } else {
                    MarkdownText(turn.answer)
                }
                if let error = turn.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
                if !turn.sources.isEmpty && (turn.done || !turn.answer.isEmpty) { sources }
            }
        }
    }

    private var sources: some View {
        FlowLayout(spacing: 6) {
            Text("Sources").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ForEach(Array(turn.sources.enumerated()), id: \.element.id) { i, c in
                Button { open = c } label: {
                    Text("[\(i + 1)] \(c.title)").font(.caption).lineLimit(1)
                        .padding(.horizontal, 7).padding(.vertical, 3)
                        .background(Capsule().fill(Color.secondary.opacity(0.15)))
                }
                .buttonStyle(.plain)
                .popover(item: Binding(get: { open?.id == c.id ? open : nil }, set: { open = $0 })) { chunk in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(chunk.title).font(.headline)
                            MarkdownText(chunk.text)
                        }
                        .padding(14)
                    }
                    .frame(width: 480, height: 360)
                }
            }
        }
    }
}

/// Light Markdown for answers and doc excerpts: inline styles per line, list markers as bullets,
/// headings bold, code fences monospaced. (SwiftUI's `Text` only does inline Markdown.)
public struct MarkdownText: View {
    let source: String
    public init(_ source: String) { self.source = source }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .code(let text):
                    Text(text).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                        .padding(8).frame(maxWidth: .infinity, alignment: .leading)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.1)))
                case .heading(let text):
                    inline(text).font(.headline).padding(.top, 4)
                case .bullet(let text, let indent):
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text("•")
                        inline(text)
                    }
                    .padding(.leading, CGFloat(indent) * 14)
                case .line(let text):
                    inline(text)
                case .gap:
                    Spacer().frame(height: 4)
                }
            }
        }
        .textSelection(.enabled)
    }

    enum Block { case code(String), heading(String), bullet(String, Int), line(String), gap }

    var blocks: [Block] {
        var out: [Block] = []
        var fence: [String]?
        for raw in source.components(separatedBy: "\n") {
            if raw.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                if let f = fence { out.append(.code(f.joined(separator: "\n"))); fence = nil } else { fence = [] }
                continue
            }
            if fence != nil { fence!.append(raw); continue }
            let line = raw.trimmingCharacters(in: .whitespaces)
            let indent = (raw.prefix { $0 == " " }.count) / 2
            if line.isEmpty {
                if case .gap = out.last {} else if !out.isEmpty { out.append(.gap) }
            } else if line.hasPrefix("#") {
                out.append(.heading(String(line.drop { $0 == "#" }).trimmingCharacters(in: .whitespaces)))
            } else if let r = line.range(of: #"^([*\-+]|\d+\.)\s+"#, options: .regularExpression) {
                out.append(.bullet(String(line[r.upperBound...]), min(indent, 3)))
            } else {
                out.append(.line(line))
            }
        }
        if let f = fence { out.append(.code(f.joined(separator: "\n"))) }
        return out
    }

    private func inline(_ text: String) -> Text {
        let attributed = (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(text)
        return Text(attributed)
    }
}

// MARK: - Feedback

private struct FeedbackView: View {
    @Environment(HelpCenter.self) private var store

    var body: some View {
        if store.diagnosticsEnabled { composer } else { disabled }
    }

    private var disabled: some View {
        VStack(spacing: 12) {
            Image(systemName: "lock.shield").font(.system(size: 40)).foregroundStyle(.secondary)
            Text("Diagnostics are off").font(.title3.weight(.semibold))
            Text("\(store.product.name) records nothing about how it runs until you turn diagnostics on. With them on, this tab helps you write a problem report or idea — collected and redacted on this Mac, and only shared if you copy, save or share it yourself.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary).frame(maxWidth: 520)
            Button("Open Settings → Diagnostics & feedback") { store.host?.openDiagnosticsSettings() }
        }
        .padding(30)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var composer: some View {
        @Bindable var store = store
        return HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            Picker("Type", selection: $store.feedbackKind) {
                                ForEach(FeedbackKind.allCases) { Text($0.title).tag($0) }
                            }
                            .helpTag("help.feedback.kind")
                            TextEditor(text: $store.feedbackDescription)
                                .font(.body)
                                .frame(minHeight: 110)
                                .helpTag("help.feedback.description")
                                .overlay(alignment: .topLeading) {
                                    if store.feedbackDescription.isEmpty {
                                        Text("What happened, or what would you like? What did you expect? Steps to reproduce help.")
                                            .foregroundStyle(.tertiary).padding(6).allowsHitTesting(false)
                                    }
                                }
                        }
                    } label: { Label("In your words", systemImage: "text.bubble") }
                    .helpTag("help.feedback.words")

                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Button { store.runDiagnostics() } label: {
                                    Label(store.feedbackSections.isEmpty ? "Run diagnostics" : "Run again", systemImage: "stethoscope")
                                }
                                .disabled(store.isCollectingDiagnostics)
                                if store.isCollectingDiagnostics { ProgressView().controlSize(.small) }
                                Spacer()
                                Text("\(store.diagnosticsLog.entries.count) events logged").font(.caption).foregroundStyle(.secondary)
                            }
                            if store.feedbackSections.isEmpty {
                                Text("Checks run on this Mac only. You choose which sections go into the report.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            ForEach($store.feedbackSections) { $section in SectionRow(section: $section).helpTag("help.feedback.section") }
                        }
                    } label: { Label("Diagnostics", systemImage: "checklist") }
                    .helpTag("help.feedback.diagnostics")

                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                if store.feedbackBusy && !store.isCollectingDiagnostics {
                                    Button("Stop") { store.stopFeedback() }
                                    ProgressView().controlSize(.small)
                                } else {
                                    Button { store.draftFeedback() } label: { Label("Draft with AI", systemImage: "wand.and.stars") }
                                        .disabled(!store.canChat || store.feedbackBusy
                                            || (store.feedbackDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && store.feedbackSections.isEmpty))
                                }
                                Spacer()
                                if !store.feedbackDraft.isEmpty { Button("Clear draft") { store.feedbackDraft = "" } }
                            }
                            Text(store.canChat
                                ? "Optional. \(store.product.modelName) (on this Mac) turns your words and the redacted diagnostics into a summary, likely cause and next steps, citing the docs. Read it — it can be wrong."
                                : "Needs the built-in language model; without it, write the report yourself.")
                                .font(.caption).foregroundStyle(.secondary)
                            if !store.feedbackDraft.isEmpty || store.feedbackBusy {
                                TextEditor(text: $store.feedbackDraft).font(.body).frame(minHeight: 180)
                                    .helpTag("help.feedback.draft")
                            }
                        }
                    } label: { Label("AI-assisted draft", systemImage: "sparkles") }
                    .helpTag("help.feedback.ai")
                }
                .padding(16)
            }
            preview.helpTag("help.feedback.preview").frame(minWidth: 320, idealWidth: 460)
        }
    }

    private var preview: some View {
        let markdown = store.feedbackMarkdown
        let redactions = store.redactionCount
        return VStack(alignment: .leading, spacing: 0) {
            HStack {
                Label("What you'll share", systemImage: "eye").font(.headline)
                Spacer()
                if redactions > 0 {
                    Label("\(redactions) personal detail\(redactions == 1 ? "" : "s") replaced", systemImage: "eye.slash")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(12)
            Divider()
            ScrollView {
                Text(markdown).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(12)
            }
            .helpTag("help.feedback.preview.text")
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Button { store.copyFeedback() } label: { Label("Copy", systemImage: "doc.on.doc") }
                    Button { store.saveFeedback() } label: { Label("Save…", systemImage: "square.and.arrow.down") }
                    ShareLink(item: markdown, subject: Text("\(store.product.name) feedback"), preview: SharePreview("\(store.product.name) feedback")) {
                        Label("Share…", systemImage: "square.and.arrow.up")
                    }
                    Spacer()
                }
                Text(store.feedbackStatus.isEmpty ? "Nothing is sent automatically — this text goes only where you copy, save or share it." : store.feedbackStatus)
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(12)
            .helpTag("help.feedback.preview.actions")
        }
    }
}

/// One diagnostics section: include toggle, problem count, items behind a disclosure.
private struct SectionRow: View {
    @Binding var section: DiagnosticSection
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 3) {
                if !section.note.isEmpty { Text(section.note).font(.caption).foregroundStyle(.secondary) }
                ForEach(Array(section.items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        icon(item.status).frame(width: 14)
                        Text(item.key).font(.caption.weight(.medium))
                        Text(item.value).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
            }
            .padding(.leading, 4)
        } label: {
            HStack {
                Toggle(isOn: $section.included) { Text(section.title) }.toggleStyle(.checkbox)
                let problems = section.problems.count
                if problems > 0 {
                    Text("\(problems)").font(.caption.weight(.semibold)).foregroundStyle(.white)
                        .padding(.horizontal, 6).background(Capsule().fill(section.problems.contains { $0.status == .fail } ? Color.red : Color.orange))
                }
            }
        }
    }

    @ViewBuilder private func icon(_ s: DiagnosticSection.Status) -> some View {
        switch s {
        case .ok: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .warn: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
        case .fail: Image(systemName: "xmark.octagon.fill").foregroundStyle(.red)
        case .info: Image(systemName: "info.circle").foregroundStyle(.secondary)
        }
    }
}
