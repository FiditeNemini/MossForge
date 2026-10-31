import Foundation
import MossForgeCore
import MossForgeEngine

/// The store as Help's host: Ask and Feedback state and logic live
/// in `HelpCenter` (`store.help`); this supplies the model, the app state and the diagnostics.
extension AppStore: HelpHost {
    var canChat: Bool { engine.capabilities.contains(.chat) }

    var diagnosticsEnabled: Bool { settings.diagnosticsEnabled }

    /// A one-line picture of the app for the assistant ("Turbo on, 1024×1024, model …") — settings
    /// only, never the prompt.
    func helpAppState() -> String {
        var parts = ["backend \(backendName)"]
        if let dit = selectedDiT { parts.append("model \(dit.name) (\(family.displayName))") }
        if let p = activeProfile { parts.append("profile “\(p.name)”") }
        parts.append(builtInTurbo != nil || activeTurbo != nil ? "Turbo on" : "Turbo off (CFG \(draft.cfg), shift \(draft.shift))")
        parts.append("size \(draft.size.width)×\(draft.size.height)")
        if !draft.loras.isEmpty { parts.append("\(draft.loras.count) LoRA(s)") }
        parts.append("mode \(draft.useLayout ? "Layout" : "Prompt")")
        return parts.joined(separator: ", ")
    }

    /// One chat on the resident Qwen3.5, queued behind any running GPU job.
    func helpChat(
        _ messages: [ChatMessage], maxTokens: Int, temperature: Float,
        status: @escaping @MainActor (String) -> Void, onText: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        let request = ChatRequest(messages: messages, maxTokens: maxTokens, temperature: temperature)
        let (engine, queue) = (engine, inferenceQueue)
        if await queue.pending > 0 { status("Waiting for the GPU (another job is running)…") }
        return try await queue.run {
            try await engine.chat(request) { p in
                Task { @MainActor in status(p.message == "loading Qwen3.5" ? "Loading Qwen3.5 (first use only)…" : "Thinking…") }
            } onText: { text in
                Task { @MainActor in onText(text) }
            }
        }
    }

    func diagnosticsContext(log: DiagnosticsLog) async -> DiagnosticsContext {
        let s = settings
        return MossDiagnostics.context(
            appVersion: appVersion, backendName: backendName, capabilities: engine.capabilities, modelsDirectory: s.modelsDirectory,
            outputDirectory: s.outputDirectory, overrides: s.overrides,
            settings: [
                ("Theme", s.theme.rawValue), ("LAN access", s.egressEnabled ? "on" : "off"), ("Server port", "\(s.serverPort)"),
                ("Server", server.running ? "running" : "stopped"),
                ("Idle unload", s.idleUnloadMinutes == 0 ? "never" : "\(s.idleUnloadMinutes) min"),
                ("Memory", s.residencyMode.rawValue),
                ("Saved profiles", "\(s.profiles.count)"),
            ],
            log: log, backendItems: await engine.diagnostics())
    }

    /// This Mac's folders → placeholders.
    func redactorPlaceholders() -> [(String, String)] {
        [(settings.modelsDirectory.standardizedFileURL.path, "<models>"), (settings.outputDirectory.standardizedFileURL.path, "<output>")]
    }

    func openDiagnosticsSettings() { selection = .settings }

    /// Record a job outcome — only while diagnostics are on. Never pass prompts or image names.
    func note(_ level: DiagnosticsLog.Level, _ area: String, _ message: String) {
        help.note(level, area, message)
    }

    static func seconds(since t0: Date) -> String { String(format: "%.1f s", Date().timeIntervalSince(t0)) }

    var appVersion: String {
        let info = Bundle.main.infoDictionary ?? [:]
        guard let v = info["CFBundleShortVersionString"] as? String else { return "\(MossForge.version) (dev build)" }
        return (info["CFBundleVersion"] as? String).map { "\(v) (\($0))" } ?? v
    }
}

extension ModelToolOperation {
    var name: String {
        switch self {
        case .convert: "convert"
        case .fuse: "fuse"
        case .merge: "merge"
        case .extractLoRA: "extract-lora"
        }
    }
}
