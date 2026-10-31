import Foundation
import MossForgeCore
import MossForgeEngine
import Observation

/// The Server screen's view of the **real** resident server: a `MossHTTPServer` (the same
/// listener + routes `moss-mlx serve` runs) over the live engine, with start/stop, the bind
/// host and a request log.
///
/// Local-first: with network access OFF (the default) it binds loopback only
/// (`127.0.0.1`), invisible to the LAN; turning access on in Settings rebinds to all
/// interfaces. The bind host is decided by `NetworkPolicy`, the single egress gate.
@MainActor
@Observable
final class ServerController {
    struct LogEntry: Identifiable {
        let id = UUID()
        let line: String
        let date = Date()
    }

    private(set) var running = false
    private(set) var log: [LogEntry] = []
    /// Last error surfaced to the UI (e.g. port in use).
    private(set) var lastError: String?

    private var server: MossHTTPServer?
    /// Shared with the app's own generation (`AppStore.inferenceQueue`).
    private let queue: InferenceQueue
    private var engine = MossEngine()
    private var modelsDirectory = URL(fileURLWithPath: ".")
    private var policy = NetworkPolicy()
    private var port = 7860
    private var idleUnloadMinutes = -1

    init(queue: InferenceQueue) {
        self.queue = queue
    }

    /// Point the server at the current engine + settings. Restarts if anything it serves
    /// with changed while running.
    func attach(engine: MossEngine, settings: AppSettings) {
        let changed = engine.backendName != self.engine.backendName
            || settings.modelsDirectory != modelsDirectory
            || settings.networkPolicy != policy || settings.serverPort != port
        self.engine = engine
        modelsDirectory = settings.modelsDirectory
        policy = settings.networkPolicy
        port = settings.serverPort
        if changed || settings.idleUnloadMinutes != idleUnloadMinutes {
            idleUnloadMinutes = settings.idleUnloadMinutes
            // Covers in-app generation too: the queue is shared, so the clock is "no GPU job".
            let minutes = idleUnloadMinutes
            let queue = queue
            Task {
                await queue.setIdleRelease(after: minutes > 0 ? .seconds(minutes * 60) : nil) {
                    await engine.releaseResidentModels()
                }
            }
        }
        if running && changed { restart() }
    }

    var bindHost: String { policy.serverBindHost }

    func toggle() { running ? stop() : start() }

    func restart() {
        stop()
        start()
    }

    func start() {
        guard server == nil else { return }
        lastError = nil
        guard let port = UInt16(exactly: port), port > 0 else {
            lastError = "Invalid port \(self.port)."
            return
        }
        let server = MossHTTPServer(router: MossServerRouter(
            engine: engine, modelsDirectory: modelsDirectory, queue: queue))
        server.onLog = { [weak self] line in Task { @MainActor in self?.append(line) } }
        self.server = server
        running = true
        let policy = policy
        Task {
            do {
                try await server.start(port: port, policy: policy)
            } catch {
                lastError = "Failed to start: \(error.localizedDescription)"
                append("start failed — \(error.localizedDescription)")
                if self.server === server { self.server = nil; running = false }
            }
        }
    }

    func stop() {
        server?.stop()
        server = nil
        running = false
    }

    private func append(_ line: String) {
        log.insert(LogEntry(line: line), at: 0)
        if log.count > 60 { log.removeLast(log.count - 60) }
    }
}
