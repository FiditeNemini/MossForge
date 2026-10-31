import Foundation
import MossForgeCore
import Network

/// The resident server's transport: a dependency-free HTTP/1.1 listener on Apple's Network
/// framework that feeds `MossServerRouter`. Shared by the app's Server screen and the
/// headless `moss-mlx serve`.
///
/// Local-first: with network access off (`NetworkPolicy`, the default) it binds loopback
/// only, invisible to the LAN; with access on it binds all interfaces.
public final class MossHTTPServer: @unchecked Sendable {
    public let router: MossServerRouter
    /// Called (on the server's queue) for lifecycle events, job start/finish, and one line
    /// per request.
    public var onLog: (@Sendable (String) -> Void)? {
        didSet { router.events.setHandler(onLog) }
    }

    private let lock = NSLock()
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "mossforge.server", qos: .userInitiated)

    public init(router: MossServerRouter) {
        self.router = router
    }

    public var isRunning: Bool { lock.withLock { listener != nil } }

    /// Start listening and wait until ready. `port` 0 picks a free one; returns the bound port.
    @discardableResult
    public func start(port: UInt16, policy: NetworkPolicy) async throws -> UInt16 {
        guard !isRunning else { throw EngineError.invalidRequest("server already running") }
        let nwPort = NWEndpoint.Port(rawValue: port) ?? .any
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        let listener: NWListener
        if policy.egressAllowed {
            listener = try NWListener(using: params, on: nwPort)
        } else {
            // Pin the local endpoint to loopback so off-box connections are never accepted.
            params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: nwPort)
            listener = try NWListener(using: params)
        }
        listener.newConnectionHandler = { [weak self] conn in self?.accept(conn) }
        lock.withLock { self.listener = listener }

        let bound: UInt16 = try await withCheckedThrowingContinuation { cont in
            let once = OnceFlag()
            listener.stateUpdateHandler = { [weak self] state in
                switch state {
                case .ready:
                    if once.claim() { cont.resume(returning: listener.port?.rawValue ?? port) }
                case .failed(let error):
                    self?.onLog?("listener failed — \(error.localizedDescription)")
                    self?.stop()
                    if once.claim() { cont.resume(throwing: error) }
                case .cancelled:
                    if once.claim() { cont.resume(throwing: EngineError.cancelled) }
                default:
                    break
                }
            }
            listener.start(queue: queue)
        }
        onLog?("listening on \(policy.serverBindHost):\(bound)")
        return bound
    }

    public func stop() {
        let l = lock.withLock { () -> NWListener? in
            defer { listener = nil }
            return listener
        }
        guard let l else { return }
        l.cancel()
        onLog?("stopped")
    }

    // MARK: - Connections

    private func accept(_ conn: NWConnection) {
        conn.start(queue: queue)
        receive(conn, buffer: Data())
    }

    private func receive(_ conn: NWConnection, buffer: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, isComplete, error in
            guard let self else { conn.cancel(); return }
            var buf = buffer
            if let data { buf.append(data) }
            switch HTTPRequestParser.parse(buf) {
            case .complete(let request):
                self.respond(conn, to: request)
            case .invalid:
                self.send(.error(400, message: "malformed or oversized request"), on: conn)
            case .incomplete:
                if error != nil || isComplete { conn.cancel() } else { self.receive(conn, buffer: buf) }
            }
        }
    }

    private func respond(_ conn: NWConnection, to request: HTTPRequest) {
        let router = router
        Task { [weak self] in
            let started = Date()
            let response = await router.handle(request)
            let ms = Int(Date().timeIntervalSince(started) * 1000)
            // Status polling (MCP clients poll jobs every ~1.5 s) would drown the log; job
            // start/finish lines already come from the router.
            let polling = request.method == "GET" && response.status == 200
                && (request.path.hasPrefix(MossServerRouter.jobPrefix) || request.path == "/v1/status")
            if !polling {
                self?.onLog?("\(response.status) · \(request.method) \(request.path) · \(ms)ms")
            }
            self?.send(response, on: conn)
        }
    }

    private func send(_ response: HTTPResponse, on conn: NWConnection) {
        conn.send(content: response.serialized(), completion: .contentProcessed { _ in conn.cancel() })
    }
}

/// Resume-once guard for the start continuation (state updates can repeat).
private final class OnceFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func claim() -> Bool { lock.withLock { defer { done = true }; return !done } }
}
