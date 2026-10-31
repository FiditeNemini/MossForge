/// Local-first network policy. MossForge runs no cloud inference; the only network
/// surfaces are the optional resident HTTP server (ingress) and future model
/// downloads (egress). This value type centralises the decision so every call site
/// — the server's bind host, a would-be downloader — asks the same gate, and the
/// Settings → Privacy toggle has a single, observable meaning.
public struct NetworkPolicy: Sendable, Equatable, Codable {
    /// Master switch (Settings → Privacy → "Allow network access"). Off by default:
    /// the app is fully usable with zero network activity.
    public var egressAllowed: Bool

    public init(egressAllowed: Bool = false) {
        self.egressAllowed = egressAllowed
    }

    /// The host the resident server binds to. With access OFF the server is
    /// loopback-only — reachable from this Mac but invisible to the LAN. With
    /// access ON it binds all interfaces so other devices can reach it.
    public var serverBindHost: String { egressAllowed ? "0.0.0.0" : "127.0.0.1" }

    /// Whether an outbound request to `host` is permitted. Loopback is always
    /// allowed (the app talking to its own resident server); anything else needs
    /// the master switch on.
    public func allowsOutbound(to host: String) -> Bool {
        egressAllowed || Self.isLoopback(host)
    }

    /// One-line explanation rendered under the Settings toggle.
    public var summary: String {
        egressAllowed
            ? "On — the resident server is reachable on your local network and model downloads are permitted."
            : "Off — loopback only. No LAN exposure, no downloads, no telemetry, no cloud."
    }

    static func isLoopback(_ host: String) -> Bool {
        host == "127.0.0.1" || host == "::1" || host == "localhost"
    }
}
