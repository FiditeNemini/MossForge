import Foundation

/// Errors surfaced by the MossForge engine.
public enum EngineError: Error, Sendable, CustomStringConvertible {
    /// The requested capability has not been implemented yet.
    case notImplemented(capability: String)
    /// An input file or directory was missing.
    case inputNotFound(URL)
    /// The request was malformed.
    case invalidRequest(String)
    /// LoRA key mapping could not place these tensor keys onto the base model.
    /// The portability layer reports unmapped keys rather than silently dropping
    /// them (FR-LORA-003).
    case unmappedKeys([String])
    /// The operation was cancelled cleanly (no corrupt temp files left behind).
    case cancelled
    /// The underlying compute backend failed.
    case backendFailure(String)
    /// A download needs network access, which the local-first policy has off.
    case networkAccessDisabled(String)

    public var description: String {
        switch self {
        case .notImplemented(let capability):
            "Not implemented yet: \(capability)"
        case .inputNotFound(let url):
            "Input not found: \(url.path)"
        case .invalidRequest(let reason):
            "Invalid request: \(reason)"
        case .unmappedKeys(let keys):
            "Unmapped LoRA keys (\(keys.count)): \(keys.prefix(5).joined(separator: ", "))"
                + (keys.count > 5 ? " …" : "")
        case .cancelled:
            "Operation cancelled"
        case .backendFailure(let reason):
            "Backend failure: \(reason)"
        case .networkAccessDisabled(let message):
            message
        }
    }
}
