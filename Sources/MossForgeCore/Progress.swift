import Foundation

/// A step-wise progress update emitted by long-running engine operations
/// (generation, training, conversion). Streamed to a UI or the CLI.
public struct GenerationProgress: Sendable, Equatable {
    /// The current step (1-based), or 0 before the loop starts.
    public var step: Int
    /// Total steps for the operation, or 0 if unknown.
    public var totalSteps: Int
    /// Short, human-readable status line.
    public var message: String
    /// A metric for charts (training: the step's loss), if the operation has one.
    public var value: Double?
    /// Files the step produced (training: that checkpoint's test images).
    public var files: [URL]

    /// Fraction complete, clamped to `0.0...1.0`.
    public var fraction: Double {
        guard totalSteps > 0 else { return 0 }
        return max(0, min(1, Double(step) / Double(totalSteps)))
    }

    public init(step: Int, totalSteps: Int, message: String = "", value: Double? = nil, files: [URL] = []) {
        self.step = step
        self.totalSteps = totalSteps
        self.message = message
        self.value = value
        self.files = files
    }
}

/// A sink for streaming progress updates from the engine to a UI or the CLI.
///
/// The facade also exposes `AsyncStream<GenerationProgress>` wrappers over this
/// for SwiftUI consumers (FR-GEN-005); the closure is the backend-facing seam.
public typealias ProgressHandler = @Sendable (GenerationProgress) -> Void
