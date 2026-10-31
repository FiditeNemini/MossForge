import Foundation
import MossForgeCore

/// The single object the app, CLI and server talk to. Swapping the stub for the MLX backend is
/// one line at construction: `MossEngine(backend: MossBackend(...))` (MossForgeMLX package).
public struct MossEngine: Sendable {
    public let backend: any EngineBackend

    public init(backend: any EngineBackend = StubBackend()) {
        self.backend = backend
    }

    public var capabilities: EngineCapabilities { backend.capabilities }
    public var backendName: String { backend.name }

    public func generate(_ request: GenerationRequest, progress: ProgressHandler? = nil) async throws -> GenerationResult {
        try request.validate()
        return try await backend.generate(request, progress: progress)
    }

    public func enhancePrompt(_ request: PromptEnhanceRequest, progress: ProgressHandler? = nil) async throws -> String {
        try await backend.enhancePrompt(request, progress: progress)
    }

    public func upscale(_ request: UpscaleRequest, progress: ProgressHandler? = nil) async throws -> UpscaleResult {
        try await backend.upscale(request, progress: progress)
    }

    public func caption(_ request: CaptionRequest, progress: ProgressHandler? = nil) async throws -> MossCaption {
        try await backend.caption(request, progress: progress)
    }

    /// Train a LoRA (validated here, before any weights load).
    public func trainLoRA(_ request: TrainingRequest, samples: [TrainingSample], progress: ProgressHandler? = nil) async throws -> TrainingResult {
        guard capabilities.contains(.loRATraining) else { throw EngineError.notImplemented(capability: "LoRA training") }
        try request.validate(sampleCount: samples.count)
        return try await backend.trainLoRA(request, samples: samples, progress: progress)
    }

    /// Run a convert / fuse / merge, then give the output its sidecar (family, built-in Turbo recipe)
    /// and copy what a built-in data-space Turbo still needs beside it.
    public func runModelTool(_ request: ModelToolRequest, progress: ProgressHandler? = nil) async throws -> ModelToolResult {
        guard capabilities.contains(.modelTooling) else { throw EngineError.notImplemented(capability: "model tooling") }
        try request.validate()
        let result = try await backend.runModelTool(request, progress: progress)
        let source = ModelSidecar.load(for: request.input, isLoRA: false)
        let (sidecar, copies) = ModelToolRequest.outputSidecar(
            source: source, operation: request.operation, precision: request.precision,
            sourceName: request.input.deletingPathExtension().lastPathComponent, output: request.output)
        for (from, to) in copies where !FileManager.default.fileExists(atPath: to.path) {
            try FileManager.default.copyItem(at: from, to: to)
        }
        try sidecar.write(for: request.output)
        return result
    }

    /// Chat with the on-device language model. The transcript must end on a user turn.
    public func chat(_ request: ChatRequest, progress: ProgressHandler? = nil, onText: TextStreamHandler? = nil) async throws -> String {
        guard capabilities.contains(.chat) else { throw EngineError.notImplemented(capability: "chat (no language model loaded)") }
        guard request.messages.last?.role == .user else { throw EngineError.invalidRequest("a chat must end with a user message") }
        return try await backend.chat(request, progress: progress, onText: onText)
    }

    public func diagnostics() async -> [DiagnosticSection.Item] {
        await backend.diagnostics()
    }

    public func inspectModel(_ url: URL) async throws -> MossModelKind {
        try await backend.inspectModel(url)
    }

    public func releaseResidentModels() async {
        await backend.releaseResidentModels()
    }

    /// Progress as an `AsyncStream` for SwiftUI; the result arrives via `onResult`.
    public func generateStream(
        _ request: GenerationRequest,
        onResult: @escaping @Sendable (Result<GenerationResult, any Error>) -> Void
    ) -> AsyncStream<GenerationProgress> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    onResult(.success(try await generate(request) { continuation.yield($0) }))
                } catch {
                    onResult(.failure(error))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
