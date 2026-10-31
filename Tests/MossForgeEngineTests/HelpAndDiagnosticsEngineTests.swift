import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// The chat seam (Help → Ask, Feedback → Draft with AI) and the diagnostics collector.
@Suite struct HelpAndDiagnosticsEngineTests {
    @Test func chatIsACapabilityTheStubDoesNotClaim() async {
        #expect(EngineCapabilities.all.names.contains("chat"))
        let engine = MossEngine()
        #expect(!engine.capabilities.contains(.chat))
        await #expect(throws: EngineError.self) {
            _ = try await engine.chat(ChatRequest(messages: [.init(.user, "hi")]))
        }
    }

    @Test func chatRequestsNeedAUserTurnLast() async {
        let engine = MossEngine(backend: EchoBackend())
        await #expect(throws: EngineError.self) { _ = try await engine.chat(ChatRequest(messages: [])) }
        await #expect(throws: EngineError.self) { _ = try await engine.chat(ChatRequest(messages: [.init(.system, "s")])) }
        let reply = try? await engine.chat(ChatRequest(messages: [.init(.system, "s"), .init(.user, "ping")]))
        #expect(reply == "echo: ping")
    }

    // MARK: diagnostics

    private func folder(_ paths: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("diag-\(UUID())")
        for p in paths {
            let url = root.appendingPathComponent(p)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: url.path, contents: Data())
        }
        return root
    }

    private func context(models: URL, log: DiagnosticsLog = .init()) -> DiagnosticsContext {
        MossDiagnostics.context(
            appVersion: "0.1.0 (12)", backendName: "stub", capabilities: [.imageGeneration], modelsDirectory: models,
            outputDirectory: models.appendingPathComponent("out"), settings: [("Theme", "native")], log: log,
            backendItems: [.init("Device", "test GPU")])
    }

    @Test func missingComponentsFailAndNothingNamesTheUsersFolders() throws {
        let models = try folder(["lora/mine/secret-portrait.safetensors"])
        defer { try? FileManager.default.removeItem(at: models) }
        let sections = DiagnosticsCollector.collect(context(models: models))
        #expect(sections.map(\.id) == ["app", "system", "storage", "models", "compute", "modelFiles", "settings", "activity"])

        let comps = try #require(sections.first { $0.id == "models" })
        let tokenizer = try #require(comps.items.first { $0.key == "Qwen3.5 tokenizer" })
        #expect(tokenizer.status == .fail)
        #expect(tokenizer.value.contains("<models>/text_encoders/qwen3.5-9b-tokenizer"))
        // Optional pieces are warnings at most, never failures.
        #expect(comps.items.first { $0.key == "VOSR2 upscaler" }?.status != .fail)

        let app = try #require(sections.first { $0.id == "app" })
        #expect(app.items.contains { $0.key == "Backend" && $0.status == .warn })  // stub = no MLX models found

        // File names only in the opt-in section, which starts off.
        let files = try #require(sections.first { $0.id == "modelFiles" })
        #expect(!files.included)
        #expect(files.items.contains { $0.value.contains("secret-portrait") })
        let elsewhere = sections.filter { $0.id != "modelFiles" }.flatMap(\.items).map { $0.key + $0.value }.joined()
        #expect(!elsewhere.contains("secret-portrait"))
        #expect(!elsewhere.contains(models.path))
    }

    @Test func presentFilesAreCheckedByTheirHeaders() throws {
        let models = try folder(["vae/flux2-vae.safetensors", "text_encoders/qwen3.5-9b-tokenizer/tokenizer.json"])
        defer { try? FileManager.default.removeItem(at: models) }
        let comps = try #require(DiagnosticsCollector.collect(context(models: models)).first { $0.id == "models" })
        // An empty file is "present" but its header can't be read.
        #expect(comps.items.first { $0.key == "FLUX.2 VAE" }?.status == .fail)
        #expect(comps.items.first { $0.key == "Qwen3.5 tokenizer" }?.status == .ok)
    }

    @Test func activityIsOptInAndListsEntries() throws {
        var log = DiagnosticsLog()
        log.append(.error, area: "generate", "out of memory", at: Date(timeIntervalSince1970: 0))
        let models = try folder([])
        defer { try? FileManager.default.removeItem(at: models) }
        let activity = try #require(DiagnosticsCollector.collect(context(models: models, log: log)).first { $0.id == "activity" })
        #expect(!activity.included)
        #expect(activity.items.count == 1)
        #expect(activity.items[0].status == .fail)
        #expect(activity.items[0].value == "out of memory")
        #expect(activity.items[0].key.hasSuffix("generate"))
    }
}

/// Answers with the last user message — the seam without a model.
private struct EchoBackend: EngineBackend {
    var name: String { "echo" }
    var capabilities: EngineCapabilities { [.chat] }
    func chat(_ request: ChatRequest, progress: ProgressHandler?, onText: TextStreamHandler?) async throws -> String {
        "echo: " + (request.messages.last?.content ?? "")
    }
    func generate(_ request: GenerationRequest, progress: ProgressHandler?) async throws -> GenerationResult { throw EngineError.cancelled }
    func enhancePrompt(_ request: PromptEnhanceRequest, progress: ProgressHandler?) async throws -> String { "" }
    func upscale(_ request: UpscaleRequest, progress: ProgressHandler?) async throws -> UpscaleResult { throw EngineError.cancelled }
    func caption(_ request: CaptionRequest, progress: ProgressHandler?) async throws -> MossCaption { throw EngineError.cancelled }
    func runModelTool(_ request: ModelToolRequest, progress: ProgressHandler?) async throws -> ModelToolResult { throw EngineError.cancelled }
    func trainLoRA(_ request: TrainingRequest, samples: [TrainingSample], progress: ProgressHandler?) async throws -> TrainingResult {
        throw EngineError.cancelled
    }
}
