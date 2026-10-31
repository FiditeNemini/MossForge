import Foundation
import MossForgeCore

// MARK: - Capability protocols (the engine seam)
// Front-ends depend only on these. More capabilities (LoRA
// training, convert/quantize, fuse/merge) are added phase by phase.

/// Text-to-image generation (plain, tag, or JSON-layout prompts).
public protocol ImageGenerating: Sendable {
    func generate(_ request: GenerationRequest, progress: ProgressHandler?) async throws -> GenerationResult
}

/// Expand a short request into a detailed prompt or a JSON layout prompt.
public protocol PromptEnhancing: Sendable {
    func enhancePrompt(_ request: PromptEnhanceRequest, progress: ProgressHandler?) async throws -> String
}

/// Enlarge an image with VOSR2 (generative) or an ESRGAN-family model (fast).
public protocol ImageUpscaling: Sendable {
    func upscale(_ request: UpscaleRequest, progress: ProgressHandler?) async throws -> UpscaleResult
}

/// Describe an image with the on-device VLM (official recaption prompt → long/medium/short).
public protocol ImageCaptioning: Sendable {
    func caption(_ request: CaptionRequest, progress: ProgressHandler?) async throws -> MossCaption
}

/// Convert / quantize, fuse LoRAs into, and merge DiTs.
public protocol ModelTooling: Sendable {
    func runModelTool(_ request: ModelToolRequest, progress: ProgressHandler?) async throws -> ModelToolResult
}

/// Train a LoRA on a folder of images + captions.
public protocol LoRATraining: Sendable {
    func trainLoRA(_ request: TrainingRequest, samples: [TrainingSample], progress: ProgressHandler?) async throws -> TrainingResult
}

/// Chat with the on-device language model (Help → Ask, Feedback → Draft with AI). `onText` streams
/// the reply so far.
public protocol TextChatting: Sendable {
    func chat(_ request: ChatRequest, progress: ProgressHandler?, onText: TextStreamHandler?) async throws -> String
}

/// Identify a checkpoint (DiT / encoder / VAE / LoRA / Turbo sampler) from its header.
public protocol ModelInspecting: Sendable {
    func inspectModel(_ url: URL) async throws -> MossModelKind
}

public struct EngineCapabilities: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let imageGeneration = EngineCapabilities(rawValue: 1 << 0)
    public static let promptEnhancing = EngineCapabilities(rawValue: 1 << 1)
    public static let modelInspection = EngineCapabilities(rawValue: 1 << 2)
    public static let loRAApply = EngineCapabilities(rawValue: 1 << 3)
    public static let turboSampling = EngineCapabilities(rawValue: 1 << 4)
    public static let upscaling = EngineCapabilities(rawValue: 1 << 5)
    public static let captioning = EngineCapabilities(rawValue: 1 << 6)
    public static let modelTooling = EngineCapabilities(rawValue: 1 << 7)
    public static let loRATraining = EngineCapabilities(rawValue: 1 << 8)
    public static let chat = EngineCapabilities(rawValue: 1 << 9)

    public static let all: EngineCapabilities = [
        .imageGeneration, .promptEnhancing, .modelInspection, .loRAApply, .turboSampling, .upscaling, .captioning, .modelTooling, .loRATraining, .chat,
    ]

    public var names: [String] {
        [
            (Self.imageGeneration, "generate"), (.promptEnhancing, "enhance"), (.modelInspection, "inspect"),
            (.loRAApply, "lora"), (.turboSampling, "turbo"), (.upscaling, "upscale"), (.captioning, "caption"),
            (.modelTooling, "tools"), (.loRATraining, "train"), (.chat, "chat"),
        ].compactMap { contains($0.0) ? $0.1 : nil }
    }
}

/// A compute backend. `StubBackend` is the default; the MLX backend lives in `MossForgeMLX`.
public protocol EngineBackend: ImageGenerating, PromptEnhancing, ModelInspecting, ImageUpscaling, ImageCaptioning, ModelTooling, LoRATraining, TextChatting {
    var capabilities: EngineCapabilities { get }
    /// Short identifier, e.g. "stub" or "mlx".
    var name: String { get }
    /// Drop resident weights so memory returns to the system.
    func releaseResidentModels() async
    /// Backend state for Feedback → diagnostics (device, memory, loaded models). Nothing personal.
    func diagnostics() async -> [DiagnosticSection.Item]
}

extension EngineBackend {
    public func releaseResidentModels() async {}

    public func diagnostics() async -> [DiagnosticSection.Item] { [] }

    /// No language model by default (the stub); `MossBackend` runs Qwen3.5.
    public func chat(_ request: ChatRequest, progress: ProgressHandler?, onText: TextStreamHandler?) async throws -> String {
        throw EngineError.notImplemented(capability: "chat")
    }

    /// Header inspection is pure logic — every backend gets it for free.
    public func inspectModel(_ url: URL) async throws -> MossModelKind {
        guard FileManager.default.fileExists(atPath: url.path) else { throw EngineError.inputNotFound(url) }
        return try MossInspector.inspect(url)
    }
}
