import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

/// The on-device prompt / layout enhancer: Qwen3.5-9B (the same file as the text encoder, shared with it
/// through `SharedQwen35`, plus its `lm_head`) writes a draft per the official `moss-image-prompt-enhancer` skill (`MossEnhancerPrompt`,
/// non-thinking chat), and `MossPromptCompiler` validates and compiles it into the MOSS prompt — a
/// caption, or one-line abs_centralized/yxyx layout JSON. One retry with the validation error.
public final class MossEnhancer: @unchecked Sendable {
    public let paths: MossModelPaths
    public var sampling = Qwen35Generator.Sampling.qwenNonThinking
    private let qwen: SharedQwen35
    private var tokenizer: MossTokenizer?

    /// - Parameter qwen: the backend's shared Qwen3.5 (default: a private one for this file).
    public init(paths: MossModelPaths, qwen: SharedQwen35? = nil) {
        self.paths = paths
        self.qwen = qwen ?? SharedQwen35(url: paths.encoder)
    }

    public func release() { qwen.trim() }

    /// Enhance a prompt. Returns the compiled MOSS prompt (caption text or layout JSON).
    public func enhance(_ request: PromptEnhanceRequest, progress: ProgressHandler? = nil) async throws -> String {
        if tokenizer == nil { tokenizer = try await MossTokenizer.load(folder: paths.tokenizer) }
        progress?(GenerationProgress(step: 0, totalSteps: 0, message: "loading Qwen3.5"))
        let (tok, gen) = (tokenizer!, try qwen.generator())
        guard let stop = tok.id(of: MossEnhancerPrompt.stopToken) else {
            throw EngineError.backendFailure("tokenizer has no \(MossEnhancerPrompt.stopToken)")
        }
        var retry: (answer: String, error: String)?
        for attempt in 1...2 {
            let chat = MossEnhancerPrompt.chat(
                prompt: request.prompt, layout: request.layout, canvas: request.canvas, retry: retry, prompts: request.prompts)
            var produced = 0
            var sampling = sampling
            sampling.seed &+= UInt64(attempt)
            let ids = try gen.generate(prompt: tok.ids(chat), stop: [stop], sampling: sampling) { _ in
                try Task.checkCancellation()
                produced += 1
                if produced % 16 == 0 {
                    progress?(GenerationProgress(step: produced, totalSteps: sampling.maxTokens, message: "writing \(produced) tokens"))
                }
            }
            let answer = tok.decode(ids).trimmingCharacters(in: .whitespacesAndNewlines)
            do {
                guard let json = MossPromptCompiler.extractJSON(from: answer) else {
                    throw EngineError.invalidRequest("no JSON object in the answer")
                }
                let out = try MossPromptCompiler.compile(MossPromptCompiler.tidyDraft(Data(json.utf8)))
                if request.layout, out.size == nil { throw EngineError.invalidRequest("expected a Layout draft, got a caption") }
                if !request.layout, out.size != nil { throw EngineError.invalidRequest("expected a caption, got a Layout draft") }
                return out.prompt
            } catch let EngineError.invalidRequest(reason) where attempt == 1 {
                retry = (answer, reason)
                progress?(GenerationProgress(step: 0, totalSteps: 0, message: "fixing draft: \(reason)"))
            }
        }
        throw EngineError.backendFailure("the enhancer could not produce a valid draft")
    }
}
