import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

/// Help → Ask and Feedback → Draft with AI: Qwen3.5-9B (the text encoder file + its `lm_head`, like
/// `MossEnhancer`) answering a ChatML transcript in non-thinking mode. Stays loaded between questions;
/// the backend releases it before a generation, caption, upscale or training job.
public final class MossAssistant: @unchecked Sendable {
    public let paths: MossModelPaths
    private let qwen: SharedQwen35
    private var tokenizer: MossTokenizer?
    /// Answered since the last release (its Qwen3.5 is the shared one, so this is about use, not memory).
    public private(set) var isLoaded = false

    /// - Parameter qwen: the backend's shared Qwen3.5 (default: a private one for this file).
    public init(paths: MossModelPaths, qwen: SharedQwen35? = nil) {
        self.paths = paths
        self.qwen = qwen ?? SharedQwen35(url: paths.encoder)
    }

    public func release() {
        guard isLoaded else { return }
        isLoaded = false
        qwen.trim()
    }

    public func chat(_ request: ChatRequest, progress: ProgressHandler? = nil, onText: TextStreamHandler? = nil) async throws -> String {
        if tokenizer == nil { tokenizer = try await MossTokenizer.load(folder: paths.tokenizer) }
        if !qwen.isLoaded { progress?(GenerationProgress(step: 0, totalSteps: 0, message: "loading Qwen3.5")) }
        let (tok, gen) = (tokenizer!, try qwen.generator())
        isLoaded = true
        guard let stop = tok.id(of: QwenChatTemplate.stopToken) else {
            throw EngineError.backendFailure("tokenizer has no \(QwenChatTemplate.stopToken)")
        }
        var sampling = Qwen35Generator.Sampling.qwenNonThinking
        sampling.temperature = request.temperature
        sampling.seed = request.seed
        sampling.maxTokens = request.maxTokens
        let prompt = tok.ids(QwenChatTemplate.render(request.messages))
        progress?(GenerationProgress(step: 0, totalSteps: request.maxTokens, message: "reading \(prompt.count) tokens"))
        var produced: [Int] = []
        let ids = try gen.generate(prompt: prompt, stop: [stop], sampling: sampling) { id in
            try Task.checkCancellation()
            produced.append(id)
            // Re-decode the whole reply (cheap at these lengths) so multi-token characters come out whole.
            if produced.count % 4 == 0 { onText?(tok.decode(produced)) }
            if produced.count % 16 == 0 {
                progress?(GenerationProgress(step: produced.count, totalSteps: request.maxTokens, message: "writing"))
            }
        }
        let text = tok.decode(ids).trimmingCharacters(in: .whitespacesAndNewlines)
        onText?(text)
        return text
    }
}
