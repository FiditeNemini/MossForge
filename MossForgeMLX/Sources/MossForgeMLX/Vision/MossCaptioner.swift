import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

/// The on-device captioner: Qwen3.5-9B's vision tower turns the image into 4096-d tokens, which are
/// spliced over the `<|image_pad|>` run of the official recaption chat (`MossCaptionPrompt`); the
/// language model (`Qwen35Generator`, same file) then writes the recaption JSON with M-RoPE positions
/// (`QwenMRoPEIndex`). One retry when the answer doesn't parse.
public final class MossCaptioner: @unchecked Sendable {
    public let paths: MossModelPaths
    public var sampling: Qwen35Generator.Sampling = {
        var s = Qwen35Generator.Sampling.qwenNonThinking
        s.maxTokens = 2048  // long captions of text-heavy images
        return s
    }()
    private let qwen: SharedQwen35
    private var tokenizer: MossTokenizer?

    /// - Parameter qwen: the backend's shared Qwen3.5 (default: a private one for this file).
    public init(paths: MossModelPaths, qwen: SharedQwen35? = nil) {
        self.paths = paths
        self.qwen = qwen ?? SharedQwen35(url: paths.encoder)
    }

    public func release() { qwen.trim() }

    public func caption(_ request: CaptionRequest, progress: ProgressHandler? = nil) async throws -> MossCaption {
        guard FileManager.default.fileExists(atPath: request.image.path) else { throw EngineError.inputNotFound(request.image) }
        if tokenizer == nil { tokenizer = try await MossTokenizer.load(folder: paths.tokenizer) }
        progress?(GenerationProgress(step: 0, totalSteps: 0, message: "loading Qwen3.5 vision"))
        let (tok, gen, tower) = (tokenizer!, try qwen.generator(), try qwen.visionTower())
        guard let stop = tok.id(of: MossCaptionPrompt.stopToken), let pad = tok.id(of: MossCaptionPrompt.imagePad) else {
            throw EngineError.backendFailure("tokenizer lacks \(MossCaptionPrompt.stopToken) / \(MossCaptionPrompt.imagePad)")
        }

        progress?(GenerationProgress(step: 0, totalSteps: 0, message: "encoding image"))
        let (patches, grid) = try QwenVisionPreprocess.preprocess(url: request.image, maxPixels: request.maxPixels)
        let imageTokens = tower(patches.asType(gen.encoder.dtype), gridTHW: [(grid.t, grid.h, grid.w)])
        eval(imageTokens)

        let ids = tok.ids(MossCaptionPrompt.chat(imageTokens: grid.tokens, reference: request.reference, prompts: request.prompts))
        let (embeddings, positions) = try Self.prefill(ids: ids, imagePad: pad, imageTokens: imageTokens, grid: grid, encoder: gen.encoder)

        var lastError = ""
        for attempt in 1...2 {
            var sampling = sampling
            sampling.seed = request.seed &+ UInt64(attempt - 1)
            var produced = 0
            let out = try gen.generate(embeddings: embeddings, positions: positions, stop: [stop], sampling: sampling) { _ in
                try Task.checkCancellation()
                produced += 1
                if produced % 16 == 0 {
                    progress?(GenerationProgress(step: produced, totalSteps: sampling.maxTokens, message: "writing \(produced) tokens"))
                }
            }
            let answer = tok.decode(out)
            do {
                return try MossCaption.parse(answer)
            } catch let EngineError.invalidRequest(reason) {
                lastError = reason
                progress?(GenerationProgress(step: 0, totalSteps: 0, message: "retrying: \(reason)"))
            }
        }
        throw EngineError.backendFailure("the captioner could not produce a valid recaption (\(lastError))")
    }

    /// Text embeddings with the image tokens spliced over the `<|image_pad|>` run, and their M-RoPE positions.
    public static func prefill(
        ids: [Int], imagePad: Int, imageTokens: MLXArray, grid: QwenVisionGeometry.Grid, encoder: Qwen35Encoder
    ) throws -> (embeddings: MLXArray, positions: (ids: [[Int]], next: Int)) {
        try prefill(ids: ids, imagePad: imagePad, images: [(imageTokens, grid)], encoder: encoder)
    }

    /// Several images, each spliced over its own `<|image_pad|>` run, in order (edit references).
    public static func prefill(
        ids: [Int], imagePad: Int, images: [(tokens: MLXArray, grid: QwenVisionGeometry.Grid)], encoder: Qwen35Encoder
    ) throws -> (embeddings: MLXArray, positions: (ids: [[Int]], next: Int)) {
        let expected = images.map(\.grid.tokens).reduce(0, +)
        guard ids.filter({ $0 == imagePad }).count == expected, images.allSatisfy({ $0.tokens.dim(0) == $0.grid.tokens })
        else { throw EngineError.backendFailure("image placeholder count mismatch (expected \(expected))") }
        let text = encoder.embed(ids)
        var pieces: [MLXArray] = []
        var cursor = 0
        var image = 0
        while let start = ids[cursor...].firstIndex(of: imagePad), image < images.count {
            pieces.append(text[cursor..<start])
            pieces.append(images[image].tokens.asType(.float32))
            cursor = start + images[image].grid.tokens
            image += 1
        }
        if cursor < ids.count { pieces.append(text[cursor...]) }
        return (concatenated(pieces, axis: 0), QwenMRoPEIndex.positions(ids: ids, imageTokenID: imagePad, grids: images.map(\.grid)))
    }
}
