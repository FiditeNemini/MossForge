import Foundation
import MLX
import MLXRandom
import MossForgeCore
import MossForgeEngine

/// Autoregressive text generation with the Qwen3.5-9B file the encoder already uses: the same
/// text stack (`Qwen35Encoder` with a `Cache`) plus `lm_head`. Prefill once, then one token per step
/// (KV cache on the 8 attention layers, conv + recurrent state on the 24 DeltaNet layers). Used by
/// the on-device prompt / layout enhancer (`MossEnhancer`).
public final class Qwen35Generator {
    public let encoder: Qwen35Encoder
    private let lmHead: MLXArray  // [vocab, hidden], working dtype

    public struct Sampling: Sendable {
        /// 0 = greedy (argmax) — deterministic, used by the parity check.
        public var temperature: Float = 0.7
        public var topP: Float = 0.8
        public var topK = 20
        public var seed: UInt64 = 0
        public var maxTokens = 1024
        public init() {}
        /// Qwen's recommended non-thinking settings (temperature 0.7, top-p 0.8, top-k 20).
        public static let qwenNonThinking = Sampling()
        public static var greedy: Sampling { var s = Sampling(); s.temperature = 0; return s }
    }

    public init(encoder: Qwen35Encoder, lmHead: MLXArray) {
        self.encoder = encoder
        self.lmHead = lmHead.asType(encoder.dtype)
    }

    public static func load(url: URL, dtype: DType = .bfloat16) throws -> Qwen35Generator {
        let encoder = try Qwen35Encoder.load(url: url, dtype: dtype)
        let head = try WeightLoader.load(url: url, prefix: "lm_head.", working: dtype)
        guard let w = head["weight"] else { throw WeightLoaderError.missingTensors(["lm_head.weight"]) }
        return Qwen35Generator(encoder: encoder, lmHead: w)
    }

    /// Logits `[vocab]` (F32) for the last position of `hidden` `[T, H]`.
    func logits(_ hidden: MLXArray) -> MLXArray {
        matmul(hidden[(hidden.dim(0) - 1)...].asType(encoder.dtype), lmHead.T)[0].asType(.float32)
    }

    /// Generate token ids after `prompt` until `stop` (excluded) or `maxTokens`. `onToken` sees each id
    /// as it is produced (streaming / cancellation via throwing).
    public func generate(
        prompt: [Int], stop: Set<Int>, sampling: Sampling = .qwenNonThinking, onToken: ((Int) throws -> Void)? = nil
    ) throws -> [Int] {
        try generate(stop: stop, sampling: sampling, onToken: onToken) { encoder.forward(prompt, cache: $0) }
    }

    /// Multimodal: generate after a prefill of `embeddings` `[T, hidden]` (image tokens already spliced
    /// in) with their M-RoPE `positions`.
    public func generate(
        embeddings: MLXArray, positions: (ids: [[Int]], next: Int), stop: Set<Int>, sampling: Sampling = .qwenNonThinking,
        onToken: ((Int) throws -> Void)? = nil
    ) throws -> [Int] {
        try generate(stop: stop, sampling: sampling, onToken: onToken) {
            encoder.forward(embeddings: embeddings, positions: positions, cache: $0)
        }
    }

    private func generate(
        stop: Set<Int>, sampling: Sampling, onToken: ((Int) throws -> Void)?, prefill: (Qwen35Encoder.Cache) -> MLXArray
    ) throws -> [Int] {
        let cache = Qwen35Encoder.Cache()
        var next = choose(logits(prefill(cache)), sampling, step: 0)
        var out: [Int] = []
        while !stop.contains(next), out.count < sampling.maxTokens {
            out.append(next)
            try onToken?(next)
            next = choose(logits(encoder.forward([next], cache: cache)), sampling, step: out.count)
        }
        return out
    }

    /// Greedy or temperature/top-k/top-p sampling.
    func choose(_ logits: MLXArray, _ s: Sampling, step: Int) -> Int {
        if s.temperature <= 0 { return argMax(logits).item(Int.self) }
        let k = min(s.topK, logits.dim(0))
        let order = argSort(-logits)[..<k]  // top-k ids, best first
        let top = logits.take(order) / s.temperature
        let probs = softmax(top)
        // Nucleus: keep the smallest prefix whose mass reaches top-p (always ≥ 1 token).
        let cumulative = cumsum(probs)
        let keep = (cumulative - probs) .< s.topP
        let masked = MLX.where(keep, log(probs), MLXArray(-Float.infinity))
        let key = MLXRandom.key(s.seed &+ UInt64(step))
        let pick = MLXRandom.categorical(masked, key: key).item(Int.self)
        return order[pick].item(Int.self)
    }
}
