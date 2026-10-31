import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

/// One Qwen3.5-9B for every user of the text-encoder file: the pipeline's prompt encoder, the
/// enhancer, the captioner and Help all read the same weights, so they share one transformer (~16 GB
/// bf16) instead of loading a copy each. `lm_head` (~2 GB) and the vision tower (~0.9 GB) load on
/// first use. Layout from prompt → Generate (or Caption → Help) reloads nothing.
///
/// Not thread-safe: the backend runs one GPU job at a time (`InferenceQueue`).
public final class SharedQwen35: @unchecked Sendable {
    public let url: URL
    public let dtype: DType
    /// The residency policy's `keepEncoderResident`: `trim()` keeps the transformer when set.
    public var keepTransformer: Bool
    private var encoder: Qwen35Encoder?
    private var lmHead: MLXArray?
    private var tower: QwenVisionTower?

    public init(url: URL, dtype: DType = .bfloat16, keepTransformer: Bool = false) {
        self.url = url
        self.dtype = dtype
        self.keepTransformer = keepTransformer
    }

    public var isLoaded: Bool { encoder != nil }

    /// The text stack (prompt encoding, and the body of every generator).
    public func transformer() throws -> Qwen35Encoder {
        if let encoder { return encoder }
        let e = try Qwen35Encoder.load(url: url, dtype: dtype)
        encoder = e
        return e
    }

    /// The transformer + `lm_head` as a text generator (enhancer, captioner, Help).
    public func generator() throws -> Qwen35Generator {
        let e = try transformer()
        if lmHead == nil {
            let head = try WeightLoader.load(url: url, prefix: "lm_head.", working: dtype)
            guard let w = head["weight"] else { throw WeightLoaderError.missingTensors(["lm_head.weight"]) }
            lmHead = w
        }
        return Qwen35Generator(encoder: e, lmHead: lmHead!)
    }

    /// The vision tower (captioner; edit layouts whose encoder sees the references).
    public func visionTower() throws -> QwenVisionTower {
        if let tower { return tower }
        let t = try QwenVisionTower.load(url: url, dtype: dtype)
        tower = t
        return t
    }

    /// After a job on a Mac that can't keep everything: drop `lm_head` and the tower, and the
    /// transformer too unless `keepTransformer`.
    public func trim() {
        lmHead = nil
        tower = nil
        if !keepTransformer { encoder = nil }
        MLX.GPU.clearCache()
    }

    /// Drop the vision tower only (an edit's reference pictures are encoded).
    public func releaseTower() {
        guard tower != nil else { return }
        tower = nil
        MLX.GPU.clearCache()
    }

    /// Idle unload, training, model tools: everything goes.
    public func releaseAll() {
        encoder = nil
        lmHead = nil
        tower = nil
        MLX.GPU.clearCache()
    }
}
