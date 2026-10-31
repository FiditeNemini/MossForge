import Foundation
import MossForgeCore

/// The DiT hyper-parameters, derived from a checkpoint header exactly the way the official
/// `comfyui/moss-image/loaders.py::model_config` does it, so any shape-compatible fine-tune
/// loads without a config file. See `docs/dev/technical-reference.md`.
public struct MossDiTConfig: Sendable, Equatable {
    /// How the block attention projections are stored.
    public enum QKVLayout: String, Sendable, Equatable {
        /// `layers.N.proj.qkv.weight` [3·dim, dim] — the Comfy-Org repack.
        case fused
        /// `layers.N.proj.{q,k,v}.weight` — diffusers / official node / toxicdog Turbo.
        case split
    }

    public var dim: Int
    public var heads: Int
    public var layers: Int
    public var ffnDim: Int
    public var latentChannels: Int
    public var textDim: Int
    public var freqDim: Int
    public var ropeBase: Double
    public var textBlocks: Int
    public var qkvLayout: QKVLayout

    public var headDim: Int { dim / heads }

    /// MOSS-Image Base / SFT as released (Comfy-Org fused layout).
    public static let release = MossDiTConfig(
        dim: 3072, heads: 24, layers: 42, ffnDim: 8192, latentChannels: 128, textDim: 4096,
        freqDim: 256, ropeBase: 10000, textBlocks: 2, qkvLayout: .fused)

    public init(
        dim: Int, heads: Int, layers: Int, ffnDim: Int, latentChannels: Int, textDim: Int,
        freqDim: Int, ropeBase: Double, textBlocks: Int, qkvLayout: QKVLayout
    ) {
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.ffnDim = ffnDim
        self.latentChannels = latentChannels
        self.textDim = textDim
        self.freqDim = freqDim
        self.ropeBase = ropeBase
        self.textBlocks = textBlocks
        self.qkvLayout = qkvLayout
    }

    /// Keys the official loader requires before it accepts a state dict as a MOSS-Image DiT.
    static let requiredKeys = [
        "video_embedder.bias", "text_embedder.proj.weight", "text_embedder.blocks.0.qkv.weight",
        "shared_block_modulation.weight", "layers.0.q_norm.weight", "layers.0.proj.ffn_up_gate.weight",
        "final_linear_video.weight", "video_head_modulation.weight", "time_embedding.0.bias",
    ]

    /// The prefix a DiT-only file keeps its tensors under: `""` for bare names, or `model.diffusion_model.` /
    /// `diffusion_model.` when every tensor sits under it (Forge's checkpoint merger saves that way). A file with
    /// anything beside the DiT (a baked-in VAE or text encoder) is an all-in-one checkpoint → `""`; extract it.
    public static func keyPrefix(_ names: some Collection<String>) -> String {
        guard !names.contains(where: { $0.hasPrefix("layers.") }) else { return "" }
        return CheckpointExtract.ditPrefixes.first { prefix in
            names.allSatisfy { $0.hasPrefix(prefix) } && names.contains { $0.hasPrefix(prefix + "layers.") }
        } ?? ""
    }

    /// `keyPrefix` of a file on disk (header only), for `WeightLoader.load(url:prefix:)`.
    public static func keyPrefix(of url: URL) throws -> String {
        keyPrefix(try SafetensorsHeader.readDescriptors(from: url).tensors.map(\.name))
    }

    /// Derive the config from header descriptors (any precision; quantized shapes keep `[out, in]`).
    public init(descriptors: [TensorDescriptor]) throws {
        let prefix = Self.keyPrefix(descriptors.map(\.name))
        let shapes = Dictionary(
            descriptors.map { (String($0.name.dropFirst(prefix.count)), $0.shape) }, uniquingKeysWith: { a, _ in a })
        let missing = Self.requiredKeys.filter { shapes[$0] == nil }
        guard missing.isEmpty else {
            throw EngineError.invalidRequest("not a MOSS-Image DiT; missing: \(missing.joined(separator: ", "))")
        }
        func first(_ key: String) -> Int { shapes[key]?.first ?? 0 }
        let dim = first("video_embedder.bias")
        let headDim = first("layers.0.q_norm.weight")
        guard dim > 0, headDim > 0, dim % headDim == 0 else {
            throw EngineError.invalidRequest("implausible DiT dims (dim \(dim), head \(headDim))")
        }
        let layers = Self.countIndices(prefix: "layers.", in: shapes.keys)
        let textBlocks = Self.countIndices(prefix: "text_embedder.blocks.", in: shapes.keys)
        self.init(
            dim: dim, heads: dim / headDim, layers: layers,
            ffnDim: first("layers.0.proj.ffn_up_gate.weight") / 2,
            latentChannels: first("final_linear_video.weight"),
            textDim: shapes["text_embedder.proj.weight"]?.last ?? 4096,
            freqDim: shapes["time_embedding.0.weight"]?.last ?? 256,
            ropeBase: 10000, textBlocks: textBlocks,
            qkvLayout: shapes["layers.0.proj.qkv.weight"] != nil ? .fused : .split)
    }

    /// Row ranges of q, k, v inside a fused `[3·dim, dim]` qkv weight.
    /// ⚠ Order q|k|v is assumed from the split module; confirm with `dit-parity`.
    public var qkvRowRanges: (q: Range<Int>, k: Range<Int>, v: Range<Int>) {
        (0..<dim, dim..<(2 * dim), (2 * dim)..<(3 * dim))
    }

    private static func countIndices(prefix: String, in keys: Dictionary<String, [Int]>.Keys) -> Int {
        var indices = Set<Int>()
        for key in keys where key.hasPrefix(prefix) {
            let rest = key.dropFirst(prefix.count)
            if let n = Int(rest.prefix { $0.isNumber }) { indices.insert(n) }
        }
        return indices.count
    }
}
