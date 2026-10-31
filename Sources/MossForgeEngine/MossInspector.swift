import Foundation
import MossForgeCore

/// What a safetensors file is, from its header alone (instant on multi-GB files).
public enum MossModelKind: Sendable, Equatable {
    case dit(MossDiTConfig, WeightPrecision)
    /// Qwen3.5 hybrid text encoder (Gated DeltaNet + every-4th full attention).
    case textEncoder(layers: Int, fullAttentionLayers: [Int])
    /// FLUX.2 VAE.
    case vae(latentChannels: Int)
    /// A DiT adapter: kohya `lora_down/up`, PEFT `lora_A/B` (± `.weight`), LyCORIS LoHa / LoKr (rank 0).
    case lora(rank: Int)
    /// Turbo data-space sampler parameters (`latent_mix.Q` + `step_codes`).
    case dmdSampler(steps: Int)
    case unknown
}

public enum MossInspector {
    public static func inspect(_ url: URL) throws -> MossModelKind {
        classify(try SafetensorsHeader.readDescriptors(from: url).tensors)
    }

    public static func classify(_ tensors: [TensorDescriptor]) -> MossModelKind {
        let byName = Dictionary(tensors.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })

        if let q = byName["latent_mix.Q"], q.shape.count == 2 {
            return .dmdSampler(steps: byName["step_codes"]?.shape.first ?? 0)
        }
        // kohya / PEFT (with or without `.weight`), LoHa (rank = w1_b rows); LoKr has no rank (0).
        if let lora = tensors.first(where: {
            $0.name.hasSuffix("lora_down.weight") || $0.name.hasSuffix("lora_A.weight") || $0.name.hasSuffix(".lora_A")
                || $0.name.hasSuffix(".hada_w1_b")
        }) {
            return .lora(rank: lora.shape.first ?? 0)
        }
        if tensors.contains(where: { $0.name.hasSuffix(".lokr_w1") }) { return .lora(rank: 0) }
        if let config = try? MossDiTConfig(descriptors: tensors) {
            return .dit(config, ditPrecision(tensors))
        }
        let encoderPrefix = "model.language_model.layers."
        if tensors.contains(where: { $0.name.hasPrefix(encoderPrefix) }) {
            var layers = Set<Int>()
            var full = Set<Int>()
            for t in tensors where t.name.hasPrefix(encoderPrefix) {
                guard let n = Int(t.name.dropFirst(encoderPrefix.count).prefix { $0.isNumber }) else { continue }
                layers.insert(n)
                if t.name.contains(".self_attn.") { full.insert(n) }
            }
            return .textEncoder(layers: layers.count, fullAttentionLayers: full.sorted())
        }
        if let pq = byName["decoder.post_quant_conv.weight"] {
            return .vae(latentChannels: pq.shape.first ?? 0)
        }
        return .unknown
    }

    /// Precision of the DiT's big block linears (norms/embedders stay bf16 in quantized files).
    static func ditPrecision(_ tensors: [TensorDescriptor]) -> WeightPrecision {
        let blocks = MossDiTConfig.keyPrefix(tensors.map(\.name)) + "layers."
        let blockWeights = tensors.filter { $0.name.hasPrefix(blocks) && $0.name.contains(".proj.") && $0.name.hasSuffix(".weight") }
        let hasComfyQuant = tensors.contains { $0.name.hasSuffix(".comfy_quant") }
        let dtypes = Set(blockWeights.map(\.dtype))
        if dtypes.contains("I8") { return hasComfyQuant ? .int8ConvRot : .int8 }
        if dtypes.contains("F8_E4M3") { return .float8 }
        if dtypes.contains("F8_E5M2") { return .float8E5M2 }
        if dtypes.contains("F32") { return .float32 }
        if dtypes.contains("F16") { return .float16 }
        return .bfloat16
    }
}

extension MossModelKind {
    /// A one-line description for model lists ("DiT · 42 blocks · bf16", "LoRA · rank 128", …).
    public var summary: String {
        switch self {
        case .dit(let c, let p): "DiT · \(c.layers) blocks · \(p.rawValue)"
        case .textEncoder(let layers, _): "Qwen3.5 encoder · \(layers) layers"
        case .vae(let ch): "VAE · \(ch) ch"
        case .lora(let rank): "LoRA · rank \(rank)"
        case .dmdSampler(let steps): "DMD sampler · \(steps) steps"
        case .unknown: "unrecognised"
        }
    }
}
