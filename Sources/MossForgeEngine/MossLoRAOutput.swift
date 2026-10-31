import Foundation
import MossForgeCore

/// The LoRA file the trainer writes: kohya keys on the canonical fused modules
/// (`diffusion_model.layers.N.proj.qkv.lora_down.weight` / `lora_up.weight` / `alpha`), the format
/// MossForge, ComfyUI and Forge already read for MOSS (Kijai's LoRAs use the same fused qkv).
public enum MossLoRAOutput {
    public static func keys(module: String) -> (down: String, up: String, alpha: String) {
        let base = "diffusion_model." + module
        return (base + ".lora_down.weight", base + ".lora_up.weight", base + ".alpha")
    }

    /// LoKr (LyCORIS) keys: `diffusion_model.<module>.lokr_w1` / `.lokr_w2` (full factors, scale 1).
    public static func lokrKeys(module: String) -> (w1: String, w2: String) {
        let base = "diffusion_model." + module
        return (base + ".lokr_w1", base + ".lokr_w2")
    }

    /// Factor shapes whose Kronecker product is `[out, in]`.
    public static func lokrShapes(out: Int, in inp: Int, factor: Int) -> (w1: [Int], w2: [Int]) {
        let (o, i) = (LoRAFactorization.split(out, factor: factor), LoRAFactorization.split(inp, factor: factor))
        return ([o.m, i.m], [o.n, i.n])
    }

    /// `[out, in]` of every trained linear (`layers.N.<module>` → shape) across the DiT's blocks.
    public static func targetShapes(config c: MossDiTConfig, targets: LoRATargetSet) -> [String: [Int]] {
        var out: [String: [Int]] = [:]
        for n in 0..<c.layers {
            for m in targets.modules {
                let shape: [Int] = switch m {
                case "proj.qkv": [3 * c.dim, c.dim]
                case "proj.o": [c.dim, c.dim]
                case "proj.ffn_up_gate": [2 * c.ffnDim, c.dim]
                default: [c.dim, c.ffnDim]  // proj.ffn_down
                }
                out["layers.\(n).\(m)"] = shape
            }
        }
        return out
    }
}
