import Foundation
import MLX
import MLXRandom
import MossForgeEngine

/// `moss-mlx dit-bench <dit.safetensors>` — times one sampler step the way `MossPipeline` runs it
/// (cond + uncond with uncond-skip block 22) at a given size, as two separate forwards and as one
/// packed `velocities` forward, and reports the packed-vs-separate error. Random latent and text
/// states (std 2, like real Qwen3.5 states); speed work only, not a quality check. `--lora` attaches a
/// runtime LoRA first (e.g. the Turbo adapter) to time its unmerged-F32 overhead.
public enum DiTBench {
    public static func run(
        dit url: URL, size: Int = 1024, positiveTokens: Int = 80, negativeTokens: Int = 24, repeats: Int = 3,
        dtype: DType = .float16, lora: URL? = nil
    ) throws -> [String] {
        var lines: [String] = []
        var start = Date()
        let dit = try MossDiT.load(url: url, dtype: dtype)
        if let lora { try dit.attachLoRA(url: lora, strength: 1) }  // runtime, unmerged F32 (e.g. the Turbo LoRA)
        lines.append(String(format: "loaded in %.1fs", Date().timeIntervalSince(start)))
        let g = size / 16
        let key = MLXRandom.key(0)
        let latent = MLXRandom.normal([1, 128, g, g], key: key)
        let positive = 2 * MLXRandom.normal([positiveTokens, 4096], key: MLXRandom.key(1))
        let negative = 2 * MLXRandom.normal([negativeTokens, 4096], key: MLXRandom.key(2))
        let branches = [MossDiT.Branch(context: positive), MossDiT.Branch(context: negative, skipLayers: [22])]
        let timestep = 700.0

        /// (min, median) seconds per call after one warm-up.
        func time(_ body: () -> [MLXArray]) -> ((Double, Double), [MLXArray]) {
            var out = body()
            eval(out)  // warm-up
            var times: [Double] = []
            for _ in 0..<repeats {
                start = Date()
                out = body()
                eval(out)
                times.append(Date().timeIntervalSince(start))
            }
            times.sort()
            return ((times[0], times[times.count / 2]), out)
        }
        let (separateTime, separate) = time {
            branches.map { dit(latent, timestep: timestep, context: $0.context, skipLayers: $0.skipLayers) }
        }
        let (packedTime, packed) = time { dit.velocities(latent, timestep: timestep, branches: branches) }
        lines.append(String(format: "%d² (%d + %d tokens), %d repeats%@", size, g * g + positiveTokens, g * g + negativeTokens, repeats, lora == nil ? "" : ", with LoRA"))
        lines.append(String(format: "  separate: %.2fs/step (min; median %.2fs)", separateTime.0, separateTime.1))
        lines.append(String(format: "  packed:   %.2fs/step (min; median %.2fs)", packedTime.0, packedTime.1))
        for (name, a, b) in [("cond", packed[0], separate[0]), ("uncond", packed[1], separate[1])] {
            let s = EncoderParity.stats(a, b)
            lines.append(String(format: "  packed vs separate %@: rel-err %.3g, row cosine min %.6f", name, s.rel, s.minCos))
        }
        return lines
    }
}
