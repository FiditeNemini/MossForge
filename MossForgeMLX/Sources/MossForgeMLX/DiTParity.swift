import Foundation
import MLX
import MossForgeEngine

/// `moss-mlx dit-parity <dit.safetensors> <fixtureDir>` — checks `MossDiT` against
/// `Scripts/dit_ref.py` (`dit.safetensors`: latent, real Qwen3.5 context, timestep → modulation,
/// text-adapter output, residual stream after layers 0 and 21, velocity). F32 is gated on a
/// relative-error budget; bf16 is reported alongside.
public enum DiTParity {
    public static func run(dit url: URL, fixtures dir: URL, dtypes: [DType] = [.float32, .bfloat16], maxRelErr: Float = 1e-3)
        throws -> (lines: [String], passed: Bool)
    {
        let fx = try MLX.loadArrays(url: dir.appendingPathComponent("dit.safetensors"))
        let latent = fx["latent"]!, context = fx["context"]!
        let timestep = Double(fx["timestep"]!.item(Float.self))
        var lines = ["latent \(latent.shape), context \(context.shape), timestep \(timestep)"]
        var passed = true
        for dtype in dtypes {
            let tag = switch dtype { case .float32: "f32"; case .float16: "f16"; default: "bf16" }
            var start = Date()
            let dit = try MossDiT.load(url: url, dtype: dtype)
            lines.append(String(format: "  [%@] loaded in %.1fs", tag, Date().timeIntervalSince(start)))
            var taps: [(String, MLXArray)] = []
            start = Date()
            let velocity = dit(latent, timestep: timestep, context: context) { taps.append(($0, $1)) }
            eval(velocity)
            let elapsed = Date().timeIntervalSince(start)
            for (name, got) in taps + [("velocity", velocity)] {
                guard let want = fx[name] else { continue }
                let s = EncoderParity.stats(got.reshaped(want.shape), want)
                let ok = dtype != .float32 || s.rel <= maxRelErr
                passed = passed && ok
                lines.append(String(
                    format: "%@ [%@] %@: rel-err %.3g, row cosine min %.6f, max-abs %.3g", dtype == .float32 ? (ok ? "✓" : "✘") : "·",
                    tag, name, s.rel, s.minCos, s.maxAbs))
            }
            lines.append(String(format: "  [%@] forward %.2fs (%d tokens)", tag, elapsed, latent.dim(2) * latent.dim(3) + context.dim(0)))
        }
        lines.append(String(format: "budget (f32): rel-err ≤ %.0e", maxRelErr))
        return (lines, passed)
    }
}
