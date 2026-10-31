import Foundation
import MLX
import MossForgeEngine

/// `moss-mlx vae-parity <vae> <fixtureDir>` — checks `FluxVAE` against `Scripts/vae_ref.py`
/// (`<fixtureDir>/vae.safetensors`): the MLX patchify vs the Engine CPU reference (exact), the
/// encoder (posterior mean + normalised tokens), the decoder on the reference's own tokens, and
/// the decoder on unit-normal tokens (the DiT's output domain). F32 must clear the budgets;
/// bf16 is reported alongside for information.
public enum VAEParity {
    public struct Report {
        public var lines: [String]
        public var passed: Bool
    }

    public static func run(vae url: URL, fixtures dir: URL, budget: Float = 2e-3) throws -> Report {
        let fx = try MLX.loadArrays(url: dir.appendingPathComponent("vae.safetensors"))
        func t(_ name: String) throws -> MLXArray {
            guard let a = fx[name] else { throw WeightLoaderError.missingTensors([name]) }
            return a
        }
        let image = try t("image"), mean = try t("vae_mean"), tokens = try t("tokens")
        let decoded = try t("decoded"), noiseTokens = try t("noise_tokens"), noiseDecoded = try t("noise_decoded")
        var lines = ["image \(image.shape) → tokens \(tokens.shape)"]
        var passed = true

        // 1. Index-level: MLX patchify ≡ the unit-tested CPU reference.
        let (c, h, w) = (mean.dim(1), mean.dim(2), mean.dim(3))
        let cpu = MossLatentFormat.patchify(mean.asArray(Float.self), channels: c, height: h, width: w)
        let gpu = FluxVAE.patchify(mean).asArray(Float.self)
        let exact = cpu == gpu
        lines.append("\(exact ? "✓" : "✘") patchify: MLX ≡ MossLatentFormat CPU reference")
        passed = passed && exact

        let weights = try WeightLoader.load(url: url).arrays
        for dtype in [DType.float32, .bfloat16] {
            let vae = try FluxVAE(weights: weights, dtype: dtype)
            let gate = dtype == .float32
            let tag = gate ? "f32" : "bf16"
            let start = Date()
            let checks: [(String, MLXArray, MLXArray, Float)] = [
                ("encode mean", vae.encodeMean(image), mean, budget),
                ("encode tokens", vae.encode(image), tokens, budget * 5),  // ÷ BN std amplifies
                ("decode", vae.decode(tokens), decoded, budget),
                ("decode noise", vae.decodeRaw(noiseTokens), noiseDecoded, budget),
            ]
            for (name, got, want, limit) in checks {
                guard got.shape == want.shape else {
                    lines.append("✘ [\(tag)] \(name): shape \(got.shape) vs \(want.shape)")
                    passed = false
                    continue
                }
                let d = abs(got.asType(.float32) - want)
                let (maxAbs, meanAbs) = (d.max().item(Float.self), d.mean().item(Float.self))
                let ok = !gate || maxAbs <= limit
                passed = passed && ok
                lines.append(
                    String(
                        format: "%@ [%@] %@: max-abs %.3g mean-abs %.3g%@", ok ? "✓" : "✘", tag, name, maxAbs, meanAbs,
                        gate ? String(format: " (budget %.3g)", limit) : ""))
            }
            lines.append(String(format: "  [%@] %.2fs", tag, Date().timeIntervalSince(start)))
        }
        return Report(lines: lines, passed: passed)
    }
}
