import Foundation
import MLX
import MLXRandom
import MossForgeEngine

/// `moss-mlx dit-denoise` — a functional, reference-independent check of the whole conditioning
/// path: encode a real image (VAE), noise it on the flow-matching line `z = (1−σ)x₀ + σε`, predict
/// `v` with a real caption (Qwen3.5 → DiT), and recover `x̂₀ = z − σ·v`. A correct model lands far
/// closer to `x₀` than `z` is, and the conditioned prediction beats an off-topic caption's.
public enum DiTDenoise {
    public static func run(
        dit ditURL: URL, vae vaeURL: URL, encoder teURL: URL, tokenizer tokURL: URL, image imageURL: URL,
        prompt: String, offPrompt: String, sigmas: [Double] = [0.3, 0.6, 0.9], out: URL? = nil
    ) async throws -> (lines: [String], passed: Bool) {
        var lines: [String] = []
        let tokenizer = try await MossTokenizer.load(folder: tokURL)
        let contexts: [MLXArray]
        do {
            let encoder = try Qwen35Encoder.load(url: teURL, dtype: .bfloat16)
            contexts = [prompt, offPrompt].map { encoder.encode(tokenizer.promptIDs($0)) }
            eval(contexts)
        }
        let vae = try FluxVAE.load(url: vaeURL)
        let image = try PixelImage.load(imageURL)
        let x0 = vae.encode(image)
        let dit = try MossDiT.load(url: ditURL, dtype: .bfloat16)
        MLXRandom.seed(42)
        let noise = MLXRandom.normal(x0.shape)
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float { (sqrt(((a - b) * (a - b)).sum() / (b * b).sum())).item(Float.self) }
        var passed = true
        lines.append("image \(image.shape) → tokens \(x0.shape)")
        for sigma in sigmas {
            let z = (1 - Float(sigma)) * x0 + Float(sigma) * noise
            let estimates = contexts.map { z - Float(sigma) * dit(z, timestep: MossTimestepEmbedding.timestep(sigma: sigma), context: $0) }
            let (noisy, on, off) = (rel(z, x0), rel(estimates[0], x0), rel(estimates[1], x0))
            let ok = on < 0.8 * noisy && on <= off + 0.01  // captions tie at low σ: the image dominates
            passed = passed && ok
            let captionEffect = rel(estimates[1], estimates[0])
            lines.append(String(format: "%@ σ %.2f: ‖z−x₀‖ %.3f → ‖x̂₀−x₀‖ %.3f (off-topic caption %.3f; caption moves x̂₀ by %.3f)", ok ? "✓" : "✘", sigma, noisy, on, off, captionEffect))
            if let out, sigma >= 0.99 {
                for (i, e) in estimates.enumerated() {
                    try PixelImage.save(vae.decode(e), to: out.deletingLastPathComponent().appendingPathComponent("x0-sigma\(Int(sigma * 100))-\(i == 0 ? "on" : "off").png"))
                }
            }
            if let out, sigma == 0.6 {
                let decoded = vae.decode(estimates[0])
                try PixelImage.save(decoded, to: out)
                lines.append(String(format: "  wrote %@ (one-step x̂₀ at σ 0.6, PSNR %.2f dB vs input)", out.path, PixelImage.psnr(decoded, image)))
            }
        }
        return (lines, passed)
    }
}
