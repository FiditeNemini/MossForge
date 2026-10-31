import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

/// `moss-mlx e2e-parity <models> <fixtureDir>` — Phase 1 step 6: the whole MLX pipeline
/// (`MossPipeline.generate`: tokenizer → Qwen3.5 → official σ grid → CFG with uncond-skip → Euler →
/// VAE) against `Scripts/e2e_ref.py` at the same fixed noise. F32 must match tightly; bf16 (the
/// shipping precision) is gated on decoded-image PSNR.
public enum E2EParity {
    public static func run(models: URL, fixtures dir: URL, minBF16PooledPSNR: Float = 20, saveImages: URL? = nil) async throws -> (lines: [String], passed: Bool) {
        let fx = try MLX.loadArrays(url: dir.appendingPathComponent("e2e.safetensors"))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("e2e.json"))) as! [String: Any]
        let request = GenerationRequest(
            prompt: meta["prompt"] as! String, negativePrompt: meta["negative"] as! String,
            width: meta["width"] as! Int, height: meta["height"] as! Int, steps: meta["steps"] as? Int,
            cfg: meta["cfg"] as? Double, shift: meta["shift"] as? Double,
            slg: .init(uncondSkipLayers: meta["uncond_skip"] as! [Int], layers: [], scale: 0, start: 0, stop: 1), seed: 0)
        var lines: [String] = []
        var passed = true
        func gate(_ ok: Bool, _ line: String) {
            passed = passed && ok
            lines.append("\(ok ? "✓" : "✘") \(line)")
        }

        // Schedule + tokenizer: exact.
        let want = fx["sigmas"]!.asArray(Float.self)
        let got = MossSamplerPlan(request: request).steps.map { Float($0.sigma) } + [0]
        gate(got.count == want.count && zip(got, want).allSatisfy { abs($0 - $1) < 1e-6 }, "σ grid: \(got.map { String(format: "%.3f", $0) }.joined(separator: " "))")
        let paths = try MossModelPaths.resolve(models: models, variant: .sft)
        let tokenizer = try await MossTokenizer.load(folder: paths.tokenizer)
        for (name, text) in [("pos", request.prompt), ("neg", request.negativePrompt)] {
            let ids = fx["ids_\(name)"]!.asArray(Int32.self).map(Int.init)
            gate(tokenizer.promptIDs(text) == ids, "tokenizer[\(name)]: \(ids.count) ids")
        }

        if let saveImages { try PixelImage.save(fx["image"]!, to: saveImages.appendingPathComponent("e2e-torch-f32.png")) }
        // (encoder, DiT) precision: all-F32 is the parity gate; bf16 DiT is the shipping path.
        for (encoderDType, dtype) in [(DType.float32, DType.float32), (.float32, .bfloat16), (.bfloat16, .bfloat16), (.bfloat16, .float16)] {
            var options = MossPipeline.Options()
            (options.ditDType, options.encoderDType) = (dtype, encoderDType)
            let tag = (encoderDType == .float32 ? "enc f32" : "enc bf16") + (dtype == .float32 ? ", dit f32" : dtype == .float16 ? ", dit f16" : ", dit bf16")
            let pipeline = MossPipeline(paths: paths, options: options)
            var seen: [String: MLXArray] = [:]
            let start = Date()
            let image = try await pipeline.generate(request, seed: 0, noise: fx["noise"]!, inspect: { seen[$0] = $1 })
            let elapsed = Date().timeIntervalSince(start)
            pipeline.release()
            for (name, key) in [("context_pos", "context_pos"), ("context_neg", "context_neg"), ("latent", "latent")] {
                let s = EncoderParity.stats(seen[name]!.reshaped(fx[key]!.shape), fx[key]!)
                let line = String(format: "[%@] %@: rel-err %.3g, row cosine min %.6f", tag, name, s.rel, s.minCos)
                // CFG 3 amplifies per-forward error over the few large steps ⇒ latent budget 2e-3.
                if dtype == .float32 { gate(s.rel <= 2e-3, line) } else { lines.append("· " + line) }
            }
            // F32: pixel parity. bf16 DiT: the trajectory drifts in detail (glyph shapes, small shifts —
            // CFG 3 over few coarse steps amplifies bf16 weight rounding), so it is gated on composition:
            // PSNR after 16×16 average pooling. Full-resolution PSNR is reported for information.
            let psnr = Double(PixelImage.psnr(image, fx["image"]!))
            let pooled = Double(PixelImage.psnr(Self.pool(image, 16), Self.pool(fx["image"]!, 16)))
            if dtype == .float32 {
                gate(psnr >= 50, String(format: "[%@] decoded image PSNR %.2f dB vs torch (budget 50 dB) — %.1fs incl. loads", tag, psnr, elapsed))
            } else {
                gate(pooled >= Double(minBF16PooledPSNR), String(format: "[%@] composition: 16×16-pooled PSNR %.2f dB (budget %.0f); full-res %.2f dB — %.1fs incl. loads", tag, pooled, Double(minBF16PooledPSNR), psnr, elapsed))
            }
            if let saveImages {
                try PixelImage.save(image, to: saveImages.appendingPathComponent("e2e-mlx-\(tag.replacingOccurrences(of: ", ", with: "-").replacingOccurrences(of: " ", with: "")).png"))
            }
        }
        return (lines, passed)
    }

    /// Average-pool `[1,C,H,W]` by `f` (H, W divisible by f).
    static func pool(_ x: MLXArray, _ f: Int) -> MLXArray {
        let (c, h, w) = (x.dim(1), x.dim(2), x.dim(3))
        return x.asType(.float32).reshaped([1, c, h / f, f, w / f, f]).mean(axes: [3, 5])
    }
}
