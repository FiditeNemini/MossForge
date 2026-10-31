import Foundation
import MLX
import MossForgeCore
import MossForgeEngine

/// `moss-mlx caption-parity <models> <fixtureDir>` — the captioner against `Scripts/caption_ref.py`:
/// preprocessing (patches), the vision tower (last block + merged tokens, f32 and bf16), the official
/// prompt's token ids, and greedy multimodal decoding (cached prefill + one-token steps vs the
/// reference's full recompute). F32 must match; bf16 (shipping) is reported.
public enum CaptionParity {
    public static func run(models: URL, fixtures dir: URL) async throws -> (lines: [String], passed: Bool) {
        let fx = try MLX.loadArrays(url: dir.appendingPathComponent("caption.safetensors"))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("caption.json"))) as! [String: Any]
        let paths = try MossModelPaths.resolve(models: models, variant: .sft)
        let tokenizer = try await MossTokenizer.load(folder: paths.tokenizer)
        var lines: [String] = []
        var passed = true
        func gate(_ ok: Bool, _ line: String) {
            passed = passed && ok
            lines.append("\(ok ? "✓" : "✘") \(line)")
        }

        let (patches, grid) = try QwenVisionPreprocess.preprocess(
            url: URL(fileURLWithPath: meta["image"] as! String, relativeTo: models.deletingLastPathComponent()),  // repo-relative
            maxPixels: meta["max_pixels"] as! Int)
        let ps = EncoderParity.stats(patches[.newAxis], fx["patches"]![.newAxis])
        gate([grid.t, grid.h, grid.w] == (meta["grid"] as! [Int]) && ps.maxAbs < 1e-5,
            String(format: "preprocess: grid %d×%d, patches max|Δ| %.2g", grid.h, grid.w, ps.maxAbs))

        let pad = tokenizer.id(of: MossCaptionPrompt.imagePad)!
        let ids = tokenizer.ids(MossCaptionPrompt.chat(imageTokens: grid.tokens, reference: nil))
        let promptIDs = fx["prompt_ids"]!.asArray(Int32.self).map(Int.init)
        gate(ids == promptIDs, "prompt: \(ids.count) ids (official recaption prompt, \(grid.tokens) image tokens)")
        lines.append("  reference: \((meta["text"] as? String ?? "").debugDescription)")

        let want = fx["generated_ids"]!.asArray(Int32.self).map(Int.init)
        for dtype in [DType.float32, .bfloat16] {
            let tag = dtype == .float32 ? "f32" : "bf16"
            let tower = try QwenVisionTower.load(url: paths.encoder, dtype: dtype)
            var last: MLXArray?
            let tokens = tower(fx["patches"]!.asType(dtype), gridTHW: [(grid.t, grid.h, grid.w)]) { last = $0 }
            eval(tokens)
            let ls = EncoderParity.stats(last!.asType(.float32)[.newAxis], fx["vision_last"]![.newAxis])
            let ts = EncoderParity.stats(tokens.asType(.float32)[.newAxis], fx["image_tokens"]![.newAxis])
            let vline = String(format: "[%@] vision: last block rel-err %.3g, image tokens rel-err %.3g", tag, ls.rel, ts.rel)
            if dtype == .float32 { gate(ts.rel < 1e-3, vline) } else { lines.append("· " + vline) }

            let generator = try Qwen35Generator.load(url: paths.encoder, dtype: dtype)
            // The reference's image tokens, so the decode check isolates the language model.
            let (embeddings, positions) = try MossCaptioner.prefill(
                ids: promptIDs, imagePad: pad, imageTokens: fx["image_tokens"]!, grid: grid, encoder: generator.encoder)
            let first = generator.logits(generator.encoder.forward(embeddings: embeddings, positions: positions, cache: nil))
            let fs = EncoderParity.stats(first[.newAxis], fx["first_logits"]![.newAxis])
            var sampling = Qwen35Generator.Sampling.greedy
            sampling.maxTokens = want.count
            let start = Date()
            let got = try generator.generate(embeddings: embeddings, positions: positions, stop: [], sampling: sampling)
            let matched = zip(got, want).prefix { $0 == $1 }.count
            let line = String(format: "[%@] first-step logits rel-err %.3g; greedy tokens %d/%d match (%.1fs)",
                tag, fs.rel, matched, want.count, Date().timeIntervalSince(start))
            if dtype == .float32 { gate(matched == want.count && fs.rel < 1e-3, line) } else { lines.append("· " + line) }
            lines.append("  [\(tag)] \(tokenizer.decode(got).debugDescription)")
            MLX.GPU.clearCache()
        }
        return (lines, passed)
    }
}
