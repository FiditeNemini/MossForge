import Foundation
import MLX
import MossForgeEngine

/// `moss-mlx generate-parity <models> <fixtureDir>` — Qwen3.5 greedy generation (cached prefill +
/// one-token decode steps) against `Scripts/generate_ref.py`, which recomputes the whole sequence each
/// step. F32 must reproduce the reference's token sequence exactly; bf16 (shipping) is reported.
public enum GenerateParity {
    public static func run(models: URL, fixtures dir: URL) async throws -> (lines: [String], passed: Bool) {
        let fx = try MLX.loadArrays(url: dir.appendingPathComponent("generate.safetensors"))
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("generate.json"))) as! [String: Any]
        let paths = try MossModelPaths.resolve(models: models, variant: .sft)
        let tokenizer = try await MossTokenizer.load(folder: paths.tokenizer)
        let promptIDs = fx["prompt_ids"]!.asArray(Int32.self).map(Int.init)
        let want = fx["generated_ids"]!.asArray(Int32.self).map(Int.init)
        var lines: [String] = []
        var passed = true
        func gate(_ ok: Bool, _ line: String) {
            passed = passed && ok
            lines.append("\(ok ? "✓" : "✘") \(line)")
        }
        gate(tokenizer.ids(meta["chat"] as! String) == promptIDs, "tokenizer: \(promptIDs.count) prompt ids")
        lines.append("  reference: \((meta["text"] as? String ?? "").debugDescription)")

        for dtype in [DType.float32, .bfloat16] {
            let tag = dtype == .float32 ? "f32" : "bf16"
            var start = Date()
            let generator = try Qwen35Generator.load(url: paths.encoder, dtype: dtype)
            lines.append(String(format: "  [%@] loaded in %.1fs", tag, Date().timeIntervalSince(start)))
            let first = generator.logits(generator.encoder.forward(promptIDs, cache: nil))
            let s = EncoderParity.stats(first[.newAxis], fx["first_logits"]![.newAxis])
            start = Date()
            var sampling = Qwen35Generator.Sampling.greedy
            sampling.maxTokens = want.count
            let got = try generator.generate(prompt: promptIDs, stop: [], sampling: sampling)
            let elapsed = Date().timeIntervalSince(start)
            let matched = zip(got, want).prefix { $0 == $1 }.count
            let line = String(format: "[%@] first-step logits rel-err %.3g; greedy tokens %d/%d match — %.1f tok/s",
                tag, s.rel, matched, want.count, Double(got.count) / elapsed)
            if dtype == .float32 { gate(matched == want.count && s.rel < 1e-3, line) } else { lines.append("· " + line) }
            lines.append("  [\(tag)] \(tokenizer.decode(got).debugDescription)")
        }
        return (lines, passed)
    }
}
