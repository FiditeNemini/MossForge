import Foundation
import MLX
import MossForgeEngine

/// `moss-mlx encoder-parity <te.safetensors> <tokenizerDir> <fixtureDir>` — checks the Qwen3.5
/// encoder against `Scripts/encoder_ref.py` (`short` / `long` `.safetensors` + `.txt`):
///
/// 1. the Swift tokenizer reproduces the Python `tokenizers` ids exactly (template + truncation);
/// 2. the gated-delta Metal kernel matches the ops-only recurrence;
/// 3. F32 hidden states (layer 0 = DeltaNet, layer 3 = first full attention, final norm) clear a
///    relative-error + per-token cosine budget; bf16 is reported alongside.
public enum EncoderParity {
    public struct Report {
        public var lines: [String]
        public var passed: Bool
    }

    public static func run(
        encoder url: URL, tokenizer folder: URL, fixtures dir: URL, dtypes: [DType] = [.float32, .bfloat16],
        maxRelErr: Float = 1e-3, minCosine: Float = 0.9999
    ) async throws -> Report {
        var lines: [String] = []
        var passed = true
        func gate(_ ok: Bool, _ line: String) {
            passed = passed && ok
            lines.append("\(ok ? "✓" : "✘") \(line)")
        }

        // 1. Tokenizer.
        let tokenizer = try await MossTokenizer.load(folder: folder)
        var cases: [(name: String, ids: [Int], fx: [String: MLXArray])] = []
        for name in ["short", "long"] {
            let fx = try MLX.loadArrays(url: dir.appendingPathComponent("\(name).safetensors"))
            let prompt = try String(contentsOf: dir.appendingPathComponent("\(name).txt"), encoding: .utf8)
            let want = fx["ids"]!.asArray(Int32.self).map(Int.init)
            let got = tokenizer.promptIDs(prompt)
            let firstDiff = zip(got, want).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            gate(got == want, "tokenizer[\(name)]: \(got.count) ids vs \(want.count)" + (firstDiff.map { ", first diff at \($0)" } ?? ""))
            cases.append((name, want, fx))
        }

        // 2. Kernel vs ops recurrence (GQA 2→4 heads, T not a multiple of anything).
        MLXRandom.seed(7)
        let (b, t, hk, hv, dk, dv) = (1, 37, 2, 4, 64, 32)
        func unit(_ x: MLXArray) -> MLXArray { x / sqrt((x * x).sum(axis: -1, keepDims: true)) }
        let q = unit(MLXRandom.normal([b, t, hk, dk])) * Float(dk).squareRoot().reciprocal
        let k = unit(MLXRandom.normal([b, t, hk, dk]))
        let v = MLXRandom.normal([b, t, hv, dv])
        let g = exp(-MLXRandom.uniform(low: 0, high: 0.5, [b, t, hv]))
        let beta = sigmoid(MLXRandom.normal([b, t, hv]))
        let kernelDiff = abs(GatedDelta.run(q: q, k: k, v: v, g: g, beta: beta).y - GatedDelta.reference(q: q, k: k, v: v, g: g, beta: beta))
            .max().item(Float.self)
        gate(kernelDiff < 1e-4, String(format: "gated-delta kernel vs ops recurrence: max-abs %.3g", kernelDiff))

        // 3. Encoder.
        for dtype in dtypes {
            let tag = dtype == .float32 ? "f32" : "bf16"
            let strict = dtype == .float32
            var start = Date()
            let encoder = try Qwen35Encoder.load(url: url, dtype: dtype)
            lines.append(String(format: "  [%@] loaded in %.1fs", tag, Date().timeIntervalSince(start)))
            for c in cases {
                var taps: [Int: MLXArray] = [:]
                start = Date()
                let hidden = encoder.encode(c.ids, taps: c.name == "short" ? [0, 3] : []) { taps[$0] = $1 }
                eval(hidden)
                let elapsed = Date().timeIntervalSince(start)
                var checks: [(String, MLXArray, MLXArray)] = [("hidden", hidden, c.fx["hidden"]!)]
                for (i, tap) in taps.sorted(by: { $0.key < $1.key }) { checks.insert(("layer\(i)", tap, c.fx["layer\(i)"]!), at: 0) }
                for (label, got, want) in checks {
                    let s = stats(got, want)
                    let ok = !strict || (s.rel <= maxRelErr && s.minCos >= minCosine)
                    let line = String(
                        format: "[%@] %@ %@: rel-err %.3g, token cosine min %.6f mean %.6f, max-abs %.3g", tag, c.name, label,
                        s.rel, s.minCos, s.meanCos, s.maxAbs)
                    if strict { gate(ok, line) } else { lines.append("· " + line) }
                }
                lines.append(String(format: "  [%@] %@: %d tokens in %.2fs", tag, c.name, c.ids.count, elapsed))
            }
        }
        if dtypes.contains(.float32) {
            lines.append(String(format: "budget (f32): rel-err ≤ %.0e, per-token cosine ≥ %.4f", maxRelErr, minCosine))
        }
        return Report(lines: lines, passed: passed)
    }

    /// Relative Frobenius error, per-token cosine (rows), max-abs — all in F32.
    static func stats(_ a: MLXArray, _ b: MLXArray) -> (rel: Float, minCos: Float, meanCos: Float, maxAbs: Float) {
        let x = a.asType(.float32), y = b.asType(.float32)
        let d = x - y
        let rel = sqrt((d * d).sum() / (y * y).sum())
        let cos = (x * y).sum(axis: -1) / maximum(sqrt((x * x).sum(axis: -1) * (y * y).sum(axis: -1)), 1e-30)
        let maxAbs = abs(d).max()
        eval(rel, cos, maxAbs)
        return (rel.item(Float.self), cos.min().item(Float.self), cos.mean().item(Float.self), maxAbs.item(Float.self))
    }
}

extension Float {
    fileprivate var reciprocal: Float { 1 / self }
}

extension EncoderParity {
    /// Diagnostic: MLX F32 vs MLX bf16 per layer on one fixture's ids — where does bf16 diverge?
    public static func layerDivergence(encoder url: URL, fixture: URL) throws -> [String] {
        let ids = try MLX.loadArrays(url: fixture)["ids"]!.asArray(Int32.self).map(Int.init)
        var ref: [Int: MLXArray] = [:]
        do {
            let f32 = try Qwen35Encoder.load(url: url, dtype: .float32)
            _ = f32.encode(ids, taps: Set(0..<f32.config.layers)) { ref[$0] = $1 }
        }
        var lines: [String] = []
        let bf16 = try Qwen35Encoder.load(url: url, dtype: .bfloat16)
        _ = bf16.encode(ids, taps: Set(0..<bf16.config.layers)) { i, x in
            let s = stats(x, ref[i]!)
            let cos = (x * ref[i]!).sum(axis: -1) / sqrt((x * x).sum(axis: -1) * (ref[i]! * ref[i]!).sum(axis: -1))
            let worst = argMin(cos).item(Int.self)
            let norms = sqrt((ref[i]! * ref[i]!).sum(axis: -1))
            lines.append(String(
                format: "layer %2d %@: rel %.3g cos min %.4f @%d (ref ‖x‖ %.1f, median ‖x‖ %.1f)", i,
                bf16.config.isFullAttention(i) ? "attn " : "delta", s.rel, s.minCos, worst,
                norms[worst].item(Float.self), sorted(norms)[norms.dim(0) / 2].item(Float.self)))
        }
        return lines
    }
}
