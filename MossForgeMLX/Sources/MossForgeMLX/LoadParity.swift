import Foundation
import MLX
import MossForgeEngine

/// `moss-mlx load-parity` — streams two checkpoints of the same model tensor by tensor and checks
/// the candidate's dequantized weights against the reference (e.g. SFT int8·convrot vs SFT bf16):
///
/// - tensors **both** files store unquantized must be bit-identical (max-abs 0);
/// - quantized tensors must clear a cosine budget (INT8's inherent error; ConvRot exists to keep
///   it small) — and a negative control re-reads the first one *without* un-rotating, which must
///   collapse, proving the Hadamard step is what recovers the basis.
public enum LoadParity {
    public struct Stats {
        public var key: String
        public var maxAbs: Float
        public var relErr: Float
        public var cosine: Float
    }

    public struct Report {
        public var lines: [String]
        public var passed: Bool
    }

    public static func run(
        reference: URL, candidate: URL, prefix: String = "", match: String? = nil, minCosine: Float = 0.98
    ) throws -> Report {
        let refPlan = try MossWeightPlan.build(url: reference, prefix: prefix)
        let candPlan = try MossWeightPlan.build(url: candidate, prefix: prefix)
        try MossWeightPlan.validate(candPlan)
        let ref = try TensorSource(url: reference)
        let cand = try TensorSource(url: candidate)
        defer {
            ref.close()
            cand.close()
        }

        let candByKey = Dictionary(candPlan.entries.map { ($0.key, $0) }, uniquingKeysWith: { a, _ in a })
        let refKeys = Set(refPlan.entries.map(\.key)).union(refPlan.fusions.map(\.key))
            .subtracting(refPlan.fusions.flatMap(\.parts))
        let candKeys = Set(candPlan.entries.map(\.key)).union(candPlan.fusions.map(\.key))
            .subtracting(candPlan.fusions.flatMap(\.parts))
        var lines: [String] = []
        var passed = true
        let onlyRef = refKeys.subtracting(candKeys), onlyCand = candKeys.subtracting(refKeys)
        if !onlyRef.isEmpty || !onlyCand.isEmpty {
            passed = false
            lines.append("✘ key sets differ: \(onlyRef.count) only in reference, \(onlyCand.count) only in candidate")
            lines += onlyRef.sorted().prefix(5).map { "    − \($0)" } + onlyCand.sorted().prefix(5).map { "    + \($0)" }
        }

        var identical: [Stats] = [], quantized: [Stats] = []
        var control: (key: String, cosine: Float)?
        let start = Date()
        for key in refKeys.intersection(candKeys).sorted() where match.map({ key.contains($0) }) ?? true {
            guard let a = try WeightLoader.tensor(key, plan: refPlan, from: ref),
                let b = try WeightLoader.tensor(key, plan: candPlan, from: cand)
            else { continue }
            guard a.shape == b.shape else {
                passed = false
                lines.append("✘ \(key): shape \(a.shape) vs \(b.shape)")
                continue
            }
            let s = stats(key, a, b)
            if let entry = candByKey[key], case let .int8ConvRot(w, sc, g) = entry.dequant, g > 1 {
                quantized.append(s)
                if control == nil {
                    let raw = try WeightLoader.materialize(
                        .init(key: key, dequant: .int8ConvRot(weightKey: w, scaleKey: sc, groupSize: 0), shape: entry.shape),
                        from: cand)
                    control = (key, stats(key, a, raw).cosine)
                }
            } else if candByKey[key]?.isPassthrough ?? false, !(refPlan.entries.first { $0.key == key }?.isPassthrough == false) {
                identical.append(s)
            } else {
                quantized.append(s)
            }
        }
        let elapsed = Date().timeIntervalSince(start)

        let diverged = identical.filter { $0.maxAbs != 0 }
        lines.append(
            "\(diverged.isEmpty ? "✓" : "✘") unquantized: \(identical.count - diverged.count)/\(identical.count) bit-identical")
        for s in diverged.prefix(5) { lines.append("    \(s.key): max-abs \(s.maxAbs)") }
        if !diverged.isEmpty { passed = false }

        if let worst = quantized.min(by: { $0.cosine < $1.cosine }) {
            let meanCos = quantized.map(\.cosine).reduce(0, +) / Float(quantized.count)
            let meanRel = quantized.map(\.relErr).reduce(0, +) / Float(quantized.count)
            let ok = worst.cosine >= minCosine
            if !ok { passed = false }
            lines.append(
                "\(ok ? "✓" : "✘") dequantized: \(quantized.count) tensors, cosine min \(fmt(worst.cosine)) "
                    + "(\(worst.key)) mean \(fmt(meanCos)); rel-err mean \(fmt(meanRel)), "
                    + "max-abs worst \(fmt(quantized.map(\.maxAbs).max() ?? 0)) — budget cosine ≥ \(minCosine)")
            for s in quantized.sorted(by: { $0.cosine < $1.cosine }).prefix(3) {
                lines.append("    \(s.key): cosine \(fmt(s.cosine)) rel-err \(fmt(s.relErr))")
            }
        }
        if let control {
            let ok = control.cosine < 0.5
            if !ok { passed = false }
            lines.append(
                "\(ok ? "✓" : "✘") control: \(control.key) without un-rotation → cosine \(fmt(control.cosine)) (must collapse)")
        }
        lines.append(String(format: "compared %d tensors in %.1fs", identical.count + quantized.count, elapsed))
        return Report(lines: lines, passed: passed)
    }

    /// Error statistics in F32 between a reference and a candidate tensor.
    static func stats(_ key: String, _ a: MLXArray, _ b: MLXArray) -> Stats {
        let x = a.asType(.float32), y = b.asType(.float32)
        let d = x - y
        let maxAbs = abs(d).max()
        let rel = sqrt(sum(d * d) / maximum(sum(x * x), 1e-30))
        let cos = sum(x * y) / maximum(sqrt(sum(x * x) * sum(y * y)), 1e-30)
        eval(maxAbs, rel, cos)
        return Stats(key: key, maxAbs: maxAbs.item(Float.self), relErr: rel.item(Float.self), cosine: cos.item(Float.self))
    }

    private static func fmt(_ v: Float) -> String { String(format: "%.5g", v) }
}

/// `moss-mlx load` — load a whole file (or a prefix of it) and report what came out, with a
/// finiteness check. The int8 Qwen3.5 encoder has no bf16 twin locally, so this is its check.
public enum LoadCheck {
    public static func run(url: URL, prefix: String = "") throws -> (summary: String, passed: Bool) {
        let start = Date()
        let weights = try WeightLoader.load(url: url, prefix: prefix)
        var elements = 0, nonFinite: [String] = []
        for (key, array) in weights.arrays.sorted(by: { $0.key < $1.key }) {
            elements += array.size
            let finite = all(isFinite(array.asType(.float32))).item(Bool.self)
            if !finite { nonFinite.append(key) }
        }
        let elapsed = Date().timeIntervalSince(start)
        var summary = String(
            format: "%@ %d tensors, %.2f G params in %.1fs", nonFinite.isEmpty ? "✓" : "✘",
            weights.tensorCount, Double(elements) / 1e9, elapsed)
        if !nonFinite.isEmpty { summary += "; non-finite: " + nonFinite.prefix(5).joined(separator: ", ") }
        return (summary, nonFinite.isEmpty)
    }
}
