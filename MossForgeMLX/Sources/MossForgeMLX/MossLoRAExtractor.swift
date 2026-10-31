import Foundation
import MLX
import MLXRandom
import MossForgeCore
import MossForgeEngine

/// Extract LoRA on the GPU: stream both DiTs one tensor at a time (dequantised to fp32, so any
/// precision pairs with any other), take `ΔW = finetune − parent`, and factorise each 2-D difference
/// with a randomized truncated SVD. Memory stays at one layer pair (≤ 16384×3072 fp32) plus the
/// growing LoRA, never two whole models.
extension MossModelTools {
    public struct Extraction: Sendable {
        public var result: Result
        public var report: String
    }

    /// Oversampling and power iterations of the randomized SVD (Halko et al. 2011): with 2 power
    /// iterations the top-`rank` subspace of a slowly decaying spectrum is close to exact SVD's.
    static let svdOversample = 16
    static let svdPowerIterations = 2

    @discardableResult
    public static func extractLoRA(
        input: URL, parent: URL, rank: Int, cutoff: LoRARankCutoff, precision: WeightPrecision, output: URL,
        progress: ProgressHandler? = nil
    ) throws -> Extraction {
        let t0 = Date()
        let sp = try MossWeightPlan.build(url: input, prefix: MossDiTConfig.keyPrefix(of: input))
        let pp = try MossWeightPlan.build(url: parent, prefix: MossDiTConfig.keyPrefix(of: parent))
        let plan = MossLoRAExtraction.plan(source: sp.logicalShapes, parent: pp.logicalShapes)
        guard plan.layers.contains(where: { $0.kind == .lowRank }) else {
            throw EngineError.invalidRequest("\(input.lastPathComponent) and \(parent.lastPathComponent) share no Linear layers")
        }
        let (ss, ps) = (try TensorSource(url: input), try TensorSource(url: parent))
        defer { ss.close(); ps.close() }
        let dtype: TensorDType = precision == .float32 ? .float32 : precision == .float16 ? .float16 : .bfloat16
        let mlxType: DType = precision == .float32 ? .float32 : precision == .float16 ? .float16 : .bfloat16

        var tensors: [SafetensorsWriter.Tensor] = []
        var unchanged = 0, diffs = 0
        var energyKept: [(key: String, fraction: Double, rank: Int)] = []
        var (keptTotal, energyTotal) = (0.0, 0.0)
        try withBoundedCache {
            for (i, layer) in plan.layers.enumerated() {
                if i % 8 == 0 {
                    progress?(GenerationProgress(step: i, totalSteps: plan.layers.count, message: "extracting \(layer.key)"))
                }
                // FileHandle reads come back as autoreleased NSData: without a pool per layer both
                // models' bytes pile up (≈ 20 GB) until the loop ends.
                let pair = try autoreleasepool { () throws -> (MLXArray, MLXArray)? in
                    guard let a = try WeightLoader.tensor(layer.key, plan: sp, from: ss),
                        let b = try WeightLoader.tensor(layer.key, plan: pp, from: ps)
                    else { return nil }
                    let d = a.asType(.float32) - b.asType(.float32)
                    eval(d)
                    return (d, (d * d).sum())
                }
                guard let (d, energySum) = pair else { continue }
                let energy = Double(energySum.item(Float.self))
                guard energy > 0 else { unchanged += 1; continue }
                let names = MossLoRAExtraction.names(for: layer)
                guard layer.kind == .lowRank else {
                    // Norm scales / biases: tiny, kept whole and fp32.
                    tensors.append(bytes(names.down, d, .float32, layer.shape))
                    diffs += 1
                    continue
                }
                let maxRank = min(rank, layer.shape[0], layer.shape[1])
                let (u, s, vt) = truncatedSVD(d, rank: maxRank)
                let sv = s.asArray(Float.self)
                let r = cutoff.rank(singularValues: sv, totalEnergy: energy, maxRank: maxRank)
                // Split √σ into both factors so neither underflows in bf16.
                let root = s[0..<r].sqrt()
                let up = (u[0..., 0..<r] * root.reshaped([1, r])).asType(mlxType)
                let down = (vt[0..<r] * root.reshaped([r, 1])).asType(mlxType)
                tensors.append(bytes(names.down, down, dtype, [r, layer.shape[1]]))
                tensors.append(bytes(names.up, up, dtype, [layer.shape[0], r]))
                tensors.append(scalar(names.alpha, Float(r)))  // alpha = rank ⇒ scale 1
                let kept = sv.prefix(r).reduce(0.0) { $0 + Double($1) * Double($1) }
                keptTotal += kept
                energyTotal += energy
                energyKept.append((layer.key, min(1, kept / energy), r))
            }
        }
        progress?(GenerationProgress(step: plan.layers.count, totalSteps: plan.layers.count, message: "writing \(output.lastPathComponent)"))
        var metadata = MossLoRAExtraction.metadata(
            source: input.deletingPathExtension().lastPathComponent, parent: parent.deletingPathExtension().lastPathComponent,
            rank: rank, cutoff: cutoff)
        metadata["precision"] = precision.rawValue
        let bytes = try SafetensorsWriter.write(tensors, metadata: metadata, to: output)

        let worst = energyKept.min { $0.fraction < $1.fraction }
        let ranks = energyKept.map(\.rank)
        var report = String(
            format: "%d low-rank layers (rank %d–%d), %.1f%% of the difference energy kept overall",
            energyKept.count, ranks.min() ?? 0, ranks.max() ?? 0, energyTotal > 0 ? 100 * keptTotal / energyTotal : 100)
        if let worst { report += String(format: " (least: %@ %.1f%%)", worst.key, 100 * worst.fraction) }
        report += "; \(diffs) norm/bias diffs; \(unchanged) tensors unchanged"
        let skipped = plan.sourceOnly.count + plan.parentOnly.count + plan.mismatched.count
        if skipped > 0 {
            report += "; skipped \(skipped): "
                + (plan.sourceOnly.map { "\($0) (source only)" } + plan.parentOnly.map { "\($0) (parent only)" }
                    + plan.mismatched.map { "\($0) (shape differs)" }).prefix(4).joined(separator: ", ")
        }
        return Extraction(result: Result(output: output, bytes: bytes, seconds: Date().timeIntervalSince(t0)), report: report + ".")
    }

    /// Top-`rank` SVD of `a` `[m, n]` (fp32): randomized range finder with power iterations on the GPU,
    /// then an exact SVD of the small `k×k` core on the CPU. Returns `U [m,k]`, `S [k]` (descending),
    /// `Vᵀ [k,n]` with `k = min(rank + oversample, m, n)`.
    static func truncatedSVD(_ a: MLXArray, rank: Int, seed: UInt64 = 0) -> (MLXArray, MLXArray, MLXArray) {
        let (m, n) = (a.dim(0), a.dim(1))
        let k = min(rank + svdOversample, m, n)
        func orth(_ y: MLXArray) -> MLXArray { qr(y, stream: .cpu).0 }
        let omega = MLXRandom.normal([n, k], key: MLXRandom.key(seed))
        var q = orth(matmul(a, omega))  // [m,k]
        for _ in 0..<svdPowerIterations {
            let z = orth(matmul(a.T, q))  // [n,k]
            q = orth(matmul(a, z))
        }
        let b = matmul(q.T, a)  // [k,n], a ≈ q·b
        // b = rᵀ·q2ᵀ with bᵀ = q2·r, so svd(rᵀ) [k,k] finishes it without an n×n Vᵀ.
        let (q2, r) = qr(b.T, stream: .cpu)  // q2 [n,k], r [k,k]
        let (ur, s, vrt) = svd(r.T, stream: .cpu)
        let u = matmul(q, ur)
        let vt = matmul(vrt, q2.T)
        eval(u, s, vt)
        return (u, s, vt)
    }

    /// Self-check: a known rank-6 difference (plus a norm and a bias change) between two tiny DiT-shaped
    /// files, extracted at rank 8, fused back onto the parent through the real LoRA planner — returns
    /// the worst relative reconstruction error per tensor (≈ 1e-6 at fp32) and the extracted rank.
    public static func extractSelfTest() throws -> (maxRelativeError: Float, ranks: [Int], fixedRank: Int) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mf-extract-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let key = MLXRandom.key(42)
        let ks = MLXRandom.split(key: key, into: 8)
        // dim 64 / ffn 96: qkv [192,64], o [64,64], up_gate [192,64], down [64,96].
        let shapes: [String: [Int]] = [
            "layers.0.proj.qkv.weight": [192, 64], "layers.0.proj.o.weight": [64, 64],
            "layers.0.proj.ffn_up_gate.weight": [192, 64], "layers.0.proj.ffn_down.weight": [64, 96],
        ]
        var parentW: [String: MLXArray] = [:], sourceW: [String: MLXArray] = [:]
        for (i, (k, s)) in shapes.sorted(by: { $0.key < $1.key }).enumerated() {
            let w = MLXRandom.normal(s, key: ks[i]) * 0.05
            let lowRank = matmul(MLXRandom.normal([s[0], 6], key: ks[i + 4]), MLXRandom.normal([6, s[1]], key: ks[(i + 5) % 8])) * 0.01
            parentW[k] = w
            sourceW[k] = w + lowRank
        }
        parentW["layers.0.norm1.weight"] = MLXArray.ones([64])
        sourceW["layers.0.norm1.weight"] = MLXArray.ones([64]) * 1.01
        parentW["video_embedder.bias"] = MLXArray.zeros([64])
        sourceW["video_embedder.bias"] = MLXArray.ones([64]) * 0.02
        let (pURL, sURL) = (dir.appendingPathComponent("parent.safetensors"), dir.appendingPathComponent("source.safetensors"))
        let out = dir.appendingPathComponent("lora.safetensors")
        try write(parentW, precision: .float32, to: pURL, op: "selftest")
        try write(sourceW, precision: .float32, to: sURL, op: "selftest")
        _ = try extractLoRA(input: sURL, parent: pURL, rank: 8, cutoff: .energy(0.999999), precision: .float32, output: out)

        let config = MossDiTConfig(
            dim: 64, heads: 1, layers: 1, ffnDim: 96, latentChannels: 64, textDim: 64, freqDim: 64, ropeBase: 10000, textBlocks: 0,
            qkvLayout: .fused)
        var fused = parentW.mapValues { $0.asType(.float32) }
        try fuseLoRA(LoRASpec(url: out), into: &fused, config: config)
        var worst: Float = 0
        for (k, s) in sourceW {
            let err = ((fused[k]! - s).abs().max() / maximum(s.abs().max(), MLXArray(Float(1e-12)))).item(Float.self)
            worst = Swift.max(worst, err)
        }
        let plan = try MossLoRAKeyMap.plan(url: out, config: config, targetShapes: parentW.mapValues(\.shape))
        // Fixed cutoff keeps the full requested rank even though the difference is rank 6.
        let fixedOut = dir.appendingPathComponent("fixed.safetensors")
        _ = try extractLoRA(input: sURL, parent: pURL, rank: 8, cutoff: .fixed, precision: .bfloat16, output: fixedOut)
        let fixedRank = try MossLoRAKeyMap.plan(url: fixedOut, config: config, targetShapes: parentW.mapValues(\.shape))
            .parts.filter { $0.kind == .lowRank }.map(\.rank).max() ?? 0
        return (worst, plan.parts.filter { $0.kind == .lowRank }.map(\.rank), fixedRank)
    }
}
