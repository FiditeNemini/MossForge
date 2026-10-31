/// MOSS-Image's sampling schedule: a uniform 1→0 grid time-shifted by a constant `shift`
/// (paper App. J; ComfyUI `ModelSamplingSD3`). There is no resolution-
/// dependent μ — the release default is a fixed shift of 5.
///
///     u_k = 1 − k/N,   σ_k = s·u_k / (1 + (s − 1)·u_k),   k = 0…N
///
/// The DiT is conditioned on `timestep = 1000·σ` (Comfy `multiplier: 1000`).
public enum MossSchedule {
    /// `steps + 1` noise levels from 1 down to 0.
    public static func sigmas(steps: Int, shift: Double) -> [Double] {
        precondition(steps > 0, "steps must be positive")
        return (0...steps).map { k in
            let u = 1.0 - Double(k) / Double(steps)
            return shift * u / (1.0 + (shift - 1.0) * u)
        }
    }

    /// The model timestep for each of the `steps` denoising updates (σ = 0 is never evaluated).
    public static func timesteps(steps: Int, shift: Double) -> [Double] {
        sigmas(steps: steps, shift: shift).dropLast().map { $0 * 1000 }
    }
}

extension MossSchedule {
    /// The **official** grid (OpenMOSS `pipeline_moss_image.py::MOSSImageScheduler`, the one
    /// `gradio_app.py` runs), which differs from `sigmas(steps:shift:)` in the tail:
    ///
    ///     v = linspace(1, s/(T+s−1), steps)        (float32, T = 1000 training steps)
    ///     σ = s·v / (1 + (s−1)·v)                  → snapped to the nearest training σ (k/T, ties
    ///                                                down), consecutive duplicates merged, then 0
    ///
    /// so 50 steps at shift 5 end …, 0.115, 0.024, 0 (the paper grid ends …, 0.093, 0). Returns the
    /// evaluated σ plus the terminal 0.
    public static func pipelineSigmas(steps: Int, shift: Double, trainingSteps: Int = 1000) -> [Double] {
        precondition(steps > 0 && shift > 0, "steps and shift must be positive")
        let s = Float(shift), total = Float(trainingSteps)
        let lower = s / (total + s - 1)
        var out: [Double] = []
        for i in 0..<steps {
            // torch.linspace fills each half from its own end.
            let step = steps > 1 ? (lower - 1) / Float(steps - 1) : 0
            let v = i < steps / 2 ? 1 + Float(i) * step : lower - Float(steps - 1 - i) * step
            let sigma = s * v / (1 + (s - 1) * v)
            // Training grid (shift 1) is k/T for k = 1…T; nearest, ties to the lower σ.
            let x = Double(sigma) * Double(trainingSteps)
            let k = min(max(x - x.rounded(.down) <= x.rounded(.up) - x ? x.rounded(.down) : x.rounded(.up), 1), Double(trainingSteps))
            let snapped = k / Double(trainingSteps)
            if out.last != snapped { out.append(snapped) }
        }
        return out + [0]
    }
}

/// One denoising update of the official flow-Euler sampler.
public struct MossSamplerStep: Sendable, Equatable {
    public var index: Int
    public var sigma: Double
    public var nextSigma: Double
    /// `1000·σ`, what the DiT is conditioned on.
    public var timestep: Double
    /// Whether the CFG negative branch runs (cfg ≠ 1).
    public var usesNegative: Bool
    /// Blocks skipped in the negative branch this step (empty = none / inactive).
    public var uncondSkipLayers: [Int]
    /// Blocks skipped in the extra standard-SLG positive pass (empty = no extra pass).
    public var slgLayers: [Int]

    /// DiT forwards this step costs.
    public var forwards: Int { 1 + (usesNegative ? 1 : 0) + (slgLayers.isEmpty ? 0 : 1) }
}

/// The per-step plan for a flow-Euler request: σ from `MossSchedule.pipelineSigmas`, which passes
/// run, and with which blocks skipped — the loop body of the official `__call__`.
public struct MossSamplerPlan: Sendable, Equatable {
    public var steps: [MossSamplerStep]
    public var cfg: Double
    public var slgScale: Double

    public init(request r: GenerationRequest) {
        let sigmas = MossSchedule.pipelineSigmas(steps: r.steps, shift: r.shift)
        let n = sigmas.count - 1
        let negative = r.cfg != 1
        steps = (0..<n).map { i in
            let active = r.slg.isActive(step: i, of: n)
            return MossSamplerStep(
                index: i, sigma: sigmas[i], nextSigma: sigmas[i + 1], timestep: 1000 * sigmas[i], usesNegative: negative,
                uncondSkipLayers: negative && active ? r.slg.uncondSkipLayers : [],
                slgLayers: active && r.slg.scale > 0 ? r.slg.layers : [])
        }
        cfg = r.cfg
        slgScale = r.slg.scale
    }

    public var totalForwards: Int { steps.reduce(0) { $0 + $1.forwards } }
}

/// The official guidance algebra (applied in F32 to whole velocity tensors by the backend; scalar
/// here so it is unit-tested).
public enum MossGuidance {
    /// `p = u + cfg·(c − u)` (or `c` without a negative branch); with SLG active `p += scale·(c − c_skip)`,
    /// where a cfg-1 run starts from `c_skip` instead (`c_skip + scale·(c − c_skip)`).
    public static func combine<T: FloatingPoint>(cond: T, uncond: T?, condSkipped: T?, cfg: T, slgScale: T) -> T {
        var p = uncond.map { $0 + cfg * (cond - $0) } ?? cond
        if let skipped = condSkipped {
            if uncond == nil { p = skipped }
            p += slgScale * (cond - skipped)
        }
        return p
    }

    /// Flow-matching Euler: `x + (σ_next − σ)·v`.
    public static func eulerStep<T: FloatingPoint>(x: T, velocity: T, sigma: T, nextSigma: T) -> T {
        x + (nextSigma - sigma) * velocity
    }
}
