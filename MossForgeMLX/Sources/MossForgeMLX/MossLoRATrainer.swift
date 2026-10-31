// AdamW, a synthetic smoke test, and the flow-match loop for MOSS (Stage-3 σ sampling,
// per-token velocity MSE, aspect buckets, fused-module adapters, checkpoints, stop = save). See NOTES.md.

import Foundation
import MLX
import MLXRandom
import MossForgeCore
import MossForgeEngine

/// Compact AdamW over a flat list of parameter tensors (bias-corrected moments, decoupled weight decay).
public final class AdamW {
    private let beta1: Float, beta2: Float, eps: Float, weightDecay: Float
    private var m: [MLXArray]
    private var v: [MLXArray]
    private var t = 0

    public init(count: Int, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8, weightDecay: Float = 0.01) {
        beta1 = betas.0
        beta2 = betas.1
        self.eps = eps
        self.weightDecay = weightDecay
        m = Array(repeating: MLXArray(Float(0)), count: count)
        v = Array(repeating: MLXArray(Float(0)), count: count)
    }

    /// One step at learning rate `lr`; returns the updated parameters.
    public func step(_ params: [MLXArray], _ grads: [MLXArray], lr: Float) -> [MLXArray] {
        t += 1
        let bc1 = 1 - powf(beta1, Float(t))
        let bc2 = 1 - powf(beta2, Float(t))
        var out: [MLXArray] = []
        out.reserveCapacity(params.count)
        for i in params.indices {
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] * grads[i])
            out.append(params[i] * (1 - lr * weightDecay) - lr * (m[i] / bc1) / (sqrt(v[i] / bc2) + eps))
        }
        eval(out + m + v)
        return out
    }
}

/// LoRA training on a MOSS DiT (Base by default): the paper's flow-match objective — `z_σ = (1−σ)x₀ + σε`,
/// target velocity `ε − x₀`, per-token MSE over the image tokens, no σ weighting — with σ drawn from the
/// Stage-3 distribution, the text encoder frozen (captions encoded once, then released), latents cached,
/// and adapters `ΔW = (α/r)·up·down` on the fused block linears (output = kohya keys MossForge loads).
public enum MossLoRATrainer {
    /// Captions + images already prepared (encoded once).
    struct Sample {
        var x0: MLXArray  // [1,128,h,w] normalised tokens
        var context: MLXArray  // [S,4096]
        var reference: MLXArray?  // edit training: the source image's clean tokens
    }

    public static func train(
        _ req: TrainingRequest, samples: [TrainingSample], paths: MossModelPaths, ditDType: DType = .float16,
        progress: ProgressHandler? = nil
    ) async throws -> TrainingResult {
        try req.validate(sampleCount: samples.count)
        let t0 = Date()
        let total = req.steps
        func report(_ step: Int, _ message: String, loss: Double? = nil) {
            progress?(GenerationProgress(step: step, totalSteps: total, message: message, value: loss))
        }

        let slider = req.mode == .slider
        // 1. Captions (or a slider's prompt triples) → Qwen3.5 states (frozen encoder, loaded once, then released).
        report(0, slider ? "encoding slider prompts" : "encoding \(samples.count) captions")
        let tokenizer = try await MossTokenizer.load(folder: paths.tokenizer)
        var contexts: [MLXArray] = []
        // Test prompts (+ the recommended negative) for the checkpoint images, encoded with the captions.
        // A slider's test images are one prompt at each test weight.
        let testPrompts = slider ? (req.sliderTestPrompt.isEmpty ? [] : req.sliderTestWeights.map { _ in req.sliderTestPrompt })
            : req.preparedSamplePrompts
        let testMultipliers: [Float] = slider ? req.sliderTestWeights.map { Float($0) } : testPrompts.map { _ in 1 }
        var testContexts: [MLXArray] = []
        var testNegative: MLXArray?
        var sliderSets: [(target: MLXArray, positive: MLXArray, negative: MLXArray)] = []
        var emptyContext: MLXArray?  // caption dropout
        do {
            let encoder = try Qwen35Encoder.load(url: paths.encoder, dtype: .bfloat16)
            for s in samples {
                try Task.checkCancellation()
                let c = encoder.encode(tokenizer.promptIDs(TrainingRequest.caption(s.caption, trigger: req.trigger))).asType(.bfloat16)
                eval(c)
                contexts.append(c)
            }
            if !slider, req.captionDropout > 0 {
                emptyContext = encoder.encode(tokenizer.promptIDs("")).asType(.bfloat16)
                eval(emptyContext!)
            }
            for p in testPrompts {
                let c = encoder.encode(tokenizer.promptIDs(p))
                eval(c)
                testContexts.append(c)
            }
            for set in slider ? req.sliderPromptSets : [] {
                let c = [set.target, set.positive, set.negative].map { encoder.encode(tokenizer.promptIDs($0)) }
                eval(c)
                sliderSets.append((c[0], c[1], c[2]))
            }
            if !testPrompts.isEmpty || slider {
                testNegative = encoder.encode(tokenizer.promptIDs(MossNegativePrompt.recommended))
                eval(testNegative!)
            }
        }
        MLX.GPU.clearCache()

        // 2. Images → aspect buckets → VAE latents (posterior mean, normalised token space).
        if !slider { report(0, "encoding \(samples.count) images") }
        var prepared: [Sample] = []
        let vae = try FluxVAE.load(url: paths.vae)  // kept for decoding test images
        do {
            for (s, c) in zip(slider ? [] : samples, contexts) {
                try Task.checkCancellation()
                guard let size = PixelImage.size(of: s.image) else { throw PixelImage.Failure.unreadable(s.image) }
                let b = TrainingBuckets.size(forWidth: size.width, height: size.height, pixels: req.pixels)
                let x0 = vae.encode(try PixelImage.load(s.image, width: b.width, height: b.height))
                // Edit pairs: the source at the target's bucket (the community recipe pairs them 1:1).
                let ref = try s.control.map { vae.encode(try PixelImage.load($0, width: b.width, height: b.height)) }
                eval([x0] + (ref.map { [$0] } ?? []))
                prepared.append(Sample(x0: x0, context: c, reference: ref))
            }
        }
        MLX.GPU.clearCache()

        // 3. The DiT (frozen) + zero-initialised adapters on the target linears.
        report(0, "loading \(paths.dit.lastPathComponent)")
        let dit = try MossDiT.load(url: paths.dit, dtype: ditDType)
        dit.checkpointBlocks = req.gradientCheckpointing
        let shapes = MossLoRAOutput.targetShapes(config: dit.config, targets: req.targets)
        let modules = shapes.keys.sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        let lokr = req.factorization == .lokr
        // LoKr factors are full (LyCORIS without decomposition), so their scale is 1; LoRA uses α/r.
        let scale = lokr ? 1 : Float(req.alpha) / Float(req.rank)
        let seed = req.seed ?? UInt64.random(in: 1...UInt64(UInt32.max))
        var params: [MLXArray] = []
        for (i, m) in modules.enumerated() {
            let (out, inp) = (shapes[m]![0], shapes[m]![1])
            if lokr {
                // w1 random, w2 zero: a no-op at step 0, like LoRA's zero `up`.
                let f = MossLoRAOutput.lokrShapes(out: out, in: inp, factor: req.lokrFactor)
                params.append(MLXRandom.normal(f.w1, key: MLXRandom.key(seed &+ UInt64(i))) * (1 / Float(f.w1[1]).squareRoot()))
                params.append(MLXArray.zeros(f.w2))
            } else {
                params.append(MLXRandom.normal([req.rank, inp], key: MLXRandom.key(seed &+ UInt64(i))) * (1 / Float(inp).squareRoot()))
                params.append(MLXArray.zeros([out, req.rank]))
            }
        }
        eval(params)

        // The per-step batch is fed through captured vars; the transform is built once.
        let sliderSize = MossDimensions.fit(pixels: req.pixels, ratio: "1:1")
        let placeholder = MLXArray.zeros([1, MossLatentFormat.tokenChannels, sliderSize.height / 16, sliderSize.width / 16])
        var zt = prepared.first?.x0 ?? placeholder, target = zt, context = prepared.first?.context ?? sliderSets[0].target
        var timestep = 500.0
        var multiplier: Float = 1  // a slider trains at a random weight across its range
        func loss(_ p: [MLXArray]) -> MLXArray {
            var adapters: [String: MossDiT.TrainableLoRA] = [:]
            for (i, m) in modules.enumerated() { adapters[m + ".weight"] = .init(down: p[2 * i], up: p[2 * i + 1], scale: scale * multiplier, kronecker: lokr) }
            dit.trainable = adapters
            let pred = dit(zt, timestep: timestep, context: context)
            let d = pred - target
            return (d * d).mean()  // mean over image tokens × channels (eq. 3)
        }
        let vg = valueAndGrad({ (p: [MLXArray]) in [loss(p)] }, argumentNumbers: Array(0..<params.count))
        let opt = AdamW(count: params.count, betas: (0.9, 0.95), weightDecay: 0.01)
        let schedule = MossNoiseSchedule.stage3
        let sliderPlan = MossSamplerPlan(request: GenerationRequest(
            prompt: "slider", variant: req.family == .sft ? .sft : .base, width: sliderSize.width, height: sliderSize.height,
            steps: req.sliderSampleSteps, cfg: 3, shift: 5, slg: .official))
        let unitEta = Float(req.sliderRange.unitStrength(full: req.sliderEta))
        var rng = SplitMix64(seed: seed)

        func save(_ url: URL, step: Int) throws {
            var tensors: [String: MLXArray] = [:]
            for (i, m) in modules.enumerated() {
                if lokr {
                    let k = MossLoRAOutput.lokrKeys(module: m)
                    tensors[k.w1] = params[2 * i].asType(.float32)  // LoKr factors are small; keep them exact
                    tensors[k.w2] = params[2 * i + 1].asType(.bfloat16)
                } else {
                    let k = MossLoRAOutput.keys(module: m)
                    tensors[k.down] = params[2 * i].asType(.bfloat16)
                    tensors[k.up] = params[2 * i + 1].asType(.bfloat16)
                    tensors[k.alpha] = MLXArray(Float(req.alpha))
                }
            }
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            var metadata = [
                "ss_network_dim": "\(req.rank)", "ss_network_alpha": "\(req.alpha)", "produced_by": "MossForge",
                "base_model": req.baseModel.lastPathComponent, "steps": "\(step)",
                "mossforge_factorization": req.factorization.rawValue, "mossforge_targets": req.targets.rawValue,
            ]
            if req.captionDropout > 0 { metadata["mossforge_caption_dropout"] = "\(req.captionDropout)" }
            // Edit LoRAs say how they saw their references, so Generate picks the layout (`MossEditLoRAInfo`).
            if req.mode == .edit { metadata["mossforge_edit_layout"] = MossEditLayout.mossForge.rawValue }
            let band = req.sliderRange.recommendedBand
            let notes: String
            if slider {
                metadata["mossforge_slider"] = "\(req.sliderPositivePrompt) ⟷ \(req.sliderNegativePrompt)"
                metadata[ModelSidecar.sliderRangeMetadataKey] = "\(band.min),\(band.max)"
                metadata["mossforge_slider_eta"] = "\(req.sliderEta)"
                notes = "Concept slider trained by MossForge on \(req.baseModel.lastPathComponent): + “\(req.sliderPositivePrompt)”, − “\(req.sliderNegativePrompt)”"
                    + (req.sliderPreservedAttributes.isEmpty ? "" : ", preserving \(req.sliderPreservedAttributes.joined(separator: ", "))")
                    + String(format: ". Use at %g…%g (trained %g…%g, η %g), %d steps, rank %d.", band.min, band.max, req.sliderRange.min, req.sliderRange.max, req.sliderEta, step, req.rank)
            } else {
                notes = "Trained by MossForge on \(req.baseModel.lastPathComponent): \(samples.count) images, \(step) steps, \(lokr ? "LoKr factor \(req.lokrFactor <= 0 ? "auto" : "\(req.lokrFactor)")" : "rank \(req.rank)/α \(req.alpha)"), lr \(req.learningRate), \(req.targets.displayName), ≤\(Int(Double(req.pixels).squareRoot()))²"
                    + (req.trigger.isEmpty ? "." : ", trigger “\(req.trigger)”.")
            }
            try MLX.save(arrays: tensors, metadata: metadata, url: url)
            try ModelSidecar(families: [req.family], strength: 1, notes: notes).write(for: url)
        }

        // Test images with the adapters as they are now: the official flow sampler (CFG + uncond-skip SLG)
        // at a fixed seed per prompt, so each row differs only by what the LoRA has learned.
        let sampleSize = MossDimensions.fit(pixels: req.samplePixels, ratio: "1:1")
        let sampleReference: MLXArray? = try req.mode == .edit
            ? samples.first?.control.map { vae.encode(try PixelImage.load($0, width: sampleSize.width, height: sampleSize.height)) } : nil
        func renderSamples(at step: Int) throws {
            guard let negative = testNegative, !testContexts.isEmpty else { return }
            dit.references = sampleReference.map { [$0] } ?? []
            defer { dit.references = [] }
            let plan = MossSamplerPlan(request: GenerationRequest(
                prompt: "test", variant: req.family == .sft ? .sft : .base, width: sampleSize.width, height: sampleSize.height,
                steps: req.sampleSteps, cfg: req.sampleCFG, shift: 5, slg: .official))
            var files: [URL] = []
            for (index, context) in testContexts.enumerated() {
                try Task.checkCancellation()
                report(step, "test image \(index + 1)/\(testContexts.count) at step \(step)")
                var adapters: [String: MossDiT.TrainableLoRA] = [:]
                for (i, m) in modules.enumerated() {
                    adapters[m + ".weight"] = .init(down: params[2 * i], up: params[2 * i + 1], scale: scale * testMultipliers[index], kronecker: lokr)
                }
                dit.trainable = testMultipliers[index] == 0 ? [:] : adapters
                // A slider's images share one seed (only the weight changes); test prompts get one each.
                var x = MLXRandom.normal([1, MossLatentFormat.tokenChannels, sampleSize.height / 16, sampleSize.width / 16],
                    key: MLXRandom.key(req.sampleSeed &+ UInt64(slider ? 0 : index)))
                for s in plan.steps {
                    var branches = [MossDiT.Branch(context: context)]
                    if s.usesNegative { branches.append(MossDiT.Branch(context: negative, skipLayers: Set(s.uncondSkipLayers))) }
                    let v = dit.velocities(x, timestep: s.timestep, branches: branches)
                    let prediction = s.usesNegative ? v[1] + Float(plan.cfg) * (v[0] - v[1]) : v[0]
                    x = x + Float(s.nextSigma - s.sigma) * prediction
                    eval(x)
                }
                let image = vae.decode(x)
                eval(image)
                let url = TrainingSamples.url(output: req.output, step: step, index: index)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try PixelImage.pngData(image).write(to: url)
                files.append(url)
            }
            MLX.GPU.clearCache()
            progress?(GenerationProgress(step: step, totalSteps: total, message: "test images at step \(step)", files: files))
        }
        try renderSamples(at: 0)
        GPU.resetPeakMemory()  // the loop's peak, not the encoder's (`moss-mlx train` prints it)

        // 4. The loop.
        var losses: [Double] = []
        var ema: Double?
        var stepTimes: [Double] = []
        var finished = 0
        for step in 0..<total {
            if Task.isCancelled { break }
            let started = Date()
            var sigma: Double
            if slider {
                // Concept Sliders (on MOSS's flow sampler): partly denoise a latent with
                // the frozen model, then train the LoRA at weight m to move the target's velocity by m·η/peak
                // along v(positive) − v(negative). The base velocities don't depend on the LoRA.
                let set = sliderSets[Int(rng.next() % UInt64(sliderSets.count))]
                dit.trainable = [:]
                let k = 1 + Int(rng.next() % UInt64(max(1, sliderPlan.steps.count - 1)))
                var x = MLXRandom.normal(placeholder.shape, key: MLXRandom.key(rng.next()))
                for st in sliderPlan.steps.prefix(k) {
                    let v = dit.velocities(x, timestep: st.timestep, branches: [
                        MossDiT.Branch(context: set.target), MossDiT.Branch(context: testNegative!, skipLayers: Set(st.uncondSkipLayers)),
                    ])
                    x = x + Float(st.nextSigma - st.sigma) * (v[1] + Float(sliderPlan.cfg) * (v[0] - v[1]))
                    eval(x)
                }
                let at = sliderPlan.steps[k]
                let base = dit.velocities(x, timestep: at.timestep, branches: [
                    MossDiT.Branch(context: set.target), MossDiT.Branch(context: set.positive), MossDiT.Branch(context: set.negative),
                ])
                multiplier = Float(req.sliderRange.multiplier(at: Double(rng.next() >> 11) / Double(1 << 53)))
                zt = x
                target = base[0] + (multiplier * unitEta) * (base[1] - base[2])
                eval(target)
                context = set.target
                timestep = at.timestep
                sigma = at.sigma
            } else {
                let s = prepared[Int(rng.next() % UInt64(prepared.count))]
                dit.references = s.reference.map { [$0] } ?? []
                sigma = schedule.sigma(u: Double(rng.next() >> 11) / Double(1 << 53))
                let noise = MLXRandom.normal(s.x0.shape, key: MLXRandom.key(rng.next()))
                zt = (1 - Float(sigma)) * s.x0 + Float(sigma) * noise
                target = noise - s.x0
                context = s.context
                // Caption dropout: now and then train on an empty caption.
                if let empty = emptyContext, Double(rng.next() >> 11) / Double(1 << 53) < req.captionDropout { context = empty }
                timestep = 1000 * sigma
            }

            let (values, grads0) = vg(params)
            var grads = grads0
            // Global-norm gradient clipping.
            let norm = sqrt(grads.reduce(MLXArray(Float(0))) { $0 + ($1 * $1).sum() }).item(Float.self)
            if req.gradientClip > 0, norm > Float(req.gradientClip) {
                let k = Float(req.gradientClip) / norm
                grads = grads.map { $0 * k }
            }
            let lr = Float(req.learningRate) * min(1, Float(step + 1) / Float(max(1, req.warmupSteps)))
            params = opt.step(params, grads, lr: lr)

            let l = Double(values[0].item(Float.self))
            losses.append(l)
            ema = ema.map { 0.95 * $0 + 0.05 * l } ?? l
            stepTimes.append(Date().timeIntervalSince(started))
            finished = step + 1
            let recent = stepTimes.suffix(20)
            report(finished, String(format: "step %d/%d · loss %.4f (avg %.4f) · σ %.2f · %.2f s/step",
                finished, total, l, ema!, sigma, recent.reduce(0, +) / Double(recent.count)), loss: l)
            if req.saveEvery > 0, finished % req.saveEvery == 0, finished < total {
                try save(checkpointURL(req.output, step: finished), step: finished)
                try renderSamples(at: finished)
            }
        }
        dit.trainable = [:]
        dit.references = []
        dit.checkpointBlocks = false
        multiplier = 1
        guard finished > 0 else { throw CancellationError() }
        report(finished, "saving \(req.output.lastPathComponent)")
        try save(req.output, step: finished)
        // A stopped run still gets its final row (cancellation is cleared for this last render).
        try? await Task { try renderSamples(at: finished) }.value
        MLX.GPU.clearCache()
        return TrainingResult(output: req.output, steps: finished, losses: losses, seconds: Date().timeIntervalSince(t0))
    }

    /// `<dir>/<name>-checkpoints/<name>-step<N>.safetensors`.
    public static func checkpointURL(_ output: URL, step: Int) -> URL {
        let name = output.deletingPathExtension().lastPathComponent
        return output.deletingLastPathComponent().appendingPathComponent("\(name)-checkpoints")
            .appendingPathComponent(String(format: "%@-step%05d.safetensors", name, step))
    }

    /// Fit a LoRA delta so a frozen linear reproduces a known low-rank perturbation: if `valueAndGrad`,
    /// AdamW and the adapter composition work, the loss collapses. `moss-mlx train-smoke`.
    public static func smoke(steps: Int = 200) -> (first: Float, last: Float) {
        let (inDim, outDim, rank, n) = (32, 16, 4, 8)
        let base = MLXRandom.normal([outDim, inDim], key: MLXRandom.key(0)) * 0.1
        let x = MLXRandom.normal([n, inDim], key: MLXRandom.key(1))
        let trueDown = MLXRandom.normal([rank, inDim], key: MLXRandom.key(3)), trueUp = MLXRandom.normal([outDim, rank], key: MLXRandom.key(4))
        let target = matmul(x, base.T) + matmul(matmul(x, trueDown.T), trueUp.T)
        var params = [MLXRandom.normal([rank, inDim], key: MLXRandom.key(2)) * (1 / Float(inDim).squareRoot()), MLXArray.zeros([outDim, rank])]
        let vg = valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            let d = matmul(x, base.T) + matmul(matmul(x, p[0].T), p[1].T) - target
            return [(d * d).mean()]
        }, argumentNumbers: [0, 1])
        let opt = AdamW(count: 2)
        var first: Float = 0, last: Float = 0
        for s in 0..<steps {
            let (values, grads) = vg(params)
            last = values[0].item(Float.self)
            if s == 0 { first = last }
            params = opt.step(params, grads, lr: 5e-2)
        }
        return (first, last)
    }

    /// `MossDiT.kroneckerApply` must equal `x · (w1 ⊗ w2)ᵀ` with the dense product (LoKr inference path).
    public static func kroneckerSmoke() -> Float {
        let w1 = MLXRandom.normal([3, 4], key: MLXRandom.key(7)), w2 = MLXRandom.normal([5, 6], key: MLXRandom.key(8))
        let x = MLXRandom.normal([2, 24], key: MLXRandom.key(9))
        let dense = (w1.reshaped([3, 1, 4, 1]) * w2.reshaped([1, 5, 1, 6])).reshaped([15, 24])
        return abs(MossDiT.kroneckerApply(x, w1: w1, w2: w2) - matmul(x, dense.T)).max().item(Float.self)
    }
}

/// Small deterministic RNG for σ / sample / noise-key draws (reproducible runs from a seed).
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
