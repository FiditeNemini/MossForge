import Foundation
import MLX
import MLXRandom
import MossForgeCore
import MossForgeEngine

/// Where the MOSS-Image components live under a `models/` folder (the repo layout, Comfy names).
public struct MossModelPaths: Sendable, Equatable {
    public var dit: URL
    public var vae: URL
    public var encoder: URL
    public var tokenizer: URL
    /// Turbo only: lodestones' data-space DMD adapter — LoRA (unmerged, F32), the F32 step-embedder +
    /// modulation-bias overrides, and `latent_mix.Q` + `step_codes`.
    public var turbo: TurboAssets?

    /// `lora` / `timeEmbedding` are `nil` when the Turbo is fused into the DiT (built in).
    public struct TurboAssets: Sendable, Equatable {
        public var lora: URL?
        public var loraStrength: Float
        public var timeEmbedding: URL?
        public var dataSpace: URL
    }

    public init(dit: URL, vae: URL, encoder: URL, tokenizer: URL, turbo: TurboAssets? = nil) {
        self.dit = dit
        self.vae = vae
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.turbo = turbo
    }

    /// Resolve a variant under `models/` through `MossModelLayout` (bf16 DiT preferred over int8·convrot).
    /// Turbo resolves like SFT; `MossBackend.route` adds its akhaliq 4-step LoRA.
    /// `dit` replaces the variant's default DiT (a library model); `overrides` are Settings' VAE / text
    /// encoder / tokenizer files. `requireDiT: false` is for users of the shared components only (the
    /// enhancer, captioner, Help → Ask), so a folder with just a library DiT — the setup bundle's
    /// Essentials — still serves them; `dit` is then the `diffusion_models` folder, never loaded.
    public static func resolve(
        models: URL, variant: MossVariant, overrides: MossModelLayout.Overrides = .init(), dit: URL? = nil,
        requireDiT: Bool = true
    ) throws -> MossModelPaths {
        func url(_ c: MossModelLayout.Component) throws -> URL {
            guard let u = MossModelLayout.find(c, in: models, overrides: overrides) else {
                throw EngineError.inputNotFound(models.appendingPathComponent(c.candidates[0]))
            }
            return u
        }
        let shared = requireDiT ? nil : MossModelLayout.find(MossModelLayout.dit(for: variant), in: models, overrides: overrides)
            ?? models.appendingPathComponent("diffusion_models")
        var paths = MossModelPaths(
            dit: try dit ?? shared ?? url(MossModelLayout.dit(for: variant)), vae: try url(.vae), encoder: try url(.textEncoder),
            tokenizer: try url(.tokenizer))
        return paths
    }
}

/// The text-to-image pipeline, mirroring OpenMOSS `pipeline_moss_image.py::__call__`:
/// prompt + negative → Qwen3.5 states (unpadded, one forward each) → fp32 Gaussian noise in token
/// space `[1,128,H/16,W/16]` → `MossSamplerPlan` loop (cond pass; negative pass with uncond-skip
/// blocks; optional standard-SLG pass; F32 guidance + Euler) → BN-denormalise → VAE decode.
///
/// The encoder is loaded, used and dropped per request unless `keepEncoderResident` (16 GB bf16;
/// `MossBackend` sets it from its residency policy); the DiT and VAE stay resident until `release()`.
public final class MossPipeline: @unchecked Sendable {
    public struct Options: Sendable {
        /// fp16, not bf16: on M1/M2 (no bf16 ALUs) its matmuls/SDPA are ~20% faster, and its extra
        /// mantissa puts it ~9× closer to the f32 reference (`dit-parity`). The weights fit
        /// (max |w| 12.75); activations stay F32 between matmuls.
        public var ditDType: DType = .float16
        public var encoderDType: DType = .bfloat16
        public var keepEncoderResident = false
        /// Turbo: apply lodestones' F32 step-embedder + modulation-bias overrides (on by default; the
        /// switch exists for A/B against the README's "LoRA only" route).
        public var turboOverrides = true
        public init() {}
    }

    public let paths: MossModelPaths
    public let options: Options
    private var tokenizer: MossTokenizer?
    /// The text encoder (+ the vision tower for edit layouts whose encoder sees the references) —
    /// shared with the enhancer, captioner and Help when the backend passes one in.
    public let qwen: SharedQwen35
    private let ownsQwen: Bool
    private var dit: MossDiT?
    private var vae: FluxVAE?
    private var attached: [String] = []  // "path@strength" of the LoRAs on `dit`
    private var dataSpace: (q: MLXArray, codes: [Double])?

    public init(paths: MossModelPaths, options: Options = .init(), qwen: SharedQwen35? = nil) {
        self.paths = paths
        self.options = options
        self.qwen = qwen ?? SharedQwen35(url: paths.encoder, dtype: options.encoderDType, keepTransformer: options.keepEncoderResident)
        ownsQwen = qwen == nil
    }

    /// Drops the DiT, VAE and adapters; a shared Qwen3.5 is the backend's to release.
    public func release() {
        if ownsQwen { qwen.releaseAll() }
        dit = nil
        vae = nil
        attached = []
        dataSpace = nil
        MLX.GPU.clearCache()
    }

    /// Generate one image `[1,3,H,W]` in [0,1]. `noise` overrides the seeded `[1,128,H/16,W/16]` draw
    /// (fixed-noise parity); `inspect` sees `context_pos`, `context_neg` and the final `latent`.
    public func generate(
        _ request: GenerationRequest, seed: UInt64, noise: MLXArray? = nil, progress: ProgressHandler? = nil,
        inspect: ((String, MLXArray) -> Void)? = nil
    ) async throws -> MLXArray {
        try request.validate()
        if request.sampler == .dataSpaceDMD {
            return try await generateDataSpace(request, seed: seed, noise: noise, progress: progress, inspect: inspect)
        }
        let plan = MossSamplerPlan(request: request)

        // 1. Text conditioning.
        progress?(GenerationProgress(step: 0, totalSteps: plan.steps.count, message: "encoding prompt"))
        if tokenizer == nil { tokenizer = try await MossTokenizer.load(folder: paths.tokenizer) }
        let enc = try qwen.transformer()
        let layout = editLayout(request)
        let pictures = try referencePictures(request, layout: layout, encoder: enc)
        let positive = try encode(request.prompt, pictures: pictures, encoder: enc)
        let negative = plan.steps.contains(where: \.usesNegative) ? try encode(request.negativePrompt, pictures: pictures, encoder: enc) : nil
        eval([positive] + (negative.map { [$0] } ?? []))
        inspect?("context_pos", positive)
        if let negative { inspect?("context_neg", negative) }
        if !options.keepEncoderResident { qwen.trim() }
        try Task.checkCancellation()

        // 2. Denoise.
        progress?(GenerationProgress(step: 0, totalSteps: plan.steps.count, message: "loading DiT"))
        let model = try preparedDiT(loras: request.loras)
        let shape = [1, MossLatentFormat.tokenChannels, request.height / 16, request.width / 16]
        var x = noise?.asType(.float32) ?? MLXRandom.normal(shape, key: MLXRandom.key(seed))  // fp32, token space
        guard x.shape == shape else { throw EngineError.invalidRequest("noise must be \(shape), got \(x.shape)") }
        let cfg = Float(plan.cfg), slgScale = Float(plan.slgScale)
        // Edit: outside the mask the latent follows the original, re-noised with this seed's noise.
        let pin = try editPin(request)
        let epsilon = x
        model.editLayout = layout
        model.references = try referenceLatents(request)
        defer { model.references = [] }
        for step in plan.steps {
            try Task.checkCancellation()
            let start = Date()
            // One packed forward for all branches (cond, uncond, standard-SLG): shared matmuls,
            // per-branch attention — the same numbers as separate forwards.
            var branches = [MossDiT.Branch(context: positive)]
            if step.usesNegative, let negative {
                branches.append(MossDiT.Branch(context: negative, skipLayers: Set(step.uncondSkipLayers)))
            }
            if !step.slgLayers.isEmpty { branches.append(MossDiT.Branch(context: positive, skipLayers: Set(step.slgLayers))) }
            let v = model.velocities(x, timestep: step.timestep, branches: branches)
            let cond = v[0]
            var prediction = cond
            if step.usesNegative, negative != nil {
                let uncond = v[1]
                prediction = uncond + cfg * (cond - uncond)
            }
            if !step.slgLayers.isEmpty {
                let skipped = v[branches.count - 1]
                if !step.usesNegative { prediction = skipped }
                prediction = prediction + slgScale * (cond - skipped)
            }
            x = x + Float(step.nextSigma - step.sigma) * prediction
            if let pin {
                let known = (1 - Float(step.nextSigma)) * pin.x0 + Float(step.nextSigma) * epsilon
                x = pin.mask * x + (1 - pin.mask) * known
            }
            eval(x)
            progress?(GenerationProgress(
                step: step.index + 1, totalSteps: plan.steps.count,
                message: String(format: "σ %.3f · %d fwd · %.2fs", step.sigma, step.forwards, Date().timeIntervalSince(start))))
        }

        inspect?("latent", x)

        // 3. Decode.
        progress?(GenerationProgress(step: plan.steps.count, totalSteps: plan.steps.count, message: "decoding"))
        if vae == nil { vae = try FluxVAE.load(url: paths.vae) }
        let image = vae!.decode(x)
        eval(image)
        return composite(image, pin)
    }
}

extension MossPipeline {
    /// The DiT with exactly the wanted LoRAs attached (turbo adapter first, then the request's).
    func preparedDiT(loras: [LoRASpec]) throws -> MossDiT {
        if dit == nil {
            dit = try MossDiT.load(url: paths.dit, dtype: options.ditDType)
            attached = []
            if let te = paths.turbo?.timeEmbedding, options.turboOverrides {
                try dit!.overrideParameters(MLX.loadArrays(url: te))
            }
        }
        let wanted = (paths.turbo.flatMap { t in t.lora.map { [LoRASpec(url: $0, strength: Double(t.loraStrength))] } } ?? []) + loras
        let keys = wanted.map { spec -> String in
            let blocks = spec.blocks.map { b in
                ":" + b.blocks.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ",") + ",*=\(b.unlisted),other=\(b.other)"
            } ?? ""
            return "\(spec.url.path)@\(spec.strength)\(blocks)"
        }
        if keys != attached {
            dit!.detachLoRAs()
            for spec in wanted { try dit!.attachLoRA(spec) }
            attached = keys
        }
        return dit!
    }

    /// Edit in Layout: the original's clean latent, the token mask `[1,1,h,w]` (1 = redraw) and the
    /// pixel mask + pixels for the final paste-back.
    struct EditPin {
        var x0: MLXArray
        var mask: MLXArray
        var pixelMask: MLXArray
        var pixels: MLXArray
    }

    func editPin(_ request: GenerationRequest) throws -> EditPin? {
        guard let edit = request.edit else { return nil }
        let pixels = try PixelImage.load(edit.source, width: request.width, height: request.height)
        if vae == nil { vae = try FluxVAE.load(url: paths.vae) }
        let x0 = vae!.encode(pixels)
        let m = edit.mask
        let pin = EditPin(
            x0: x0, mask: MLXArray(m.values, [1, 1, m.height, m.width]),
            pixelMask: MLXArray(m.pixels(scale: 16), [1, 1, m.height * 16, m.width * 16]), pixels: pixels)
        eval(pin.x0, pin.mask, pin.pixelMask, pin.pixels)
        return pin
    }

    /// The request's edit layout: explicit, else told by its (non-zero) LoRAs' metadata.
    func editLayout(_ request: GenerationRequest) -> MossEditLayout {
        guard !request.references.isEmpty else { return .mossForge }
        let infos = request.loras.filter { $0.strength != 0 }.map { MossEditLoRAInfo(url: $0.url) }
        return MossEditLayout.resolve(explicit: request.editLayout, loras: infos)
    }

    /// The encoder's view of the references when the layout shows them to it: each area-scaled to
    /// ≤ 384² and run through Qwen3.5's vision tower (ComfyUI `TextEncodeMOSSImageEdit`).
    func referencePictures(
        _ request: GenerationRequest, layout: MossEditLayout, encoder enc: Qwen35Encoder
    ) throws -> [(tokens: MLXArray, grid: QwenVisionGeometry.Grid)] {
        guard layout.encoderSeesReferences, !request.references.isEmpty else { return [] }
        let tower = try qwen.visionTower()
        defer { if !options.keepEncoderResident { qwen.releaseTower() } }
        return try request.references.map { url in
            let image = try RGB8.load(url: url)
            let size = QwenVisionGeometry.editReferenceSize(height: image.height, width: image.width)
            let (patches, grid) = QwenVisionPreprocess.preprocess(image, height: size.height, width: size.width)
            let tokens = tower(patches.asType(enc.dtype), gridTHW: [(grid.t, grid.h, grid.w)])
            eval(tokens)
            return (tokens, grid)
        }
    }

    /// Prompt → Qwen3.5 states; with pictures, "Picture N:" + their vision tokens lead the
    /// instruction and positions are M-RoPE.
    func encode(_ text: String, pictures: [(tokens: MLXArray, grid: QwenVisionGeometry.Grid)], encoder enc: Qwen35Encoder) throws -> MLXArray {
        guard !pictures.isEmpty else { return enc.encode(tokenizer!.promptIDs(text)) }
        guard let pad = tokenizer!.id(of: MossEditPrompt.imagePad) else {
            throw EngineError.backendFailure("tokenizer lacks \(MossEditPrompt.imagePad)")
        }
        let ids = tokenizer!.promptIDs(MossEditPrompt.instruction(text, pictureTokens: pictures.map(\.grid.tokens)))
        let (embeddings, positions) = try MossCaptioner.prefill(ids: ids, imagePad: pad, images: pictures, encoder: enc)
        return enc.forward(embeddings: embeddings, positions: positions, cache: nil)
    }

    /// Edit-LoRA references → clean latents at the request's pixel budget (aspect kept, ×16).
    func referenceLatents(_ request: GenerationRequest) throws -> [MLXArray] {
        guard !request.references.isEmpty else { return [] }
        if vae == nil { vae = try FluxVAE.load(url: paths.vae) }
        return try request.references.map { url in
            guard let size = PixelImage.size(of: url) else { throw PixelImage.Failure.unreadable(url) }
            let b = TrainingBuckets.size(forWidth: size.width, height: size.height, pixels: request.width * request.height)
            let z = vae!.encode(try PixelImage.load(url, width: b.width, height: b.height))
            eval(z)
            return z
        }
    }

    /// Final paste-back: the original's pixels outside the (feathered) mask.
    func composite(_ image: MLXArray, _ pin: EditPin?) -> MLXArray {
        guard let pin else { return image }
        let out = pin.pixelMask * image + (1 - pin.pixelMask) * pin.pixels
        eval(out)
        return out
    }

    /// lodestones' data-space DMD loop (no CFG, no negative prompt):
    /// `x₀ = 0; for r: x = z + Q·x₀; v = DiT(x, code_r); x₀ = z − v` → decode `x₀`.
    /// Any prefix of the 8 trained steps works.
    func generateDataSpace(
        _ request: GenerationRequest, seed: UInt64, noise: MLXArray?, progress: ProgressHandler?,
        inspect: ((String, MLXArray) -> Void)?
    ) async throws -> MLXArray {
        guard let turbo = paths.turbo else {
            throw EngineError.invalidRequest("the data-space DMD sampler needs the Turbo variant's adapter files")
        }
        if dataSpace == nil {
            let ds = try MLX.loadArrays(url: turbo.dataSpace)
            guard let q = ds["latent_mix.Q"], let codes = ds["step_codes"] else {
                throw WeightLoaderError.missingTensors(["latent_mix.Q", "step_codes"])
            }
            dataSpace = (q.asType(.float32), codes.asArray(Float.self).map(Double.init))
        }
        let (q, codes) = dataSpace!
        let steps = min(request.steps, codes.count)

        progress?(GenerationProgress(step: 0, totalSteps: steps, message: "encoding prompt"))
        if tokenizer == nil { tokenizer = try await MossTokenizer.load(folder: paths.tokenizer) }
        let enc = try qwen.transformer()
        let layout = editLayout(request)
        let positive = try encode(request.prompt, pictures: try referencePictures(request, layout: layout, encoder: enc), encoder: enc)
        eval(positive)
        inspect?("context_pos", positive)
        if !options.keepEncoderResident { qwen.trim() }

        progress?(GenerationProgress(step: 0, totalSteps: steps, message: "loading DiT + Turbo adapter"))
        let model = try preparedDiT(loras: request.loras)
        let (c, h, w) = (MossLatentFormat.tokenChannels, request.height / 16, request.width / 16)
        let z = noise?.asType(.float32) ?? MLXRandom.normal([1, c, h, w], key: MLXRandom.key(seed))
        guard z.shape == [1, c, h, w] else { throw EngineError.invalidRequest("noise must be [1,\(c),\(h),\(w)]") }
        var x0 = MLXArray.zeros(z.shape)
        let pin = try editPin(request)
        model.editLayout = layout
        model.references = try referenceLatents(request)
        defer { model.references = [] }
        for r in 0..<steps {
            try Task.checkCancellation()
            let start = Date()
            let x = z + matmul(q, x0.reshaped([c, h * w])).reshaped([1, c, h, w])  // einsum("oc,bchw->bohw")
            x0 = z - model(x, timestep: codes[r], context: positive)
            if let pin { x0 = pin.mask * x0 + (1 - pin.mask) * pin.x0 }  // edit: keep the original's estimate outside
            eval(x0)
            progress?(GenerationProgress(
                step: r + 1, totalSteps: steps, message: String(format: "code %.0f · %.2fs", codes[r], Date().timeIntervalSince(start))))
        }
        inspect?("latent", x0)
        progress?(GenerationProgress(step: steps, totalSteps: steps, message: "decoding"))
        if vae == nil { vae = try FluxVAE.load(url: paths.vae) }
        let image = vae!.decode(x0)
        eval(image)
        return composite(image, pin)
    }
}

/// The MLX `EngineBackend`: injected in one line at the composition root
/// (`MossEngine(backend: MossBackend(models:))`); front-ends only see the protocol.
public actor MossBackend: EngineBackend {
    public nonisolated let name = "mlx"
    public nonisolated let capabilities: EngineCapabilities = [.imageGeneration, .promptEnhancing, .modelInspection, .loRAApply, .turboSampling, .upscaling, .captioning, .modelTooling, .loRATraining, .chat]
    private let models: URL
    private let options: MossPipeline.Options
    private let turboLoRA: LoRASpec?
    private let overrides: MossModelLayout.Overrides
    /// What stays loaded between jobs (Settings → Memory); `modelsCoexist` turns off the evictions below.
    public nonisolated let residency: MossResidencyPolicy
    /// One resident pipeline (DiT + VAE + adapters), keyed by what it loaded.
    private var resident: (paths: MossModelPaths, pipeline: MossPipeline)?
    private var enhancer: MossEnhancer?
    private let upscaler: MossUpscaler
    private var captioner: MossCaptioner?
    private var assistant: MossAssistant?
    /// The one Qwen3.5-9B the pipeline's encoder, the enhancer, the captioner and Help share.
    private var qwen: SharedQwen35?

    /// - Parameter turboLoRA: replaces the Turbo variant's akhaliq 4-step LoRA (e.g. lodestones `lora_latest` at 0.7).
    /// - Parameter overrides: Settings' VAE / text encoder / tokenizer files.
    /// - Parameter residency: `nil` = `.auto` for this Mac's recommended GPU working set.
    public init(
        models: URL, options: MossPipeline.Options = .init(), turboLoRA: LoRASpec? = nil, overrides: MossModelLayout.Overrides = .init(),
        residency: MossResidencyPolicy? = nil
    ) {
        let residency = residency ?? Self.residency(for: .auto)
        var options = options
        options.keepEncoderResident = options.keepEncoderResident || residency.keepEncoderResident
        self.models = models
        self.options = options
        self.residency = residency
        self.turboLoRA = turboLoRA
        self.overrides = overrides
        self.upscaler = MossUpscaler(models: models)
    }

    /// The policy a Settings choice means on this Mac.
    public static func residency(for mode: MossResidencyMode) -> MossResidencyPolicy {
        MossResidencyPolicy.resolve(mode, recommendedWorkingSet: Int(GPU.deviceInfo().maxRecommendedWorkingSetSize))
    }

    /// The Turbo variant's LoRA: `turboLoRA`, else akhaliq 4-step from the layout at its sidecar strength.
    private func turboVariantLoRA() throws -> LoRASpec {
        if let turboLoRA { return turboLoRA }
        guard let url = MossModelLayout.find(.turboLoRA, in: models) else {
            throw EngineError.inputNotFound(models.appendingPathComponent(MossVariant.turboLoRAPath))
        }
        return LoRASpec(url: url, strength: ModelSidecar.load(for: url, isLoRA: true).strength ?? 1)
    }

    /// The shared Qwen3.5 for this text-encoder file (Settings can point it elsewhere).
    private func sharedQwen(_ url: URL) -> SharedQwen35 {
        if let qwen, qwen.url == url { return qwen }
        qwen?.releaseAll()
        let q = SharedQwen35(url: url, dtype: options.encoderDType, keepTransformer: options.keepEncoderResident)
        qwen = q
        return q
    }

    /// The paths of the shared components (encoder, tokenizer) without needing a DiT.
    private func sharedPaths() throws -> MossModelPaths {
        try MossModelPaths.resolve(models: models, variant: .sft, overrides: overrides, requireDiT: false)
    }

    public func generate(_ request: GenerationRequest, progress: ProgressHandler?) async throws -> GenerationResult {
        if !residency.modelsCoexist { qwen?.trim() }  // lm_head / vision tower (and, saving memory, Qwen3.5) vs the DiT
        let (pipeline, request) = try route(request)
        let seed = request.seed ?? UInt64.random(in: 1...UInt64(UInt32.max))
        let image = try await pipeline.generate(request, seed: seed, progress: progress)
        return GenerationResult(imagePNG: try PixelImage.pngData(image), width: request.width, height: request.height, seed: seed)
    }

    public func enhancePrompt(_ request: PromptEnhanceRequest, progress: ProgressHandler?) async throws -> String {
        if enhancer == nil { let p = try sharedPaths(); enhancer = MossEnhancer(paths: p, qwen: sharedQwen(p.encoder)) }
        // One big model at a time on smaller Macs: Qwen3.5 (~18 GB bf16, shared) and a resident DiT.
        if !residency.modelsCoexist { resident?.pipeline.release() }
        defer { if !residency.modelsCoexist { enhancer?.release() } }
        return try await enhancer!.enhance(request, progress: progress)
    }

    public func caption(_ request: CaptionRequest, progress: ProgressHandler?) async throws -> MossCaption {
        if captioner == nil { let p = try sharedPaths(); captioner = MossCaptioner(paths: p, qwen: sharedQwen(p.encoder)) }
        // On smaller Macs Qwen3.5 + vision (~19 GB bf16) doesn't share the GPU with a resident DiT or
        // VOSR2. It stays loaded for the next image (dataset captioning) until a generation or idle unload.
        if !residency.modelsCoexist {
            resident?.pipeline.release()
            upscaler.release()
        }
        return try await captioner!.caption(request, progress: progress)
    }

    public func upscale(_ request: UpscaleRequest, progress: ProgressHandler?) async throws -> UpscaleResult {
        // VOSR2 is ~7 GB: it doesn't share the GPU with a resident DiT.
        if request.method == .vosr, !residency.modelsCoexist {
            resident?.pipeline.release()
            qwen?.releaseAll()
        }
        return try upscaler.upscale(request, progress: progress)
    }

    public func runModelTool(_ request: ModelToolRequest, progress: ProgressHandler?) async throws -> ModelToolResult {
        // A DiT's worth of fp32 maths doesn't share the GPU with resident models.
        await releaseResidentModels()
        let r: MossModelTools.Result
        var report = ""
        switch request.operation {
        case .convert:
            r = try MossModelTools.convert(input: request.input, precision: request.precision, output: request.output, progress: progress)
        case .fuse(let loras, let overrides):
            r = try MossModelTools.fuse(
                input: request.input, loras: loras, overrides: overrides, precision: request.precision, output: request.output, progress: progress)
        case .merge(let others, let mode, let alpha, let loras):
            r = try MossModelTools.merge(
                models: [request.input] + others, mode: mode, alpha: alpha, loras: loras, precision: request.precision,
                output: request.output, progress: progress)
        case .extractLoRA(let parent, let rank, let cutoff):
            let x = try MossModelTools.extractLoRA(
                input: request.input, parent: parent, rank: rank, cutoff: cutoff, precision: request.precision, output: request.output,
                progress: progress)
            (r, report) = (x.result, x.report)
        }
        return ModelToolResult(output: r.output, bytes: r.bytes, seconds: r.seconds, report: report)
    }

    public func trainLoRA(_ request: TrainingRequest, samples: [TrainingSample], progress: ProgressHandler?) async throws -> TrainingResult {
        // Training owns the GPU: drop resident models first.
        await releaseResidentModels()
        let paths = try MossModelPaths.resolve(models: models, variant: .base, overrides: overrides, dit: request.baseModel)
        return try await MossLoRATrainer.train(request, samples: samples, paths: paths, ditDType: options.ditDType, progress: progress)
    }

    public func releaseResidentModels() async {
        resident?.pipeline.release()
        resident = nil
        enhancer?.release()
        upscaler.release()
        captioner?.release()
        assistant?.release()
        qwen?.releaseAll()
    }

    /// Help → Ask / Feedback drafts. Qwen3.5 (shared) stays loaded for the next question; on smaller
    /// Macs a resident DiT, the captioner's vision tower and VOSR2 are released first.
    public func chat(_ request: ChatRequest, progress: ProgressHandler?, onText: TextStreamHandler?) async throws -> String {
        if assistant == nil { let p = try sharedPaths(); assistant = MossAssistant(paths: p, qwen: sharedQwen(p.encoder)) }
        if !residency.modelsCoexist {
            resident?.pipeline.release()
            qwen?.releaseTower()
            upscaler.release()
        }
        return try await assistant!.chat(request, progress: progress, onText: onText)
    }

    /// GPU state for Feedback → diagnostics.
    public func diagnostics() async -> [DiagnosticSection.Item] {
        let info = GPU.deviceInfo()
        let gb = { (b: Int) in String(format: "%.1f GB", Double(b) / 1_073_741_824) }
        let loaded = [
            resident.map { _ in "DiT pipeline" }, qwen?.isLoaded == true ? "Qwen3.5 (shared)" : nil,
        ].compactMap { $0 }
        return [
            .init("GPU", info.architecture),
            .init("Recommended working set", gb(Int(info.maxRecommendedWorkingSetSize))),
            .init("MLX memory in use", gb(Memory.activeMemory) + " active · " + gb(Memory.cacheMemory) + " cache · " + gb(Memory.peakMemory) + " peak"),
            .init("Loaded models", loaded.isEmpty ? "none" : loaded.joined(separator: ", ")),
            .init("Residency", residency.summary),
        ]
    }

    /// The pipeline a request needs and the request as it should run. A library DiT (`model`) replaces
    /// the variant's default; a distillation LoRA runs either through the data-space loop (its adapters
    /// go on the pipeline) or as an ordinary LoRA on the flow sampler. `.turbo` without one = SFT + the
    /// akhaliq 4-step LoRA (or `turboLoRA`) on flow Euler, at its sidecar strength.
    private func route(_ request: GenerationRequest) throws -> (MossPipeline, GenerationRequest) {
        var request = request
        let variant: MossVariant = request.variant == .turbo ? .sft : request.variant
        if request.variant == .turbo, request.distill == nil {
            request.loras.insert(try turboVariantLoRA(), at: 0)
        }
        var paths = try MossModelPaths.resolve(models: models, variant: variant, overrides: overrides, dit: request.model)
        if let d = request.distill {
            switch d.route {
            case .dataSpace:
                // A Turbo LoRA brings its time embedding; a built-in one has it fused (no LoRA, no embedding).
                guard let ds = d.dataSpace, d.lora == nil || d.timeEmbedding != nil else {
                    throw EngineError.invalidRequest("the data-space Turbo needs its dataspace (+ time_embedding) files")
                }
                paths.turbo = .init(lora: d.lora?.url, loraStrength: Float(d.lora?.strength ?? 1), timeEmbedding: d.timeEmbedding, dataSpace: ds)
            case .flow:
                if let lora = d.lora { request.loras.insert(lora, at: 0) }
            }
        }
        if request.editRunsOnFlow, let turbo = paths.turbo {
            // Edit references don't survive the data-space loop: its LoRA alone on flow Euler (Edit-HQ's
            // stack). A fused Turbo has the loop's time embedding baked in, so it can't leave the loop.
            guard turbo.lora != nil else {
                throw EngineError.invalidRequest(
                    "edits can't run on a model with a built-in Turbo — pick the SFT model with a Turbo LoRA (or none)")
            }
            if let lora = turbo.lora { request.loras.insert(LoRASpec(url: lora, strength: Double(turbo.loraStrength)), at: 0) }
            paths.turbo = nil
            request.moveTurboToFlow()
        }
        if let resident, resident.paths == paths { return (resident.pipeline, request) }
        // One resident DiT at a time: switching models drops the previous one (and, on smaller Macs,
        // VOSR2 / the captioner).
        resident?.pipeline.release()
        if !residency.modelsCoexist {
            upscaler.release()
            qwen?.releaseTower()
        }
        let p = MossPipeline(paths: paths, options: options, qwen: sharedQwen(paths.encoder))
        resident = (paths, p)
        return (p, request)
    }
}
