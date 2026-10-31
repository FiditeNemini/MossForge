import Foundation
import MLX
import MLXRandom
import MossForgeCore
import MossForgeEngine
import MossForgeMLX

// `moss-mlx` — grows a subcommand per porting step (load-parity, vae-parity, encoder-parity,
// dit-parity, generate, serve).

/// Inside MossForge.app, MLX finds its metallib through the main bundle, which Foundation derives from
/// the path we were launched by. Via a symlink (`/usr/local/bin/moss-mlx`) that path is outside the
/// app, so re-exec through the real one before touching MLX.
func reexecThroughRealPathIfSymlinked() {
    var size: UInt32 = 0
    _NSGetExecutablePath(nil, &size)
    var buffer = [CChar](repeating: 0, count: Int(size))
    guard _NSGetExecutablePath(&buffer, &size) == 0, let real = realpath(buffer, nil) else { return }
    defer { free(real) }
    let launched = String(cString: buffer), resolved = String(cString: real)
    guard resolved != launched, resolved.contains(".app/Contents/MacOS/") else { return }
    var argv: [UnsafeMutablePointer<CChar>?] = CommandLine.arguments.map { strdup($0) } + [nil]
    execv(resolved, &argv)  // only returns on failure — then carry on as we are
}
reexecThroughRealPathIfSymlinked()

let args = Array(CommandLine.arguments.dropFirst())

/// `--name value` from the remaining arguments.
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

/// Positional arguments after the subcommand (options and their values removed).
func positionals() -> [String] {
    var out: [String] = []
    var skip = false
    for a in args.dropFirst() {
        if skip { skip = false; continue }
        if a.hasPrefix("--") { skip = true; continue }
        out.append(a)
    }
    return out
}

let usage = """
    moss-mlx generate --prompt "…" [--negative "…"] [--size WxH] [--steps 50] [--cfg 3] [--shift 5] [--seed N]
                      [--slg official|off|standard:L1,L2[@scale]] [--variant sft|base] [--models ../models]
                      [--encoder-dtype bf16|f32] [--dit-dtype f16|bf16] [--lora file@strength[:blocks]]… [--out moss.png]
                      blocks: "21-41" (only those) or "0-20=0,21-41=0.5,other=0"
                      --variant turbo: lodestones 8-step data-space DMD (cfg 1; --steps 1…8)
                      [--turbo-lora file@strength] [--no-turbo-overrides]
                      [--reference source.png …]  for edit LoRAs (≤ 4)
                      [--edit-layout auto|aiToolkit|aiToolkitNoCache|mossImageEdit|mossForge]
                      [--edit original.png --edit-box y0,x0,y1,x1 …]  redraw only those boxes (same seed = closest)
                      [--edit original.png --edit-mask mask.png]  …or where the mask is white (with --reference:
                                         a masked edit-LoRA edit)
                      [--model dit.safetensors]  a library DiT instead of the variant's default
                      [--distill lora.safetensors]  a few-step LoRA, run as its .mfmeta.json sidecar (or name) says
    moss-mlx convert <dit.safetensors> --precision fp32|bf16|fp16|int8-convrot|fp8|fp8-e5m2|int4-convrot|nvfp4 [--out f]
    moss-mlx fuse <dit.safetensors> --lora file@strength[:blocks]… [--overrides time_embedding.safetensors]
                  [--precision bf16] [--out f]   (fp32 maths; source must be bf16/fp16/fp32)
    moss-mlx merge <A> <B> [C] [--mode weighted|add-difference] [--alpha 0.5] [--lora …] [--precision bf16] [--out f]
    moss-mlx quant-selftest   (encode → write → load round trip per export precision)
    moss-mlx extract-lora <finetune.safetensors> [--parent dit] [--rank 64] [--cutoff fixed|energy:0.9|threshold:0.02]
                          [--precision bf16|fp16|fp32] [--out f] [--models dir]   (parent: the official model of its family)
    moss-mlx extract-selftest  (known low-rank diff → extract → fuse back → reconstruction error)
    moss-mlx train <dataset folder> [--model ../models/diffusion_models/moss_image_base_bf16.safetensors] [--out lora.safetensors]
                   [--rank 16] [--alpha 16] [--lr 1e-4] [--steps 1000] [--res 1024] [--targets attention|attention-ffn]
                   [--save-every 250] [--trigger word] [--seed N] [--models ../models]   (images + name.txt captions)
      slider: moss-mlx train --positive "old" --negative "young" --target "a portrait of a person" [--range -1:1]
              [--eta 1] [--latent-steps 10] [--preserve "male, female"] --out lora/trained/age.safetensors
      edit (experimental): moss-mlx train <folder: targets + name.txt instructions, sources in control/> --edit-pairs
      [--lokr [--lokr-factor 8]] [--caption-dropout 0.05] [--no-checkpointing]
      [--sample "test prompt" ×≤3] [--sample-steps 20] [--sample-res 512]   test images at each checkpoint
    moss-mlx train-smoke      (synthetic LoRA fit: the loss must collapse)
    moss-mlx enhance --prompt "…" [--layout] [--size WxH] [--models ../models]
    moss-mlx upscale <in.png> [--scale 2] [--method vosr|esrgan] [--esrgan-model file.safetensors] [--out out.png]
                     [--seed 42] [--tile 512] [--overlap 256] [--soften off|light|medium|strong]
                     [--color wavelet|adain|none] [--batch 4] [--models ../models]
    moss-mlx caption <image> [--detail long|medium|short] [--reference "…"] [--max-pixels 1048576] [--seed 0] [--json]
    moss-mlx ask "<question>" [--models dir]   Help → Ask: answer from the bundled docs with Qwen3.5
                     [--models ../models]
    moss-mlx serve [--models ../models] [--port 7860] [--lan] [--keep-encoder] [--residency auto|keepLoaded|saveMemory] [--idle-unload minutes(10; 0 = never)]
    moss-mlx smoke
    moss-mlx load <file.safetensors> [--prefix model.language_model.]
    moss-mlx load-parity <reference.safetensors> <candidate.safetensors> [--prefix p] [--match s] [--min-cosine 0.98]
    moss-mlx vae-parity <flux2-vae.safetensors> <fixtureDir>
    moss-mlx encoder-parity <qwen3.5.safetensors> <tokenizerDir> <fixtureDir> [--dtype f32|bf16|both]
    moss-mlx encoder-diverge <qwen3.5.safetensors> <fixture.safetensors>   (bf16 vs f32 per layer)
    moss-mlx dit-parity <moss_image.safetensors> <fixtureDir> [--dtype f32|bf16|f16|both|all]
    moss-mlx dit-bench <dit> [--size 1024] [--repeats 3] [--dtype f16|bf16] [--lora file]
    moss-mlx dit-denoise <dit> <vae> <qwen3.5> <tokenizerDir> <image.png> --prompt "…" [--off "…"] [--out x0.png]
    moss-mlx e2e-parity <modelsDir> <fixtureDir> [--save-images dir]  (full pipeline vs Scripts/e2e_ref.py, fixed noise)
    moss-mlx generate-parity <modelsDir> <fixtureDir>   (Qwen3.5 greedy decode vs Scripts/generate_ref.py)
    moss-mlx caption-parity <modelsDir> <fixtureDir>    (vision tower + multimodal decode vs Scripts/caption_ref.py)
    moss-mlx vae-roundtrip <flux2-vae.safetensors> <image.png> [--out out.png] [--dtype bf16]
    """

do {
    switch args.first {
    case "generate":
        guard let prompt = option("--prompt") else { print(usage); exit(2) }
        var request = GenerationRequest(
            prompt: prompt, negativePrompt: option("--negative") ?? MossNegativePrompt.recommended,
            variant: MossVariant(rawValue: option("--variant") ?? "sft") ?? .sft,
            steps: option("--steps").flatMap(Int.init), cfg: option("--cfg").flatMap(Double.init),
            shift: option("--shift").flatMap(Double.init), seed: option("--seed").flatMap(UInt64.init))
        if let size = option("--size") {
            let wh = size.split(separator: "x").compactMap { Int($0) }
            guard wh.count == 2 else { print(usage); exit(2) }
            (request.width, request.height) = (wh[0], wh[1])
        }
        switch option("--slg") ?? "official" {
        case "official": request.slg = .official
        case "off": request.slg = .off
        case let spec where spec.hasPrefix("standard:"):
            let body = spec.dropFirst("standard:".count).split(separator: "@")
            request.slg = .standard(
                layers: body[0].split(separator: ",").compactMap { Int($0) },
                scale: body.count > 1 ? Double(body[1]) ?? 3 : 3)
        default: print(usage); exit(2)
        }
        func loraSpec(_ s: String) -> LoRASpec {
            guard let spec = LoRASpec(parsing: s) else {
                FileHandle.standardError.write(Data("bad LoRA spec '\(s)' (path[@strength[:blocks]])\n".utf8))
                exit(2)
            }
            return spec
        }
        // Repeatable: --lora a.safetensors@0.8 --lora kijai.safetensors@-0.5:21-41
        request.loras = args.indices.filter { args[$0] == "--lora" && $0 + 1 < args.count }.map { loraSpec(args[$0 + 1]) }
        // Edit LoRAs: --reference source.png (repeatable, ≤ 4) — clean reference tokens on RoPE r = 1…;
        // --edit-layout ai-toolkit | Moss-Image-Edit | … (default: from the LoRA's metadata).
        request.references = args.indices.filter { args[$0] == "--reference" && $0 + 1 < args.count }.map { URL(fileURLWithPath: args[$0 + 1]) }
        if let name = option("--edit-layout"), name.lowercased() != "auto" {
            guard let layout = MossEditLayout(name: name) else {
                FileHandle.standardError.write(Data("unknown --edit-layout '\(name)' (\(MossEditLayout.allCases.map(\.rawValue).joined(separator: " | ")))\n".utf8))
                exit(2)
            }
            request.editLayout = layout
        }
        // Edit: --edit original.png --edit-box y0,x0,y1,x1 (abs_centralized px, repeatable) — redraw only those boxes.
        if let source = option("--edit") {
            let boxes = args.indices.filter { args[$0] == "--edit-box" && $0 + 1 < args.count }
                .map { args[$0 + 1].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) } }
            let mask: EditMask
            if let file = option("--edit-mask") {
                // White = change: luminance area-averaged onto the token grid.
                let img = try RGB8.load(url: URL(fileURLWithPath: file))
                let lum = stride(from: 0, to: img.pixels.count, by: 3).map { Float(img.pixels[$0]) / 255 }
                mask = EditMask.resampled(from: lum, width: img.width, height: img.height, toWidth: request.width / 16, toHeight: request.height / 16)
            } else {
                mask = LayoutEditMask.tokenMask(boxes: boxes, canvas: Dimensions(width: request.width, height: request.height))
            }
            request.edit = ImageEditSpec(source: URL(fileURLWithPath: source), mask: mask)
        }
        if let model = option("--model") {
            let url = URL(fileURLWithPath: model)
            request.model = url
            // A Turbo fused into the model runs at its recipe (unless --distill picks another).
            if option("--distill") == nil, let info = ModelSidecar.read(for: url)?.distill, let spec = DistillSpec(builtInto: url, info: info) {
                let explicitSteps = option("--steps").flatMap(Int.init)
                request.apply(spec)
                if let explicitSteps { request.steps = explicitSteps }
            }
        }
        if let path = option("--distill") {
            let url = URL(fileURLWithPath: path)
            // The sidecar's strength, like Generate (lodestones lora_latest runs at 0.7).
            let sidecar = ModelSidecar.load(for: url, isLoRA: true)
            guard let info = sidecar.distill, let spec = DistillSpec(lora: url, info: info, strength: sidecar.strength ?? 1) else {
                FileHandle.standardError.write(Data("\(url.lastPathComponent) has no usable distillation recipe (sidecar)\n".utf8))
                exit(2)
            }
            let explicitSteps = option("--steps").flatMap(Int.init)
            request.apply(spec)
            if let explicitSteps { request.steps = explicitSteps }
        }
        var options = MossPipeline.Options()
        if option("--encoder-dtype") == "f32" { options.encoderDType = .float32 }
        if option("--dit-dtype") == "bf16" { options.ditDType = .bfloat16 }
        if args.contains("--no-turbo-overrides") { options.turboOverrides = false }
        let models = URL(fileURLWithPath: option("--models") ?? "../models")
        let turboLoRA = option("--turbo-lora").map(loraSpec)
        let engine = MossEngine(backend: MossBackend(models: models, options: options, turboLoRA: turboLoRA))  // the one-line injection
        let start = Date()
        let result = try await engine.generate(request) { p in
            FileHandle.standardError.write(Data("\r\u{1B}[K[\(p.step)/\(p.totalSteps)] \(p.message)".utf8))
        }
        let out = URL(fileURLWithPath: option("--out") ?? "moss.png")
        try result.imagePNG.write(to: out)
        print(String(format: "\nwrote %@ (%@, %d×%d, seed %llu, %d steps, cfg %.1f) in %.1fs", out.path, request.variant.rawValue,
            result.width, result.height, result.seed, request.steps, request.cfg, Date().timeIntervalSince(start)))
    case "enhance":
        guard let prompt = option("--prompt") else { print(usage); exit(2) }
        var canvas = Dimensions(width: 1024, height: 1024)
        if let size = option("--size") {
            let wh = size.split(separator: "x").compactMap { Int($0) }
            guard wh.count == 2 else { print(usage); exit(2) }
            canvas = Dimensions(width: wh[0], height: wh[1])
        }
        let engine = MossEngine(backend: MossBackend(models: URL(fileURLWithPath: option("--models") ?? "../models")))
        let start = Date()
        let text = try await engine.enhancePrompt(PromptEnhanceRequest(prompt: prompt, layout: args.contains("--layout"), canvas: canvas)) { p in
            FileHandle.standardError.write(Data("\r\u{1B}[K\(p.message)".utf8))
        }
        FileHandle.standardError.write(Data(String(format: "\r\u{1B}[Kenhanced in %.1fs\n", Date().timeIntervalSince(start)).utf8))
        print(text)
    case "train-smoke":
        let (first, last) = MossLoRATrainer.smoke()
        print(String(format: "loss %.5f → %.7f %@", first, last, last < first * 1e-3 ? "✓" : "✗"))
        let kron = MossLoRATrainer.kroneckerSmoke()
        print(String(format: "LoKr apply vs dense: max |Δ| %.2e %@", kron, kron < 1e-4 ? "✓" : "✗"))

    case "train":
        let isSlider = option("--positive") != nil
        guard let folder = positionals().first ?? (isSlider ? "slider" : nil) else { print(usage); exit(2) }
        let models = URL(fileURLWithPath: option("--models") ?? "../models")
        let dataset = URL(fileURLWithPath: folder)
        let isEdit = args.contains("--edit-pairs")
        let samples = isSlider ? []
            : isEdit ? EditDatasetScanner.scan(directory: dataset).pairs.map { TrainingSample(image: $0.target, caption: $0.instruction, control: $0.control) }
            : DatasetScanner.scan(directory: dataset)
        let base = option("--model").map { URL(fileURLWithPath: $0) } ?? MossModelLayout.find(.baseDiT, in: models)!
        let out = option("--out").map { URL(fileURLWithPath: $0) }
            ?? models.appendingPathComponent("lora/trained/\(dataset.lastPathComponent).safetensors")
        let res = option("--res").flatMap(Int.init) ?? 1024
        let request = TrainingRequest(
            baseModel: base, dataset: dataset, output: out, rank: option("--rank").flatMap(Int.init) ?? 16,
            alpha: option("--alpha").flatMap(Double.init) ?? 16, learningRate: option("--lr").flatMap(Double.init) ?? 1e-4,
            steps: option("--steps").flatMap(Int.init) ?? 1000, pixels: res * res,
            targets: option("--targets") == "attention-ffn" ? .attentionAndFFN : .attention,
            saveEvery: option("--save-every").flatMap(Int.init) ?? 250, trigger: option("--trigger") ?? "",
            seed: option("--seed").flatMap(UInt64.init),
            family: ModelSidecar.load(for: base, isLoRA: false).families.first ?? .base)
        var trainRequest = request
        // --sample "prompt" (repeatable, up to 3): test images before training, at checkpoints and at the end.
        trainRequest.samplePrompts = args.indices.filter { args[$0] == "--sample" && $0 + 1 < args.count }.map { args[$0 + 1] }
        if let n = option("--sample-steps").flatMap(Int.init) { trainRequest.sampleSteps = n }
        if let r = option("--sample-res").flatMap(Int.init) { trainRequest.samplePixels = r * r }
        if args.contains("--lokr") {
            trainRequest.factorization = .lokr
            trainRequest.lokrFactor = option("--lokr-factor").flatMap(Int.init) ?? -1
        }
        if let p = option("--caption-dropout").flatMap(Double.init) { trainRequest.captionDropout = p }
        trainRequest.gradientCheckpointing = !args.contains("--no-checkpointing")
        if isEdit { trainRequest.mode = .edit }
        if isSlider {
            // --positive "old" --negative "young" [--target "a portrait of a person"] [--range -1:1] [--eta 1]
            // [--latent-steps 10] [--preserve "male, female"] — an image-free Concept Slider.
            trainRequest.mode = .slider
            trainRequest.sliderPositive = option("--positive") ?? ""
            trainRequest.sliderNegative = option("--negative") ?? ""
            trainRequest.sliderTarget = option("--target") ?? ""
            if let range = option("--range")?.split(separator: ":").compactMap({ Double($0) }), range.count == 2 {
                trainRequest.sliderRange = SliderRange(min: range[0], max: range[1])
            }
            if let eta = option("--eta").flatMap(Double.init) { trainRequest.sliderEta = eta }
            if let n = option("--latent-steps").flatMap(Int.init) { trainRequest.sliderSampleSteps = n }
            trainRequest.sliderPreserve = SliderPromptSet.attributes(from: option("--preserve") ?? "")
        }
        let engine = MossEngine(backend: MossBackend(models: models))  // the one-line injection
        print("training on \(samples.count) images → \(out.path)")
        let r = try await engine.trainLoRA(trainRequest, samples: samples) { p in
            print("\r\u{1B}[K\(p.message)", terminator: "")
            for f in p.files { print("\n  sample: \(f.path)") }
            fflush(stdout)
        }
        print(String(format: "\npeak MLX memory %.1f GB", Double(Memory.peakMemory) / 1_073_741_824))
        print(String(format: "wrote %@ after %d steps in %.0f s (loss %.4f → %.4f)", r.output.path, r.steps, r.seconds,
            r.losses.prefix(20).reduce(0, +) / Double(min(20, r.losses.count)), r.losses.suffix(20).reduce(0, +) / Double(min(20, r.losses.count))))

    case "quant-selftest":
        for (p, c) in MossModelTools.selfTest() { print(String(format: "%-14@ corr %.6f", p.rawValue as NSString, c)) }

    case "extract-selftest":
        let r = try MossModelTools.extractSelfTest()
        print(String(format: "max relative error %.2e, extracted ranks %@ (true rank 6), fixed-cutoff rank %d", r.maxRelativeError,
            r.ranks.map(String.init).joined(separator: ",") as NSString, r.fixedRank))
        if r.maxRelativeError > 1e-4 || r.ranks.contains(where: { $0 != 6 }) || r.fixedRank != 8 { print("FAIL"); exit(1) }
        print("PASS")

    case "extract-lora":
        guard let source = positionals().first.map({ URL(fileURLWithPath: $0) }) else { print(usage); exit(2) }
        let models = URL(fileURLWithPath: option("--models") ?? "../models")
        let parent: URL
        if let p = option("--parent") {
            parent = URL(fileURLWithPath: p)
        } else {
            // Same pick as the app: the official model of the source's family.
            let family = ModelSidecar.load(for: source, isLoRA: false).families.first ?? .sft
            guard let found = MossLoRAExtraction.defaultParent(for: source, family: family, among: ModelLibrary.scan(models: models).dits) else {
                FileHandle.standardError.write(Data("no \(family.displayName) parent DiT in \(models.path); pass --parent\n".utf8))
                exit(2)
            }
            parent = found.url
        }
        let rank = option("--rank").flatMap(Int.init) ?? 64
        guard let cutoff = LoRARankCutoff(parsing: option("--cutoff") ?? "fixed"),
            let precision = WeightPrecision(rawValue: option("--precision") ?? "bf16")
        else { print(usage); exit(2) }
        let out = option("--out").map { URL(fileURLWithPath: $0) }
            ?? MossLoRAExtraction.outputURL(
                stem: MossLoRAExtraction.outputStem(source: source.deletingPathExtension().lastPathComponent, rank: rank), models: models)
        print("extracting \(source.lastPathComponent) − \(parent.lastPathComponent) at rank \(rank) (\(cutoff))")
        let engine = MossEngine(backend: MossBackend(models: models))
        let r = try await engine.runModelTool(
            ModelToolRequest(input: source, operation: .extractLoRA(parent: parent, rank: rank, cutoff: cutoff), precision: precision, output: out),
            progress: { p in print("  \(p.message)") })
        print(r.report)
        print(String(format: "wrote %@ (%.0f MB) in %.0f s", r.output.path, Double(r.bytes) / 1e6, r.seconds))

    case "convert", "fuse", "merge":
        let cmd = args.first!
        let inputs = positionals().map { URL(fileURLWithPath: $0) }
        guard let first = inputs.first else { print(usage); exit(2) }
        let precision = WeightPrecision(rawValue: option("--precision") ?? (cmd == "convert" ? "" : "bf16"))
        guard let precision, MossExport.precisions.contains(precision) else {
            FileHandle.standardError.write(Data("--precision: one of \(MossExport.precisions.map(\.rawValue).joined(separator: ", "))\n".utf8))
            exit(2)
        }
        let loras = args.indices.filter { args[$0] == "--lora" && $0 + 1 < args.count }.compactMap { LoRASpec(parsing: args[$0 + 1]) }
        let suffix = cmd == "convert" ? nil : cmd == "merge" ? "merge" : loras.map { $0.url.deletingPathExtension().lastPathComponent }.joined(separator: "+")
        let out = option("--out").map { URL(fileURLWithPath: $0) }
            ?? first.deletingLastPathComponent().appendingPathComponent(
                MossExport.outputStem(first.deletingPathExtension().lastPathComponent, precision: precision, suffix: suffix) + ".safetensors")
        let report: ProgressHandler = { p in print("  \(p.message)") }
        // Through MossEngine, like the app: it also writes the output's sidecar (a fused Turbo becomes a
        // built-in recipe) and copies a data-space Turbo's `dataspace` file beside it.
        var request: ModelToolRequest
        switch cmd {
        case "convert":
            request = ModelToolRequest(input: first, operation: .convert, precision: precision, output: out)
        case "fuse":
            // A data-space Turbo's time embedding comes from its sidecar unless --overrides names one.
            request = .fuse(dit: first, loras: loras, precision: precision, output: out)
            if let overrides = option("--overrides").map({ URL(fileURLWithPath: $0) }) {
                request.operation = .fuse(loras: loras, overrides: overrides)
            }
        default:
            let mode: ModelMergeMode = option("--mode") == "add-difference" ? .addDifference : .weightedSum
            request = ModelToolRequest(
                input: first, operation: .merge(others: Array(inputs.dropFirst()), mode: mode,
                    alpha: option("--alpha").flatMap(Double.init) ?? 0.5, loras: loras),
                precision: precision, output: out)
        }
        let engine = MossEngine(backend: MossBackend(models: URL(fileURLWithPath: option("--models") ?? "../models")))
        let r = try await engine.runModelTool(request, progress: report)
        print(String(format: "wrote %@ (%.2f GB) in %.0f s", r.output.path, Double(r.bytes) / 1e9, r.seconds))

    case "upscale":
        // VOSR2 (generative, default) or an ESRGAN/SPAN model from models/esrgan, through MossEngine.
        guard let input = positionals().first, let scale = Double(option("--scale") ?? "2"),
            let method = UpscaleMethod(rawValue: option("--method") ?? "vosr")
        else { print(usage); exit(2) }
        let d = UpscaleSettings()
        let request = UpscaleRequest(
            image: URL(fileURLWithPath: input), scale: scale, method: method, esrganModel: option("--esrgan-model"),
            settings: UpscaleSettings(
                seed: option("--seed").flatMap(UInt64.init) ?? d.seed, tile: option("--tile").flatMap(Int.init) ?? d.tile,
                overlap: option("--overlap").flatMap(Int.init) ?? d.overlap,
                soften: option("--soften").flatMap(UpscaleSoftening.init(rawValue:)) ?? d.soften,
                colorFix: option("--color").flatMap(UpscaleColorFix.init(rawValue:)) ?? d.colorFix,
                batch: option("--batch").flatMap(Int.init) ?? d.batch),
            output: option("--out").map { URL(fileURLWithPath: $0) })
        let engine = MossEngine(backend: MossBackend(models: URL(fileURLWithPath: option("--models") ?? "../models")))
        let start = Date()
        let result = try await engine.upscale(request) { p in FileHandle.standardError.write(Data("\r\u{1B}[K\(p.message)".utf8)) }
        print(String(format: "\nwrote %@ (%@, %d×%d) in %.1fs", result.output.path, method.rawValue,
            result.width, result.height, Date().timeIntervalSince(start)))
    case "caption":
        // Qwen3.5-9B vision + LM with the official recaption prompt, through MossEngine.
        guard let image = positionals().first, let detail = CaptionDetail(rawValue: option("--detail") ?? "long") else {
            print(usage); exit(2)
        }
        let request = CaptionRequest(
            image: URL(fileURLWithPath: image), detail: detail, reference: option("--reference"),
            maxPixels: option("--max-pixels").flatMap(Int.init) ?? CaptionRequest.defaultMaxPixels,
            seed: option("--seed").flatMap(UInt64.init) ?? 0)
        let engine = MossEngine(backend: MossBackend(models: URL(fileURLWithPath: option("--models") ?? "../models")))
        let start = Date()
        let caption = try await engine.caption(request) { p in FileHandle.standardError.write(Data("\r\u{1B}[K\(p.message)".utf8)) }
        FileHandle.standardError.write(Data(String(format: "\r\u{1B}[Kcaptioned in %.1fs\n", Date().timeIntervalSince(start)).utf8))
        if args.contains("--json") {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
            print(String(decoding: try encoder.encode(caption), as: UTF8.self))
        } else {
            print(caption.text(detail))
        }
    case "ask":
        // Help → Ask from the terminal: retrieve doc sections, answer with Qwen3.5 (streams).
        let question = positionals().joined(separator: " ")
        guard !question.isEmpty else { print(usage); exit(2) }
        let excerpts = HelpRetriever(chunks: HelpCorpus.chunks).search(question)
        let engine = MossEngine(backend: MossBackend(models: URL(fileURLWithPath: option("--models") ?? "../models")))
        let start = Date()
        let printed = LockedCounter()
        _ = try await engine.chat(
            ChatRequest(messages: HelpAssistantPrompt.messages(product: .mossForge, question: question, history: [], excerpts: excerpts, appState: nil))
        ) { p in
            if printed.value == 0 { FileHandle.standardError.write(Data("\r\u{1B}[K\(p.message)".utf8)) }
        } onText: { text in
            let n = printed.value
            if n == 0 { FileHandle.standardError.write(Data("\r\u{1B}[K".utf8)) }
            let chars = Array(text)
            if chars.count > n { print(String(chars[n...]), terminator: ""); fflush(stdout); printed.value = chars.count }
        }
        print("")
        for (i, c) in excerpts.enumerated() { FileHandle.standardError.write(Data("[\(i + 1)] \(c.title)\n".utf8)) }
        FileHandle.standardError.write(Data(String(format: "answered in %.1fs\n", Date().timeIntervalSince(start)).utf8))
    case "serve":
        // The resident server (Phase 3) on the MLX backend: /health, /v1/status, /v1/models,
        // /v1/images/generations, /v1/moss/{generate,jobs,enhance,upscale,caption,edit,options,loras}. Loopback-only unless
        // --lan; one job at a time (InferenceQueue); models unload after --idle-unload minutes idle.
        guard let port = UInt16(option("--port") ?? "7860"), let idle = Double(option("--idle-unload") ?? "10"), idle >= 0 else {
            print(usage); exit(2)
        }
        let models = URL(fileURLWithPath: option("--models") ?? "../models")
        var serveOptions = MossPipeline.Options()
        serveOptions.keepEncoderResident = args.contains("--keep-encoder")  // +16 GB, skips the per-request encoder load
        guard let mode = MossResidencyMode(rawValue: option("--residency") ?? "auto") else { print(usage); exit(2) }
        let residency = MossBackend.residency(for: mode)
        let engine = MossEngine(backend: MossBackend(models: models, options: serveOptions, residency: residency))  // the one-line injection
        let queue = InferenceQueue()
        let server = MossHTTPServer(router: MossServerRouter(engine: engine, modelsDirectory: models, queue: queue))
        setvbuf(stdout, nil, _IOLBF, 0)  // one line per request, even when piped to a log
        server.onLog = { print($0) }
        if idle > 0 {
            await queue.setIdleRelease(after: .seconds(idle * 60)) {
                await engine.releaseResidentModels()
                print("idle \(idle.formatted()) min — models unloaded (reload on next request)")
            }
        }
        let bound = try await server.start(port: port, policy: NetworkPolicy(egressAllowed: args.contains("--lan")))
        print("residency \(mode.rawValue): \(residency.summary)")
        print("moss-mlx serve · models \(models.path) · http://\(NetworkPolicy(egressAllowed: args.contains("--lan")).serverBindHost):\(bound)")
        while true { try await Task.sleep(for: .seconds(3600)) }
    case "smoke":
        print(MLXSmoke.run())
    case "load":
        guard let path = positionals().first else { print(usage); exit(2) }
        let (summary, passed) = try LoadCheck.run(url: URL(fileURLWithPath: path), prefix: option("--prefix") ?? "")
        print(summary)
        exit(passed ? 0 : 1)
    case "load-parity":
        let files = positionals()
        guard files.count == 2 else { print(usage); exit(2) }
        let report = try LoadParity.run(
            reference: URL(fileURLWithPath: files[0]), candidate: URL(fileURLWithPath: files[1]),
            prefix: option("--prefix") ?? "", match: option("--match"),
            minCosine: option("--min-cosine").flatMap(Float.init) ?? 0.98)
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "vae-parity":
        let paths = positionals()
        guard paths.count == 2 else { print(usage); exit(2) }
        let report = try VAEParity.run(vae: URL(fileURLWithPath: paths[0]), fixtures: URL(fileURLWithPath: paths[1]))
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "encoder-parity":
        let paths = positionals()
        guard paths.count == 3 else { print(usage); exit(2) }
        let dtypes: [DType] = switch option("--dtype") {
        case "f32": [.float32]
        case "bf16": [.bfloat16]
        default: [.float32, .bfloat16]
        }
        let report = try await EncoderParity.run(
            encoder: URL(fileURLWithPath: paths[0]), tokenizer: URL(fileURLWithPath: paths[1]),
            fixtures: URL(fileURLWithPath: paths[2]), dtypes: dtypes)
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "encoder-diverge":
        let paths = positionals()
        guard paths.count == 2 else { print(usage); exit(2) }
        try EncoderParity.layerDivergence(encoder: URL(fileURLWithPath: paths[0]), fixture: URL(fileURLWithPath: paths[1])).forEach { print($0) }
    case "dit-bench":
        let p = positionals()
        guard p.count == 1 else { print(usage); exit(2) }
        let lines = try DiTBench.run(
            dit: URL(fileURLWithPath: p[0]), size: option("--size").flatMap(Int.init) ?? 1024,
            repeats: option("--repeats").flatMap(Int.init) ?? 3, dtype: option("--dtype") == "bf16" ? .bfloat16 : .float16,
            lora: option("--lora").map { URL(fileURLWithPath: $0) })
        lines.forEach { print($0) }
    case "dit-parity":
        let paths = positionals()
        guard paths.count == 2 else { print(usage); exit(2) }
        let dtypes: [DType] = switch option("--dtype") {
        case "f32": [.float32]
        case "bf16": [.bfloat16]
        case "f16": [.float16]
        case "all": [.float32, .bfloat16, .float16]
        default: [.float32, .bfloat16]
        }
        let report = try DiTParity.run(dit: URL(fileURLWithPath: paths[0]), fixtures: URL(fileURLWithPath: paths[1]), dtypes: dtypes)
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "dit-denoise":
        let p = positionals()
        guard p.count == 5, let prompt = option("--prompt") else { print(usage); exit(2) }
        let report = try await DiTDenoise.run(
            dit: URL(fileURLWithPath: p[0]), vae: URL(fileURLWithPath: p[1]), encoder: URL(fileURLWithPath: p[2]),
            tokenizer: URL(fileURLWithPath: p[3]), image: URL(fileURLWithPath: p[4]), prompt: prompt,
            offPrompt: option("--off") ?? "A bowl of ramen on a wooden table, top-down food photography.",
            sigmas: option("--sigmas").map { $0.split(separator: ",").compactMap { Double($0) } } ?? [0.3, 0.6, 0.9],
            out: option("--out").map { URL(fileURLWithPath: $0) })
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "e2e-parity":
        let p = positionals()
        guard p.count == 2 else { print(usage); exit(2) }
        let report = try await E2EParity.run(
            models: URL(fileURLWithPath: p[0]), fixtures: URL(fileURLWithPath: p[1]),
            saveImages: option("--save-images").map { URL(fileURLWithPath: $0) })
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "generate-parity":
        let p = positionals()
        guard p.count == 2 else { print(usage); exit(2) }
        let report = try await GenerateParity.run(models: URL(fileURLWithPath: p[0]), fixtures: URL(fileURLWithPath: p[1]))
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "caption-parity":
        let p = positionals()
        guard p.count == 2 else { print(usage); exit(2) }
        let report = try await CaptionParity.run(models: URL(fileURLWithPath: p[0]), fixtures: URL(fileURLWithPath: p[1]))
        report.lines.forEach { print($0) }
        exit(report.passed ? 0 : 1)
    case "vae-roundtrip":
        let paths = positionals()
        guard paths.count == 2 else { print(usage); exit(2) }
        let vae = try FluxVAE.load(url: URL(fileURLWithPath: paths[0]), dtype: option("--dtype") == "bf16" ? .bfloat16 : .float32)
        let image = try PixelImage.load(URL(fileURLWithPath: paths[1]))
        let start = Date()
        let tokens = vae.encode(image)
        let decoded = vae.decode(tokens)
        eval(decoded)
        let elapsed = Date().timeIntervalSince(start)
        print("image \(image.shape) → tokens \(tokens.shape)")
        print(String(format: "tokens mean %.4f std %.4f; round-trip PSNR %.2f dB in %.2fs",
            tokens.mean().item(Float.self), sqrt(tokens.variance()).item(Float.self), PixelImage.psnr(decoded, image), elapsed))
        if let out = option("--out") { try PixelImage.save(decoded, to: URL(fileURLWithPath: out)); print("wrote \(out)") }
    default:
        print(usage)
    }
} catch {
    FileHandle.standardError.write(Data("moss-mlx: \(error)\n".utf8))
    exit(1)
}

/// A counter the streaming callback can update from another task.
final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var v = 0
    var value: Int {
        get { lock.withLock { v } }
        set { lock.withLock { v = newValue } }
    }
}
