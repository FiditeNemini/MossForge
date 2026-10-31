import AppKit
import Foundation
import MossForgeCore
import MossForgeEngine
import MossForgeMLX
import Observation

/// Persisted app settings (UserDefaults, JSON).
struct AppSettings: Codable, Equatable {
    var theme: ThemeID = .native
    var modelsDirectory: URL = AppSettings.defaultModelsDirectory
    var outputDirectory: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Pictures/MossForge")
    var egressEnabled = false
    var serverPort = 7860
    var idleUnloadMinutes = 10
    /// Settings → Network & server → Memory: what stays loaded between jobs.
    var residencyMode = MossResidencyMode.auto
    /// Settings → Models: explicit VAE / text encoder / tokenizer (unset = the models-folder layout).
    var overrides = MossModelLayout.Overrides()
    /// Generate → Profile: the user's saved combinations (the built-ins live in `GenerationProfile`).
    var profiles: [GenerationProfile] = []
    /// Settings → Diagnostics & feedback. Off by default: nothing is recorded until the user turns it on.
    var diagnosticsEnabled = false
    /// Settings → Models: the Hugging Face repo setup downloads the model bundle from.
    var modelBundleRepo = ModelBundleSource.defaultRepo
    /// Settings → Prompts: edits to the enhancer, captioner and slider instructions (only edits are stored).
    var prompts = MossPrompts()

    init() {}

    enum CodingKeys: String, CodingKey {
        case theme, modelsDirectory, outputDirectory, egressEnabled, serverPort, idleUnloadMinutes, overrides, profiles, diagnosticsEnabled
        case modelBundleRepo, residencyMode, prompts
    }

    /// Tolerant: a field added after the settings were saved keeps its default instead of resetting all.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = AppSettings()
        theme = (try? c.decode(ThemeID.self, forKey: .theme)) ?? d.theme
        modelsDirectory = (try? c.decode(URL.self, forKey: .modelsDirectory)) ?? d.modelsDirectory
        outputDirectory = (try? c.decode(URL.self, forKey: .outputDirectory)) ?? d.outputDirectory
        egressEnabled = (try? c.decode(Bool.self, forKey: .egressEnabled)) ?? d.egressEnabled
        serverPort = (try? c.decode(Int.self, forKey: .serverPort)) ?? d.serverPort
        idleUnloadMinutes = (try? c.decode(Int.self, forKey: .idleUnloadMinutes)) ?? d.idleUnloadMinutes
        overrides = (try? c.decode(MossModelLayout.Overrides.self, forKey: .overrides)) ?? d.overrides
        profiles = (try? c.decode([GenerationProfile].self, forKey: .profiles)) ?? d.profiles
        diagnosticsEnabled = (try? c.decode(Bool.self, forKey: .diagnosticsEnabled)) ?? d.diagnosticsEnabled
        modelBundleRepo = (try? c.decode(String.self, forKey: .modelBundleRepo)) ?? d.modelBundleRepo
        residencyMode = (try? c.decode(MossResidencyMode.self, forKey: .residencyMode)) ?? d.residencyMode
        prompts = (try? c.decode(MossPrompts.self, forKey: .prompts)) ?? d.prompts
    }

    var networkPolicy: NetworkPolicy { NetworkPolicy(egressAllowed: egressEnabled) }

    /// `MOSSFORGE_MODELS`, else a `models/` folder next to (or above) the working directory — the repo
    /// layout when run with `swift run` — else Application Support.
    static var defaultModelsDirectory: URL {
        if let env = ProcessInfo.processInfo.environment["MOSSFORGE_MODELS"] { return URL(fileURLWithPath: env) }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for candidate in [cwd.appendingPathComponent("models"), cwd.deletingLastPathComponent().appendingPathComponent("models")]
        where FileManager.default.fileExists(atPath: candidate.appendingPathComponent("diffusion_models").path) {
            return candidate.standardizedFileURL
        }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MossForge/models")
    }

    static let key = "mossforge.settings"

    static func load() -> AppSettings {
        guard let data = UserDefaults.standard.data(forKey: key), let s = try? JSONDecoder().decode(AppSettings.self, from: data) else {
            return AppSettings()
        }
        return s
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.key) }
    }
}

/// What the Generate screen is editing.
struct GenerateDraft: Equatable {
    struct LoRAEntry: Identifiable, Equatable {
        let id = UUID()
        var url: URL
        var strength: Double = 1
        var blocks = ""
    }

    var prompt = ""
    var negativePrompt = MossNegativePrompt.recommended
    /// The library DiT (relative path); `nil` = the first SFT model.
    var model: String?
    /// The Turbo (distillation) LoRA (relative path); `nil` = the first one that suits the model.
    var turbo: String?
    /// Turbo on by default (fast); off = the model's full 50-step CFG sampling.
    var turboOff = false
    /// The model is already few-step (no sidecar recipe): run it at 8 steps, CFG 1, with no Turbo LoRA.
    var alreadyDistilled = false
    var pixels = 1_048_576
    var ratio = "1:1"
    var steps: Int?
    var cfg = 3.0
    var shift = 5.0
    var seed: UInt64?
    var lockSeed = false
    var skipBlock22 = true
    var useLayout = false
    var loras: [LoRAEntry] = []
    /// Edit in Layout: the original image's exact size (the pixel/ratio pickers are bypassed).
    var exactSize: Dimensions?
    /// Reference images for an edit LoRA (≤ 4, "Picture 1…4").
    var references: [URL] = []
    /// How the edit LoRA sees them; `nil` = from the LoRA's metadata.
    var editLayout: MossEditLayout?
    /// Where Picture 1 may change (Edit → Mask…); applies only while `source` is still Picture 1.
    var editMask: ReferenceMask?

    struct ReferenceMask: Equatable {
        var source: URL
        var mask: EditMask
    }

    var size: Dimensions { exactSize ?? MossDimensions.fit(pixels: pixels, ratio: ratio) }
}

/// The app's state: settings, the engine (MLX when MOSS weights are present, else the stub), the
/// Generate draft + layout canvas, the gallery, and the resident server — which shares the one
/// `InferenceQueue` with in-app generation so they never contend for the GPU.
@MainActor
@Observable
final class AppStore {
    var selection: Screen = .generate
    var settings: AppSettings { didSet { settings.save(); if settings != oldValue { reconfigure(old: oldValue) } } }
    private(set) var engine: MossEngine
    let inferenceQueue = InferenceQueue()
    let server: ServerController

    var draft = GenerateDraft()
    var layout = LayoutDraft(canvas: Dimensions(width: 1024, height: 1024))
    private(set) var isGenerating = false
    private(set) var progress: GenerationProgress?
    private(set) var lastImage: NSImage?
    /// Where the last image was saved (Generate preview → Share / Save a Copy / Show in Gallery).
    private(set) var lastImageURL: URL?
    private(set) var lastSeed: UInt64?
    /// `editInfo(_:)`'s per-file cache (LoRA headers don't change under a URL often enough to watch).
    @ObservationIgnored var editInfoCache: [URL: MossEditLoRAInfo] = [:]
    private(set) var lastError: String?
    private(set) var gallery: [URL] = []
    private(set) var isEnhancing = false
    /// Settings → Prompts → Try it (`AppStore+Prompts.swift`).
    var promptLab = PromptLab()
    private var task: Task<Void, Never>?

    var backendName: String { engine.backendName }

    init() {
        let settings = AppSettings.load()
        self.settings = settings
        engine = Self.makeEngine(settings)
        server = ServerController(queue: inferenceQueue)
        server.attach(engine: engine, settings: settings)
        refreshGallery()
        refreshLibrary()
        help.host = self
    }

    /// The one-line backend injection: MLX when the encoder, tokenizer, VAE and any DiT are present
    /// (Generate always names its library model, so the release SFT file isn't required).
    static func makeEngine(_ s: AppSettings) -> MossEngine {
        let dits = (try? FileManager.default.contentsOfDirectory(atPath: s.modelsDirectory.appendingPathComponent("diffusion_models").path)) ?? []
        if dits.contains(where: { $0.hasSuffix(".safetensors") && !$0.contains("-dataspace") }),
            (try? MossModelPaths.resolve(models: s.modelsDirectory, variant: .sft, overrides: s.overrides, requireDiT: false)) != nil
        {
            return MossEngine(backend: MossBackend(
                models: s.modelsDirectory, overrides: s.overrides, residency: MossBackend.residency(for: s.residencyMode)))
        }
        return MossEngine()
    }

    /// What Settings → Memory means on this Mac (the composition root asks the MLX side).
    var residency: MossResidencyPolicy { MossBackend.residency(for: settings.residencyMode) }

    /// After setup downloads models: a fresh engine (MLX now that the files are here) and library.
    func modelsChanged() {
        let previous = engine
        Task { await previous.releaseResidentModels() }
        engine = Self.makeEngine(settings)
        server.attach(engine: engine, settings: settings)
        refreshLibrary()
    }

    private func reconfigure(old: AppSettings) {
        if !settings.diagnosticsEnabled, old.diagnosticsEnabled { help.forgetDiagnostics() }
        if settings.modelsDirectory != old.modelsDirectory || settings.overrides != old.overrides
            || settings.residencyMode != old.residencyMode
        {
            let previous = engine
            Task { await previous.releaseResidentModels() }
            engine = Self.makeEngine(settings)
            refreshLibrary()
        }
        server.attach(engine: engine, settings: settings)
        if settings.outputDirectory != old.outputDirectory { refreshGallery() }
    }

    /// The request the draft (and, in layout mode, the canvas) describes.
    func makeRequest() throws -> GenerationRequest {
        let size = draft.size
        layout.canvas = size
        let prompt: String
        if draft.useLayout {
            if layout.caption.isEmpty { layout.caption = draft.prompt }
            prompt = try layout.layoutPrompt().jsonString()
        } else {
            prompt = draft.prompt
        }
        var r = try makeRequest(for: draft, prompt: prompt)
        if let e = editing {
            r.seed = e.seed
            let boxes = LayoutEditMask.changedBoxes(original: e.original, edited: try layout.layoutPrompt())
            r.edit = ImageEditSpec(source: e.source, mask: LayoutEditMask.tokenMask(boxes: boxes, canvas: size))
        }
        try r.validate()
        return r
    }

    /// The request a Generate draft describes for `prompt` — model, Turbo, sampling, LoRAs, references.
    /// Shared by Generate and Character Studio (which re-aims a copy of the draft at each image).
    func makeRequest(for draft: GenerateDraft, prompt: String) throws -> GenerationRequest {
        let size = draft.size
        var r = GenerationRequest(
            prompt: prompt, negativePrompt: draft.negativePrompt, variant: family.variant,
            width: size.width, height: size.height, cfg: draft.cfg, shift: draft.shift,
            slg: draft.skipBlock22 ? .official : .off, seed: draft.lockSeed ? draft.seed : nil)
        guard let dit = selectedDiT else { throw EngineError.invalidRequest("no DiT in the library — import one on the Models screen") }
        r.model = dit.url
        if let info = builtInTurbo {
            guard let spec = DistillSpec(builtInto: dit.url, info: info) else {
                throw EngineError.invalidRequest("\(dit.name)'s built-in Turbo is missing its dataspace file")
            }
            r.apply(spec)
        } else if let t = activeTurbo, let info = t.sidecar.distill {
            guard let spec = DistillSpec(lora: t.url, info: info, strength: t.sidecar.strength ?? 1) else {
                throw EngineError.invalidRequest("\(t.name) is missing its data-space adapter files")
            }
            r.apply(spec)
        }
        if let steps = draft.steps { r.steps = steps }
        r.references = draft.references
        r.editLayout = draft.editLayout
        // Masked edit: pin everything outside the mask to Picture 1 (resampled if the size changed).
        if r.edit == nil, let m = draft.editMask, m.source == draft.references.first {
            let (tw, th) = (size.width / 16, size.height / 16)
            let mask = (m.mask.width, m.mask.height) == (tw, th)
                ? m.mask : EditMask.resampled(from: m.mask.values, width: m.mask.width, height: m.mask.height, toWidth: tw, toHeight: th)
            r.edit = ImageEditSpec(source: m.source, mask: mask)
        }
        r.loras = try draft.loras.map { entry in
            var blocks: LoRABlockWeights?
            if !entry.blocks.trimmingCharacters(in: .whitespaces).isEmpty {
                guard let b = LoRABlockWeights(parsing: entry.blocks) else {
                    throw EngineError.invalidRequest("LoRA \(entry.url.lastPathComponent): '\(entry.blocks)' is not a block spec")
                }
                blocks = b
            }
            return LoRASpec(url: entry.url, strength: entry.strength, blocks: blocks)
        }
        try r.validate()
        return r
    }

    // MARK: Edit in Layout

    /// The image being edited: its file, seed and layout as first loaded (what "changed" is measured against).
    struct EditSession: Equatable {
        var source: URL
        var seed: UInt64
        var original: LayoutPrompt
    }
    var editing: EditSession? { didSet { editBackground = editing.flatMap { NSImage(contentsOf: $0.source) } } }
    /// The edited image, shown under the layout canvas.
    private(set) var editBackground: NSImage?

    /// Share of the image the current edit will redraw (0 = nothing changed yet).
    var editCoverage: Double {
        guard let e = editing, let current = try? layout.layoutPrompt() else { return 0 }
        return LayoutEditMask.tokenMask(boxes: LayoutEditMask.changedBoxes(original: e.original, edited: current), canvas: draft.size).coverage
    }

    /// Right-click → Edit in Layout…: restore the image's prompt (as a layout), seed, size, model,
    /// Turbo, sampling and LoRAs from its sidecar, then Generate redraws only boxes you change.
    func beginEdit(_ url: URL) {
        let meta = GallerySidecar.read(for: url)
        guard let rep = NSImageRep(contentsOf: url) else { lastError = "Can't read \(url.lastPathComponent)"; return }
        let size = Dimensions(width: rep.pixelsWide / 16 * 16, height: rep.pixelsHigh / 16 * 16)
        let seed = meta["seed"] as? UInt64 ?? (meta["seed"] as? Int).map(UInt64.init) ?? 0
        if let model = meta["model"] as? String, let dit = library.dits.first(where: { $0.url.lastPathComponent == model }) {
            draft.model = dit.relativePath
        }
        let turbo = meta["turbo"] as? String ?? ""
        draft.alreadyDistilled = false
        if turbo.isEmpty {
            draft.turboOff = true
        } else if turbo == "built in" {
            // A fused Turbo's sidecar brings its own recipe; a model without one was run as "already 8-step".
            draft.alreadyDistilled = selectedDiT?.sidecar.distill == nil
        } else if let t = turboChoices.first(where: { $0.url.lastPathComponent == turbo }) {
            draft.turboOff = false
            draft.turbo = t.relativePath
        }
        if let n = meta["steps"] as? Int {
            let recipe = (builtInTurbo ?? activeTurbo?.sidecar.distill)?.steps ?? (draft.turboOff ? family.variant.defaults.steps : nil)
            draft.steps = n == recipe ? nil : n  // the recipe's own steps stay "default", so the profile still matches
        }
        if let c = meta["cfg"] as? Double { draft.cfg = c }
        if let sh = meta["shift"] as? Double { draft.shift = sh }
        if let neg = meta["negative_prompt"] as? String { draft.negativePrompt = neg }
        if let loras = meta["loras"] as? [[String: Any]] {
            draft.loras = loras.compactMap { l in
                guard let path = l["path"] as? String, FileManager.default.fileExists(atPath: path) else { return nil }
                var e = GenerateDraft.LoRAEntry(url: URL(fileURLWithPath: path))
                e.strength = l["strength"] as? Double ?? 1
                return e
            }
        }
        draft.exactSize = size
        draft.seed = seed
        draft.lockSeed = true
        let prompt = LayoutEditMask.layout(fromPrompt: meta["prompt"] as? String ?? "")
        layout = LayoutDraft(prompt, canvas: size)
        draft.useLayout = true
        // Measure changes against the layout as the canvas holds it (same rounding as later).
        editing = EditSession(source: url, seed: seed, original: (try? layout.layoutPrompt()) ?? prompt)
        lastImage = NSImage(contentsOf: url)
        lastImageURL = url
        lastError = nil
        selection = .generate
    }

    /// Gallery / preview → Use as Edit Reference: a source image for an edit LoRA (Generate).
    func useAsReference(_ url: URL) {
        if draft.references.count == GenerationRequest.maxReferences { draft.references.removeFirst() }
        addReference(url)
        selection = .generate
    }

    func stopEditing() {
        editing = nil
        draft.exactSize = nil
    }

    func generate() {
        guard !isGenerating else { return }
        let request: GenerationRequest
        do { request = try makeRequest() } catch {
            lastError = "\(error)"
            return
        }
        isGenerating = true
        lastError = nil
        progress = GenerationProgress(step: 0, totalSteps: request.steps, message: "queued")
        let (engine, queue, output) = (engine, inferenceQueue, settings.outputDirectory)
        let t0 = Date()
        task = Task {
            do {
                let result = try await queue.run {
                    try await engine.generate(request) { p in Task { @MainActor in self.progress = p } }
                }
                let url = try Self.save(result, request: request, in: output)
                lastImage = NSImage(data: result.imagePNG)
                lastImageURL = url
                // Editing: the result is the next edit's starting point (its layout = the new baseline).
                if let e = editing, let current = try? layout.layoutPrompt() {
                    editing = EditSession(source: url, seed: e.seed, original: current)
                }
                note(.info, "generate", "\(result.width)×\(result.height), \(request.steps) steps, CFG \(request.cfg)\(request.distill != nil ? ", Turbo" : "")\(request.loras.isEmpty ? "" : ", \(request.loras.count) LoRA(s)") in \(Self.seconds(since: t0))")
                lastSeed = result.seed
                draft.seed = result.seed
                refreshGallery()
            } catch is CancellationError {
                lastError = "cancelled"
            } catch {
                lastError = "\(error)"
                note(.error, "generate", "failed after \(Self.seconds(since: t0)): \(error)")
            }
            isGenerating = false
            progress = nil
        }
    }

    func cancel() { task?.cancel() }

    /// Gallery with the last generated image selected.
    func showInGallery(_ url: URL) {
        galleryCollection = nil
        refreshGallery()
        gallerySelection.clear()
        gallerySelection.click(url, in: visibleGallery)
        selection = .gallery
    }

    /// Run the on-device enhancer (Qwen3.5) on the prompt: rewrite it as a detailed caption, or —
    /// `layout` — turn it into boxes on the layout canvas for the current size.
    func enhance(layout: Bool) {
        guard !isEnhancing, !isGenerating else { return }
        let source = draft.useLayout && !self.layout.caption.isEmpty ? self.layout.caption : draft.prompt
        guard !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        isEnhancing = true
        lastError = nil
        progress = GenerationProgress(step: 0, totalSteps: 0, message: layout ? "drafting a layout" : "enhancing prompt")
        let (engine, queue, size, prompts) = (engine, inferenceQueue, draft.size, settings.prompts)
        task = Task {
            do {
                let text = try await queue.run {
                    try await engine.enhancePrompt(PromptEnhanceRequest(prompt: source, layout: layout, canvas: size, prompts: prompts)) { p in
                        Task { @MainActor in self.progress = p }
                    }
                }
                if layout {
                    let prompt = try JSONDecoder().decode(LayoutPrompt.self, from: Data(text.utf8))
                    self.layout = LayoutDraft(prompt, canvas: size)
                    draft.useLayout = true
                } else {
                    draft.prompt = text
                }
            } catch is CancellationError {
                lastError = "cancelled"
            } catch {
                lastError = "\(error)"
                note(.error, layout ? "layout" : "enhance", "failed: \(error)")
            }
            isEnhancing = false
            progress = nil
        }
    }

    /// PNG + a JSON sidecar with the request (prompt, variant, seed, …) for the gallery.
    static func save(_ result: GenerationResult, request: GenerationRequest, in dir: URL) throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
        let url = dir.appendingPathComponent("moss-\(stamp)-\(result.seed).png")
        try result.imagePNG.write(to: url)
        var meta: [String: Any] = [
            "prompt": request.prompt, "negative_prompt": request.negativePrompt, "variant": request.variant.rawValue,
            "width": result.width, "height": result.height, "steps": request.steps, "cfg": request.cfg,
            "shift": request.shift, "seed": result.seed, "loras": request.loras.map { ["path": $0.url.path, "strength": $0.strength] },
            "model": request.model?.lastPathComponent ?? "", "turbo": request.distill.map { $0.lora?.url.lastPathComponent ?? "built in" } ?? "",
        ]
        if let edit = request.edit { meta["edited_from"] = edit.source.lastPathComponent }
        try? GallerySidecar.write(meta, for: url)
        return url
    }

    // MARK: Character Studio state (behaviour in AppStore+Character.swift)

    /// The Character Studio form, persisted so a half-built plan survives relaunch.
    var character = AppStore.loadCharacter() { didSet { if character != oldValue { Self.persistCharacter(character) } } }
    /// The planned images for the current run, with their prompts and results.
    var characterRows: [CharacterRow] = []
    var characterRunning = false
    var characterStatus = "Ready."
    var characterProgress = 0.0
    var characterTask: Task<Void, Never>?

    // MARK: help + feedback (this store is its HelpHost, AppStore+Help.swift)

    let help = HelpCenter(product: .mossForge)

    // MARK: gallery management

    var gallerySelection = GallerySelection<URL>()
    /// Set to ask the Gallery to confirm moving the selection to the Trash.
    var confirmGalleryDelete = false
    private(set) var galleryStatus = ""

    func selectAllGallery() { gallerySelection.selectAll(visibleGallery) }

    // MARK: collections (tags in each image's sidecar)

    /// Each image's collections, read from the sidecars on refresh.
    private(set) var collectionsByImage: [URL: Set<String>] = [:]
    /// The Gallery filter: `nil` = all images.
    var galleryCollection: String? { didSet { gallerySelection.clear() } }

    var allCollections: [String] { Array(Set(collectionsByImage.values.flatMap { $0 })).sorted() }

    var visibleGallery: [URL] {
        guard let c = galleryCollection else { return gallery }
        return gallery.filter { collectionsByImage[$0]?.contains(c) == true }
    }

    func membership(of collection: String, in images: [URL]) -> CollectionTagging.Membership {
        CollectionTagging.membership(of: collection, in: images.map { collectionsByImage[$0] ?? [] })
    }

    /// Add `images` to, or remove them from, a collection (explicit, never a toggle).
    func tag(_ op: CollectionTagging.Operation, collection: String, images: [URL]) {
        let name = collection.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        for image in images {
            let meta = GallerySidecar.read(for: image)
            let tags = CollectionTagging.apply(op, tag: name, to: GallerySidecar.collections(in: meta))
            try? GallerySidecar.write(GallerySidecar.setting(tags, in: meta), for: image)
            collectionsByImage[image] = tags
        }
        sidecarVersion += 1
        let n = images.count
        galleryStatus = (op == .add ? "Added \(n) image\(n == 1 ? "" : "s") to " : "Removed \(n) image\(n == 1 ? "" : "s") from ") + "“\(name)”."
        if op == .remove, galleryCollection == name { gallerySelection.prune(keeping: Set(visibleGallery)) }
    }

    /// Move the selected images (and their sidecars) to the Trash — recoverable, like Finder.
    func trashSelectedGallery() {
        let items = gallerySelection.ordered(in: gallery)
        var failed: [String] = []
        for image in items {
            for file in GallerySidecar.files(for: image) {
                do { try FileManager.default.trashItem(at: file, resultingItemURL: nil) } catch {
                    if file == image { failed.append(image.lastPathComponent) }
                }
            }
        }
        refreshGallery()
        galleryStatus = failed.isEmpty
            ? "Moved \(items.count) image\(items.count == 1 ? "" : "s") to the Trash."
            : "Couldn't move: " + failed.joined(separator: ", ")
    }

    /// Move one image (and its sidecar) to the Trash — the single-image menus outside the Gallery grid.
    func trashImage(_ url: URL) {
        for file in GallerySidecar.files(for: url) { try? FileManager.default.trashItem(at: file, resultingItemURL: nil) }
        if lastImageURL == url {
            lastImage = nil
            lastImageURL = nil
        }
        refreshGallery()
    }

    func refreshGallery() {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: settings.outputDirectory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        gallery = urls.filter { $0.pathExtension.lowercased() == "png" }.sorted {
            let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return a > b
        }
        collectionsByImage = Dictionary(uniqueKeysWithValues: gallery.map { ($0, GallerySidecar.collections(in: GallerySidecar.read(for: $0))) })
        if let c = galleryCollection, !allCollections.contains(c) { galleryCollection = nil }
        gallerySelection.prune(keeping: Set(visibleGallery))
    }

    // MARK: upscale (Gallery → Upscale…)

    var upscaleMethod: UpscaleMethod = .esrgan
    var upscaleScale = 2.0
    var upscaleSettings = UpscaleSettings()
    /// ESRGAN model filename; empty = the backend default.
    var upscaleESRGAN = ""
    private(set) var isUpscaling = false
    private(set) var upscaleProgress = 0.0
    private(set) var upscaleStatus = ""
    private var upscaleTask: Task<Void, Never>?

    var availableUpscalers: [String] { UpscaleModels.esrganModels(in: settings.modelsDirectory) }
    var missingVOSR: [String] { UpscaleModels.missingVOSR(in: settings.modelsDirectory) }

    /// Upscale a gallery image through the shared GPU queue; the result (PNG + sidecar recording its
    /// source) lands in the output folder, so it shows up in the gallery beside the original.
    func upscale(_ url: URL) {
        guard !isUpscaling else { return }
        isUpscaling = true
        upscaleProgress = 0
        upscaleStatus = "Starting…"
        var draft = UpscaleRequest(
            image: url, scale: upscaleScale, method: upscaleMethod,
            esrganModel: upscaleESRGAN.isEmpty ? nil : upscaleESRGAN, settings: upscaleSettings)
        draft.output = settings.outputDirectory.appendingPathComponent(draft.resolvedOutput.lastPathComponent)
        let request = draft
        let (engine, queue) = (engine, inferenceQueue)
        upscaleTask = Task {
            do {
                if await queue.pending > 0 { upscaleStatus = "Queued behind another GPU job…" }
                let t0 = Date()
                let result = try await queue.run {
                    if Task.isCancelled { throw EngineError.cancelled }
                    return try await engine.upscale(request) { p in
                        Task { @MainActor in
                            self.upscaleProgress = p.fraction
                            self.upscaleStatus = p.message
                        }
                    }
                }
                try? GallerySidecar.write(
                    GallerySidecar.upscaled(GallerySidecar.read(for: url), request: request, result: result), for: result.output)
                refreshGallery()
                note(.info, "upscale", "\(request.method.rawValue) ×\(request.scale) → \(result.width)×\(result.height) in \(Self.seconds(since: t0))")
                upscaleStatus = String(
                    format: "Done — %d × %d in %.0f s, added to the gallery.", result.width, result.height, Date().timeIntervalSince(t0))
            } catch is CancellationError {
                upscaleStatus = "Cancelled."
            } catch let error as EngineError {
                if case .cancelled = error { upscaleStatus = "Cancelled." } else { upscaleStatus = error.description; note(.error, "upscale", error.description) }
            } catch {
                upscaleStatus = error.localizedDescription
                note(.error, "upscale", error.localizedDescription)
            }
            upscaleProgress = 0
            isUpscaling = false
            upscaleTask = nil
        }
    }

    func cancelUpscale() {
        upscaleStatus = "Cancelling…"
        upscaleTask?.cancel()
    }

    // MARK: caption (Gallery → Caption)

    var captionDetail: CaptionDetail = .long
    /// The image being captioned, if any.
    private(set) var captioning: URL?
    private(set) var captionStatus = ""
    /// Bumped when a sidecar changes so the Gallery detail re-reads it.
    private(set) var sidecarVersion = 0

    /// A running batch: done / total, and how many were skipped or failed.
    struct CaptionBatch: Equatable {
        var done = 0
        var total = 0
        var failed = 0
        var current = ""
    }
    private(set) var captionBatch: CaptionBatch?
    private var captionTask: Task<Void, Never>?

    /// Caption one gallery image (the detail pane's button).
    func caption(_ url: URL) { captionGallery([url], overwrite: true) }

    /// Caption gallery images with the on-device VLM (Qwen3.5 vision) into their sidecars, one after
    /// another through the GPU queue (the captioner stays loaded between images). Each image's prompt,
    /// when it isn't layout JSON, is the reference the model corrects. Already-captioned images are
    /// skipped unless `overwrite`.
    func captionGallery(_ images: [URL], overwrite: Bool) {
        let todo = overwrite ? images : images.filter { GallerySidecar.caption(in: GallerySidecar.read(for: $0)) == nil }
        runCaptions(todo, label: "image") { url in
            let meta = GallerySidecar.read(for: url)
            return (meta["prompt"] as? String).flatMap { $0.hasPrefix("{") ? nil : $0 }
        } save: { url, caption, _ in
            try GallerySidecar.write(GallerySidecar.adding(caption, to: GallerySidecar.read(for: url)), for: url)
        }
    }

    /// Caption a folder of training images as kohya `name.txt` files (`trigger, caption`).
    func captionFolder(_ folder: URL, overwrite: Bool, trigger: String) {
        let todo = FolderCaptioning.pending(in: folder, overwrite: overwrite)
        runCaptions(todo, label: "image") { url in
            // An existing caption is the reference to correct.
            (try? String(contentsOf: FolderCaptioning.captionURL(for: url), encoding: .utf8))
        } save: { url, caption, detail in
            try FolderCaptioning.write(caption.text(detail), trigger: trigger, for: url)
        }
    }

    func cancelCaptions() {
        captionTask?.cancel()
        captionStatus = "Cancelling…"
    }

    private func runCaptions(
        _ images: [URL], label: String, reference: @escaping (URL) -> String?,
        save: @escaping (URL, MossCaption, CaptionDetail) throws -> Void
    ) {
        guard captionBatch == nil else { return }
        guard !images.isEmpty else {
            captionStatus = "Nothing to caption — every \(label) already has one."
            return
        }
        let (engine, queue, detail) = (engine, inferenceQueue, captionDetail)
        captionBatch = CaptionBatch(total: images.count)
        captionStatus = "Captioning…"
        let t0 = Date()
        captionTask = Task {
            for url in images {
                if Task.isCancelled { break }
                captioning = url
                captionBatch?.current = url.lastPathComponent
                do {
                    if await queue.pending > 0 { captionStatus = "Queued behind another GPU job…" }
                    let request = CaptionRequest(image: url, detail: detail, reference: reference(url), prompts: settings.prompts)
                    let caption = try await queue.run {
                        try await engine.caption(request) { p in Task { @MainActor in self.captionStatus = p.message } }
                    }
                    try save(url, caption, detail)
                    sidecarVersion += 1
                } catch is CancellationError {
                    break
                } catch {
                    captionBatch?.failed += 1
                    captionStatus = "\(url.lastPathComponent): \(error)"
                    note(.error, "caption", "\(error)")
                }
                captionBatch?.done += 1
            }
            let b = captionBatch ?? CaptionBatch()
            let ok = b.done - b.failed
            captionStatus = Task.isCancelled
                ? "Cancelled after \(ok) of \(b.total)."
                : String(format: "Captioned %d of %d in %.0f s", ok, b.total, Date().timeIntervalSince(t0)) + (b.failed > 0 ? " (\(b.failed) failed)." : ".")
            note(b.failed > 0 ? .warning : .info, "caption", "\(ok) of \(b.total) captioned in \(Self.seconds(since: t0))")
            captioning = nil
            captionBatch = nil
            captionTask = nil
        }
    }

    // MARK: training (Train screen)

    /// What the Train screen is editing.
    struct TrainDraft: Equatable {
        enum Preset: String, CaseIterable, Identifiable {
            case quick = "Quick test", standard = "Standard", detailed = "Detailed"
            var id: String { rawValue }
            var detail: String {
                switch self {
                case .quick: "512², rank 8, 200 steps — checks a dataset in a few minutes"
                case .standard: "1024², rank 16, 1000 steps — style / subject LoRAs"
                case .detailed: "1024², rank 32, attention + FFN, 2000 steps — concepts, typography"
                }
            }
            /// With a LoRA type: the resolution, and this share of the type's dataset-scaled steps.
            var runScale: Double { self == .quick ? 0.1 : self == .standard ? 1 : 1.5 }
            var typeDetail: String {
                switch self {
                case .quick: "512², a tenth of the steps — checks a dataset in a few minutes"
                case .standard: "1024², the type's steps for this dataset"
                case .detailed: "1024², half as many steps again"
                }
            }
        }
        var preset: Preset = .standard
        /// What the LoRA is for (dataset mode); a type fills rank, layers, LR and steps.
        var profile: LoRATrainingProfile = .custom
        var dataset: URL?
        var model: String?  // library DiT; nil = the first Base
        var name = ""
        var trigger = ""
        var steps = 1000
        var rank = 16
        var learningRate = 1e-4
        var resolution = 1024
        var targets: LoRATargetSet = .attention
        var saveEvery = 250
        /// Test prompts rendered before training, at each checkpoint and at the end (empty = skipped).
        var testPrompts = ["", "", ""]
        var testSteps = 20
        var testResolution = 512
        // Slider mode (image-free Concept Slider).
        var mode: TrainingMode = .dataset
        var sliderDescription = ""
        var sliderTarget = ""
        var sliderPositive = ""
        var sliderNegative = ""
        var sliderMin = -1.0
        var sliderMax = 1.0
        var sliderEta = TrainingRequest.defaultSliderEta
        var sliderLatentSteps = 10
        var sliderPreserve = ""
        var factorization: LoRAFactorization = .lora
        var lokrFactor = -1
        var captionDropout = 0.0
        var checkpointing = true

        /// The type's settings for a dataset of `imageCount` images at the current preset; `nil` for Custom.
        func profileSettings(imageCount: Int) -> LoRATrainingProfile.Settings? {
            profile.settings(imageCount: imageCount, runScale: preset.runScale)
        }

        /// The hyperparameters no longer match the selected type.
        func profileModified(imageCount: Int) -> Bool {
            guard let p = profileSettings(imageCount: imageCount) else { return false }
            return rank != p.rank || learningRate != p.learningRate || steps != p.steps || targets != p.targets
                || factorization != p.factorization
        }

        /// Fill the hyperparameters from the type (Custom: the preset's fixed values).
        mutating func applyProfile(imageCount: Int) {
            guard let p = profileSettings(imageCount: imageCount) else { return }
            (rank, learningRate, steps, targets, factorization, saveEvery) = (p.rank, p.learningRate, p.steps, p.targets, p.factorization, p.saveEvery)
        }

        /// Apply the preset: with a type, its resolution + the type's settings at its run length.
        mutating func apply(_ p: Preset, imageCount: Int) {
            preset = p
            guard profile == .custom || mode != .dataset else {
                resolution = p == .quick ? 512 : 1024
                applyProfile(imageCount: imageCount)
                return
            }
            apply(p)
        }

        mutating func apply(_ p: Preset) {
            switch p {
            case .quick: (resolution, rank, steps, targets, saveEvery) = (512, 8, 200, .attention, 100)
            case .standard: (resolution, rank, steps, targets, saveEvery) = (1024, 16, 1000, .attention, 250)
            case .detailed: (resolution, rank, steps, targets, saveEvery) = (1024, 32, 2000, .attentionAndFFN, 500)
            }
        }
    }

    var trainDraft = TrainDraft()
    private(set) var trainProgress: GenerationProgress?
    private(set) var trainLosses: [Double] = []
    private(set) var trainStatus = ""
    private(set) var lastTrained: URL?
    /// The run's test images, one row per checkpoint (step 0 = before training).
    private(set) var trainSampleRows: [TrainingSamples.Row] = []
    private(set) var isSuggestingPrompts = false
    var isTraining: Bool { trainProgress != nil }
    private var trainTask: Task<Void, Never>?

    /// The DiT to train on: the chosen one, else the first Base, else the first.
    var trainBase: ModelItem? {
        library.dits.first { $0.relativePath == trainDraft.model }
            ?? library.dits.first { $0.sidecar.supports(.base) } ?? library.dits.first
    }

    /// Where the LoRA lands: `<models>/lora/trained/<name>.safetensors`.
    var trainOutput: URL? {
        let name = trainDraft.name.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty, !name.contains("/") else { return nil }
        return settings.modelsDirectory.appendingPathComponent("lora/trained/\(name).safetensors")
    }

    /// Ready to train: a name and a base, plus a dataset (images) or both slider descriptors.
    var canStartTraining: Bool {
        guard !isTraining, trainBase != nil, trainOutput != nil else { return false }
        switch trainDraft.mode {
        case .dataset: return trainDraft.dataset.map { !DatasetScanner.scan(directory: $0).isEmpty } ?? false
        case .edit:
            return trainDraft.dataset.map { !EditDatasetScanner.scan(directory: $0).pairs.isEmpty } ?? false
        case .slider:
            return !trainDraft.sliderPositive.trimmingCharacters(in: .whitespaces).isEmpty
                && !trainDraft.sliderNegative.trimmingCharacters(in: .whitespaces).isEmpty
                && SliderRange(min: trainDraft.sliderMin, max: trainDraft.sliderMax).isValid
        }
    }

    func startTraining() {
        guard canStartTraining, let base = trainBase, let output = trainOutput else { return }
        let dataset = trainDraft.dataset ?? output.deletingLastPathComponent()
        let d = trainDraft
        let request0 = TrainingRequest(
            baseModel: base.url, dataset: dataset, output: output, rank: d.rank, alpha: Double(d.rank), learningRate: d.learningRate,
            steps: d.steps, pixels: d.resolution * d.resolution, targets: d.targets, saveEvery: d.saveEvery, trigger: d.trigger,
            family: base.sidecar.families.first ?? .base)
        var withTests = request0
        withTests.samplePrompts = d.testPrompts
        withTests.sampleSteps = d.testSteps
        withTests.samplePixels = d.testResolution * d.testResolution
        if d.mode == .slider {
            withTests.mode = .slider
            withTests.sliderTarget = d.sliderTarget
            withTests.sliderPositive = d.sliderPositive
            withTests.sliderNegative = d.sliderNegative
            withTests.sliderRange = SliderRange(min: d.sliderMin, max: d.sliderMax)
            withTests.sliderEta = d.sliderEta
            withTests.sliderSampleSteps = d.sliderLatentSteps
            withTests.sliderPreserve = SliderPromptSet.attributes(from: d.sliderPreserve)
            withTests.trigger = ""
        }
        withTests.factorization = d.factorization
        withTests.lokrFactor = d.lokrFactor
        withTests.captionDropout = d.mode == .dataset ? d.captionDropout : 0
        withTests.gradientCheckpointing = d.checkpointing
        if d.mode == .edit { withTests.mode = .edit }
        let request = withTests
        let samples: [TrainingSample] = switch d.mode {
        case .slider: []
        case .edit: EditDatasetScanner.scan(directory: dataset).pairs.map { TrainingSample(image: $0.target, caption: $0.instruction, control: $0.control) }
        case .dataset: DatasetScanner.scan(directory: dataset)
        }
        trainSampleRows = []
        trainLosses = []
        trainStatus = ""
        lastTrained = nil
        trainProgress = GenerationProgress(step: 0, totalSteps: d.steps, message: "queued")
        let (engine, queue) = (engine, inferenceQueue)
        trainTask = Task {
            do {
                let r = try await queue.run {
                    try await engine.trainLoRA(request, samples: samples) { p in
                        Task { @MainActor in
                            self.trainProgress = p
                            if let v = p.value { self.trainLosses.append(v) }
                            if !p.files.isEmpty { self.addSampleRow(p.files) }
                        }
                    }
                }
                lastTrained = r.output
                trainStatus = String(format: "Saved %@ after %d steps in %.0f min.", r.output.lastPathComponent, r.steps, r.seconds / 60)
                note(.info, "train", String(format: "rank %d, %d steps at %d² in %.0f min", d.rank, r.steps, d.resolution, r.seconds / 60))
                refreshLibrary()
            } catch is CancellationError {
                trainStatus = "Stopped before the first step — nothing saved."
            } catch {
                trainStatus = "\(error)"
                note(.error, "train", "rank \(d.rank), \(d.resolution)²: \(error)")
            }
            trainProgress = nil
            trainTask = nil
        }
    }

    private func addSampleRow(_ files: [URL]) {
        let merged = TrainingSamples.rows(trainSampleRows.flatMap(\.images) + files)
        trainSampleRows = merged
    }

    /// Fill the test prompts: Qwen3.5 reads some dataset captions and suggests three (close to the data,
    /// a new scene, a harder test); without the model, three captions spread across the dataset.
    func suggestTestPrompts() {
        guard !isSuggestingPrompts, !isTraining, let dataset = trainDraft.dataset else { return }
        let captions = DatasetScanner.scan(directory: dataset).map(\.caption)
        let trigger = trainDraft.trigger
        let fallback = TrainingPromptSuggester.fallback(captions: captions, trigger: trigger)
        guard canChat else {
            trainDraft.testPrompts = fallback
            return
        }
        isSuggestingPrompts = true
        let request = ChatRequest(messages: TrainingPromptSuggester.messages(captions: captions, trigger: trigger), maxTokens: 300, temperature: 0.7)
        let (engine, queue) = (engine, inferenceQueue)
        Task {
            let answer = (try? await queue.run { try await engine.chat(request) }) ?? ""
            let parsed = TrainingPromptSuggester.parse(answer)
            trainDraft.testPrompts = (parsed + fallback.dropFirst(parsed.count)).prefix(TrainingSamples.maxPrompts).map { $0 }
            isSuggestingPrompts = false
        }
    }

    private(set) var isDraftingSlider = false

    /// Slider wizard: "a young to old slider" → name, target, positive, negative (Qwen3.5; an "X to Y"
    /// reading when the model is missing or its reply can't be parsed).
    func draftSlider() {
        let description = trainDraft.sliderDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !description.isEmpty, !isDraftingSlider else { return }
        let apply = { (d: SliderConceptDraft) in
            self.trainDraft.name = d.name
            self.trainDraft.sliderTarget = d.target
            self.trainDraft.sliderPositive = d.positive
            self.trainDraft.sliderNegative = d.negative
        }
        guard canChat else {
            if let d = SliderConceptDraft.heuristic(description) { apply(d) } else { trainStatus = "Describe it as “X to Y”, e.g. “young to old”." }
            return
        }
        isDraftingSlider = true
        let request = ChatRequest(messages: SliderConceptDraft.messages(for: description, prompts: settings.prompts), maxTokens: 200, temperature: 0.3)
        let (engine, queue) = (engine, inferenceQueue)
        Task {
            let reply = (try? await queue.run { try await engine.chat(request) }) ?? ""
            if let d = SliderConceptDraft.parse(reply) ?? SliderConceptDraft.heuristic(description) {
                apply(d)
            } else {
                trainStatus = "Couldn't read a slider from that — try “X to Y”."
            }
            isDraftingSlider = false
        }
    }

    /// Stop after the current step; what's trained so far is saved.
    func stopTraining() {
        trainTask?.cancel()
        trainStatus = "Stopping — saving what's trained so far…"
    }

    /// Generate with the new LoRA: a Turbo profile for its family, the LoRA added at strength 1.
    func tryInGenerate(_ lora: URL) {
        let family = ModelSidecar.read(for: lora)?.families.first ?? .base
        if let p = GenerationProfile.builtIn.first(where: { $0.id == (family == .base ? "base-turbo" : "fast") }),
            p.missing(in: libraryPaths).isEmpty
        {
            apply(p)
        }
        draft.loras.removeAll { $0.url == lora }
        draft.loras.append(.init(url: lora))
        if !trainDraft.trigger.isEmpty, !draft.prompt.lowercased().contains(trainDraft.trigger.lowercased()) {
            draft.prompt = trainDraft.trigger + ", " + draft.prompt
        }
        selection = .generate
    }

    // MARK: model tools (Models → Convert / Fuse / Merge)

    private(set) var toolProgress: GenerationProgress?
    private(set) var toolStatus = ""
    var isRunningTool: Bool { toolProgress != nil }
    private var toolTask: Task<Void, Never>?

    /// Run a convert / fuse / merge through the engine (one GPU job; resident models are released),
    /// then rescan so the new DiT shows up. `onDone` gets the output's library id.
    func runModelTool(_ request: ModelToolRequest, onDone: @escaping (String) -> Void = { _ in }) {
        guard !isRunningTool else { return }
        toolProgress = GenerationProgress(step: 0, totalSteps: 0, message: "queued")
        toolStatus = ""
        let (engine, queue, models) = (engine, inferenceQueue, settings.modelsDirectory)
        toolTask = Task {
            do {
                let r = try await queue.run {
                    try await engine.runModelTool(request) { p in Task { @MainActor in self.toolProgress = p } }
                }
                toolStatus = String(format: "Wrote %@ (%.2f GB) in %.0f s.", r.output.lastPathComponent, Double(r.bytes) / 1e9, r.seconds)
                    + (r.report.isEmpty ? "" : " " + r.report)
                note(.info, "tools", String(format: "%@ → %@ in %.0f s", request.operation.name, request.precision.rawValue, r.seconds))
                refreshLibrary()
                onDone(ModelLibrary.relativePath(r.output, in: models))
            } catch {
                toolStatus = "\(error)"
                note(.error, "tools", "\(request.operation.name): \(error)")
                try? FileManager.default.removeItem(at: request.output.deletingLastPathComponent()
                    .appendingPathComponent(".\(request.output.lastPathComponent).partial"))
            }
            toolProgress = nil
            toolTask = nil
        }
    }

    func runModelToolRefused(_ message: String) { toolStatus = message }

    // MARK: library (Models screen; Generate's model / Turbo / LoRA pickers)

    private(set) var library = ModelLibrary()
    private(set) var isScanningLibrary = false

    func refreshLibrary() {
        let models = settings.modelsDirectory
        isScanningLibrary = true
        Task {
            library = await Task.detached { ModelLibrary.scan(models: models) }.value
            isScanningLibrary = false
            // Nothing picked yet → start on "Fast — 4 steps" when its files are here.
            if draft.model == nil, draft.turbo == nil, !draft.turboOff, !draft.alreadyDistilled,
                let fast = GenerationProfile.builtIn.first(where: { $0.id == "fast" }), fast.missing(in: libraryPaths).isEmpty
            {
                apply(fast)
            }
        }
    }

    /// The DiT Generate runs: the chosen one, else the first SFT model, else the first.
    var selectedDiT: ModelItem? {
        library.dits.first { $0.relativePath == draft.model }
            ?? library.dits.first { $0.sidecar.supports(.sft) } ?? library.dits.first
    }

    var family: MossFamily { selectedDiT?.sidecar.families.first ?? .sft }

    /// A Turbo fused into the selected model (its sidecar's recipe), or the user's "already 8-step"
    /// pick for a few-step model without one: no Turbo LoRA is stacked on it.
    var builtInTurbo: DistillInfo? {
        selectedDiT?.sidecar.distill ?? (draft.alreadyDistilled ? .alreadyDistilled : nil)
    }

    /// Turbo LoRAs that suit the selected model.
    var turboChoices: [ModelItem] { library.distills(for: family) }

    /// The Turbo LoRA in use, or `nil` when off (or none suits the model).
    var activeTurbo: ModelItem? {
        guard !draft.turboOff, builtInTurbo == nil else { return nil }
        let choices = turboChoices
        return choices.first { $0.relativePath == draft.turbo }
            ?? choices.first { $0.relativePath == MossVariant.turboLoRAPath } ?? choices.first
    }

    // MARK: profiles (Generate → Profile)

    var allProfiles: [GenerationProfile] { GenerationProfile.builtIn + settings.profiles }
    var libraryPaths: Set<String> { Set((library.dits + library.loras).map(\.relativePath)) }

    /// The profile the current model / Turbo / steps match, if any ("Custom" otherwise).
    var activeProfile: GenerationProfile? {
        GenerationProfile.matching(
            model: selectedDiT?.relativePath, turbo: activeTurbo?.relativePath, steps: draft.steps, in: allProfiles)
    }

    func apply(_ profile: GenerationProfile) {
        draft.model = profile.model
        draft.turboOff = profile.turbo == nil
        draft.alreadyDistilled = false
        if let t = profile.turbo { draft.turbo = t }
        draft.steps = profile.steps
    }

    /// Save the current model + Turbo + steps as a named profile.
    func saveProfile(named name: String) {
        let clean = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty, let model = selectedDiT?.relativePath else { return }
        let turbo = activeTurbo
        let detail = [selectedDiT?.name, turbo?.name].compactMap { $0 }.joined(separator: " + ")
            + (draft.steps.map { ", \($0) steps" } ?? "")
        settings.profiles.append(GenerationProfile(
            id: "user-\(UUID().uuidString)", name: clean, detail: detail, model: model, turbo: turbo?.relativePath, steps: draft.steps))
    }

    func deleteProfile(_ id: String) { settings.profiles.removeAll { $0.id == id } }

    /// Plain LoRAs that suit the selected model.
    var availableLoRAs: [URL] { library.plainLoRAs(for: family).map(\.url) }

    /// A library LoRA's properties (default strength, slider range); the filename guess when it isn't listed.
    func loraSidecar(_ url: URL) -> ModelSidecar {
        library.loras.first { $0.url.standardizedFileURL == url.standardizedFileURL }?.sidecar
            ?? ModelSidecar.inferred(fileName: url.lastPathComponent, isLoRA: true, siblings: [])
    }
}
