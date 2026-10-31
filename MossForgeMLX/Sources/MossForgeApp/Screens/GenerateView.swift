import AppKit
import MossForgeCore
import SwiftUI
import Tweakbench

/// Generate: prompt (or a layout drawn on the canvas), model + Turbo LoRA, size, sampling, LoRAs → image.
/// Turbo (a few-step distillation LoRA suited to the model, no CFG) is the default; off exposes CFG,
/// shift, negative prompt and SLG.
struct GenerateView: View {
    @Environment(AppStore.self) private var store
    @Environment(\.theme) private var theme
    @State private var namingProfile = false
    @State private var profileName = ""
    @State private var upscaling: URL?
    @State private var gettingModels = false

    var body: some View {
        @Bindable var store = store
        PersistentHSplit(key: "generate", sized: .leading, defaultWidth: 440, range: 360...760, otherMin: 420) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if store.library.dits.isEmpty, !store.isScanningLibrary { setupBanner.tweakable("generate.setup") }
                    if store.editing != nil { editBanner }
                    promptSection.tweakable("generate.prompt")
                    EditReferencesSection().tweakable("generate.edit")
                    if store.draft.useLayout { layoutSection.tweakable("generate.layout") }
                    modelSection.tweakable("generate.model")
                    sizeSection.tweakable("generate.size")
                    samplingSection.tweakable("generate.sampling")
                    loraSection.tweakable("generate.loras")
                }
                .padding(16)
            }
        } trailing: {
            preview.frame(maxWidth: .infinity, maxHeight: .infinity)
                .tweakable("generate.preview")
                .sheet(item: $upscaling) { UpscaleSheet(url: $0) }
                .sheet(isPresented: $gettingModels) { ModelSetupSheet() }
        }
    }

    // MARK: - Form

    /// First run: no diffusion model yet.
    private var setupBanner: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                Label("No models yet", systemImage: "shippingbox").font(.callout.weight(.semibold))
                Text("Download the Essentials (15.5 GB: the 4-step Turbo model, text encoder and VAE) and start generating — or point Settings → Folders at a models folder you already have.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Button("Get Models…") { gettingModels = true }.buttonStyle(.borderedProminent)
            }
        }
    }

    /// Edit in Layout: what's being edited and how much will be redrawn.
    private var editBanner: some View {
        let coverage = store.editCoverage
        return GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Label("Editing \(store.editing?.source.lastPathComponent ?? "")", systemImage: "pencil.and.outline")
                        .font(.callout.weight(.semibold)).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button("Stop editing") { store.stopEditing() }
                }
                Text(coverage == 0
                    ? "Change a box's text or caption, move it, add a box or delete one — only those areas are redrawn; the rest of the image stays exactly as it is. Same seed, model and settings as the original."
                    : String(format: "%.0f%% of the image will be redrawn (the changed boxes plus a small margin). Everything else is kept pixel for pixel.", coverage * 100))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var promptSection: some View {
        @Bindable var store = store
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Picker("", selection: $store.draft.useLayout) {
                    Text("Prompt").tag(false)
                    Text("Layout").tag(true)
                }
                .pickerStyle(.segmented).labelsHidden()
                .tweakable("generate.prompt.mode")
                enhanceButtons.tweakable("generate.prompt.enhance")
                if !store.draft.useLayout {
                    TextEditor(text: $store.draft.prompt)
                        .font(.body)
                        .resizableHeight("generate.prompt", default: 140, range: 70...700)
                        .overlay(alignment: .topLeading) {
                            if store.draft.prompt.isEmpty {
                                Text("Describe the image. Write visible text literally, in quotes — MOSS renders it.")
                                    .foregroundStyle(.tertiary).padding(6).allowsHitTesting(false)
                            }
                        }
                        .tweakable("generate.prompt.editor")
                }
            }
        } label: { Label("Prompt", systemImage: "text.cursor") }
    }

    /// On-device Qwen3.5 enhancer: rewrite the prompt, or draft a layout from it.
    private var enhanceButtons: some View {
        HStack {
            Button { store.enhance(layout: false) } label: { Label("Enhance", systemImage: "wand.and.stars") }
                .disabled(store.draft.useLayout || busy)
                .help("Rewrite the prompt as a detailed MOSS-Image caption (Qwen3.5, on device)")
            Button { store.enhance(layout: true) } label: { Label("Layout from prompt", systemImage: "rectangle.3.group") }
                .disabled(busy)
                .help("Draft text and object boxes on the layout canvas for the current size")
            Spacer()
            if store.isEnhancing { ProgressView().controlSize(.small) }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
    }

    private var busy: Bool {
        store.isEnhancing || store.isGenerating
            || (store.draft.useLayout ? store.layout.caption : store.draft.prompt).trimmingCharacters(in: .whitespaces).isEmpty
    }

    private var layoutSection: some View {
        @Bindable var store = store
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                LayoutCanvas(draft: $store.layout, aspectRatio: aspect, background: store.editBackground)
                    .resizableHeight("generate.canvas", default: 280, range: 160...800)
                    .onChange(of: store.draft.size) { _, size in store.layout.canvas = size }
                    .tweakable("generate.layout.canvas")
                LayoutInspector(draft: $store.layout).tweakable("generate.layout.inspector")
                HStack {
                    Button { addBox(.text) } label: { Label("Text box", systemImage: "textformat") }
                    Button { addBox(.object) } label: { Label("Object box", systemImage: "square.dashed") }
                    Spacer()
                    Text("\(store.layout.items.count) box\(store.layout.items.count == 1 ? "" : "es")").font(.caption).foregroundStyle(.secondary)
                }
                .tweakable("generate.layout.addBoxes")
            }
        } label: { Label("Layout (abs_centralized boxes)", systemImage: "square.grid.3x3.square") }
    }

    private func addBox(_ label: LayoutPrompt.Label) {
        let offset = Double(store.layout.items.count % 5) * 0.05
        store.layout.add(label: label, frame: .init(top: 0.1 + offset, left: 0.2 + offset, bottom: 0.3 + offset, right: 0.8))
    }

    private var aspect: Double { Double(store.draft.size.width) / Double(store.draft.size.height) }

    /// The Turbo picker's tag for "None — model is already 8-step" (never a library path).
    private static let alreadyDistilledTag = "@already-distilled"

    private var modelSection: some View {
        let model = Binding<String>(
            get: { store.selectedDiT?.relativePath ?? "" },
            set: { store.draft.model = $0; store.draft.steps = nil })
        let turbo = Binding<String>(
            get: { store.draft.alreadyDistilled ? Self.alreadyDistilledTag : store.activeTurbo?.relativePath ?? "" },
            set: {
                store.draft.alreadyDistilled = $0 == Self.alreadyDistilledTag
                store.draft.turboOff = $0.isEmpty || store.draft.alreadyDistilled
                if !store.draft.turboOff { store.draft.turbo = $0 }
                store.draft.steps = nil
            })
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                profileMenu.tweakable("generate.model.profile")
                if store.library.dits.isEmpty {
                    Text(store.isScanningLibrary ? "Scanning models…" : "No DiT in the library — import one on the Models screen.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Picker("Model", selection: model) {
                        ForEach(store.library.dits) { dit in
                            Text("\(dit.name)  ·  \(dit.sidecar.families.map(\.displayName).joined(separator: "/"))").tag(dit.relativePath)
                        }
                    }
                    .tweakable("generate.model.picker")
                    if let built = store.selectedDiT?.sidecar.distill {
                        LabeledContent("Turbo", value: "built in · \(built.steps) steps, CFG 1")
                    } else {
                    Picker("Turbo", selection: turbo) {
                        Text("Off — \(store.family.displayName) 50 steps, CFG").tag("")
                        Text("None — model is already \(DistillInfo.alreadyDistilled.steps)-step, CFG 1").tag(Self.alreadyDistilledTag)
                        ForEach(store.turboChoices) { t in
                            Text("\(t.name)  ·  \(t.sidecar.distill!.steps) steps").tag(t.relativePath)
                        }
                    }
                    .tweakable("generate.model.turbo")
                    }
                    if store.builtInTurbo == nil && store.turboChoices.isEmpty {
                        Text("No Turbo LoRA marked for \(store.family.displayName) — set one up on the Models screen.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        } label: { Label("Model", systemImage: "cube") }
    }

    /// One pick for model + Turbo + steps: built-in profiles (greyed out when their files are missing)
    /// and the user's own.
    private var profileMenu: some View {
        let library = store.libraryPaths
        return HStack {
            Menu {
                Section("Profiles") {
                    ForEach(GenerationProfile.builtIn) { p in profileButton(p, missing: p.missing(in: library)) }
                }
                if !store.settings.profiles.isEmpty {
                    Section("Yours") {
                        ForEach(store.settings.profiles) { p in profileButton(p, missing: p.missing(in: library)) }
                    }
                }
                Divider()
                Button("Save current as profile…") { profileName = ""; namingProfile = true }
                if !store.settings.profiles.isEmpty {
                    Menu("Delete") {
                        ForEach(store.settings.profiles) { p in Button(p.name, role: .destructive) { store.deleteProfile(p.id) } }
                    }
                }
            } label: {
                Label(store.activeProfile?.name ?? "Custom", systemImage: "slider.horizontal.below.square.filled.and.square")
            }
            .fixedSize()
            if let p = store.activeProfile {
                Text(p.detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
        }
        .alert("Save profile", isPresented: $namingProfile) {
            TextField("name", text: $profileName)
            Button("Save") { store.saveProfile(named: profileName) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Saves the current model, Turbo and steps.")
        }
    }

    private func profileButton(_ p: GenerationProfile, missing: [String]) -> some View {
        Button {
            store.apply(p)
        } label: {
            Text(missing.isEmpty ? p.name : "\(p.name) — missing \(missing.map { ($0 as NSString).lastPathComponent }.joined(separator: ", "))")
        }
        .disabled(!missing.isEmpty)
    }

    private var sizeSection: some View {
        @Bindable var store = store
        return GroupBox {
            if let exact = store.draft.exactSize {
                Text("\(exact.width)×\(exact.height) — the original's size while editing").monospacedDigit().foregroundStyle(.secondary)
            } else {
            HStack {
                Picker("Pixels", selection: $store.draft.pixels) {
                    ForEach(MossDimensions.pixelBudgets, id: \.self) { p in
                        Text(String(format: "%.2g MP", Double(p) / 1_048_576)).tag(p)
                    }
                }
                .tweakable("generate.size.pixels")
                Picker("Ratio", selection: $store.draft.ratio) {
                    ForEach(MossDimensions.ratios, id: \.self) { Text($0).tag($0) }
                }
                .tweakable("generate.size.ratio")
                Text("\(store.draft.size.width)×\(store.draft.size.height)").monospacedDigit().foregroundStyle(.secondary)
            }
            }
        } label: { Label("Size", systemImage: "aspectratio") }
    }

    private var samplingSection: some View {
        @Bindable var store = store
        let distill = store.builtInTurbo ?? store.activeTurbo?.sidecar.distill
        let turbo = distill != nil
        let defaultSteps = distill?.steps ?? store.family.variant.defaults.steps
        let steps = Binding(get: { Double(store.draft.steps ?? defaultSteps) }, set: { store.draft.steps = Int($0) })
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                slider("Steps", steps, turbo ? 1...Double(max(8, defaultSteps)) : 10...100, step: 1, format: "%.0f")
                if !turbo {
                    slider("CFG", $store.draft.cfg, 1...8, step: 0.5, format: "%.1f")
                    slider("Shift", $store.draft.shift, 1...10, step: 0.5, format: "%.1f")
                    Toggle("Skip block 22 in the negative branch (official SLG)", isOn: $store.draft.skipBlock22)
                        .tweakable("generate.sampling.slg")
                    TextField("Negative prompt", text: $store.draft.negativePrompt, axis: .vertical).lineLimit(1...3)
                        .tweakable("generate.sampling.negative")
                }
                HStack {
                    Toggle("Lock seed", isOn: $store.draft.lockSeed)
                    TextField("seed", value: $store.draft.seed, format: .number.grouping(.never))
                        .frame(width: 140).disabled(!store.draft.lockSeed)
                }
                .tweakable("generate.sampling.seed")
            }
        } label: { Label("Sampling", systemImage: "dial.medium") }
    }

    private func slider(_ name: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double, format: String) -> some View {
        HStack {
            Text(name).frame(width: 46, alignment: .leading)
            Slider(value: value, in: range, step: step)
            Text(String(format: format, value.wrappedValue)).monospacedDigit().frame(width: 36, alignment: .trailing)
        }
        .tweakable("generate.sampling.\(name.lowercased())")
    }

    /// 0.05 on the usual ±2 slider; coarser on wide ones (±10 → 0.25) so it doesn't crawl.
    static func strengthStep(_ range: SliderRange) -> Double {
        let span = range.max - range.min
        return span <= 4 ? 0.05 : span <= 10 ? 0.1 : 0.25
    }

    private var loraSection: some View {
        @Bindable var store = store
        return GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                ForEach($store.draft.loras) { $entry in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(entry.url.deletingPathExtension().lastPathComponent).lineLimit(1).truncationMode(.middle)
                            Spacer()
                            // Capture the id first: reading `entry` (a binding into `store.draft`) inside the
                            // mutating `removeAll` closure is an exclusivity violation — it crashed the app.
                            Button {
                                let id = entry.id
                                store.draft.loras.removeAll { $0.id == id }
                            } label: { Image(systemName: "xmark.circle") }
                                .buttonStyle(.borderless)
                        }
                        HStack {
                            let range = store.loraSidecar(entry.url).sliderRange
                            Slider(value: $entry.strength, in: range.min...range.max, step: Self.strengthStep(range))
                                .tweakable("generate.loras.strength")
                            // Typed values may go past the slider's ends (the engine doesn't clamp).
                            TextField("1.0", value: $entry.strength, format: .number.precision(.fractionLength(0...2)))
                                .monospacedDigit().multilineTextAlignment(.trailing)
                                .frame(width: 52).tweakable("generate.loras.value")
                            TextField("blocks e.g. 21-41", text: $entry.blocks).frame(width: 130)
                        }
                    }
                    .tweakable("generate.loras.row")
                    Divider()
                }
                Menu {
                    ForEach(store.availableLoRAs, id: \.self) { url in
                        Button(url.deletingPathExtension().lastPathComponent) {
                            store.draft.loras.append(.init(url: url, strength: store.loraSidecar(url).strength ?? 1))
                        }
                    }
                } label: { Label("Add LoRA", systemImage: "plus") }
                .disabled(store.availableLoRAs.isEmpty)
                .tweakable("generate.loras.add")
            }
        } label: { Label("LoRAs", systemImage: "slider.horizontal.3") }
    }

    // MARK: - Preview

    /// Right-click on the result (Prompt and Layout alike): share, save, copy, edit, upscale, find, trash.
    @ViewBuilder private var previewMenu: some View {
        if let url = store.lastImageURL {
            ImageItemMenu(url: url, onUpscale: { upscaling = $0 })
        }
    }

    private var preview: some View {
        VStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4))
                if let image = store.lastImage {
                    Image(nsImage: image).resizable().interpolation(.high).aspectRatio(contentMode: .fit).padding(8)
                        .contextMenu { previewMenu }
                } else {
                    Text("Your image appears here").foregroundStyle(.secondary)
                }
            }
            .tweakable("generate.preview.image")
            if let p = store.progress {
                ProgressView(value: p.fraction) { Text(p.message).font(.caption) }
            }
            if let err = store.lastError { Text(err).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            HStack {
                // Verbatim: a seed is an identifier, not a quantity (no thousands separators).
                if let seed = store.lastSeed {
                    Text(verbatim: "seed \(seed)").font(.callout).monospacedDigit().foregroundStyle(.secondary).textSelection(.enabled)
                        .tweakable("generate.preview.seed")
                }
                Spacer()
                if let url = store.lastImageURL, !store.isGenerating {
                    ShareLink(item: url) { Label("Share", systemImage: "square.and.arrow.up") }
                        .help("Share the image")
                        .tweakable("generate.preview.share")
                    Button { ImageActions.saveCopies([url]) } label: { Label("Save…", systemImage: "square.and.arrow.down") }
                        .help("Save a copy…")
                        .tweakable("generate.preview.save")
                }
                if store.isGenerating {
                    Button("Cancel", role: .cancel) { store.cancel() }
                }
                Button { store.generate() } label: {
                    Label(store.editing != nil ? "Redraw changes" : "Generate", systemImage: store.editing != nil ? "pencil.and.outline" : "sparkles").frame(minWidth: 110)
                }
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .disabled(store.isGenerating || store.isEnhancing || (!store.draft.useLayout && store.draft.prompt.trimmingCharacters(in: .whitespaces).isEmpty))
                    .tweakable("generate.preview.generate")
            }
            .tweakable("generate.preview.actions")
        }
        .padding(16)
    }
}
