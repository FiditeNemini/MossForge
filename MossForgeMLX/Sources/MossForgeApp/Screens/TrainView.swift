import AppKit
import Charts
import MossForgeCore
import MossForgeEngine
import SwiftUI
import Tweakbench

/// Train: a LoRA from a folder of images with `name.txt` captions, on a library DiT (Base by default —
/// MOSS's recommendation; the text encoder stays frozen). Settings on the left; progress, the loss
/// curve and the result on the right.
struct TrainView: View {
    @Environment(AppStore.self) private var store
    @State private var captioning: BatchCaptionJob?
    @State private var enlarged: URL?

    var body: some View {
        PersistentHSplit(key: "train", sized: .leading, defaultWidth: 440, range: 380...720, otherMin: 420) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Picker("", selection: Bindable(store).trainDraft.mode) {
                        ForEach(TrainingMode.allCases) { Text($0.displayName).tag($0) }
                    }
                    .pickerStyle(.segmented).labelsHidden()
                    .onChange(of: store.trainDraft.mode) { _, mode in
                        // Sliders converge fast; 300 steps at 1e-4 is the tested default.
                        if mode == .slider, store.trainDraft.steps > 500 { store.trainDraft.steps = 300 }
                    }
                    .tweakable("train.mode")
                    switch store.trainDraft.mode {
                    case .dataset: datasetSection.tweakable("train.dataset")
                    case .slider: sliderSection.tweakable("train.slider")
                    case .edit: editDatasetSection.tweakable("train.editPairs")
                    }
                    modelSection.tweakable("train.model")
                    if store.trainDraft.mode == .dataset { profileSection.tweakable("train.loraType") }
                    settingsSection.tweakable("train.settings")
                    testPromptsSection.tweakable("train.testPrompts")
                }
                .padding(16)
            }
            .tweakable("train.form")
        } trailing: {
            progressPane.frame(maxWidth: .infinity, maxHeight: .infinity)
                .tweakable("train.progress")
        }
        .sheet(item: $captioning) { BatchCaptionSheet(job: $0) }
        .sheet(item: $enlarged) { url in SampleViewer(url: url, rows: store.trainSampleRows, prompts: store.trainDraft.testPrompts) }
    }

    // MARK: form

    private var dataset: [TrainingSample] { store.trainDraft.dataset.map { DatasetScanner.scan(directory: $0) } ?? [] }

    private var datasetSection: some View {
        let samples = dataset
        let uncaptioned = store.trainDraft.dataset.map { FolderCaptioning.pending(in: $0, overwrite: false).count } ?? 0
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(store.trainDraft.dataset?.path ?? "Choose a folder of images + name.txt captions")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(2).truncationMode(.middle)
                    Spacer()
                    Button("Choose…") { pickDataset() }
                }
                .tweakable("train.dataset.folder")
                if store.trainDraft.dataset != nil {
                    HStack {
                        Text("\(samples.count) images · \(samples.count - uncaptioned) captioned")
                        Spacer()
                        if uncaptioned > 0 {
                            Button("Caption \(uncaptioned) missing…") { captioning = .folder(store.trainDraft.dataset!) }
                                .disabled(store.captionBatch != nil || store.isTraining)
                        }
                    }
                    .font(.callout).tweakable("train.dataset.count")
                    if uncaptioned > 0 {
                        Text("Images without a .txt train on their file name — caption them first (on-device, ~15–20 s each).")
                            .font(.caption).foregroundStyle(.orange)
                    }
                    if !samples.isEmpty {
                        ScrollView(.horizontal) {
                            HStack(spacing: 6) {
                                ForEach(samples.prefix(30), id: \.image) { s in
                                    if let img = NSImage(contentsOf: s.image) {
                                        Image(nsImage: img).resizable().aspectRatio(contentMode: .fill)
                                            .frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 4))
                                            .help(s.caption)
                                    }
                                }
                            }
                        }
                        .tweakable("train.dataset.thumbnails")
                    }
                }
            }
        } label: { Label("Dataset", systemImage: "photo.stack") }
    }

    private var modelSection: some View {
        @Bindable var store = store
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Train on", selection: Binding(
                    get: { store.trainBase?.relativePath ?? "" }, set: { store.trainDraft.model = $0 })
                ) {
                    ForEach(store.library.dits) { d in
                        Text("\(d.name) · \(d.sidecar.families.map(\.displayName).joined(separator: "/"))").tag(d.relativePath)
                    }
                }
                .tweakable("train.model.base")
                if store.trainBase?.sidecar.supports(.base) == false {
                    Text("MOSS recommends training on Base; LoRAs trained on SFT work only with SFT.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Text("Name")
                    TextField("my-lora", text: $store.trainDraft.name)
                }
                .tweakable("train.model.name")
                if let out = store.trainOutput {
                    Text("→ lora/trained/\(out.lastPathComponent)" + (FileManager.default.fileExists(atPath: out.path) ? " (exists — will be replaced)" : ""))
                        .font(.caption).foregroundStyle(.secondary)
                }
                if store.trainDraft.mode != .slider {
                    HStack {
                        Text("Trigger word")
                        TextField("optional, e.g. mossfox", text: $store.trainDraft.trigger)
                    }
                    .tweakable("train.model.trigger")
                    Text("Prefixed to every caption as “trigger, caption” — use it in prompts to call the LoRA up.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        } label: { Label("Model", systemImage: "cube") }
    }

    // MARK: LoRA type

    private var usesProfile: Bool { store.trainDraft.mode == .dataset && store.trainDraft.profile != .custom }

    /// Character / Concept / Style / Custom: a type fills rank,
    /// layers, LR and steps scaled to the dataset; the preset below stays the run length.
    private var profileSection: some View {
        @Bindable var store = store
        let n = dataset.count
        let profile = store.trainDraft.profile
        return GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                Picker("", selection: $store.trainDraft.profile) {
                    ForEach(LoRATrainingProfile.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden()
                .tweakable("train.loraType.picker")
                .onChange(of: store.trainDraft.profile) { _, _ in store.trainDraft.applyProfile(imageCount: n) }
                .onChange(of: n) { old, _ in
                    // Re-scale the steps to a new dataset unless they've been tuned by hand.
                    if let before = store.trainDraft.profileSettings(imageCount: old), store.trainDraft.steps == before.steps {
                        store.trainDraft.applyProfile(imageCount: n)
                    }
                }
                Text(profile.summary).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if profile != .custom {
                    let band = profile.recommendedImages
                    Text(n == 0
                        ? "Aim for \(band.lowerBound)–\(band.upperBound) images."
                        : "\(n) images · aim for \(band.lowerBound)–\(band.upperBound). Steps scale with the dataset "
                            + "(≈ \(max(1, store.trainDraft.steps / max(n, 1))) epochs).")
                        .font(.caption).foregroundStyle(n > 0 && n < band.lowerBound ? .orange : .secondary)
                    Label(profile.captionTip, systemImage: "text.bubble")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    if store.trainDraft.profileModified(imageCount: n) {
                        HStack(spacing: 6) {
                            Text("Modified").font(.caption).foregroundStyle(.secondary)
                            Button("Reset") { store.trainDraft.applyProfile(imageCount: n) }.controlSize(.small)
                        }
                    }
                    Text("Community starting points for Flux-class DiTs — not yet measured on MOSS-Image.")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
        } label: { Label("LoRA type", systemImage: "person.crop.square.badge.camera") }
    }

    private var settingsSection: some View {
        @Bindable var store = store
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Picker("Preset", selection: $store.trainDraft.preset) { ForEach(AppStore.TrainDraft.Preset.allCases) { Text($0.rawValue).tag($0) } }
                        .fixedSize()
                    Button("Apply") { store.trainDraft.apply(store.trainDraft.preset, imageCount: dataset.count) }
                }
                .tweakable("train.settings.preset")
                Text(usesProfile ? store.trainDraft.preset.typeDetail : store.trainDraft.preset.detail)
                    .font(.caption).foregroundStyle(.secondary)
                Divider()
                Stepper("Steps: \(store.trainDraft.steps)", value: $store.trainDraft.steps, in: 50...20000, step: 50)
                    .tweakable("train.settings.steps")
                Picker("Adapter", selection: $store.trainDraft.factorization) {
                    ForEach(LoRAFactorization.allCases) { Text($0.displayName).tag($0) }
                }
                .pickerStyle(.segmented)
                .tweakable("train.settings.adapter")
                if store.trainDraft.factorization == .lora {
                    Picker("Rank", selection: $store.trainDraft.rank) { ForEach([4, 8, 16, 32, 64, 128], id: \.self) { Text("\($0)").tag($0) } }
                        .tweakable("train.settings.rank")
                } else {
                    Picker("Kronecker factor", selection: $store.trainDraft.lokrFactor) {
                        Text("Auto (balanced)").tag(-1)
                        ForEach([4, 8, 16], id: \.self) { Text("\($0)").tag($0) }
                    }
                    Text("LoKr: ΔW = w1 ⊗ w2 (LyCORIS). Often more expressive per parameter; smaller factors = smaller file.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Picker("Resolution", selection: $store.trainDraft.resolution) {
                    ForEach([512, 768, 1024], id: \.self) { Text("≤ \($0)²").tag($0) }
                }
                .tweakable("train.settings.resolution")
                Picker("Train", selection: $store.trainDraft.targets) {
                    ForEach(LoRATargetSet.allCases) { Text($0.displayName).tag($0) }
                }
                .pickerStyle(.segmented)
                .tweakable("train.settings.targets")
                HStack {
                    Text("Learning rate")
                    TextField("1e-4", value: $store.trainDraft.learningRate, format: .number).frame(width: 90)
                    Spacer()
                    Stepper("Save every \(store.trainDraft.saveEvery)", value: $store.trainDraft.saveEvery, in: 0...5000, step: 50)
                }
                .tweakable("train.settings.learningRate")
                if store.trainDraft.mode == .dataset {
                    HStack {
                        Text("Caption dropout")
                        Slider(value: $store.trainDraft.captionDropout, in: 0...0.3, step: 0.05)
                        Text(String(format: "%.0f%%", store.trainDraft.captionDropout * 100)).monospacedDigit().frame(width: 40, alignment: .trailing)
                    }
                    .help("Share of steps trained with an empty caption — keeps the LoRA working with prompts unlike its captions. 5–10% is typical.")
                    .tweakable("train.settings.captionDropout")
                }
                Toggle("Save memory (gradient checkpointing)", isOn: $store.trainDraft.checkpointing)
                    .tweakable("train.settings.checkpointing")
                    .help("Recomputes each block in the backward pass. Same results; about half the memory — and usually faster at 1024², where training otherwise runs into memory pressure.")
                Text("α = rank. Each image trains at its own aspect, scaled to the resolution. Checkpoints go to <name>-checkpoints/.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } label: { Label("Training", systemImage: "dial.medium") }
    }

    /// Experimental edit LoRA: targets + instructions in the folder, sources with the same names in
    /// `control/` (AI Toolkit's layout). The source rides along as reference tokens on RoPE r = 1.
    private var editDatasetSection: some View {
        let scan = store.trainDraft.dataset.map { EditDatasetScanner.scan(directory: $0) }
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(store.trainDraft.dataset?.path ?? "Choose a folder: edited targets + name.txt instructions, sources in control/")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(2).truncationMode(.middle)
                    Spacer()
                    Button("Choose…") { pickDataset() }
                }
                if let scan {
                    Text("\(scan.pairs.count) pairs" + (scan.missingControl.isEmpty ? "" : " · \(scan.missingControl.count) targets without a control image (skipped)"))
                        .font(.callout)
                    if let first = scan.pairs.first {
                        HStack(spacing: 6) {
                            thumb(first.control)
                            Image(systemName: "arrow.right").foregroundStyle(.secondary)
                            thumb(first.target)
                            Text(first.instruction).font(.caption).lineLimit(3)
                        }
                    }
                }
                Text("Experimental (community method): the source image is fed as clean reference tokens and the LoRA learns the instruction. Needs hundreds of pairs to work well; use the result in Generate with a reference image. Test images use the first pair's source.")
                    .font(.caption).foregroundStyle(.orange)
            }
        } label: { Label("Edit pairs", systemImage: "arrow.left.and.right.square") }
    }

    private func thumb(_ url: URL) -> some View {
        Group {
            if let img = NSImage(contentsOf: url) {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fill)
            } else {
                Rectangle().fill(.quaternary)
            }
        }
        .frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 4))
    }

    /// Concept Slider (no images): the wizard drafts target / positive / negative from a description;
    /// the LoRA learns the direction between them, usable at any weight across the range.
    private var sliderSection: some View {
        @Bindable var store = store
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("Describe it, e.g. “a young to old slider”", text: $store.trainDraft.sliderDescription)
                        .onSubmit { store.draftSlider() }
                    Button { store.draftSlider() } label: { Label("Suggest", systemImage: "wand.and.stars") }
                        .disabled(store.trainDraft.sliderDescription.trimmingCharacters(in: .whitespaces).isEmpty || store.isDraftingSlider)
                    if store.isDraftingSlider { ProgressView().controlSize(.small) }
                }
                .tweakable("train.slider.describe")
                Text("Qwen3.5 fills the fields below (“X to Y”: X is −, Y is +). Review them before training.")
                    .font(.caption).foregroundStyle(.secondary)
                Divider()
                labeled("Target", "the neutral subject, e.g. a portrait of a person", $store.trainDraft.sliderTarget)
                labeled("Positive (+)", "descriptors, e.g. old, grey hair", $store.trainDraft.sliderPositive)
                labeled("Negative (−)", "descriptors, e.g. young, smooth skin", $store.trainDraft.sliderNegative)
                labeled("Preserve", "optional, e.g. male, female", $store.trainDraft.sliderPreserve)
                HStack {
                    Text("Range")
                    TextField("", value: $store.trainDraft.sliderMin, format: .number).frame(width: 50)
                    Text("…")
                    TextField("", value: $store.trainDraft.sliderMax, format: .number).frame(width: 50)
                    Spacer()
                    Text("Strength η")
                    TextField("", value: $store.trainDraft.sliderEta, format: .number).frame(width: 50)
                }
                .tweakable("train.slider.range")
                Stepper("Latent steps: \(store.trainDraft.sliderLatentSteps)", value: $store.trainDraft.sliderLatentSteps, in: 2...30)
                    .tweakable("train.slider.latentSteps")
                let range = SliderRange(min: store.trainDraft.sliderMin, max: store.trainDraft.sliderMax)
                Text(range.isValid
                    ? String(format: "Trained for weights %g…%g; use it at %g…%g (sliders usually need the extra push on the + side). η 1 keeps the rest of the image steady — raise it only if +2 is still too subtle. Preserve keeps listed attributes out of the direction.",
                        range.min, range.max, range.recommendedBand.min, range.recommendedBand.max)
                    : "The range must contain 0 (0 = off).")
                    .font(.caption).foregroundStyle(range.isValid ? Color.secondary : Color.orange)
            }
        } label: { Label("Slider", systemImage: "slider.horizontal.3") }
    }

    private func labeled(_ title: String, _ prompt: String, _ text: Binding<String>) -> some View {
        HStack {
            Text(title).frame(width: 84, alignment: .leading)
            TextField(prompt, text: text)
        }
        .tweakable("train.slider.field")
    }

    /// Up to three prompts rendered before training and at every checkpoint (typed, or suggested by
    /// Qwen3.5 from the dataset's captions), so the LoRA's effect can be watched as it trains.
    private var testPromptsSection: some View {
        @Bindable var store = store
        return GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(0..<(store.trainDraft.mode == .slider ? 1 : TrainingSamples.maxPrompts), id: \.self) { i in
                    HStack(alignment: .top) {
                        Text("\(i + 1)").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 12).padding(.top, 4)
                        TextField("test prompt \(i + 1)", text: $store.trainDraft.testPrompts[i], axis: .vertical).lineLimit(1...3)
                            .tweakable("train.testPrompts.prompt")
                    }
                }
                if store.trainDraft.mode != .slider {
                HStack {
                    Button { store.suggestTestPrompts() } label: { Label("Suggest from captions", systemImage: "wand.and.stars") }
                        .disabled(store.trainDraft.dataset == nil || store.isSuggestingPrompts || store.isTraining)
                        .help(store.canChat ? "Qwen3.5 reads the captions and writes three test prompts" : "Picks three captions from the dataset")
                    if store.isSuggestingPrompts { ProgressView().controlSize(.small) }
                    Spacer()
                    if store.trainDraft.testPrompts.contains(where: { !$0.isEmpty }) {
                        Button("Clear") { store.trainDraft.testPrompts = ["", "", ""] }
                    }
                }
                .tweakable("train.testPrompts.suggest")
                }
                HStack {
                    Stepper("\(store.trainDraft.testSteps) steps", value: $store.trainDraft.testSteps, in: 4...50, step: 2)
                    Picker("Size", selection: $store.trainDraft.testResolution) {
                        ForEach([384, 512, 768], id: \.self) { Text("\($0)²").tag($0) }
                    }
                    .fixedSize()
                }
                .tweakable("train.testPrompts.options")
                Text(store.trainDraft.mode == .slider
                    ? "One prompt (empty = the target) at the range's − end, 0 and + end — before training, at every save and at the end, same seed. Each set adds about \(estimate)."
                    : "Rendered before training, at every save and at the end — same seed each time, so only the LoRA changes. "
                        + "The trigger word is added. Each set adds about \(estimate) to the run.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } label: { Label("Test images", systemImage: "photo.on.rectangle.angled") }
    }

    /// Rough time per set of test images (CFG = two passes per step; measured ≈0.12 s/pass at 512² on an M2 Ultra).
    private var estimate: String {
        let n = store.trainDraft.mode == .slider ? 3 : max(1, store.trainDraft.testPrompts.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count)
        let side = Double(store.trainDraft.testResolution) / 512
        let seconds = Double(n * store.trainDraft.testSteps * 2) * 0.12 * side * side + Double(n)
        return seconds < 60 ? String(format: "%.0f s", seconds) : String(format: "%.1f min", seconds / 60)
    }

    private func pickDataset() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.message = "Choose a folder of training images (each with an optional name.txt caption)"
        if panel.runModal() == .OK, let dir = panel.url {
            store.trainDraft.dataset = dir
            if store.trainDraft.name.isEmpty { store.trainDraft.name = dir.lastPathComponent }
        }
    }

    // MARK: progress

    /// Loss on top, test images below, with a draggable splitter between them (each starts at about
    /// half the height); status and the Train / Stop buttons stay pinned underneath.
    private var progressPane: some View {
        VStack(alignment: .leading, spacing: 12) {
            VSplitView {
                lossPane
                    .padding(.bottom, 8)
                    .frame(minHeight: 160, idealHeight: 300, maxHeight: .infinity)
                    .tweakable("train.loss")
                samplesPane
                    .padding(.top, 8)
                    .frame(minHeight: 160, idealHeight: 300, maxHeight: .infinity)
                    .tweakable("train.samples")
            }
            if !store.trainStatus.isEmpty {
                Text(store.trainStatus).font(.callout).textSelection(.enabled).tweakable("train.status")
            }
            HStack {
                if let url = store.lastTrained, !store.isTraining {
                    Button("Try in Generate") { store.tryInGenerate(url) }.tweakable("train.tryInGenerate")
                    Button("Show in Models") { store.selection = .models }.tweakable("train.showInModels")
                }
                Spacer()
                if store.isTraining {
                    Button("Stop & save", role: .cancel) { store.stopTraining() }
                } else {
                    Button { store.startTraining() } label: { Label("Train", systemImage: "dumbbell").frame(minWidth: 100) }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!store.canStartTraining)
                        .tweakable("train.trainButton")
                }
            }
        }
        .padding(16)
    }

    private var lossPane: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 4) {
                Group {
                    if store.trainLosses.isEmpty {
                        ZStack {
                            RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4))
                            Text(store.isTraining ? "Preparing — encoding captions and images…" : "The loss curve appears here while training")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    } else {
                        lossChart
                    }
                }
                .frame(maxHeight: .infinity).tweakable("train.loss.chart")
                ProgressView(value: store.trainProgress?.fraction ?? (store.trainLosses.isEmpty ? 0 : 1))
                    .tweakable("train.loss.progress")
                HStack {
                    Text(store.trainProgress?.message ?? "—")
                    Spacer()
                    if let last = store.trainLosses.last {
                        Text(String(format: "loss %.4f · avg %.4f", last, Self.ema(store.trainLosses).last ?? last))
                    }
                }
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .tweakable("train.loss.readout")
            }
        } label: { Label("Loss", systemImage: "chart.xyaxis.line") }
    }

    private var samplesPane: some View {
        GroupBox {
            if store.trainSampleRows.isEmpty {
                ZStack {
                    RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4))
                    Text("Test images appear here before training starts and at each save — click one to inspect it at full size.")
                        .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center).padding()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                sampleRows
            }
        } label: { Label("Test images", systemImage: "photo.on.rectangle.angled") }
    }

    /// Exponential moving average (α 0.05) — the trend through the noisy per-step losses.
    static func ema(_ values: [Double]) -> [Double] {
        var out: [Double] = []
        out.reserveCapacity(values.count)
        for v in values { out.append(out.last.map { 0.95 * $0 + 0.05 * v } ?? v) }
        return out
    }

    /// Per-step loss (faint — σ is random each step, so it's noisy) and its moving average.
    private var lossChart: some View {
        let losses = store.trainLosses
        let ema = Self.ema(losses)
        return Chart {
            ForEach(Array(losses.enumerated()), id: \.offset) { i, l in
                PointMark(x: .value("Step", i + 1), y: .value("Loss", l))
                    .symbolSize(8).foregroundStyle(Color.secondary.opacity(0.35))
            }
            ForEach(Array(ema.enumerated()), id: \.offset) { i, l in
                LineMark(x: .value("Step", i + 1), y: .value("Average", l))
                    .foregroundStyle(Color.accentColor).lineStyle(StrokeStyle(lineWidth: 2))
            }
            ForEach(store.trainSampleRows.filter { $0.step > 0 }) { row in
                RuleMark(x: .value("Checkpoint", row.step))
                    .foregroundStyle(Color.secondary.opacity(0.25)).lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
        }
        .chartXScale(domain: 0...max(1, store.trainProgress?.totalSteps ?? max(losses.count, store.trainDraft.steps)))
        .chartXAxisLabel("step")
        .chartYAxisLabel("loss")
        .frame(maxHeight: .infinity)
    }

    /// One row per checkpoint: the step, then each test prompt's image (click to enlarge, right-click to share).
    private var sampleRows: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if store.trainDraft.mode == .slider {
                        HStack(spacing: 8) {
                            Text("weight").font(.caption).foregroundStyle(.secondary).frame(width: 58, alignment: .leading)
                            ForEach([store.trainDraft.sliderMin, store.trainDraft.sliderMin < 0 && store.trainDraft.sliderMax > 0 ? 0 : (store.trainDraft.sliderMin + store.trainDraft.sliderMax) / 2, store.trainDraft.sliderMax], id: \.self) { w in
                                Text(String(format: "%+g", w)).font(.caption.monospacedDigit()).frame(maxWidth: 200)
                            }
                        }
                    }
                    ForEach(store.trainSampleRows) { row in
                        HStack(alignment: .top, spacing: 8) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(row.step == 0 ? "Before" : "Step").font(.caption).foregroundStyle(.secondary)
                                Text(row.step == 0 ? "training" : "\(row.step)").font(.callout.monospacedDigit().weight(.semibold))
                            }
                            .frame(width: 58, alignment: .leading)
                            ForEach(row.images, id: \.self) { url in
                                SampleThumb(url: url)
                                    .onTapGesture { enlarged = url }
                                    .contextMenu { ImageShareMenuItems(images: [url]) }
                            }
                        }
                        .id(row.step)
                    }
                }
                .padding(.vertical, 4)
            }
            .onChange(of: store.trainSampleRows.count) { _, _ in
                if let last = store.trainSampleRows.last { withAnimation { proxy.scrollTo(last.step, anchor: .bottom) } }
            }
        }
        .frame(maxHeight: .infinity)
    }
}

/// A square test image, loaded off disk (re-read when the file changes between runs).
private struct SampleThumb: View {
    let url: URL
    var body: some View {
        Group {
            if let img = NSImage(contentsOf: url) {
                Image(nsImage: img).resizable().aspectRatio(1, contentMode: .fit)
            } else {
                Rectangle().fill(.quaternary).aspectRatio(1, contentMode: .fit)
            }
        }
        .frame(maxWidth: 200)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
    }
}

/// Click a test image: a modal for close inspection — fit to the window, or 100–400% pixels in a
/// scrollable view — with the same prompt at every checkpoint along the bottom, so the point where
/// the LoRA stops improving (or starts to overcook) is easy to spot.
private struct SampleViewer: View {
    let url: URL
    let rows: [TrainingSamples.Row]
    let prompts: [String]
    @Environment(\.dismiss) private var dismiss
    @State private var current: URL?
    /// 0 = fit to the window; otherwise pixel scale (1 = actual pixels).
    @State private var zoom: Double = 0

    var body: some View {
        let shown = current ?? url
        let id = TrainingSamples.parse(shown)
        let column = rows.compactMap { row in row.images.first { TrainingSamples.parse($0)?.index == id?.index } }
        let image = NSImage(contentsOf: shown)
        let pixels = image.flatMap { $0.representations.first }.map { CGSize(width: $0.pixelsWide, height: $0.pixelsHigh) }
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(id.map { $0.step == 0 ? "Before training" : "Step \($0.step)" } ?? "").font(.headline)
                if let i = id?.index, i < prompts.count { Text(prompts[i]).foregroundStyle(.secondary).lineLimit(2) }
                Spacer()
                if let pixels { Text("\(Int(pixels.width))×\(Int(pixels.height))").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                Picker("", selection: $zoom) {
                    Text("Fit").tag(0.0)
                    Text("100%").tag(1.0)
                    Text("200%").tag(2.0)
                    Text("400%").tag(4.0)
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
                Button { NSWorkspace.shared.activateFileViewerSelecting([shown]) } label: { Image(systemName: "folder") }
                    .help("Show in Finder")
                ShareLink(item: shown) { Image(systemName: "square.and.arrow.up") }
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Group {
                if let image {
                    if zoom == 0 {
                        Image(nsImage: image).resizable().interpolation(.high).aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        // Point size = pixel size × zoom ÷ the screen scale, so 100% is one image pixel per screen pixel.
                        let px = pixels ?? image.size
                        let screen = NSScreen.main?.backingScaleFactor ?? 2
                        ScrollView([.horizontal, .vertical]) {
                            Image(nsImage: image).resizable().interpolation(.none)
                                .frame(width: px.width * zoom / screen, height: px.height * zoom / screen)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                } else {
                    Text("Couldn't read \(shown.lastPathComponent)").foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .background(Color.black.opacity(0.25))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(column, id: \.self) { u in
                        VStack(spacing: 2) {
                            SampleThumb(url: u).frame(width: 90)
                                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(u == shown ? Color.accentColor : .clear, lineWidth: 2))
                            Text(TrainingSamples.parse(u).map { $0.step == 0 ? "before" : "\($0.step)" } ?? "").font(.caption2.monospacedDigit())
                        }
                        .onTapGesture { current = u }
                    }
                }
            }
            .frame(height: 120)
        }
        .padding(16)
        .frame(minWidth: 900, idealWidth: 1100, minHeight: 800, idealHeight: 1000)
    }
}
