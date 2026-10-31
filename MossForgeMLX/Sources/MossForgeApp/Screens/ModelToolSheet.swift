import MossForgeCore
import MossForgeEngine
import SwiftUI

/// Models → a DiT → Tools: convert / quantize it, fuse LoRAs into it, or merge it with other DiTs.
/// The new file lands beside it in diffusion_models with a sidecar (family; a fused Turbo LoRA
/// becomes a built-in recipe, so Generate runs it at its steps and CFG 1 with no LoRA). Extract LoRA
/// instead writes `lora/extracted/<name>.safetensors`: the DiT's difference from its parent.
struct ModelToolSheet: View {
    enum Mode: String, CaseIterable, Identifiable {
        case convert = "Convert", fuse = "Fuse LoRAs", merge = "Merge", extract = "Extract LoRA"
        var id: String { rawValue }
    }

    @Environment(AppStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let dit: ModelItem
    let onDone: (String) -> Void

    @State private var mode: Mode
    @State private var precision: WeightPrecision = .bfloat16
    @State private var picked: [String: Double] = [:]  // LoRA id → strength
    @State private var modelB = ""
    @State private var modelC = ""
    @State private var mergeMode: ModelMergeMode = .weightedSum
    @State private var alpha = 0.5
    @State private var name = ""
    @State private var parentID = ""
    @State private var rank = 64
    @State private var cutoffKind = CutoffKind.fixed
    @State private var cutoffValue = 0.9

    enum CutoffKind: String, CaseIterable, Identifiable {
        case fixed = "Fixed rank", energy = "Energy", threshold = "Threshold"
        var id: String { rawValue }
    }

    init(dit: ModelItem, mode: Mode, onDone: @escaping (String) -> Void) {
        self.dit = dit
        self.onDone = onDone
        _mode = State(initialValue: mode)
    }

    private var family: MossFamily { dit.sidecar.families.first ?? .sft }
    private var sourcePrecision: WeightPrecision? {
        dit.summary.split(separator: "·").last.flatMap { WeightPrecision(rawValue: $0.trimmingCharacters(in: .whitespaces)) }
    }
    private var sourceIsQuantized: Bool { sourcePrecision.map(MossExport.isQuantized) ?? false }
    private var loras: [ModelItem] { store.library.loras.filter { $0.sidecar.supports(family) } }
    private var pickedSpecs: [LoRASpec] {
        loras.filter { picked[$0.id] != nil }.map { LoRASpec(url: $0.url, strength: picked[$0.id]!) }
    }
    private var others: [ModelItem] { store.library.dits.filter { $0.id != dit.id } }
    private var parent: ModelItem? { others.first { $0.id == parentID } }
    private var cutoff: LoRARankCutoff {
        switch cutoffKind {
        case .fixed: .fixed
        case .energy: .energy(cutoffValue)
        case .threshold: .threshold(cutoffValue)
        }
    }
    private var precisions: [WeightPrecision] { mode == .extract ? [.bfloat16, .float16, .float32] : MossExport.precisions }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(dit.name).font(.title3.bold())
            Picker("", selection: $mode) { ForEach(Mode.allCases) { Text($0.rawValue).tag($0) } }
                .pickerStyle(.segmented).labelsHidden()
                .disabled(store.isRunningTool)

            Group {
                switch mode {
                case .convert: Text("Re-encode this DiT at another precision. Quantized formats keep the 168 block linears quantized and everything else bf16, like the Comfy-Org files.")
                case .fuse: fuseForm
                case .merge: mergeForm
                case .extract: extractForm
                }
            }
            .font(.callout)

            Picker("Output", selection: $precision) {
                // DiT sizes don't apply to a LoRA: just the format.
                ForEach(precisions, id: \.self) { Text(mode == .extract ? $0.rawValue : MossExport.displayName($0)).tag($0) }
            }
            .tweakable("models.tools.precision")
            if mode == .fuse, precision == .bfloat16, pickedSpecs.contains(where: { ModelSidecar.read(for: $0.url)?.distill != nil }) {
                Label("bf16 loses most of a Turbo LoRA's delta — int8·convrot (best measured) or fp16 keep it.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            HStack {
                Text("Name")
                TextField("file name", text: $name)
                Text(".safetensors").foregroundStyle(.secondary)
            }
            .tweakable("models.tools.name")
            Group {
                if let p = store.toolProgress {
                    ProgressView(value: p.fraction) { Text(p.message).font(.caption) }
                } else if !store.toolStatus.isEmpty {
                    Text(store.toolStatus).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
            .tweakable("models.tools.progress")
            HStack {
                Text("Needs ~\(memoryHint) GB of memory while it runs.").font(.caption).foregroundStyle(.tertiary)
                Spacer()
                Button(store.isRunningTool ? "Hide" : "Close") { dismiss() }
                Button("Run") { run() }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(!canRun)
                    .tweakable("models.tools.run")
            }
        }
        .padding(20)
        .frame(width: 520)
        .tweakable("models.tools")
        .onAppear {
            // Measured: a fused Turbo survives best as int8·convrot (fp32 fuse → quantize), bf16 loses it.
            if mode == .fuse { precision = .int8ConvRot }
            parentID = MossLoRAExtraction.defaultParent(for: dit.url, family: family, among: store.library.dits)?.id ?? ""
            updateName()
        }
        .onChange(of: precision) { _, _ in updateName() }
        .onChange(of: rank) { _, _ in updateName() }
        .onChange(of: mode) { _, m in
            if m == .extract, !precisions.contains(precision) { precision = .bfloat16 }
            updateName()
        }
        .onChange(of: picked) { _, _ in updateName() }
    }

    @ViewBuilder private var fuseForm: some View {
        VStack(alignment: .leading, spacing: 6) {
            if sourceIsQuantized {
                Label("This DiT is quantized — a LoRA's delta is lost in quantized weights. Fuse into the bf16 / fp32 model, then pick a quantized output.",
                    systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
            HStack {
                Text("LoRAs marked for \(family.displayName) — fused in fp32, then encoded. A Turbo LoRA becomes built in.")
                    .foregroundStyle(.secondary)
                Spacer()
                presets
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(loras) { l in
                        HStack {
                            Toggle(isOn: Binding(
                                get: { picked[l.id] != nil },
                                set: { on in
                                    if on {
                                        // One Turbo at a time: they're whole-model recipes.
                                        if l.isDistill { for d in loras where d.isDistill { picked[d.id] = nil } }
                                        picked[l.id] = l.sidecar.strength ?? 1
                                    } else { picked[l.id] = nil }
                                })
                            ) {
                                Text(l.name).lineLimit(1).truncationMode(.middle)
                            }
                            if let d = l.sidecar.distill { Badge(text: "Turbo \(d.steps)", tint: .orange) }
                            Spacer()
                            if picked[l.id] != nil {
                                TextField("1", value: Binding(get: { picked[l.id] ?? 1 }, set: { picked[l.id] = $0 }), format: .number)
                                    .frame(width: 50)
                            }
                        }
                    }
                }
            }
            .frame(maxHeight: 160)
        }
    }

    /// One-pick export recipes (measured: a fused Turbo keeps best as int8·convrot).
    private var presets: some View {
        let distribute = loras.first { $0.relativePath == MossVariant.turboLoRAPath }
        let eight = loras.first { $0.relativePath == "lora/lodestones/lodestones-8step-sft-latest.safetensors" }
        return Menu("Preset") {
            if let distribute {
                Button("Distribute — akhaliq 4-step → int8·convrot (one 5.4 GB file, stock Euler sampler)") {
                    picked = [distribute.id: 1]
                    precision = .int8ConvRot
                }
            }
            if let eight {
                Button("8 steps — \(eight.name) at \((eight.sidecar.strength ?? 0.7).formatted()) → int8·convrot (stock Euler sampler)") {
                    picked = [eight.id: eight.sidecar.strength ?? 0.7]
                    precision = .int8ConvRot
                }
            }
            if let any = distribute ?? eight {
                Button("Archive — \(any.name) → fp32 (lossless, 20 GB)") {
                    picked = [any.id: 1]
                    precision = .float32
                }
            }
        }
        .fixedSize()
    }

    @ViewBuilder private var mergeForm: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Mode", selection: $mergeMode) { ForEach(ModelMergeMode.allCases) { Text($0.displayName).tag($0) } }
                .pickerStyle(.segmented)
            Picker("B", selection: $modelB) {
                Text("Choose…").tag("")
                ForEach(others) { Text($0.name).tag($0.id) }
            }
            if mergeMode == .addDifference {
                Picker("C (base of B)", selection: $modelC) {
                    Text("Choose…").tag("")
                    ForEach(others) { Text($0.name).tag($0.id) }
                }
            }
            HStack {
                Text("α")
                Slider(value: $alpha, in: 0...1, step: 0.05)
                Text(String(format: "%.2f", alpha)).monospacedDigit().frame(width: 40)
            }
            Text(mergeMode == .weightedSum ? "(1 − α)·this + α·B" : "this + α·(B − C)").font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var extractForm: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Save what this model changed as a LoRA on its parent: the weight difference of each layer, factorised by truncated SVD. Load the LoRA on the parent at strength 1 and the finetune file can go.")
                .foregroundStyle(.secondary)
            Picker("Parent", selection: $parentID) {
                Text("Choose…").tag("")
                ForEach(others) { m in
                    Text("\(m.name) (\(m.sidecar.families.map(\.displayName).joined(separator: ", ")))").tag(m.id)
                }
            }
            .tweakable("models.tools.extract.parent")
            if let parent, !parent.sidecar.families.contains(family) {
                Label("\(parent.name) isn't a \(family.displayName) model — the LoRA will carry the whole family difference.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            HStack {
                Text("Rank")
                Picker("", selection: $rank) { ForEach([16, 32, 64, 128, 256], id: \.self) { Text("\($0)").tag($0) } }
                    .labelsHidden().pickerStyle(.segmented).frame(width: 240)
                    .tweakable("models.tools.extract.rank")
                Text("max per layer").font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Picker("Cutoff", selection: $cutoffKind) { ForEach(CutoffKind.allCases) { Text($0.rawValue).tag($0) } }
                    .frame(width: 210)
                    .tweakable("models.tools.extract.cutoff")
                if cutoffKind != .fixed {
                    TextField("", value: $cutoffValue, format: .number).frame(width: 60)
                        .tweakable("models.tools.extract.cutoffValue")
                    Text(cutoffKind == .energy ? "of ‖ΔW‖² kept (0–1)" : "× the largest singular value")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .onChange(of: cutoffKind) { _, k in cutoffValue = k == .energy ? 0.9 : 0.02 }
            Text("Norm and bias changes are stored whole. Rank 64 ≈ 0.3 GB, 128 ≈ 0.6 GB in bf16. Saved to lora/extracted/.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var canRun: Bool {
        guard !store.isRunningTool, !name.isEmpty else { return false }
        switch mode {
        case .convert: return true
        case .fuse: return !pickedSpecs.isEmpty && !sourceIsQuantized
        case .merge: return !modelB.isEmpty && (mergeMode == .weightedSum || !modelC.isEmpty)
        case .extract: return parent != nil && (cutoffKind == .fixed || (cutoffValue > 0 && cutoffValue <= 1))
        }
    }

    private var memoryHint: Int {
        switch mode {
        case .convert: 25
        case .fuse: 30
        case .merge: mergeMode == .weightedSum ? 45 : 60
        case .extract: 6
        }
    }

    private func updateName() {
        if mode == .extract {
            name = MossLoRAExtraction.outputStem(source: dit.name, rank: rank)
            return
        }
        let suffix: String? = switch mode {
        case .convert: nil
        case .fuse: pickedSpecs.map { $0.url.deletingPathExtension().lastPathComponent }.joined(separator: "+")
        case .merge, .extract: "merge"
        }
        name = MossExport.outputStem(dit.name, precision: precision, suffix: suffix)
    }

    private func run() {
        let output = mode == .extract
            ? MossLoRAExtraction.outputURL(stem: name, models: store.settings.modelsDirectory)
            : dit.url.deletingLastPathComponent().appendingPathComponent(name + ".safetensors")
        let request: ModelToolRequest
        switch mode {
        case .convert:
            request = ModelToolRequest(input: dit.url, operation: .convert, precision: precision, output: output)
        case .fuse:
            request = .fuse(dit: dit.url, loras: pickedSpecs, precision: precision, output: output)
        case .merge:
            let urls = [modelB, modelC].prefix(mergeMode == .weightedSum ? 1 : 2).compactMap { id in others.first { $0.id == id }?.url }
            request = ModelToolRequest(input: dit.url, operation: .merge(others: urls, mode: mergeMode, alpha: alpha, loras: []), precision: precision, output: output)
        case .extract:
            guard let parent else { return }
            request = ModelToolRequest(input: dit.url, operation: .extractLoRA(parent: parent.url, rank: rank, cutoff: cutoff), precision: precision, output: output)
        }
        if FileManager.default.fileExists(atPath: output.path) {
            store.runModelToolRefused("\(output.lastPathComponent) already exists — pick another name.")
            return
        }
        store.runModelTool(request) { id in onDone(id) }
    }
}
