import AppKit
import MossForgeCore
import MossForgeEngine
import SwiftUI
import Tweakbench

/// Models: the user's diffusion models (DiTs) and LoRAs, with an inspector for the selected file —
/// its properties (family, Turbo recipe, default strength, notes; saved in a `.mfmeta.json` sidecar)
/// and file actions (rename, move to a subfolder, reveal, trash). Import brings in a DiT, a LoRA, an
/// all-in-one checkpoint (its DiT is extracted) or a diffusers `transformer/` folder (shards merged).
/// The VAE and text encoder are shared components and live in Settings.
struct ModelsView: View {
    @Environment(AppStore.self) private var store
    @State private var selected: String?
    @State private var importing: URL?
    @State private var status = ""
    @State private var gettingModels = false

    private var items: [ModelItem] { store.library.dits + store.library.loras }
    private var selectedItem: ModelItem? { items.first { $0.id == selected } }

    var body: some View {
        // The inspector can be dragged wider for long paths and notes; its width is remembered.
        PersistentHSplit(key: "models", sized: .trailing, defaultWidth: 360, range: 300...760, otherMin: 380) {
            list.tweakable("models.list")
        } trailing: {
            Group {
                if let item = selectedItem {
                    ModelInspector(item: item, status: $status, onChange: { selected = $0 }).id(item.id)
                } else {
                    VStack(spacing: 6) {
                        Image(systemName: "shippingbox").font(.largeTitle).foregroundStyle(.secondary)
                        Text("Select a model or LoRA").foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxHeight: .infinity).tweakable("models.inspector")
        }
        .sheet(item: $importing) { url in ImportSheet(source: url) { selected = $0 } }
        .sheet(isPresented: $gettingModels) { ModelSetupSheet() }
        .onAppear { store.refreshLibrary() }
    }

    private var list: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Your diffusion models and LoRAs. The VAE and text encoder are set in Settings.")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                if store.isScanningLibrary { ProgressView().controlSize(.small) }
                Button { gettingModels = true } label: { Label("Get Models…", systemImage: "icloud.and.arrow.down") }
                    .help("Download MOSS-Image, its encoder and VAE, and Turbo LoRAs from the model bundle")
                    .tweakable("models.list.get")
                Button { pickImport() } label: { Label("Import…", systemImage: "square.and.arrow.down") }
                Button { store.refreshLibrary() } label: { Label("Rescan", systemImage: "arrow.clockwise") }
            }
            .padding(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14)).tweakable("models.list.header")
            if !status.isEmpty {
                Text(status).font(.caption).foregroundStyle(.secondary).padding(.horizontal, 14).padding(.bottom, 6)
            }
            List(selection: $selected) {
                Section("Diffusion models · \(store.library.dits.count)") {
                    ForEach(store.library.dits) { row($0).tweakable("models.list.dit") }
                }
                Section("LoRAs · \(store.library.loras.count)") {
                    ForEach(store.library.loras) { row($0).tweakable("models.list.lora") }
                }
            }
            .listStyle(.inset)
            .scrollContentBackground(.hidden)
        }
    }

    private func row(_ item: ModelItem) -> some View {
        HStack(spacing: 8) {
            Image(systemName: item.kind == .dit ? "cube" : item.isDistill ? "hare" : "slider.horizontal.3")
                .foregroundStyle(.secondary).frame(width: 16)
            VStack(alignment: .leading, spacing: 1) {
                Text(item.name).lineLimit(1).truncationMode(.middle)
                if !item.folder.isEmpty { Text(item.folder).font(.caption2).foregroundStyle(.tertiary) }
            }
            ForEach(item.sidecar.families) { Badge(text: $0.displayName) }
            if let d = item.sidecar.distill { Badge(text: "Turbo \(d.steps)", tint: .orange) }
            if let precision = item.summary.split(separator: "·").last, item.kind == .dit {
                Badge(text: precision.trimmingCharacters(in: .whitespaces), tint: .secondary)
            }
            Spacer()
            Text(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
        .tag(item.id)
        .padding(.vertical, 2)
    }

    private func pickImport() {
        let panel = NSOpenPanel()
        panel.message = "Choose a .safetensors DiT, checkpoint or LoRA — or a diffusers transformer/ folder"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { importing = url }
    }
}

/// A small capsule label.
struct Badge: View {
    let text: String
    var tint: Color = .accentColor
    var body: some View {
        Text(text).font(.caption2.weight(.medium))
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(Capsule().fill(tint.opacity(0.18)))
            .foregroundStyle(tint)
    }
}

// MARK: - Inspector

private struct ModelInspector: View {
    @Environment(AppStore.self) private var store
    let item: ModelItem
    @Binding var status: String
    /// Called with the item's new id after a rename / move.
    let onChange: (String?) -> Void

    @State private var sidecar: ModelSidecar
    @State private var name: String
    @State private var newFolder = ""
    @State private var askingFolder = false
    @State private var confirmTrash = false
    @State private var tool: ModelToolSheet.Mode?
    /// Set while the inspector itself reloads `sidecar` (reset), so that change isn't saved back.
    @State private var reloading = false

    init(item: ModelItem, status: Binding<String>, onChange: @escaping (String?) -> Void) {
        self.item = item
        _status = status
        self.onChange = onChange
        _sidecar = State(initialValue: item.sidecar)
        _name = State(initialValue: item.name)
    }

    private var models: URL { store.settings.modelsDirectory }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header.tweakable("models.inspector.header")
                GroupBox("Properties") { properties.frame(maxWidth: .infinity, alignment: .leading) }
                    .tweakable("models.inspector.properties")
                if item.kind == .lora, sidecar.distill != nil {
                    GroupBox("Turbo recipe") { recipe.frame(maxWidth: .infinity, alignment: .leading) }
                        .tweakable("models.inspector.recipe")
                }
                if item.kind == .dit {
                    GroupBox("Tools") {
                        VStack(alignment: .leading, spacing: 6) {
                            FlowLayout(spacing: 6) {
                                Button("Convert…") { tool = .convert }
                                Button("Fuse LoRAs…") { tool = .fuse }
                                Button("Merge…") { tool = .merge }
                                Button("Extract LoRA…") { tool = .extract }
                                    .help("Save what this finetune changed as a LoRA on the official model, so the big file can go.")
                                    .tweakable("models.inspector.tools.extract")
                            }
                            .controlSize(.small)
                            .disabled(store.isRunningTool)
                            if let p = store.toolProgress {
                                ProgressView(value: p.fraction) { Text(p.message).font(.caption) }
                            }
                            if let d = sidecar.distill {
                                Text("Turbo built in: \(d.steps) steps, \(d.route.displayName), CFG 1.").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .tweakable("models.inspector.tools")
                }
                GroupBox("File") { fileActions.frame(maxWidth: .infinity, alignment: .leading) }
                    .tweakable("models.inspector.file")
            }
            .padding(14)
        }
        .onChange(of: sidecar) { _, new in
            if reloading { reloading = false } else { save(new) }
        }
        .confirmationDialog("Move “\(item.name)” to the Trash?", isPresented: $confirmTrash) {
            Button("Move to Trash", role: .destructive) { trash() }
            Button("Cancel", role: .cancel) {}
        } message: {
            let extras = store.library.supportingFiles(of: item).map(\.lastPathComponent)
            Text(extras.isEmpty ? "Its settings go too. Recoverable from the Trash." : "Its settings and \(extras.joined(separator: ", ")) go too. Recoverable from the Trash.")
        }
        .sheet(item: $tool) { mode in ModelToolSheet(dit: item, mode: mode) { onChange($0) } }
        .alert("New folder", isPresented: $askingFolder) {
            TextField("folder name", text: $newFolder)
            Button("Move") { move(to: (item.folder.isEmpty ? "" : item.folder + "/") + newFolder) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Created inside \(item.kind == .dit ? "diffusion_models" : "lora")/\(item.folder.isEmpty ? "" : item.folder + "/")")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(item.name).font(.headline).textSelection(.enabled)
            Text(item.relativePath).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            HStack {
                Text(item.summary)
                Spacer()
                Text(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file))
            }
            .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var properties: some View {
        VStack(alignment: .leading, spacing: 8) {
            if item.kind == .dit {
                Picker("Family", selection: Binding(
                    get: { sidecar.families.first ?? .sft }, set: { sidecar.families = [$0] })
                ) {
                    ForEach(MossFamily.allCases) { Text($0.displayName).tag($0) }
                }
                .pickerStyle(.segmented)
                .tweakable("models.inspector.family")
                Text("Which MOSS-Image family this DiT is (finetunes and merges keep their parent's). Generate uses it to offer matching Turbo LoRAs and LoRAs.")
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                HStack {
                    Text("Works with")
                    ForEach(MossFamily.allCases) { f in
                        Toggle(f.displayName, isOn: Binding(
                            get: { sidecar.supports(f) },
                            set: { on in sidecar.families = on ? sidecar.families + [f] : sidecar.families.filter { $0 != f } }))
                    }
                }
                .tweakable("models.inspector.worksWith")
                Toggle("Turbo — a few-step distillation LoRA", isOn: Binding(
                    get: { sidecar.distill != nil },
                    set: { sidecar.distill = $0 ? (sidecar.distill ?? DistillInfo(steps: 8, route: .flow)) : nil }))
                    .tweakable("models.inspector.turbo")
                if sidecar.distill == nil {
                    HStack {
                        Text("Default strength")
                        TextField("1.0", value: Binding(get: { sidecar.strength ?? 1 }, set: { sidecar.strength = $0 }), format: .number)
                            .frame(width: 60)
                    }
                    .tweakable("models.inspector.strength")
                    strengthRangeFields
                }
            }
            TextField("Source (URL)", text: Binding(get: { sidecar.source ?? "" }, set: { sidecar.source = $0.isEmpty ? nil : $0 }))
                .tweakable("models.inspector.source")
            TextField("Notes", text: Binding(get: { sidecar.notes ?? "" }, set: { sidecar.notes = $0.isEmpty ? nil : $0 }), axis: .vertical)
                .lineLimit(2...6)
                .tweakable("models.inspector.notes")
            HStack {
                Text(item.hasSavedSidecar ? "Saved beside the file (.mfmeta.json)." : "Guessed from the file name — edits are saved beside it.")
                    .font(.caption2).foregroundStyle(.secondary)
                Spacer()
                if item.hasSavedSidecar {
                    Button("Reset to auto") { resetSidecar() }.controlSize(.small)
                }
            }
        }
    }

    @ViewBuilder private var recipe: some View {
        let siblings = siblingFiles
        VStack(alignment: .leading, spacing: 8) {
            Stepper("Steps: \(sidecar.distill?.steps ?? 8)", value: Binding(
                get: { sidecar.distill?.steps ?? 8 }, set: { sidecar.distill?.steps = $0 }), in: 1...16)
                .tweakable("models.inspector.recipe.steps")
            Picker("Sampler", selection: Binding(get: { sidecar.distill?.route ?? .flow }, set: { sidecar.distill?.route = $0 })) {
                ForEach(DistillRoute.allCases) { Text($0.displayName).tag($0) }
            }
            .tweakable("models.inspector.recipe.sampler")
            if sidecar.distill?.route == .dataSpace {
                adapterPicker("Data space", siblings, \.dataSpace)
                adapterPicker("Time embedding", siblings, \.timeEmbedding)
                if store.library.supportingFiles(of: item).count < 2 && item.hasSavedSidecar {
                    Label("Pick both adapter files (they sit beside the LoRA).", systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
            } else {
                Text("Runs on the stock flow-Euler sampler at CFG 1, shift \(sidecar.distill?.shift ?? 5, specifier: "%.0f").")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func adapterPicker(_ title: String, _ files: [String], _ key: WritableKeyPath<DistillInfo, String?>) -> some View {
        Picker(title, selection: Binding(
            get: { sidecar.distill?[keyPath: key] ?? "" },
            set: { sidecar.distill?[keyPath: key] = $0.isEmpty ? nil : $0 })
        ) {
            Text("None").tag("")
            ForEach(files, id: \.self) { Text($0).tag($0) }
        }
    }

    /// Non-LoRA `.safetensors` beside the item (adapter candidates).
    private var siblingFiles: [String] {
        let dir = item.url.deletingLastPathComponent()
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        let loras = Set(store.library.loras.map(\.url.lastPathComponent))
        return files.filter { $0.hasSuffix(".safetensors") && !loras.contains($0) }.sorted()
    }

    @ViewBuilder private var fileActions: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField("Name", text: $name).onSubmit(rename)
                Button("Rename", action: rename).disabled(name == item.name || name.isEmpty)
            }
            .tweakable("models.inspector.file.name")
            HStack {
                Text("Folder")
                Menu(item.folder.isEmpty ? "(top level)" : item.folder) {
                    ForEach(ModelLibrary.subfolders(item.kind, models: models), id: \.self) { f in
                        Button(f.isEmpty ? "(top level)" : f) { move(to: f) }.disabled(f == item.folder)
                    }
                    Divider()
                    Button("New folder…") { newFolder = ""; askingFolder = true }
                }
                .fixedSize()
            }
            .tweakable("models.inspector.file.folder")
            if item.kind == .lora, !store.library.supportingFiles(of: item).isEmpty {
                Text("Moves with: " + store.library.supportingFiles(of: item).map(\.lastPathComponent).joined(separator: ", "))
                    .font(.caption2).foregroundStyle(.secondary)
            }
            HStack {
                Button { NSWorkspace.shared.activateFileViewerSelecting([item.url]) } label: { Label("Reveal", systemImage: "folder") }
                Spacer()
                Button(role: .destructive) { confirmTrash = true } label: { Label("Move to Trash…", systemImage: "trash") }
            }
            .tweakable("models.inspector.file.actions")
        }
    }

    /// Generate's strength slider span for this LoRA (±2 unless set; a trained slider's header fills it).
    private var strengthRangeFields: some View {
        let range = sidecar.sliderRange
        let bound = { (keyPath: WritableKeyPath<SliderRange, Double>) in
            Binding<Double>(
                get: { range[keyPath: keyPath] },
                set: { v in
                    var r = sidecar.strengthRange ?? range
                    r[keyPath: keyPath] = v
                    sidecar.strengthRange = r
                })
        }
        return VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text("Slider range")
                TextField("-2", value: bound(\.min), format: .number)
                    .frame(width: 60).tweakable("models.inspector.strengthRange.min")
                Text("to")
                TextField("2", value: bound(\.max), format: .number)
                    .frame(width: 60).tweakable("models.inspector.strengthRange.max")
            }
            Text(sidecar.strengthRange.map { $0.isValid ? "Generate's strength slider spans this range." : "The range must include 0 — Generate uses −2 to 2 until it does." }
                 ?? "Generate's strength slider spans −2 to 2. Widen it for LoRAs made for bigger weights (e.g. −10 to 10).")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .tweakable("models.inspector.strengthRange")
    }

    // MARK: actions

    private func save(_ s: ModelSidecar) {
        do {
            try s.write(for: item.url)
            store.refreshLibrary()
        } catch { status = error.localizedDescription }
    }

    private func resetSidecar() {
        try? FileManager.default.removeItem(at: ModelSidecar.url(for: item.url))
        let auto = ModelSidecar.detected(for: item.url, isLoRA: item.kind == .lora)
        if auto != sidecar {
            reloading = true
            sidecar = auto
        }
        store.refreshLibrary()
        status = "\(item.name): back to the defaults guessed from its name."
    }

    private func rename() {
        do {
            let url = try ModelLibrary.rename(item.url, to: name, models: models)
            done("Renamed to \(url.lastPathComponent).", url)
        } catch { status = error.localizedDescription }
    }

    private func move(to folder: String) {
        do {
            let url = try ModelLibrary.move(item.url, toSubfolder: folder, models: models)
            done("Moved \(item.name) to \(folder.isEmpty ? "the top level" : folder).", url)
        } catch { status = error.localizedDescription }
    }

    private func trash() {
        do {
            try ModelLibrary.trash(item.url, models: models)
            done("Moved \(item.name) to the Trash.", nil)
        } catch { status = error.localizedDescription }
    }

    private func done(_ message: String, _ url: URL?) {
        status = message
        let root = item.kind == .dit ? ModelLibrary.ditFolder : ModelLibrary.loraFolder
        onChange(url.map { root + "/" + ModelLibrary.relativePath($0, in: models.appendingPathComponent(root)) })
        store.refreshLibrary()
    }
}

// MARK: - Import

private struct ImportSheet: View {
    @Environment(AppStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let source: URL
    let onImported: (String) -> Void

    @State private var detected: Detected = .scanning
    @State private var kind: ModelItem.Kind = .lora
    @State private var folder = ""
    @State private var name = ""
    @State private var move = true
    @State private var progress: Double?
    @State private var error = ""

    enum Detected: Equatable {
        case scanning, dit, lora, checkpoint(vae: Bool, te: Bool), diffusers, unknown(String)
    }

    private var models: URL { store.settings.modelsDirectory }
    /// Extraction and shard merging write a new file and leave the source alone.
    private var writesNewFile: Bool {
        switch detected {
        case .checkpoint, .diffusers: true
        default: false
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Import").font(.title2.bold())
            Text(source.path).font(.caption).foregroundStyle(.secondary).lineLimit(2).truncationMode(.middle)
            Text(description).font(.callout)
            if case .unknown = detected {} else if detected != .scanning {
                Picker("Import as", selection: $kind) {
                    Text("Diffusion model (DiT)").tag(ModelItem.Kind.dit)
                    Text("LoRA").tag(ModelItem.Kind.lora)
                }
                .pickerStyle(.segmented)
                .disabled(writesNewFile)
                HStack {
                    Text("Folder")
                    TextField(kind == .dit ? "diffusion_models/…" : "lora/… e.g. akhaliq", text: $folder)
                    Menu("") {
                        ForEach(ModelLibrary.subfolders(kind, models: models), id: \.self) { f in
                            Button(f.isEmpty ? "(top level)" : f) { folder = f }
                        }
                    }
                    .fixedSize()
                }
                HStack {
                    Text("Name")
                    TextField("file name", text: $name)
                    Text(".safetensors").foregroundStyle(.secondary)
                }
                if writesNewFile {
                    Text("A new file is written; the source stays where it is.").font(.caption).foregroundStyle(.secondary)
                } else {
                    Picker("", selection: $move) {
                        Text("Move").tag(true)
                        Text("Copy").tag(false)
                    }
                    .pickerStyle(.segmented).labelsHidden().frame(width: 160)
                }
            }
            if let progress { ProgressView(value: progress) }
            if !error.isEmpty { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.disabled(progress != nil)
                Button("Import") { run() }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(progress != nil || name.isEmpty || detected == .scanning || { if case .unknown = detected { return true }; return false }())
            }
        }
        .padding(20)
        .frame(width: 480)
        .task { await detect() }
    }

    private var description: String {
        switch detected {
        case .scanning: "Reading the header…"
        case .dit: "A MOSS-Image diffusion model."
        case .lora: "A LoRA."
        case .checkpoint(let vae, let te):
            "An all-in-one checkpoint" + (vae || te ? " (with \([vae ? "VAE" : nil, te ? "text encoder" : nil].compactMap { $0 }.joined(separator: " + ")) baked in)" : "")
                + ". Its DiT will be extracted into diffusion_models."
        case .diffusers: "A diffusers transformer folder. Its shards will be merged into one file."
        case .unknown(let why): why
        }
    }

    private func detect() async {
        let url = source
        let result: (Detected, String) = await Task.detached {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
            if isDir.boolValue || url.lastPathComponent.hasSuffix(".index.json") {
                let stem = url.lastPathComponent == "transformer" ? url.deletingLastPathComponent().lastPathComponent : url.deletingPathExtension().lastPathComponent
                return (.diffusers, stem)
            }
            let stem = url.deletingPathExtension().lastPathComponent
            switch (try? MossInspector.inspect(url)) ?? .unknown {
            case .dit: return (.dit, stem)
            case .lora: return (.lora, stem)
            default:
                if let names = try? SafetensorsHeader.readEntries(from: url).entries.keys.map({ $0 }),
                    let plan = CheckpointExtract.plan(tensorNames: names)
                {
                    return (.checkpoint(vae: plan.hasVAE, te: plan.hasTextEncoder), stem)
                }
                return (.unknown("Not a MOSS-Image DiT, checkpoint or LoRA. (VAE and text encoder files are set in Settings.)"), stem)
            }
        }.value
        detected = result.0
        name = result.1
        kind = result.0 == .lora ? .lora : .dit
    }

    private func run() {
        let (src, kind, folder, name, move, models) = (source, kind, folder, name, move, models)
        progress = 0
        error = ""
        Task {
            do {
                let url = try await Task.detached {
                    try ModelLibrary.importFile(src, as: kind, subfolder: folder, models: models, move: move, name: name) { f in
                        Task { @MainActor in progress = f }
                    }
                }.value
                let root = kind == .dit ? ModelLibrary.ditFolder : ModelLibrary.loraFolder
                store.refreshLibrary()
                onImported(root + "/" + ModelLibrary.relativePath(url, in: models.appendingPathComponent(root)))
                dismiss()
            } catch {
                self.error = error.localizedDescription
                progress = nil
            }
        }
    }
}
