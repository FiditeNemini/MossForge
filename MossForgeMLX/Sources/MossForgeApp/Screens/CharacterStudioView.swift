import AppKit
import MossForgeCore
import MossForgeEngine
import SwiftUI
import Tweakbench

// No reference-photo
// well (text-only for now), MOSS pixel budgets instead of long side + steps (Generate's model and Turbo
// set the sampling), gallery URLs, and `ImageItemMenu` on the rows.

/// Character Studio: build a character-LoRA dataset in one go. Each seed-prompt row (one per framing,
/// close-up to full body) is enhanced into several prompts — each asking for a different camera angle —
/// and each prompt renders with its own seed. Results land in a gallery collection whose prompts are
/// the training captions, ready to cull and export as png + txt pairs with the trigger word.
struct CharacterStudioView: View {
    @Environment(AppStore.self) private var store
    @State private var upscaling: URL?
    @State private var exporting: DatasetExportJob?

    var body: some View {
        PersistentHSplit(key: "character", sized: .leading, defaultWidth: 440, range: 360...620, otherMin: 420) {
            form.tweakable("character.form")
        } trailing: {
            results.frame(maxWidth: .infinity, maxHeight: .infinity).tweakable("character.results")
        }
        .sheet(item: $upscaling) { UpscaleSheet(url: $0) }
        .sheet(item: $exporting) { DatasetExportSheet(job: $0) }
    }

    // MARK: Form

    private var form: some View {
        @Bindable var store = store
        return ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 8) {
                            field("Name (gallery collection)") { TextField("e.g. Mara", text: $store.character.name).tweakable("character.name") }
                            field("Trigger word") { TextField("e.g. mra woman", text: $store.character.trigger).tweakable("character.trigger") }
                        }
                        field("Description") {
                            TextField(
                                "a woman in her thirties with short red hair, freckles and green eyes",
                                text: $store.character.description, axis: .vertical
                            ).lineLimit(2...4).wrappingField().tweakable("character.description")
                        }
                        Text("Written into every prompt. Describe what never changes: face, hair, build, age.")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        referencePhoto.tweakable("character.reference")
                    }
                    .textFieldStyle(.roundedBorder)
                } label: { Label("Character", systemImage: "person.crop.square") }
                .tweakable("character.identity")

                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("One row per framing. Each row is enhanced into \(store.character.variations) prompts, "
                            + "each from a different camera angle, then rendered with its own seed. Vary outfits "
                            + "and settings across rows so the LoRA learns the person, not the clothes.")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        ForEach($store.character.seeds) { $seed in
                            HStack(alignment: .firstTextBaseline, spacing: 6) {
                                Picker("", selection: $seed.shot) {
                                    ForEach(CharacterShot.allCases) { Text($0.title).tag($0) }
                                }
                                .labelsHidden().frame(width: 130).tweakable("character.seed.shot")
                                TextField("scenario", text: $seed.prompt, axis: .vertical)
                                    .lineLimit(1...4).wrappingField().tweakable("character.seed.prompt")
                                Button { store.character.seeds.removeAll { $0.id == seed.id } } label: {
                                    Image(systemName: "minus.circle")
                                }
                                .buttonStyle(.borderless).foregroundStyle(.secondary)
                                .disabled(store.character.seeds.count <= 1)
                            }
                            .tweakable("character.seed.row")
                        }
                        HStack {
                            Button { store.character.seeds.append(CharacterSeed(shot: .fullBody, prompt: "")) } label: {
                                Label("Add row", systemImage: "plus")
                            }
                            Spacer()
                            Button("Reset to defaults") { store.character.seeds = CharacterDatasetPlan.defaultSeeds }
                                .buttonStyle(.link).font(.caption)
                        }
                        .tweakable("character.seeds.buttons")
                    }
                } label: { Label("Seed prompts", systemImage: "list.bullet.rectangle") }
                .tweakable("character.seeds")

                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Stepper(value: $store.character.variations, in: CharacterDatasetPlan.variationRange) {
                            Text("Variations per row: \(store.character.variations)")
                        }
                        .tweakable("character.output.variations")
                        Picker("Size", selection: $store.character.pixels) {
                            ForEach(CharacterStudioDraft.pixelChoices, id: \.self) { Text(String(format: "%.1f MP", Double($0) / 1_048_576)).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        .tweakable("character.output.size")
                        Text("Close shots are 1:1, body shots 2:3, the action row 3:2. Full-body faces are small: 1.5–2 MP keeps "
                            + "them sharper (slower). Model, Turbo, sampling and style LoRAs come from Generate.")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                } label: { Label("Output", systemImage: "photo.stack") }
                .tweakable("character.output")

                summary.tweakable("character.summary")
                actions.tweakable("character.actions")
            }
            .padding(16)
        }
    }

    private var summary: some View {
        let n = store.character.plannedCount
        let keep = CharacterDatasetPlan.recommendedKeep
        return VStack(alignment: .leading, spacing: 2) {
            Text("\(n) images planned").font(.headline)
            Text("Character LoRAs train best on about \(keep.lowerBound)–\(keep.upperBound) varied images "
                + "(15 at the least). Generate a few more than that and keep the ones that look most alike.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if n < keep.lowerBound {
                Text("Fewer than \(keep.lowerBound) — add rows or variations for a full pool.")
                    .font(.caption).foregroundStyle(.orange)
            }
        }
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if store.characterRunning {
                    ProgressView(value: store.characterProgress).frame(maxWidth: .infinity)
                    Button(role: .cancel, action: store.cancelCharacter) { Label("Cancel", systemImage: "stop.fill") }
                        .keyboardShortcut(".", modifiers: [.command])
                } else {
                    Button { store.writeCharacterPrompts() } label: { Label("Write prompts", systemImage: "wand.and.stars") }
                        .help("Enhance every seed prompt into its variations, so you can read and edit them first.")
                    Spacer()
                    Button { store.generateCharacterDataset() } label: { Label(generateLabel, systemImage: "sparkles") }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.return, modifiers: [.command])
                }
            }
            .disabled(store.character.plannedCount == 0 || (!store.characterRunning && store.isGenerating))
            Text(store.characterStatus).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
        }
    }

    private var generateLabel: String {
        let remaining = store.characterRows.filter { ![.done, .removed].contains($0.state) }.count
        if store.characterRows.isEmpty { return "Build \(store.character.plannedCount) images" }
        return remaining == 0 ? "All done" : "Generate \(remaining) images"
    }

    // MARK: Results

    private var results: some View {
        @Bindable var store = store
        let rows = store.characterRows
        return VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Dataset").font(.headline)
                if !rows.isEmpty {
                    Text("\(rows.filter { $0.state == .done }.count) / \(rows.filter { $0.state != .removed }.count)")
                        .font(.caption).padding(.horizontal, 7).padding(.vertical, 2)
                        .background(.quaternary, in: Capsule())
                }
                Spacer()
                if !rows.isEmpty, !store.characterRunning {
                    Button("Clear plan") { store.characterRows = [] }.buttonStyle(.link).font(.caption)
                }
                Button("Open Gallery") {
                    store.galleryCollection = store.allCollections.contains(store.character.collectionName) ? store.character.collectionName : nil
                    store.selection = .gallery
                }
                .buttonStyle(.link).font(.caption)
                Button {
                    exporting = DatasetExportJob(
                        title: "Export “\(store.character.collectionName)” dataset", images: store.characterDatasetImages,
                        captions: true, trigger: store.character.trigger)
                } label: { Label("Export Dataset…", systemImage: "square.and.arrow.up") }
                    .controlSize(.small)
                    .disabled(store.characterRunning || store.characterDatasetImages.isEmpty)
                    .help("Write every image left in “\(store.character.collectionName)” as name.png + name.txt "
                        + "(trigger, prompt). Right-click an image → Move to Trash to leave it out.")
            }
            .padding(14).tweakable("character.results.header")
            Divider()
            if rows.isEmpty {
                ContentUnavailableView(
                    "No dataset yet", systemImage: "person.2.crop.square.stack",
                    description: Text("Write prompts to preview them, or Build to write and render in one go."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach($store.characterRows) { $row in rowView($row).tweakable("character.results.row") }
                    }
                    .padding(14)
                }
            }
        }
    }

    private func rowView(_ row: Binding<CharacterRow>) -> some View {
        let r = row.wrappedValue
        let image = r.imageURL.flatMap { FileManager.default.fileExists(atPath: $0.path) ? NSImage(contentsOf: $0) : nil }
        let editable = !store.characterRunning && ![.done, .removed].contains(r.state)
        return HStack(alignment: .top, spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.5))
                if let image {
                    Image(nsImage: image).resizable().interpolation(.medium).aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                } else {
                    switch r.state {
                    case .enhancing, .generating: ProgressView().controlSize(.small)
                    case .failed: Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                    case .removed, .done: Label("Deleted", systemImage: "trash").font(.caption2).foregroundStyle(.tertiary)
                    default: Image(systemName: "photo").foregroundStyle(.tertiary)
                    }
                }
            }
            .frame(width: 96, height: 96).tweakable("character.results.thumbnail")
            .contextMenu {
                if let url = r.imageURL, image != nil {
                    ImageItemMenu(url: url, onUpscale: { upscaling = $0 }, onDelete: store.deleteCharacterImage)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(r.job.shot.title).font(.caption.bold())
                    Text(CharacterDatasetPlan.angles[r.job.variation]).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Text("seed \(String(r.job.seed))").font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                }
                if editable {
                    TextField("prompt (written by the enhancer)", text: row.prompt, axis: .vertical)
                        .font(.caption).lineLimit(2...6).wrappingField()
                        .tweakable("character.results.prompt")
                        .onChange(of: row.wrappedValue.prompt) { _, new in
                            if row.wrappedValue.state == .pending, !new.isEmpty { row.wrappedValue.state = .ready }
                        }
                } else {
                    Text(r.prompt.isEmpty ? "…" : r.prompt).font(.caption).foregroundStyle(.secondary)
                        .lineLimit(5).textSelection(.enabled)
                }
            }
        }
        .opacity(image == nil && [.done, .removed].contains(r.state) ? 0.45 : 1)
    }

    // MARK: Helpers

    private func field<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }
}

extension View {
    /// A vertical-axis `TextField` that wraps and grows. On macOS `.roundedBorder` draws a single-line
    /// bezel that truncates instead, so use the plain style inside a field-like rounded frame.
    func wrappingField() -> some View {
        textFieldStyle(.plain)
            .padding(.horizontal, 6).padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 5).fill(.background.opacity(0.6)))
            .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(.separator))
    }
}

extension CharacterStudioView {
    /// A photo of the character, restaged in every row by an edit LoRA — far less drift than the
    /// description alone. Drop it here or choose one.
    @ViewBuilder var referencePhoto: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.5))
                if let url = store.character.referencePhoto, let image = NSImage(contentsOf: url) {
                    Image(nsImage: image).resizable().aspectRatio(contentMode: .fill).frame(width: 80, height: 80)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                } else {
                    RoundedRectangle(cornerRadius: 6).strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4])).foregroundStyle(.secondary)
                    VStack(spacing: 2) {
                        Image(systemName: "person.crop.square.badge.plus")
                        Text("Drop or\nclick").font(.caption2).multilineTextAlignment(.center)
                    }.foregroundStyle(.secondary)
                }
            }
            .frame(width: 80, height: 80)
            .contentShape(Rectangle())
            .onTapGesture { chooseReference() }
            .dropDestination(for: URL.self) { urls, _ in
                guard let url = urls.first else { return false }
                store.character.referencePhoto = url
                return true
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Reference photo (optional)").font(.caption.weight(.semibold))
                if store.character.referencePhoto == nil {
                    Text("Without one, the description alone keeps the character consistent — expect some drift and keep the images that look most alike.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                } else if let lora = store.characterEditLoRA {
                    Text("Every image restages this person through \(lora.deletingPathExtension().lastPathComponent) (picture 1); captions stay the row prompts.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 6) {
                        Text("Strength").font(.caption).foregroundStyle(.secondary)
                        Slider(value: Binding(get: { store.character.referenceStrengthValue }, set: { store.character.referenceStrength = $0 }),
                               in: 0.4...1, step: 0.05)
                            .frame(width: 140)
                            .help("Lower lets each shot re-pose the person; 1.0 copies the photo's pose and framing.")
                            .tweakable("character.reference.strength")
                        Text(String(format: "%.2f", store.character.referenceStrengthValue)).font(.caption).monospacedDigit()
                        Button("Remove") { store.character.referencePhoto = nil }.controlSize(.small)
                    }
                } else {
                    Label("Needs an edit LoRA in models/lora — chinoll/Moss-Image-Edit restages people best.", systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                    Button("Remove") { store.character.referencePhoto = nil }.controlSize(.small)
                }
            }
        }
    }

    private func chooseReference() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url { store.character.referencePhoto = url }
    }
}
