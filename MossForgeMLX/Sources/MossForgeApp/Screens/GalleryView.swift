import AppKit
import MossForgeCore
import MossForgeEngine
import SwiftUI
import Tweakbench

/// Gallery: everything saved to the output folder (newest first), with the request sidecar. Images
/// can be grouped into collections (tags in the sidecar), captioned in batches, exported as a
/// kohya / ai-toolkit dataset, upscaled and trashed.
struct GalleryView: View {
    @Environment(AppStore.self) private var store
    @State private var upscaling: URL?
    @State private var exporting: DatasetExportJob?
    @State private var batch: BatchCaptionJob?
    @State private var naming: [URL]?
    @State private var newCollection = ""

    var body: some View {
        PersistentHSplit(key: "gallery", sized: .trailing, defaultWidth: 460, range: 320...1000, otherMin: 380) {
            VStack(spacing: 0) {
                filterBar.tweakable("gallery.filterBar")
                Divider()
                grid.tweakable("gallery.grid")
            }
        } trailing: {
            detail.frame(maxWidth: .infinity, maxHeight: .infinity)
                .tweakable("gallery.detail")
        }
        .onAppear { store.refreshGallery() }
        .confirmationDialog(deleteTitle, isPresented: Bindable(store).confirmGalleryDelete) {
            Button("Move to Trash", role: .destructive) { store.trashSelectedGallery() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The images and their JSON sidecars go to the Trash, so you can put them back.")
        }
        .sheet(item: $upscaling) { UpscaleSheet(url: $0) }
        .sheet(item: $exporting) { DatasetExportSheet(job: $0) }
        .sheet(item: $batch) { BatchCaptionSheet(job: $0) }
        .alert("New collection", isPresented: Binding(get: { naming != nil }, set: { if !$0 { naming = nil } })) {
            TextField("name", text: $newCollection)
            Button("Add") {
                if let images = naming { store.tag(.add, collection: newCollection, images: images) }
                naming = nil
            }
            Button("Cancel", role: .cancel) { naming = nil }
        } message: {
            Text("Adds \(naming?.count ?? 0) image\(naming?.count == 1 ? "" : "s") to a new collection.")
        }
    }

    // MARK: filter bar

    /// All / each collection, plus what applies to the whole visible set.
    private var filterBar: some View {
        HStack(spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    chip("All", count: store.gallery.count, active: store.galleryCollection == nil) { store.galleryCollection = nil }
                    ForEach(store.allCollections, id: \.self) { c in
                        chip(c, count: store.collectionsByImage.values.filter { $0.contains(c) }.count, active: store.galleryCollection == c) {
                            store.galleryCollection = c
                        }
                    }
                }
            }
            .tweakable("gallery.filterBar.chips")
            Menu {
                let scope = store.galleryCollection.map { "“\($0)”" } ?? "all images"
                Button("Caption \(scope)…") { batch = .gallery(title: "Caption \(scope)", images: store.visibleGallery) }
                Button("Export \(scope) as dataset…") {
                    exporting = DatasetExportJob(title: "Export \(scope)", images: store.visibleGallery, captions: true)
                }
                Divider()
                Button("Caption a folder of images…") { pickFolderToCaption() }
            } label: {
                Label("Dataset", systemImage: "square.stack.3d.up")
            }
            .menuStyle(.borderlessButton).fixedSize()
            .disabled(store.captionBatch != nil)
            .tweakable("gallery.filterBar.dataset")
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }

    private func chip(_ title: String, count: Int, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Text(title)
                Text("\(count)").foregroundStyle(active ? Color.white.opacity(0.8) : Color.secondary)
            }
            .font(.caption.weight(.medium))
            .padding(.horizontal, 9).padding(.vertical, 4)
            .background(Capsule().fill(active ? Color.accentColor : Color.secondary.opacity(0.15)))
            .foregroundStyle(active ? Color.white : Color.primary)
        }
        .buttonStyle(.plain)
    }

    private func pickFolderToCaption() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Caption"
        panel.message = "Choose a folder of training images — each gets a name.txt caption beside it"
        if panel.runModal() == .OK, let dir = panel.url { batch = .folder(dir) }
    }

    // MARK: grid

    private var grid: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150, maximum: 220), spacing: 10)], spacing: 10) {
                ForEach(store.visibleGallery, id: \.self) { url in
                    let isSelected = store.gallerySelection.selected.contains(url)
                    Thumbnail(url: url)
                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(isSelected ? Color.accentColor : .clear, lineWidth: 3))
                        .overlay(alignment: .topTrailing) {
                            if isSelected && store.gallerySelection.selected.count > 1 {
                                Image(systemName: "checkmark.circle.fill").font(.title3)
                                    .foregroundStyle(.white, Color.accentColor).padding(6)
                            }
                        }
                        .overlay(alignment: .bottomLeading) {
                            if store.captioning == url { ProgressView().controlSize(.small).padding(6) }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { click(url) }
                        .contextMenu { contextMenu(for: url) }
                        .tweakable("gallery.grid.thumbnail")
                }
            }
            .padding(12)
        }
        .background(Color.clear.contentShape(Rectangle()).onTapGesture { store.gallerySelection.clear() })
    }

    /// Right-click acts on the selection when the image is part of it, else on that image alone.
    @ViewBuilder private func contextMenu(for url: URL) -> some View {
        let images = store.gallerySelection.selected.contains(url) ? store.gallerySelection.ordered(in: store.visibleGallery) : [url]
        let n = images.count
        ImageShareMenuItems(images: images)
        Divider()
        if n == 1 {
            Button("Edit in Layout…") { store.beginEdit(url) }
            Button("Use as Edit Reference") { store.useAsReference(url) }
            Button("Upscale…") { upscaling = url }
        }
        Button(n == 1 ? "Caption" : "Caption \(n) images…") {
            if n == 1 { store.caption(url) } else { batch = .gallery(title: "Caption \(n) images", images: images) }
        }
        .disabled(store.captionBatch != nil)
        collectionMenu(images)
        Button(n == 1 ? "Export…" : "Export \(n) images…") { exporting = DatasetExportJob(title: "Export \(n) image\(n == 1 ? "" : "s")", images: images, captions: true) }
        if n == 1 { Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) } }
        Divider()
        Button("Move to Trash…", role: .destructive) {
            if !store.gallerySelection.selected.contains(url) { store.gallerySelection.click(url, in: store.visibleGallery) }
            store.confirmGalleryDelete = true
        }
    }

    /// Add to / remove from a collection — explicit operations, never a toggle.
    private func collectionMenu(_ images: [URL]) -> some View {
        Menu("Collections") {
            ForEach(store.allCollections, id: \.self) { c in
                switch store.membership(of: c, in: images) {
                case .all:
                    Button("Remove from “\(c)”") { store.tag(.remove, collection: c, images: images) }
                case .some:
                    Button("Add all to “\(c)”") { store.tag(.add, collection: c, images: images) }
                    Button("Remove from “\(c)”") { store.tag(.remove, collection: c, images: images) }
                case .none:
                    Button("Add to “\(c)”") { store.tag(.add, collection: c, images: images) }
                }
            }
            if !store.allCollections.isEmpty { Divider() }
            Button("New collection…") { newCollection = ""; naming = images }
        }
    }

    private var deleteTitle: String {
        let n = store.gallerySelection.selected.count
        return n == 1 ? "Move this image to the Trash?" : "Move \(n) images to the Trash?"
    }

    /// Plain click selects one; ⌘ toggles; ⇧ extends the range (Finder's rules).
    private func click(_ url: URL) {
        let flags = NSEvent.modifierFlags
        let modifier: GallerySelection<URL>.Modifier = flags.contains(.command) ? .command : flags.contains(.shift) ? .shift : .none
        store.gallerySelection.click(url, in: store.visibleGallery, modifier: modifier)
    }

    // MARK: detail

    @ViewBuilder private var detail: some View {
        let count = store.gallerySelection.selected.count
        if count > 1 {
            let images = store.gallerySelection.ordered(in: store.visibleGallery)
            VStack(spacing: 10) {
                Image(systemName: "photo.stack").font(.system(size: 40)).foregroundStyle(.secondary)
                Text("\(count) images selected").font(.headline)
                HStack {
                    collectionMenu(images).fixedSize()
                    Button("Caption…") { batch = .gallery(title: "Caption \(count) images", images: images) }
                        .disabled(store.captionBatch != nil)
                    Button("Export…") { exporting = DatasetExportJob(title: "Export \(count) images", images: images, captions: true) }
                }
                HStack {
                    ShareLink(items: images) { Label("Share…", systemImage: "square.and.arrow.up") }
                    Button("Save \(count) Copies…") { ImageActions.saveCopies(images) }
                }
                HStack {
                    Button("Deselect") { store.gallerySelection.clear() }
                    Button("Move to Trash…", role: .destructive) { store.confirmGalleryDelete = true }
                }
                status
            }
            .padding(12).tweakable("gallery.detail.multiple")
        } else if count == 1, let url = store.gallerySelection.primary, let image = NSImage(contentsOf: url) {
            VStack(spacing: 0) {
                // Image above, details below; drag the splitter to trade one for the other (remembered).
                PersistentVSplit(key: "gallery.detail", defaultHeight: 440, minTop: 160, minBottom: 120) {
                    Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity).padding(12)
                        .contextMenu { contextMenu(for: url) }
                        .tweakable("gallery.detail.image")
                } bottom: {
                    ScrollView { info(url).padding(12) }.tweakable("gallery.detail.info")
                }
                Divider()
                HStack {
                    Button("Edit…") { store.beginEdit(url) }
                        .help("Edit in Layout: change boxes and redraw only those areas")
                    Button("Upscale…") { upscaling = url }
                    Button(store.captioning == url ? "Captioning…" : "Caption") { store.caption(url) }
                        .disabled(store.captionBatch != nil)
                    Spacer()
                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }.help("Share")
                    Button { ImageActions.saveCopies([url]) } label: { Image(systemName: "square.and.arrow.down") }
                        .help("Save a copy…")
                    Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: { Image(systemName: "folder") }
                        .help("Reveal in Finder")
                    Button(role: .destructive) { store.confirmGalleryDelete = true } label: { Image(systemName: "trash") }
                        .help("Move to Trash (⌘⌫)")
                }
                .padding(12).tweakable("gallery.detail.actions")
            }
        } else {
            VStack(spacing: 6) {
                Text(store.gallery.isEmpty ? "No images yet — generate one." : "Select an image").foregroundStyle(.secondary)
                if !store.gallery.isEmpty {
                    Text("⌘-click or ⇧-click to select several, ⌘A for all.").font(.caption).foregroundStyle(.tertiary)
                }
                status
            }
        }
    }

    /// Everything known about an image: collections, prompt, settings, caption, provenance, raw sidecar.
    @ViewBuilder private func info(_ url: URL) -> some View {
        let _ = store.sidecarVersion  // re-read after a caption / collection change lands
        let meta = GallerySidecar.read(for: url)
        VStack(alignment: .leading, spacing: 12) {
            collectionChips(url).tweakable("gallery.detail.collections")
            if let prompt = meta["prompt"] as? String, !prompt.isEmpty {
                section("Prompt") {
                    if let layout = try? JSONDecoder().decode(LayoutPrompt.self, from: Data(prompt.utf8)) {
                        Text(layout.summary.caption.flatMap { $0.isEmpty ? nil : $0 } ?? "Layout prompt").textSelection(.enabled)
                        Text("\(layout.items.count) layout box\(layout.items.count == 1 ? "" : "es"): "
                            + layout.items.map { $0.text.map { "“\($0)”" } ?? $0.caption ?? "box" }.joined(separator: ", "))
                            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    } else {
                        Text(prompt).textSelection(.enabled)
                    }
                }
                .tweakable("gallery.detail.prompt")
            }
            let settings = settingsLine(meta)
            if !settings.isEmpty {
                Text(settings).font(.caption.monospacedDigit()).foregroundStyle(.secondary).textSelection(.enabled)
                    .tweakable("gallery.detail.settings")
            }
            if let caption = GallerySidecar.caption(in: meta) {
                section("Caption") {
                    Picker("", selection: Bindable(store).captionDetail) {
                        ForEach(CaptionDetail.allCases) { Text($0.rawValue.capitalized).tag($0) }
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize()
                    Text(caption.text(store.captionDetail)).textSelection(.enabled)
                }
                .tweakable("gallery.detail.caption")
            }
            if let up = meta["upscale"] as? [String: Any] {
                Text("Upscaled ×\(up["scale"] ?? "") with \(up["method"] ?? "") from \(up["from"] ?? "")")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !meta.isEmpty, let data = try? Data(contentsOf: GallerySidecar.url(for: url)), let text = String(data: data, encoding: .utf8) {
                DisclosureGroup("Metadata (JSON sidecar)") {
                    Text(text).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            status
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// "SFT · akhaliq-4step-sft · 4 steps · CFG 1 · 1024×1024 · seed 42"
    private func settingsLine(_ meta: [String: Any]) -> String {
        var parts: [String] = []
        if let model = meta["model"] as? String, !model.isEmpty {
            parts.append((model as NSString).deletingPathExtension)
        } else if let v = meta["variant"] as? String {
            parts.append(v.uppercased())
        }
        if let t = meta["turbo"] as? String, !t.isEmpty { parts.append((t as NSString).deletingPathExtension) }
        if let n = meta["steps"] as? Int { parts.append("\(n) steps") }
        if let c = meta["cfg"] as? Double { parts.append(String(format: "CFG %g", c)) }
        if let w = meta["width"] as? Int, let h = meta["height"] as? Int { parts.append("\(w)×\(h)") }
        if let seed = meta["seed"] as? UInt64 ?? (meta["seed"] as? Int).map(UInt64.init) { parts.append("seed \(seed)") }
        return parts.joined(separator: " · ")
    }

    /// The image's collections (× removes) and an add menu.
    private func collectionChips(_ url: URL) -> some View {
        FlowLayout(spacing: 6) {
            ForEach((store.collectionsByImage[url] ?? []).sorted(), id: \.self) { c in
                HStack(spacing: 3) {
                    Text(c)
                    Button { store.tag(.remove, collection: c, images: [url]) } label: { Image(systemName: "xmark") }
                        .buttonStyle(.plain).help("Remove from “\(c)”")
                }
                .font(.caption)
                .padding(.horizontal, 7).padding(.vertical, 3)
                .background(Capsule().fill(Color.accentColor.opacity(0.18)))
            }
            collectionMenu([url]).menuStyle(.borderlessButton).fixedSize().font(.caption)
        }
    }

    @ViewBuilder private var status: some View {
        if let b = store.captionBatch {
            ProgressView(value: Double(b.done), total: Double(max(b.total, 1))) {
                Text("Captioning \(min(b.done + 1, b.total)) of \(b.total) — \(b.current)").font(.caption)
            }
        }
        ForEach([store.captionStatus, store.galleryStatus].filter { !$0.isEmpty }, id: \.self) {
            Text($0).font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct Thumbnail: View {
    let url: URL
    var body: some View {
        Group {
            if let image = NSImage(contentsOf: url) {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fit)
            } else {
                Rectangle().fill(.quaternary)
            }
        }
        .frame(height: 150)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
