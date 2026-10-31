import AppKit
import MossForgeCore
import SwiftUI
import UniformTypeIdentifiers

/// Share / Save a Copy / Copy for saved images — the same items in the Generate preview's and the
/// Gallery's right-click menus (and the Gallery detail pane). Everything stays local until the user
/// picks a share service; saving never overwrites (`ImageExport`).
enum ImageActions {
    /// Save one image (a save panel, a prompt-based name) or several (a folder panel, numbered on clash).
    @MainActor static func saveCopies(_ images: [URL]) {
        guard !images.isEmpty else { return }
        if images.count == 1, let src = images.first {
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.png]
            panel.nameFieldStringValue = suggestedName(for: src)
            panel.canCreateDirectories = true
            panel.message = "Save a copy of the image"
            guard panel.runModal() == .OK, let dest = panel.url else { return }
            // The panel already asked about replacing an existing file.
            try? FileManager.default.removeItem(at: dest)
            try? FileManager.default.copyItem(at: src, to: dest)
        } else {
            let panel = NSOpenPanel()
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.canCreateDirectories = true
            panel.prompt = "Save \(images.count) Copies"
            panel.message = "Choose a folder — files with the same name are kept, copies get a number"
            guard panel.runModal() == .OK, let folder = panel.url else { return }
            let existing = Set((try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [])
            for (src, dest) in zip(images, ImageExport.destinations(for: images, in: folder, existing: existing)) {
                try? FileManager.default.copyItem(at: src, to: dest)
            }
        }
    }

    /// Put the images on the pasteboard (file URLs + the first image's pixels, so both Finder and
    /// image editors can paste).
    @MainActor static func copy(_ images: [URL]) {
        let pb = NSPasteboard.general
        pb.clearContents()
        var objects: [NSPasteboardWriting] = images.map { $0 as NSURL }
        if let first = images.first, let image = NSImage(contentsOf: first) { objects.insert(image, at: 0) }
        pb.writeObjects(objects)
    }

    /// The prompt (from the sidecar) + seed, else the file's own name.
    static func suggestedName(for url: URL) -> String {
        let meta = GallerySidecar.read(for: url)
        guard let prompt = meta["prompt"] as? String, !prompt.isEmpty else { return url.lastPathComponent }
        let seed = meta["seed"] as? UInt64 ?? (meta["seed"] as? Int).map(UInt64.init) ?? 0
        return ImageExport.suggestedName(prompt: prompt, seed: seed)
    }
}

/// The share / save / copy items for a right-click menu.
struct ImageShareMenuItems: View {
    let images: [URL]

    var body: some View {
        let n = images.count
        ShareLink(items: images) { Label(n == 1 ? "Share…" : "Share \(n) Images…", systemImage: "square.and.arrow.up") }
        Button(n == 1 ? "Save a Copy…" : "Save \(n) Copies…") { ImageActions.saveCopies(images) }
        Button(n == 1 ? "Copy Image" : "Copy \(n) Images") { ImageActions.copy(images) }
    }
}

/// The right-click menu for one saved image outside the Gallery grid — Generate's preview and Character
/// Studio's rows. `onDelete` replaces the plain
/// Move to Trash (Character Studio marks its row removed too).
struct ImageItemMenu: View {
    @Environment(AppStore.self) private var store
    let url: URL
    var onUpscale: (URL) -> Void
    var onDelete: ((URL) -> Void)?

    var body: some View {
        if FileManager.default.fileExists(atPath: url.path) {
            ImageShareMenuItems(images: [url])
            Divider()
            Button("Edit in Layout…") { store.beginEdit(url) }
            Button("Use as Edit Reference") { store.useAsReference(url) }
            Button("Upscale…") { onUpscale(url) }
            Button("Caption") { store.caption(url) }.disabled(store.captionBatch != nil)
            Button("Show in Gallery") { store.showInGallery(url) }
            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            Divider()
            Button("Move to Trash", role: .destructive) { (onDelete ?? store.trashImage)(url) }
        } else {
            Text("The saved file has moved or been deleted")
        }
    }
}
