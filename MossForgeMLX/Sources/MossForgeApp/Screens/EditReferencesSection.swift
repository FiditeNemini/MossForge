import AppKit
import MossForgeCore
import MossForgeMLX
import SwiftUI
import Tweakbench

// The well: thumbnail +
// Replace/Remove, a dashed empty state. MOSS has no task presets, so: up to four wells with drag and
// drop, and a layout picker for the edit LoRA. See NOTES.md.

/// Generate's Edit section: reference images for an edit LoRA ("Picture 1…4" in the instruction),
/// how the LoRA sees them, and what's missing to run.
struct EditReferencesSection: View {
    @Environment(AppStore.self) private var store
    @Environment(\.theme) private var theme
    @State private var targeted: Int?
    @State private var masking: URL?

    var body: some View {
        @Bindable var store = store
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Text("Drop an image to edit (or up to four for multi-image edits), then describe the change: "
                    + "\"make it night\", \"replace the text with OPEN\". With several, name them: \"the cat from picture 2\".")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(alignment: .top, spacing: 8) {
                    ForEach(0..<slots, id: \.self) { well($0) }
                    Spacer(minLength: 0)
                }
                .tweakable("generate.edit.wells")
                if !store.draft.references.isEmpty { settings.tweakable("generate.edit.settings") }
            }
        } label: { Label("Edit", systemImage: "photo.on.rectangle.angled") }
        .sheet(item: $masking) { url in
            let size = store.draft.size
            let existing = store.draft.editMask.flatMap { $0.source == url ? $0.mask : nil }
            EditMaskSheet(source: url, mask: existing.map {
                ($0.width, $0.height) == (size.width / 16, size.height / 16)
                    ? $0 : EditMask.resampled(from: $0.values, width: $0.width, height: $0.height, toWidth: size.width / 16, toHeight: size.height / 16)
            } ?? .empty(width: size.width / 16, height: size.height / 16))
        }
    }

    /// Well size: four fit across the form's default width.
    static let side: CGFloat = 96

    /// Filled wells plus one empty one (four at most).
    private var slots: Int { min(store.draft.references.count + 1, GenerationRequest.maxReferences) }

    private func well(_ index: Int) -> some View {
        let refs = store.draft.references
        let url = index < refs.count ? refs[index] : nil
        return VStack(spacing: 3) {
            ZStack(alignment: .topTrailing) {
                RoundedRectangle(cornerRadius: 6).fill(theme.well)
                if let url, let image = NSImage(contentsOf: url) {
                    Image(nsImage: image).resizable().aspectRatio(contentMode: .fill)
                        .frame(width: Self.side, height: Self.side).clipShape(RoundedRectangle(cornerRadius: 6))
                    Button { store.draft.references.remove(at: index) } label: {
                        Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                            .foregroundStyle(.white, .black.opacity(0.6))
                    }
                    .buttonStyle(.plain).padding(3).help("Remove")
                } else {
                    RoundedRectangle(cornerRadius: 6).strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4]))
                        .foregroundStyle(.secondary)
                    VStack(spacing: 3) {
                        Image(systemName: "plus")
                        Text("Drop or\nclick").font(.caption2).multilineTextAlignment(.center)
                    }.foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(width: Self.side, height: Self.side)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.accentColor, lineWidth: targeted == index ? 2 : 0))
            .contentShape(Rectangle())
            .onTapGesture { choose(replacing: url == nil ? nil : index) }
            .dropDestination(for: URL.self) { urls, _ in
                guard let first = urls.first(where: { PixelImage.size(of: $0) != nil }) else { return false }
                if url != nil { store.draft.references[index] = first } else { urls.forEach(store.addReference) }
                return true
            } isTargeted: { targeted = $0 ? index : (targeted == index ? nil : targeted) }
            .help(url?.lastPathComponent ?? "Choose or drop a reference image")
            Text(url == nil ? " " : "Picture \(index + 1)").font(.caption2).foregroundStyle(.secondary)
            if index == 0, let url {
                let masked = store.draft.editMask?.source == url
                Button { masking = url } label: {
                    Label(masked ? String(format: "Mask %.0f%%", (store.draft.editMask?.mask.coverage ?? 0) * 100) : "Mask…",
                          systemImage: masked ? "paintbrush.fill" : "paintbrush")
                        .font(.caption2)
                }
                .buttonStyle(.borderless)
                .help("Paint where Picture 1 may change; the rest stays exactly as it is")
                .tweakable("generate.edit.mask")
            }
        }
        .tweakable("generate.edit.well")
    }

    @ViewBuilder private var settings: some View {
        @Bindable var store = store
        VStack(alignment: .leading, spacing: 6) {
            if !store.hasActiveEditLoRA {
                HStack {
                    Label("No edit LoRA in the stack — these images only steer a plain model.", systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                    if !store.editLoRAs.isEmpty {
                        Menu("Add") {
                            ForEach(store.editLoRAs, id: \.self) { url in
                                Button(url.deletingPathExtension().lastPathComponent) { store.draft.loras.append(.init(url: url)) }
                            }
                        }.fixedSize()
                    }
                }
            }
            Picker("Layout", selection: $store.draft.editLayout) {
                Text("Auto — \(store.detectedEditLayout.displayName)").tag(MossEditLayout?.none)
                Divider()
                ForEach(MossEditLayout.allCases) { Text($0.displayName).tag(MossEditLayout?.some($0)) }
            }
            .help("How the edit LoRA was trained to see its references. Auto reads it from the LoRA file; "
                + "ai-toolkit LoRAs (Edit-HQ) also show the references to the text encoder.")
            .tweakable("generate.edit.layout")
            if store.builtInTurbo != nil {
                Label("This model has a built-in Turbo, which can't edit — pick the SFT model with a Turbo LoRA.",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            } else if store.activeTurbo != nil {
                Text("With Turbo, edits run its LoRA on flow Euler at CFG 1 — fast drafts. Turn Turbo off for "
                    + "50 steps, CFG 3 (the edit LoRAs' recommended quality pass).")
                    .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func choose(replacing index: Int?) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = index == nil
        guard panel.runModal() == .OK else { return }
        if let index, let url = panel.url { store.draft.references[index] = url } else { panel.urls.forEach(store.addReference) }
    }
}
