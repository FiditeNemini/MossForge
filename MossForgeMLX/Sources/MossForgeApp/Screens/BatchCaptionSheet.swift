import AppKit
import MossForgeCore
import SwiftUI

/// What to caption in a batch: gallery images (into their sidecars) or a folder of training images
/// (as kohya `name.txt` files).
enum BatchCaptionJob: Identifiable {
    case gallery(title: String, images: [URL])
    case folder(URL)

    var id: String {
        switch self {
        case .gallery(let title, let images): "g:\(title):\(images.count)"
        case .folder(let url): "f:\(url.path)"
        }
    }
}

/// Batch captioning with the on-device VLM (Qwen3.5 vision), one image after another; the captioner
/// stays loaded between images. Already-captioned images are skipped unless "Re-caption" is on.
struct BatchCaptionSheet: View {
    @Environment(AppStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let job: BatchCaptionJob
    @State private var overwrite = false
    @AppStorage("folderCaptionTrigger") private var trigger = ""
    @State private var started = false

    var body: some View {
        @Bindable var store = store
        VStack(alignment: .leading, spacing: 14) {
            Text(title).font(.title2.bold())
            Text(summary).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Picker("Caption", selection: $store.captionDetail) {
                ForEach(CaptionDetail.allCases) { Text($0.rawValue.capitalized).tag($0) }
            }
            .pickerStyle(.segmented)
            .disabled(started)
            if case .folder = job {
                TextField("Trigger word (optional)", text: $trigger).textFieldStyle(.roundedBorder).disabled(started)
                Text("Written as “trigger, caption”. An existing .txt is passed to the model as a reference to correct.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Text("The full long / medium / short answer is stored; this picks the one shown and exported. Each image's prompt is the reference the model corrects.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Toggle("Re-caption images that already have a caption", isOn: $overwrite).disabled(started)
            if let b = store.captionBatch {
                ProgressView(value: Double(b.done), total: Double(max(b.total, 1))) {
                    Text("\(min(b.done + 1, b.total)) of \(b.total) — \(b.current)").font(.caption)
                }
            }
            if started, !store.captionStatus.isEmpty {
                Text(store.captionStatus).font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Text("~15–20 s per image").font(.caption).foregroundStyle(.tertiary)
                Spacer()
                if store.captionBatch != nil {
                    Button("Stop", role: .cancel) { store.cancelCaptions() }
                    Button("Hide") { dismiss() }
                } else {
                    Button(started ? "Done" : "Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                    if !started {
                        Button("Start") { start() }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                            .disabled(pending == 0)
                    }
                }
            }
        }
        .padding(20)
        .frame(width: 460)
    }

    private var title: String {
        switch job {
        case .gallery(let title, _): title
        case .folder(let url): "Caption “\(url.lastPathComponent)”"
        }
    }

    /// Images the run would caption with the current settings.
    private var pending: Int {
        switch job {
        case .gallery(_, let images):
            overwrite ? images.count : images.filter { GallerySidecar.caption(in: GallerySidecar.read(for: $0)) == nil }.count
        case .folder(let url):
            FolderCaptioning.pending(in: url, overwrite: overwrite).count
        }
    }

    private var summary: String {
        let n = pending
        switch job {
        case .gallery(_, let images):
            return "\(n) of \(images.count) image\(images.count == 1 ? "" : "s") to caption (into their gallery sidecars)."
        case .folder(let url):
            let total = FolderCaptioning.pending(in: url, overwrite: true).count
            return "\(n) of \(total) image\(total == 1 ? "" : "s") in the folder to caption, each as name.txt beside it."
        }
    }

    private func start() {
        started = true
        switch job {
        case .gallery(_, let images): store.captionGallery(images, overwrite: overwrite)
        case .folder(let url): store.captionFolder(url, overwrite: overwrite, trigger: trigger)
        }
    }
}
