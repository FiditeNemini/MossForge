import AppKit
import MossForgeCore
import MossForgeEngine
import SwiftUI

// Gallery file URLs + JSON sidecars, and a caption-length choice (MOSS recaptions are
// long / medium / short).

/// What the Gallery asked to export.
struct DatasetExportJob: Identifiable {
    let id = UUID()
    var title: String
    var images: [URL]
    /// `true` → kohya / ai-toolkit dataset with caption `.txt` files; `false` → images only.
    var captions: Bool
    /// Pre-fills the trigger word (Character Studio), whose prompts are the intended captions.
    var trigger: String?
}

/// Gallery → Export…: pick a folder, copy the images under unique filesystem-safe names (never
/// overwriting), and for a dataset write each caption (sidecar caption at the chosen length, else
/// the prompt) with an optional trigger-word prefix.
struct DatasetExportSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State var job: DatasetExportJob
    @AppStorage("datasetExportTrigger") private var trigger = ""
    @AppStorage("datasetExportDetail") private var detail: CaptionDetail = .long
    @State private var outcome: String?
    @State private var folder: URL?

    private var uncaptioned: Int {
        job.images.filter { GallerySidecar.caption(in: GallerySidecar.read(for: $0)) == nil }.count
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(job.title).font(.title2.bold())
            Toggle("Write caption .txt files (training dataset)", isOn: $job.captions)
            Text(summary).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if job.captions {
                Picker("Caption", selection: $detail) {
                    ForEach(CaptionDetail.allCases) { Text($0.rawValue.capitalized).tag($0) }
                }
                .pickerStyle(.segmented)
                TextField("Trigger word (optional)", text: $trigger).textFieldStyle(.roundedBorder)
                Text("Prefixes every caption as “trigger, caption”.").font(.caption).foregroundStyle(.secondary)
            }
            if let outcome { Text(outcome).font(.callout).fixedSize(horizontal: false, vertical: true) }
            HStack {
                if let folder, outcome != nil { Button("Show in Finder") { NSWorkspace.shared.open(folder) } }
                Spacer()
                Button(outcome == nil ? "Cancel" : "Done") { dismiss() }.keyboardShortcut(.cancelAction)
                if outcome == nil {
                    Button("Choose Folder…") { run() }.keyboardShortcut(.defaultAction).disabled(job.images.isEmpty)
                }
            }
        }
        .padding(20)
        .frame(width: 440)
        .onAppear { if let t = job.trigger { trigger = t } }
    }

    private var summary: String {
        let n = job.images.count
        var text = job.captions
            ? "\(n) image\(n == 1 ? "" : "s") as name.png + name.txt (kohya / ai-toolkit layout)."
            : "Copies \(n) image\(n == 1 ? "" : "s") into a folder."
        if job.captions && uncaptioned > 0 && job.trigger == nil {
            text += " \(uncaptioned) without a caption use their prompt — caption them first for better training text."
        }
        return text + " Existing files are never overwritten."
    }

    private func run() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Export"
        guard panel.runModal() == .OK, let dir = panel.url else { return }
        folder = dir
        let items = job.images.map { GalleryDataset.item(image: $0, meta: GallerySidecar.read(for: $0), detail: detail) }
        do {
            let result = try DatasetExporter.export(items, to: dir, captions: job.captions, trigger: job.captions ? trigger : "")
            var text = "Wrote \(result.written.count) image\(result.written.count == 1 ? "" : "s")"
            if job.captions { text += " and captions" }
            text += " to \(dir.lastPathComponent)."
            if !result.skipped.isEmpty { text += " Skipped \(result.skipped.count) missing file(s)." }
            outcome = text
        } catch {
            outcome = "Export failed: \(error.localizedDescription)"
        }
    }
}
