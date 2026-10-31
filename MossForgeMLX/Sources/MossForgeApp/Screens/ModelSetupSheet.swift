import MossForgeCore
import MossForgeEngine
import SwiftUI
import Tweakbench

/// Models → Get Models… (and Generate's first-run prompt): pick groups from the model bundle
/// (Settings → Models → Model bundle repo) and download them into the models folder, checksummed.
/// Already-installed files are skipped and a part-downloaded one resumes from its finished pieces.
struct ModelSetupSheet: View {
    @Environment(AppStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    @State private var manifest: ModelBundleManifest?
    @State private var installed: [String: Int64] = [:]
    @State private var chosen: Set<String> = []
    @State private var progress: ModelBundleProgress?
    @State private var task: Task<Void, Never>?
    @State private var message: String?
    @State private var failed = false

    private var downloader: ModelBundleDownloader {
        ModelBundleDownloader(
            source: ModelBundleSource(repo: store.settings.modelBundleRepo), models: store.settings.modelsDirectory,
            policy: store.settings.networkPolicy)
    }

    var body: some View {
        @Bindable var store = store
        VStack(alignment: .leading, spacing: 12) {
            Text("Get Models").font(.title2.weight(.semibold))
            Text("Downloads MOSS-Image and its text encoder and VAE into \(store.settings.modelsDirectory.path), "
                + "laid out the way MossForge reads them. Add your own weights and LoRAs there any time.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack {
                Text("From").foregroundStyle(.secondary)
                TextField("owner/name", text: $store.settings.modelBundleRepo).frame(maxWidth: 260)
                    .onSubmit { load() }
                    .tweakable("setup.repo")
                Button("Reload") { load() }.disabled(task != nil)
            }
            if !store.settings.egressEnabled {
                VStack(alignment: .leading, spacing: 4) {
                    Toggle("Allow network access", isOn: $store.settings.egressEnabled)
                        .onChange(of: store.settings.egressEnabled) { _, on in if on { load() } }
                    Text("MossForge stays offline until you allow it (Settings → Privacy). This also lets the resident server answer on your network.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                .tweakable("setup.network")
            }
            if let manifest { groups(manifest).tweakable("setup.groups") }
            if let progress {
                ProgressView(value: progress.fraction) {
                    Text("\(progress.fileIndex + 1) of \(progress.fileCount): \((progress.file as NSString).lastPathComponent)")
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                } currentValueLabel: {
                    Text(progressLine(progress)).font(.caption).monospacedDigit()
                }
                .tweakable("setup.progress")
            }
            if let message {
                Text(message).font(.caption).foregroundStyle(failed ? .red : .secondary).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                if task != nil {
                    Button("Cancel", role: .cancel) { task?.cancel() }
                } else {
                    Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                }
                Button(resumable > 0 ? "Resume — \(gb(toFetch - resumable)) left" : "Download \(gb(toFetch))") { start() }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(task != nil || manifest == nil || toFetch == 0 || !store.settings.egressEnabled)
                    .tweakable("setup.download")
            }
        }
        .padding(20)
        .frame(width: 560)
        .onAppear { load() }
        .onDisappear { task?.cancel() }
    }

    private func groups(_ m: ModelBundleManifest) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(m.groups) { g in
                let missing = m.pending(groups: [g.id], existingSizes: installed)
                HStack(alignment: .firstTextBaseline) {
                    Toggle(isOn: Binding(get: { chosen.contains(g.id) }, set: { if $0 { chosen.insert(g.id) } else { chosen.remove(g.id) } })) {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(g.title)
                            Text(g.summary).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .disabled(missing.isEmpty || task != nil)
                    Spacer()
                    Text(missing.isEmpty ? "Installed" : gb(missing.reduce(0) { $0 + $1.size }))
                        .font(.caption).monospacedDigit().foregroundStyle(missing.isEmpty ? .green : .secondary)
                }
            }
        }
    }

    /// Bytes still to fetch for the chosen groups.
    private var toFetch: Int64 {
        manifest?.pending(groups: chosen, existingSizes: installed).reduce(0) { $0 + $1.size } ?? 0
    }

    /// Bytes of the chosen groups already fetched by an interrupted download (kept as finished pieces).
    private var resumable: Int64 {
        guard let manifest, task == nil else { return 0 }
        return downloader.resumableBytes(groups: chosen, manifest: manifest)
    }

    /// "1.2 GB of 15.5 GB · 42 MB/s · about 6 min left"
    private func progressLine(_ p: ModelBundleProgress) -> String {
        var line = "\(gb(p.bytes)) of \(gb(p.totalBytes))"
        if p.bytesPerSecond > 0 { line += String(format: " · %.0f MB/s", p.bytesPerSecond / 1e6) }
        if let left = p.secondsLeft, left.isFinite {
            line += left < 90 ? " · under 2 min left" : " · about \(Int((left / 60).rounded())) min left"
        }
        return line
    }

    private func gb(_ n: Int64) -> String { n >= 1_000_000_000 ? String(format: "%.1f GB", Double(n) / 1e9) : String(format: "%.0f MB", Double(n) / 1e6) }

    private func load() {
        let source = ModelBundleSource(repo: store.settings.modelBundleRepo)
        guard source.isValid else {
            (message, failed) = ("The repo must look like owner/name.", true)
            return
        }
        guard store.settings.egressEnabled else { return }
        let d = downloader
        message = "Reading \(source.repo)…"
        failed = false
        Task {
            do {
                let m = try await d.fetchManifest()
                installed = d.existingSizes(m)
                manifest = m
                chosen = m.defaultGroups.filter { !m.pending(groups: [$0], existingSizes: installed).isEmpty }
                message = nil
            } catch {
                (manifest, message, failed) = (nil, "\(error.localizedDescription)", true)
            }
        }
    }

    private func start() {
        guard let manifest else { return }
        let d = downloader
        let groups = chosen
        (message, failed) = (nil, false)
        task = Task {
            do {
                let written = try await d.download(groups: groups, manifest: manifest) { p in
                    Task { @MainActor in progress = p }
                }
                message = "Downloaded \(written.count) file(s). Ready to generate."
            } catch where Task.isCancelled || error is CancellationError {
                message = "Cancelled — what's downloaded is kept; Resume carries on from there."
            } catch {
                (message, failed) = ("\(error.localizedDescription) What's downloaded is kept — Resume carries on from there.", true)
            }
            installed = d.existingSizes(manifest)
            progress = nil
            task = nil
            store.modelsChanged()
        }
    }
}
