import AppKit
import MossForgeCore
import MossForgeEngine
import SwiftUI
import Tweakbench

/// Settings: General — appearance (theme picker), folders, network + server, memory,
/// diagnostics (opt-in) — and Prompts (`PromptsSettingsView`).
struct SettingsView: View {
    enum Tab: String, CaseIterable, Identifiable {
        case general = "General", prompts = "Prompts"
        var id: String { rawValue }
    }

    @Environment(AppStore.self) private var store
    @AppStorage("settings.tab") private var tab = Tab.general

    var body: some View {
        VStack(spacing: 0) {
            Picker("Settings", selection: $tab) {
                ForEach(Tab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .padding(.top, 12)
            .tweakable("settings.tabs")
            switch tab {
            case .general: general
            case .prompts: PromptsSettingsView()
            }
        }
    }

    private var general: some View {
        @Bindable var store = store
        return Form {
            Section("Appearance") { ThemePicker(selection: $store.settings.theme).tweakable("settings.theme") }
            Section("Folders") {
                folderRow("Models", store.settings.modelsDirectory) { store.settings.bookmarks.remember($0); store.settings.modelsDirectory = $0 }
                folderRow("Output", store.settings.outputDirectory) { store.settings.bookmarks.remember($0); store.settings.outputDirectory = $0 }
                HStack {
                    TextField("Model bundle", text: $store.settings.modelBundleRepo, prompt: Text(ModelBundleSource.defaultRepo))
                        .help("The Hugging Face repo Models → Get Models… downloads from (owner/name)")
                    if store.settings.modelBundleRepo != ModelBundleSource.defaultRepo {
                        Button("Default") { store.settings.modelBundleRepo = ModelBundleSource.defaultRepo }
                    }
                }
                .tweakable("settings.modelBundle")
                Text("Backend: \(store.backendName) — MLX loads when an SFT/Base DiT, the Qwen3.5 encoder + tokenizer and the FLUX.2 VAE are found.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                componentRow(.vae, \.vae, folder: false)
                componentRow(.textEncoder, \.textEncoder, folder: false)
                componentRow(.tokenizer, \.tokenizer, folder: true)
            } header: {
                Text("Shared components")
            } footer: {
                Text("Used by every diffusion model. Unset = the models folder's standard place. Upscalers are read from models/vosr and models/esrgan.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Network & server") {
                Toggle("Allow network access (server reachable on the LAN)", isOn: $store.settings.egressEnabled)
                    .tweakable("settings.network")
                Stepper("Server port \(store.settings.serverPort)", value: $store.settings.serverPort, in: 1024...65535)
                    .tweakable("settings.port")
                Stepper(
                    store.settings.idleUnloadMinutes == 0 ? "Never unload idle models" : "Unload models after \(store.settings.idleUnloadMinutes) idle min",
                    value: $store.settings.idleUnloadMinutes, in: 0...120)
                    .tweakable("settings.idleUnload")
            }
            Section {
                Picker("Memory", selection: $store.settings.residencyMode) {
                    ForEach(MossResidencyMode.allCases) { Text($0.title).tag($0) }
                }
                .tweakable("settings.memory")
            } header: {
                Text("Memory")
            } footer: {
                Text(store.residency.summary
                    + (store.settings.residencyMode == .auto ? " (Automatic: chosen from this Mac's GPU memory.)" : ""))
                    .font(.caption).foregroundStyle(.secondary)
                    .tweakable("settings.memory.summary")
            }
            Section {
                Toggle("Enable diagnostics", isOn: $store.settings.diagnosticsEnabled)
                    .tweakable("settings.diagnostics")
                if store.settings.diagnosticsEnabled {
                    HStack {
                        Button("Write feedback…") { store.help.tab = .feedback; store.selection = .help }
                        Button("Clear log (\(store.help.diagnosticsLog.entries.count) events)") { store.help.diagnosticsLog.clear() }
                            .disabled(store.help.diagnosticsLog.entries.isEmpty)
                    }
                    .tweakable("settings.diagnostics.actions")
                }
            } header: {
                Text("Diagnostics & feedback")
            } footer: {
                Text(store.settings.diagnosticsEnabled
                    ? "On: job outcomes and errors (never prompts or images) are kept in memory — the last 200, gone when you quit or turn this off. Help → Feedback runs local checks and helps you write a redacted report. Nothing is sent: you copy, save or share it yourself."
                    : "Off: nothing about how MossForge runs is recorded. Turn on to report a problem or idea from Help → Feedback — collected and redacted on this Mac, shared only if you choose to.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .tweakable("settings.form")
    }

    /// One shared component: the file in use (override or the layout's), Choose…, Reset.
    private func componentRow(
        _ c: MossModelLayout.Component, _ key: WritableKeyPath<MossModelLayout.Overrides, URL?>, folder: Bool
    ) -> some View {
        let models = store.settings.modelsDirectory
        let override = store.settings.overrides[keyPath: key]
        let inUse = MossModelLayout.find(c, in: models, overrides: store.settings.overrides)
        return HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(c.title)
                Group {
                    if let inUse {
                        Text((override != nil ? "" : "default · ") + inUse.path)
                    } else {
                        Text("missing — expected \(models.appendingPathComponent(c.candidates[0]).path)").foregroundStyle(.orange)
                    }
                }
                .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            Spacer()
            if override != nil {
                Button("Reset") { store.settings.overrides[keyPath: key] = nil }
            }
            Button("Choose…") {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = folder
                panel.canChooseFiles = !folder
                panel.directoryURL = (inUse ?? models).deletingLastPathComponent()
                if panel.runModal() == .OK, let chosen = panel.url {
                    store.settings.bookmarks.remember(chosen)
                    store.settings.overrides[keyPath: key] = chosen
                }
            }
        }
        .tweakable("settings.component")
    }

    private func folderRow(_ name: String, _ url: URL, set: @escaping (URL) -> Void) -> some View {
        HStack {
            Text(name)
            Text(url.path).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            Spacer()
            Button("Choose…") {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.directoryURL = url
                if panel.runModal() == .OK, let chosen = panel.url { set(chosen) }
            }
        }
        .tweakable("settings.folder")
    }
}
