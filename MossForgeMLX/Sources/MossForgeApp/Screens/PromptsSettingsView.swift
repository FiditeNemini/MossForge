import AppKit
import MossForgeCore
import SwiftUI
import Tweakbench

/// Settings → Prompts: the instructions sent to Qwen3.5 (enhancer, captioner, slider designer), editable
/// with a per-prompt Reset, and a Try it run that shows the answer the edit produces.
struct PromptsSettingsView: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        @Bindable var store = store
        // A ScrollView, not a grouped Form: the Form re-measures its TextEditor rows on every settings
        // change and jumps back to the top while you type.
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Edit the instructions MossForge sends to the on-device Qwen3.5 model. Saved as you type; used by the app (not the server or CLI). Try it runs one with your edits.")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Reset all") { store.settings.prompts.resetAll() }
                        .disabled(store.settings.prompts.isDefault)
                        .tweakable("settings.prompts.resetAll")
                }
                ForEach(MossPrompts.Kind.allCases) { kind in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(kind.title).font(.headline)
                            Text("edited").font(.caption2).foregroundStyle(.orange)
                                .opacity(store.settings.prompts.isDefault(kind) ? 0 : 1)
                            Spacer()
                            Button("Reset") { store.settings.prompts.reset(kind) }.controlSize(.small)
                                .disabled(store.settings.prompts.isDefault(kind))
                        }
                        Text(kind.detail).font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $store.settings.prompts[kind])
                            .font(.system(.callout, design: .monospaced))
                            .frame(height: kind == .caption || kind == .enhancerLayout || kind == .slider ? 200 : 120)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                            .tweakable("settings.prompts.editor")
                        tryIt(kind)
                    }
                    .padding(12)
                    .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.35)))
                    .tweakable("settings.prompts.card")
                }
            }
            .padding(20)
            .frame(maxWidth: 820)
            .frame(maxWidth: .infinity)
        }
        .tweakable("settings.prompts.form")
    }

    /// The sample input, the Run button and the last answer for one prompt.
    @ViewBuilder private func tryIt(_ kind: MossPrompts.Kind) -> some View {
        @Bindable var store = store
        let input = Binding(get: { store.promptLab.inputs[kind] ?? "" }, set: { store.promptLab.inputs[kind] = $0 })
        let sample = PromptLab.samples[kind] ?? ""
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(kind == .caption ? "Test image and reference caption" : "Test input").font(.caption.bold()).foregroundStyle(.secondary)
                Spacer()
                if input.wrappedValue != sample {
                    Button("Reset input") { input.wrappedValue = sample }.controlSize(.small)
                }
            }
            if kind == .caption {
                HStack {
                    Text(store.promptLabImage.map { "Image: \($0.lastPathComponent)" } ?? "No image — choose one, or generate one first")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button("Choose…") {
                        let panel = NSOpenPanel()
                        panel.allowedContentTypes = [.image]
                        panel.directoryURL = store.promptLabImage?.deletingLastPathComponent() ?? store.settings.outputDirectory
                        if panel.runModal() == .OK, let url = panel.url { store.promptLab.image = url }
                    }
                }
            }
            TextField(
                kind == .caption ? "Optional: an existing caption for {{reference}} (empty = none)"
                    : kind == .slider ? "Describe a slider, e.g. young to old" : "An image idea to enhance",
                text: input, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)
                .tweakable("settings.prompts.input")
        }
        HStack {
            Spacer()
            if store.promptLab.running == kind {
                ProgressView().controlSize(.small)
            }
            Button("Try it") { store.tryPrompt(kind) }
                .disabled(!store.canTryPrompt(kind))
                .help(store.canChat ? "Run this prompt as edited and show the answer" : "Needs the MLX backend (Qwen3.5 found)")
        }
        .tweakable("settings.prompts.try")
        if let result = store.promptLab.results[kind] {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(result.failed ? "Failed" : "Answer").font(.caption.bold())
                        .foregroundStyle(result.failed ? .red : .secondary)
                    Text(String(format: "%.1f s", result.seconds)).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(result.text, forType: .string)
                    }
                    .controlSize(.small)
                }
                Text(result.text)
                    .font(.system(.callout, design: kind == .enhancerLayout ? .monospaced : .default))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.5)))
            }
            .tweakable("settings.prompts.result")
        }
    }
}
