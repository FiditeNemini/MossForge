import AppKit
import MossForgeCore
import MossForgeEngine
import SwiftUI

// No in-app downloads
// (MossForge's models are user-installed), works on a gallery file URL.

/// Gallery → Upscale…: enlarge one image with an ESRGAN-family model (fast) or VOSR2 (generative,
/// rebuilds texture). The result is saved to the output folder and shows up in the gallery.
struct UpscaleSheet: View {
    @Environment(AppStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let url: URL

    var body: some View {
        @Bindable var store = store
        let missing = store.missingVOSR
        VStack(alignment: .leading, spacing: 14) {
            Text("Upscale").font(.title2.bold())
            Text("\(url.lastPathComponent) · \(sizeText)").font(.callout).foregroundStyle(.secondary)

            Picker("Method", selection: $store.upscaleMethod) {
                ForEach(UpscaleMethod.allCases) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)

            Picker("Scale", selection: $store.upscaleScale) {
                Text("×2").tag(2.0)
                Text("×3").tag(3.0)
                Text("×4").tag(4.0)
            }
            .pickerStyle(.segmented)

            if store.upscaleMethod == .vosr {
                if !missing.isEmpty {
                    GroupBox {
                        Text("VOSR2 needs its weights under the models folder:\n" + missing.joined(separator: "\n"))
                            .font(.caption).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                Picker("Input softening", selection: $store.upscaleSettings.soften) {
                    ForEach(UpscaleSoftening.allCases) { Text($0.displayName).tag($0) }
                }
                Text("Smooths fine generation grain so VOSR doesn't turn it into fabric-weave texture. Off keeps every input detail.")
                    .font(.caption2).foregroundStyle(.secondary)
                Picker("Colour match", selection: $store.upscaleSettings.colorFix) {
                    Text("Wavelet").tag(UpscaleColorFix.wavelet)
                    Text("AdaIN").tag(UpscaleColorFix.adain)
                    Text("None").tag(UpscaleColorFix.none)
                }
                HStack {
                    Text("Seed")
                    TextField("42", value: $store.upscaleSettings.seed, format: .number.grouping(.never)).frame(width: 90)
                    Spacer()
                    Stepper(
                        "Overlap \(store.upscaleSettings.overlap) px", value: $store.upscaleSettings.overlap,
                        in: UpscaleSettings.overlapRange, step: 64)
                }
            } else if store.availableUpscalers.isEmpty {
                Text("No ESRGAN model in models/esrgan.").font(.caption).foregroundStyle(.secondary)
            } else {
                Picker("Model", selection: $store.upscaleESRGAN) {
                    Text("Auto").tag("")
                    ForEach(store.availableUpscalers, id: \.self) {
                        Text($0.replacingOccurrences(of: ".safetensors", with: "")).tag($0)
                    }
                }
            }

            if store.isUpscaling {
                ProgressView(value: store.upscaleProgress) { Text(store.upscaleStatus).font(.caption) }
            } else if !store.upscaleStatus.isEmpty {
                Text(store.upscaleStatus).font(.caption).foregroundStyle(.secondary)
            }

            HStack {
                Spacer()
                Button("Close") { dismiss() }
                if store.isUpscaling {
                    Button("Cancel", role: .cancel) { store.cancelUpscale() }
                } else {
                    Button("Upscale") { store.upscale(url) }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.defaultAction)
                        .disabled(
                            (store.upscaleMethod == .vosr && !missing.isEmpty)
                                || (store.upscaleMethod == .esrgan && store.availableUpscalers.isEmpty))
                }
            }
        }
        .padding(20)
        .frame(width: 440)
    }

    private var sizeText: String {
        guard let rep = NSImageRep(contentsOf: url) else { return "" }
        let (w, h) = UpscaleTiling.targetSize(width: rep.pixelsWide, height: rep.pixelsHigh, scale: store.upscaleScale)
        return "\(rep.pixelsWide) × \(rep.pixelsHigh) → \(w) × \(h)"
    }
}

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}
