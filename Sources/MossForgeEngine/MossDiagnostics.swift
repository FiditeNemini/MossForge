import Foundation
import MossForgeCore

/// What the app hands the collector: its own state (never prompts, images or gallery names).
/// MossForge's own diagnostics sections for the generic `DiagnosticsCollector`: the MOSS components, GPU state and the opt-in model file names. Local
/// checks only; paths are reported relative to `<models>`.
public enum MossDiagnostics {
    /// Sections placed after Storage: Models, GPU / compute, Model files (opt-in).
    public static func sections(
        modelsDirectory: URL, overrides: MossModelLayout.Overrides, backendName: String, backendItems: [DiagnosticSection.Item]
    ) -> [DiagnosticSection] {
        [models(modelsDirectory, overrides: overrides), compute(backendName: backendName, items: backendItems), modelFiles(modelsDirectory)]
    }

    /// The context for `DiagnosticsCollector.collect`, with MossForge's folders, sections and the
    /// component overrides listed under Settings.
    public static func context(
        appVersion: String, backendName: String, capabilities: EngineCapabilities, modelsDirectory: URL, outputDirectory: URL,
        overrides: MossModelLayout.Overrides = .init(), settings: [(String, String)], log: DiagnosticsLog,
        backendItems: [DiagnosticSection.Item]
    ) -> DiagnosticsContext {
        let set = MossModelLayout.Component.allCases.filter { overrides.url(for: $0) != nil }.map(\.title)
        return DiagnosticsContext(
            appVersion: appVersion, backendName: backendName, capabilities: capabilities.names,
            folders: [("Models folder", modelsDirectory, true), ("Output folder", outputDirectory, false)],
            settings: settings + [("Component overrides", set.isEmpty ? "none" : set.joined(separator: ", "))], log: log,
            extraSections: sections(modelsDirectory: modelsDirectory, overrides: overrides, backendName: backendName, backendItems: backendItems),
            minimumMemoryGB: 32, minimumFreeGB: 20)
    }

    /// The components MossForge needs (✗ when missing), then the optional ones (⚠︎ / info).
    static func models(_ dir: URL, overrides: MossModelLayout.Overrides) -> DiagnosticSection {
        let required: [MossModelLayout.Component] = [.sftDiT, .baseDiT, .textEncoder, .tokenizer, .vae]
        var items: [DiagnosticSection.Item] = []
        let hasDiT = [MossModelLayout.Component.sftDiT, .baseDiT].contains { MossModelLayout.find($0, in: dir, overrides: overrides) != nil }
            || count(in: dir.appendingPathComponent(ModelLibrary.ditFolder)) > 0
        for comp in MossModelLayout.Component.allCases {
            let isRequired = required.contains(comp)
            guard let url = MossModelLayout.find(comp, in: dir, overrides: overrides) else {
                let expected = "<models>/" + comp.candidates[0]
                let status: DiagnosticSection.Status = switch comp {
                case .sftDiT, .baseDiT: hasDiT ? .info : .fail
                default: isRequired ? .fail : .info
                }
                let value = isRequired ? "missing — expected \(expected)" : "not installed (optional: \(comp.usedBy))"
                items.append(.init(comp.title, value, status: status))
                continue
            }
            let overridden = overrides.url(for: comp) != nil
            items.append(check(comp, url, overridden: overridden))
        }
        let dits = count(in: dir.appendingPathComponent(ModelLibrary.ditFolder))
        let loras = count(in: dir.appendingPathComponent("lora"))
        items.append(.init("Diffusion models", "\(dits)", status: dits == 0 ? .fail : .ok))
        items.append(.init("LoRAs", "\(loras)"))
        return DiagnosticSection(id: "models", title: "Models", items: items, note: "Which components are present, their sizes and whether their headers read correctly")
    }

    /// Size + a header check: the file must read as the kind the component expects.
    static func check(_ comp: MossModelLayout.Component, _ url: URL, overridden: Bool) -> DiagnosticSection.Item {
        let source = overridden ? "set in Settings" : "standard place"
        if comp == .tokenizer {
            let ok = FileManager.default.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path)
            return .init(comp.title, ok ? "present (\(source))" : "folder has no tokenizer.json", status: ok ? .ok : .fail)
        }
        let size = ((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize).map { String(format: "%.1f GB", Double($0) / 1e9) } ?? "?"
        let kind = try? MossInspector.inspect(url)
        let expected: Bool = switch (comp, kind) {
        case (.sftDiT, .dit?), (.baseDiT, .dit?), (.vosrDiT, _), (.vosrDINOv2, _), (.vosrVAE, _): kind != nil
        case (.textEncoder, .textEncoder?), (.vae, .vae?), (.turboLoRA, .lora?): true
        default: false
        }
        // MossInspector only knows MOSS formats: other readable files (VOSR2, DINOv2) are just "present".
        let described = kind.map { $0 == .unknown ? "present" : describe($0) } ?? "unreadable header"
        return .init(comp.title, "\(size) · \(described) (\(source))", status: expected ? .ok : .fail)
    }

    static func describe(_ kind: MossModelKind) -> String {
        switch kind {
        case .dit(let config, let precision): "DiT \(config.layers) blocks, \(precision)"
        case .textEncoder(let layers, _): "text encoder, \(layers) layers"
        case .vae(let ch): "VAE, \(ch) latent channels"
        case .lora(let rank): "LoRA rank \(rank)"
        case .dmdSampler(let steps): "Turbo sampler, \(steps) steps"
        case .unknown: "unrecognised"
        }
    }

    static func compute(backendName: String, items: [DiagnosticSection.Item]) -> DiagnosticSection {
        DiagnosticSection(id: "compute", title: "GPU / compute", items: items.isEmpty ? [.init("Backend", "no GPU state (\(backendName))")] : items)
    }

    /// DiT and LoRA file names (relative to `<models>`) — opt-in: names can be personal.
    static func modelFiles(_ models: URL) -> DiagnosticSection {
        var items: [DiagnosticSection.Item] = []
        for folder in [ModelLibrary.ditFolder, "lora"] {
            for rel in files(in: models.appendingPathComponent(folder)) {
                items.append(.init(folder == "lora" ? "LoRA" : "DiT", "<models>/\(folder)/\(rel)"))
            }
        }
        return DiagnosticSection(id: "modelFiles", title: "Model files", items: items.isEmpty ? [.init("Files", "none")] : items,
            included: false, note: "Names of your diffusion models and LoRAs — leave off if they're personal")
    }

    static func files(in folder: URL) -> [String] {
        guard let e = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return [] }
        let base = folder.standardizedFileURL.path
        return e.compactMap { $0 as? URL }
            .filter { $0.pathExtension == "safetensors" }
            .map { String($0.standardizedFileURL.path.dropFirst(base.count + 1)) }
            .sorted()
    }

    static func count(in folder: URL) -> Int { files(in: folder).count }
}
