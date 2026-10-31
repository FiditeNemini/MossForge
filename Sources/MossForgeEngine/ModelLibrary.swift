import Foundation
import MossForgeCore

public enum ModelLibraryError: Error, LocalizedError, Equatable {
    case notA(ModelItem.Kind, URL)
    case alreadyExists(URL)
    case outsideModelsDirectory(String)
    case badName(String)

    public var errorDescription: String? {
        switch self {
        case .notA(let kind, let url): "\(url.lastPathComponent) isn't a MOSS-Image \(kind == .dit ? "DiT or checkpoint" : "LoRA")."
        case .alreadyExists(let url): "\(url.lastPathComponent) already exists there."
        case .outsideModelsDirectory(let p): "\(p) is outside the models folder."
        case .badName(let n): "“\(n)” isn't a usable file name."
        }
    }
}

extension ModelSidecar {
    /// `load` with a LoRA's safetensors header metadata (a trained slider's strength range) read from disk.
    public static func detected(for model: URL, isLoRA: Bool) -> ModelSidecar {
        let metadata = isLoRA ? (try? SafetensorsHeader.readEntries(from: model).metadata) ?? [:] : [:]
        return load(for: model, isLoRA: isLoRA, metadata: metadata)
    }
}

/// One DiT or LoRA in the library.
public struct ModelItem: Identifiable, Sendable, Equatable, Hashable {
    public enum Kind: String, Sendable { case dit, lora }

    public var url: URL
    /// Relative to the models folder, e.g. `lora/akhaliq/akhaliq-4step-sft.safetensors`.
    public var relativePath: String
    public var kind: Kind
    public var sidecar: ModelSidecar
    /// Whether `sidecar` was saved (else it's the filename guess).
    public var hasSavedSidecar: Bool
    public var sizeBytes: Int64
    /// `MossModelKind.summary` ("DiT · 42 blocks · bf16", "LoRA · rank 128").
    public var summary: String

    public var id: String { relativePath }
    public var name: String { url.deletingPathExtension().lastPathComponent }
    /// The subfolder under `diffusion_models/` or `lora/` ("" at the top).
    public var folder: String {
        let parts = relativePath.split(separator: "/").dropFirst().dropLast()
        return parts.joined(separator: "/")
    }
    public var isDistill: Bool { sidecar.distill != nil }

    public func hash(into h: inout Hasher) { h.combine(relativePath) }
}

/// The user-managed half of the models folder: DiTs under `diffusion_models/` and LoRAs under `lora/`
/// (any depth). VAE, text encoder and upscalers are configured in Settings, not managed here.
public struct ModelLibrary: Sendable, Equatable {
    public static let ditFolder = "diffusion_models"
    public static let loraFolder = "lora"

    public var dits: [ModelItem] = []
    public var loras: [ModelItem] = []

    public init(dits: [ModelItem] = [], loras: [ModelItem] = []) {
        self.dits = dits
        self.loras = loras
    }

    /// Header reads only (instant on multi-GB files), but still disk I/O — call off the main thread.
    public static func scan(models: URL) -> ModelLibrary {
        var lib = ModelLibrary()
        for (folder, kind) in [(ditFolder, ModelItem.Kind.dit), (loraFolder, .lora)] {
            let dir = models.appendingPathComponent(folder)
            guard let walker = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else { continue }
            for case let url as URL in walker where url.pathExtension == "safetensors" {
                let detected = (try? MossInspector.inspect(url)) ?? .unknown
                var isLoRA = false
                if case .lora = detected { isLoRA = true }
                // Only real LoRAs are listed under lora/; adapters (dataspace, time_embedding) are supporting files.
                if kind == .lora && !isLoRA { continue }
                let saved = ModelSidecar.read(for: url)
                let item = ModelItem(
                    url: url, relativePath: relativePath(url, in: models), kind: kind,
                    sidecar: ModelSidecar.detected(for: url, isLoRA: kind == .lora), hasSavedSidecar: saved != nil,
                    sizeBytes: (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0,
                    summary: detected.summary)
                if kind == .dit { lib.dits.append(item) } else { lib.loras.append(item) }
            }
        }
        // A fused Turbo's `dataspace` file sits beside its DiT (the sampler needs it there): a
        // supporting file, not a model to pick.
        let supporting = Set(lib.dits.flatMap { supportingFiles(of: $0.url, sidecar: $0.sidecar) }.map(\.standardizedFileURL))
        lib.dits.removeAll { supporting.contains($0.url.standardizedFileURL) }
        let byPath: (ModelItem, ModelItem) -> Bool = { $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending }
        lib.dits.sort(by: byPath)
        lib.loras.sort(by: byPath)
        return lib
    }

    /// The adapter files a distillation LoRA's sidecar names (those that exist).
    public func supportingFiles(of item: ModelItem) -> [URL] {
        Self.supportingFiles(of: item.url, sidecar: item.sidecar)
    }

    static func supportingFiles(of url: URL, sidecar: ModelSidecar?) -> [URL] {
        guard let d = sidecar?.distill else { return [] }
        let dir = url.deletingLastPathComponent()
        return [d.dataSpace, d.timeEmbedding].compactMap { $0 }.map { dir.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Distillation (Turbo) LoRAs that work with `family` and have what their route needs.
    public func distills(for family: MossFamily) -> [ModelItem] {
        loras.filter { item in
            guard item.sidecar.supports(family), let d = item.sidecar.distill, DistillSpec(lora: item.url, info: d) != nil else { return false }
            return d.route == .flow || supportingFiles(of: item).count == 2
        }
    }

    /// Ordinary LoRAs that work with `family`.
    public func plainLoRAs(for family: MossFamily) -> [ModelItem] {
        loras.filter { !$0.isDistill && $0.sidecar.supports(family) }
    }

    // MARK: file operations

    /// Bring a file into the library. A LoRA must classify as one. For a DiT: a bare DiT is moved or
    /// copied; an all-in-one checkpoint has its DiT extracted (the source is left alone); a diffusers
    /// `transformer/` folder (or its `*.index.json`) has its shards merged into one file.
    @discardableResult
    public static func importFile(
        _ source: URL, as kind: ModelItem.Kind, subfolder: String, models: URL, move: Bool, name: String? = nil,
        progress: ((Double) -> Void)? = nil
    ) throws -> URL {
        let dir = try folder(kind, subfolder, models: models)
        var isDir: ObjCBool = false
        FileManager.default.fileExists(atPath: source.path, isDirectory: &isDir)

        if kind == .dit, isDir.boolValue || source.lastPathComponent.hasSuffix(".index.json") {
            let index = isDir.boolValue ? try shardIndex(in: source) : source
            let folderName = index.deletingLastPathComponent().lastPathComponent
            let stem = name ?? (folderName == "transformer" ? index.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent : folderName)
            let dest = try destination(dir, stem: stem)
            try mergeShards(index: index, to: dest, progress: progress)
            return dest
        }

        let detected = (try? MossInspector.inspect(source)) ?? .unknown
        let stem = name ?? source.deletingPathExtension().lastPathComponent
        switch (kind, detected) {
        case (.lora, .lora), (.dit, .dit):
            let dest = try destination(dir, stem: stem)
            if move { try FileManager.default.moveItem(at: source, to: dest) } else { try FileManager.default.copyItem(at: source, to: dest) }
            for extra in [ModelSidecar.url(for: source)] where FileManager.default.fileExists(atPath: extra.path) {
                try? FileManager.default.copyItem(at: extra, to: ModelSidecar.url(for: dest))
            }
            progress?(1)
            return dest
        case (.dit, _):
            let names = try SafetensorsHeader.readEntries(from: source).entries.keys.map { $0 }
            guard let plan = CheckpointExtract.plan(tensorNames: names) else { throw ModelLibraryError.notA(.dit, source) }
            let dest = try destination(dir, stem: stem)
            try SafetensorsRepack.write(
                plan.dit.map { .init(source: source, name: $0, as: String($0.dropFirst(plan.prefix.count))) }, to: dest, progress: progress)
            return dest
        default:
            throw ModelLibraryError.notA(kind, source)
        }
    }

    /// Move a model (with its sidecar and adapters) to another subfolder of its kind's root.
    @discardableResult
    public static func move(_ url: URL, toSubfolder subfolder: String, models: URL) throws -> URL {
        let kind: ModelItem.Kind = relativePath(url, in: models).hasPrefix(ditFolder + "/") ? .dit : .lora
        let dir = try folder(kind, subfolder, models: models)
        guard dir.standardizedFileURL != url.deletingLastPathComponent().standardizedFileURL else { return url }
        let dest = dir.appendingPathComponent(url.lastPathComponent)
        guard !FileManager.default.fileExists(atPath: dest.path) else { throw ModelLibraryError.alreadyExists(dest) }
        let sidecar = ModelSidecar.read(for: url)
        let companions = [ModelSidecar.url(for: url)] + supportingFiles(of: url, sidecar: sidecar)
        try FileManager.default.moveItem(at: url, to: dest)
        for c in companions where FileManager.default.fileExists(atPath: c.path) {
            try? FileManager.default.moveItem(at: c, to: dir.appendingPathComponent(c.lastPathComponent))
        }
        return dest
    }

    /// Rename a model in place (its sidecar follows; adapters keep their names, which the sidecar holds).
    @discardableResult
    public static func rename(_ url: URL, to stem: String, models: URL) throws -> URL {
        let clean = stem.trimmingCharacters(in: .whitespaces)
        guard !clean.isEmpty, !clean.contains("/"), !clean.hasPrefix(".") else { throw ModelLibraryError.badName(stem) }
        let dest = url.deletingLastPathComponent().appendingPathComponent(clean + ".safetensors")
        guard dest != url else { return url }
        guard !FileManager.default.fileExists(atPath: dest.path) else { throw ModelLibraryError.alreadyExists(dest) }
        try FileManager.default.moveItem(at: url, to: dest)
        let sidecar = ModelSidecar.url(for: url)
        if FileManager.default.fileExists(atPath: sidecar.path) { try? FileManager.default.moveItem(at: sidecar, to: ModelSidecar.url(for: dest)) }
        return dest
    }

    /// Move a model, its sidecar and its adapters to the Trash.
    @discardableResult
    public static func trash(_ url: URL, models: URL) throws -> [URL] {
        let adapters = supportingFiles(of: url, sidecar: ModelSidecar.read(for: url))
        var moved = try ModelFileOps.moveToTrash(url, root: models)
        for a in adapters where (try? FileManager.default.trashItem(at: a, resultingItemURL: nil)) != nil { moved.append(a) }
        return moved
    }

    /// Existing subfolders of a kind's root (for the Move menu), "" first.
    public static func subfolders(_ kind: ModelItem.Kind, models: URL) -> [String] {
        let root = models.appendingPathComponent(kind == .dit ? ditFolder : loraFolder)
        let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey])
        var out = [""]
        while let url = walker?.nextObject() as? URL {
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                out.append(String(relativePath(url, in: root)))
            }
        }
        return out.sorted()
    }

    // MARK: helpers

    public static func relativePath(_ url: URL, in root: URL) -> String {
        let base = root.standardizedFileURL.path + "/"
        let path = url.standardizedFileURL.path
        return path.hasPrefix(base) ? String(path.dropFirst(base.count)) : url.lastPathComponent
    }

    /// `<models>/<kind root>/<subfolder>`, created, refusing anything that escapes the kind's root.
    static func folder(_ kind: ModelItem.Kind, _ subfolder: String, models: URL) throws -> URL {
        let root = models.appendingPathComponent(kind == .dit ? ditFolder : loraFolder).standardizedFileURL
        let trimmed = subfolder.trimmingCharacters(in: CharacterSet(charactersIn: "/ "))
        let dir = (trimmed.isEmpty ? root : root.appendingPathComponent(trimmed)).standardizedFileURL
        guard dir == root || dir.path.hasPrefix(root.path + "/") else { throw ModelLibraryError.outsideModelsDirectory(subfolder) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func destination(_ dir: URL, stem: String) throws -> URL {
        guard !stem.isEmpty, !stem.contains("/") else { throw ModelLibraryError.badName(stem) }
        let dest = dir.appendingPathComponent(stem + ".safetensors")
        guard !FileManager.default.fileExists(atPath: dest.path) else { throw ModelLibraryError.alreadyExists(dest) }
        return dest
    }

    static func shardIndex(in folder: URL) throws -> URL {
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        let candidates = files + files.filter { $0.lastPathComponent == "transformer" }.flatMap {
            (try? FileManager.default.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil)) ?? []
        }
        guard let index = candidates.first(where: { $0.lastPathComponent.hasSuffix(".safetensors.index.json") }) else {
            throw ModelLibraryError.notA(.dit, folder)
        }
        return index
    }

    /// Merge a diffusers `*.safetensors.index.json` + shards into one file (names unchanged).
    static func mergeShards(index: URL, to dest: URL, progress: ((Double) -> Void)?) throws {
        let data = try Data(contentsOf: index)
        guard let map = (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["weight_map"] as? [String: String] else {
            throw ModelLibraryError.notA(.dit, index)
        }
        let dir = index.deletingLastPathComponent()
        let picks = map.map { SafetensorsRepack.Pick(source: dir.appendingPathComponent($0.value), name: $0.key, as: $0.key) }
        try SafetensorsRepack.write(picks, to: dest, progress: progress)
        guard case .dit = (try? MossInspector.inspect(dest)) ?? .unknown else {
            try? FileManager.default.removeItem(at: dest)
            throw ModelLibraryError.notA(.dit, index)
        }
    }
}
