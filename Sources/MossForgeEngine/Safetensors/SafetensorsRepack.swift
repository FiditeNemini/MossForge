import Foundation
import MossForgeCore

/// Write a new safetensors file from tensors picked out of existing ones (optionally renamed), copying
/// their bytes in chunks — never holding a multi-GB checkpoint in memory. Used to extract the DiT from
/// an all-in-one checkpoint and to merge a diffusers `transformer/` folder's shards into one file.
public enum SafetensorsRepack {
    public struct Pick: Sendable, Equatable {
        public var source: URL
        public var name: String
        public var newName: String
        public init(source: URL, name: String, as newName: String) {
            self.source = source
            self.name = name
            self.newName = newName
        }
    }

    static let chunk = 64 << 20

    /// Writes to a temporary file beside `url` and moves it into place, so a failed or cancelled copy
    /// never leaves a half-written model behind. `progress` gets the fraction of bytes copied.
    public static func write(
        _ picks: [Pick], metadata: [String: String] = [:], to url: URL, progress: ((Double) -> Void)? = nil
    ) throws {
        var headers: [URL: [String: SafetensorsHeader.Entry]] = [:]
        for source in Set(picks.map(\.source)) { headers[source] = try SafetensorsHeader.readEntries(from: source).entries }
        var header: [String: Any] = [:]
        var offset = 0
        var plan: [(Pick, SafetensorsHeader.Entry)] = []
        for p in picks.sorted(by: { $0.newName < $1.newName }) {
            guard let e = headers[p.source]?[p.name] else { throw EngineError.backendFailure("safetensors: no tensor named '\(p.name)'") }
            guard header[p.newName] == nil else { throw EngineError.invalidRequest("duplicate tensor '\(p.newName)'") }
            let size = e.end - e.start
            header[p.newName] = ["dtype": e.dtype, "shape": e.shape, "data_offsets": [offset, offset + size]]
            offset += size
            plan.append((p, e))
        }
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerData.count % 8 != 0 { headerData.append(0x20) }

        let fm = FileManager.default
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let temp = url.deletingLastPathComponent().appendingPathComponent(".\(url.lastPathComponent).partial")
        fm.createFile(atPath: temp.path, contents: nil)
        do {
            let out = try FileHandle(forWritingTo: temp)
            defer { try? out.close() }
            var len = UInt64(headerData.count).littleEndian
            try out.write(contentsOf: withUnsafeBytes(of: &len) { Data($0) })
            try out.write(contentsOf: headerData)
            var handles: [URL: FileHandle] = [:]
            defer { handles.values.forEach { try? $0.close() } }
            var copied = 0
            for (p, e) in plan {
                if handles[p.source] == nil { handles[p.source] = try FileHandle(forReadingFrom: p.source) }
                let input = handles[p.source]!
                try input.seek(toOffset: UInt64(e.start))
                var remaining = e.end - e.start
                while remaining > 0 {
                    guard let data = try input.read(upToCount: min(chunk, remaining)), !data.isEmpty else {
                        throw EngineError.backendFailure("safetensors: truncated tensor '\(p.name)'")
                    }
                    try out.write(contentsOf: data)
                    remaining -= data.count
                    copied += data.count
                    progress?(offset == 0 ? 1 : Double(copied) / Double(offset))
                }
                if Task.isCancelled { throw CancellationError() }
            }
        } catch {
            try? fm.removeItem(at: temp)
            throw error
        }
        if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
        try fm.moveItem(at: temp, to: url)
    }
}

/// Finds the MOSS DiT inside an all-in-one checkpoint (Comfy "checkpoint" layout: the DiT under
/// `model.diffusion_model.`, with the VAE and text encoder baked in beside it).
public enum CheckpointExtract {
    public struct Plan: Sendable, Equatable {
        public var prefix: String
        /// Full names of the DiT tensors (strip `prefix` to get the stand-alone names).
        public var dit: [String]
        public var hasVAE: Bool
        public var hasTextEncoder: Bool
    }

    static let ditPrefixes = ["model.diffusion_model.", "diffusion_model."]

    /// `nil` when the names are already a bare DiT (or hold no DiT at all).
    public static func plan(tensorNames: [String]) -> Plan? {
        for prefix in ditPrefixes {
            let dit = tensorNames.filter { $0.hasPrefix(prefix) }.sorted()
            guard dit.contains(where: { $0.dropFirst(prefix.count).hasPrefix("layers.") }) else { continue }
            let rest = tensorNames.filter { !$0.hasPrefix(prefix) }
            return Plan(
                prefix: prefix, dit: dit,
                hasVAE: rest.contains { $0.hasPrefix("vae.") || $0.hasPrefix("first_stage_model.") },
                hasTextEncoder: rest.contains { $0.hasPrefix("text_encoders.") || $0.hasPrefix("cond_stage_model.") })
        }
        return nil
    }
}
