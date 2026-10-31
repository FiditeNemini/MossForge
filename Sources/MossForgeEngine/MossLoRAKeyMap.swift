import Foundation
import MossForgeCore

/// One update aimed at (a row slice of) a canonical DiT tensor: a low-rank `ΔW = scale·up·down`, or a
/// ComfyUI full delta — `.diff` (weight, e.g. Kijai's norm diffs) / `.diff_b` (bias).
public struct MossLoRAPart: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case lowRank
        /// `target += strength · diff` (`down` names the diff tensor; `up` is empty).
        case weightDiff
        /// Same, for `<module>.bias`.
        case biasDiff
        /// LyCORIS LoKr: `ΔW = scale · (w1 ⊗ w2)` (`down` = w1 `[a, b]`, `up` = w2 `[c, d]`, a·c × b·d).
        case kronecker
        /// LyCORIS LoHa: `ΔW = scale · (up·down) ⊙ (up2·down2)` (`hada_w1_a·hada_w1_b` ⊙ `hada_w2_a·hada_w2_b`).
        /// Dense, so it is merged onto a saved copy of the target rather than applied unmerged.
        case hadamard
    }

    public var kind: Kind = .lowRank
    /// DiT block (`layers.N`) the update belongs to; nil for embedders, modulation, text adapter, head.
    public var block: Int?
    /// Canonical weight key in MossForge's layout, e.g. `layers.3.proj.qkv.weight`.
    public var target: String
    /// Output rows of `target` this update covers (q|k|v slices of the fused qkv); `nil` = all rows.
    public var rows: Range<Int>?
    /// Raw tensor names in the LoRA file: `down` `[rank, in]`, `up` `[out, rank]`.
    public var down: String
    public var up: String
    /// LoHa's second factor pair (`hada_w2_b` `[rank, in]`, `hada_w2_a` `[out, rank]`); empty otherwise.
    public var down2 = ""
    public var up2 = ""
    public var rank: Int
    /// `alpha / rank` (1 when the file has no alpha); the user strength multiplies on top.
    public var scale: Float
}

public struct MossLoRAPlan: Sendable, Equatable {
    public var parts: [MossLoRAPart]
    /// LoRA modules with no DiT counterpart — reported, never guessed.
    public var unmapped: [String]

    /// Parts for one target, in row order (q, k, v for a fused qkv).
    public func parts(for target: String) -> [MossLoRAPart] {
        parts.filter { $0.target == target }.sorted { ($0.rows?.lowerBound ?? 0) < ($1.rows?.lowerBound ?? 0) }
    }
}

/// Maps LoRA tensor names onto the MOSS DiT. Conventions:
/// - **kohya** (lodestones DMD, ComfyUI): `diffusion_model.<module>.lora_down.weight`, `.lora_up.weight`, `.alpha`
/// - **PEFT/diffusers**: `[transformer.]<module>.lora_A.weight` / `.lora_B.weight` (no alpha ⇒ scale 1)
/// - **moss-edit-lora** (chinoll, Edit-HQ): bare `<module>.lora_A` / `.lora_B`, alphas in the
///   `adapter_config` metadata JSON (`adapterConfigAlphas`)
/// - **LyCORIS**: LoKr `lokr_w1` / `lokr_w2`, LoHa `hada_w{1,2}_{a,b}` (+ `.alpha`)
///
/// Both name the official split `layers.N.proj.{q,k,v}`, which land on the q|k|v row slices of the
/// canonical fused `layers.N.proj.qkv` (`MossDiTConfig.qkvRowRanges`). Applied **unmerged at runtime**
/// (`y = Wx + s·up(down(x))` in F32) because distillation deltas like lodestones' sit below bf16
/// rounding — merging them into bf16 weights would erase them.
public enum MossLoRAKeyMap {
    private static let prefixes = ["diffusion_model.", "transformer.", "model.diffusion_model.", "base_model.model."]
    private static let downSuffixes = [".lora_down.weight", ".lora_A.weight", ".lora_A"]
    private static let upFor = [".lora_down.weight": ".lora_up.weight", ".lora_A.weight": ".lora_B.weight", ".lora_A": ".lora_B"]

    /// Plan straight from a file (reads the scalar `alpha` tensors). Pass the DiT's tensor shapes
    /// (`targetShapes`) to accept updates on any module the DiT has, not just the block projections.
    public static func plan(url: URL, config: MossDiTConfig, targetShapes: [String: [Int]]? = nil) throws -> MossLoRAPlan {
        let reader = try SafetensorsReader(url: url)
        defer { reader.close() }
        var alphas = adapterConfigAlphas(metadata: (try? SafetensorsHeader.readEntries(from: url).metadata) ?? [:])
        for d in reader.descriptors where d.name.hasSuffix(".alpha") {
            let bytes = try reader.bytes(d.name)
            if let dtype = TensorDType(rawValue: d.dtype), let v = try SafetensorsCodec.decodeFloats(bytes, dtype: dtype).first {
                alphas[d.name] = v
            }
        }
        return try plan(descriptors: reader.descriptors, alphas: alphas, config: config, targetShapes: targetShapes)
    }

    /// `moss-edit-lora` files keep each module's alpha in `adapter_config.modules.<module>.alpha`;
    /// returned under the `<module>.alpha` names the planner looks up.
    public static func adapterConfigAlphas(metadata: [String: String]) -> [String: Float] {
        guard let data = metadata["adapter_config"]?.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modules = json["modules"] as? [String: Any]
        else { return [:] }
        var out: [String: Float] = [:]
        for (module, value) in modules {
            if let alpha = (value as? [String: Any])?["alpha"] as? NSNumber { out[module + ".alpha"] = alpha.floatValue }
        }
        return out
    }

    public static func plan(
        descriptors: [TensorDescriptor], alphas: [String: Float], config: MossDiTConfig, targetShapes: [String: [Int]]? = nil
    ) throws -> MossLoRAPlan {
        let shapes = Dictionary(descriptors.map { ($0.name, $0.shape) }, uniquingKeysWith: { a, _ in a })
        var parts: [MossLoRAPart] = []
        var unmapped: [String] = []
        func strip(_ raw: String) -> String { prefixes.reduce(raw) { m, p in m.hasPrefix(p) ? String(m.dropFirst(p.count)) : m } }
        for d in descriptors {
            // ComfyUI full deltas: `<m>.diff` → `<m>.weight`, `<m>.diff_b` → `<m>.bias`.
            if d.name.hasSuffix(".diff") || d.name.hasSuffix(".diff_b") {
                let isBias = d.name.hasSuffix(".diff_b")
                let module = strip(String(d.name.dropLast(isBias ? ".diff_b".count : ".diff".count)))
                let target = module + (isBias ? ".bias" : ".weight")
                guard let shape = targetShapes?[target] else {
                    unmapped.append(module + (isBias ? " (diff_b)" : " (diff)"))
                    continue
                }
                guard shapes[d.name] == shape else {
                    throw EngineError.invalidRequest("LoRA diff \(d.name) \(shapes[d.name] ?? []) doesn't fit \(target) \(shape)")
                }
                parts.append(.init(
                    kind: isBias ? .biasDiff : .weightDiff, block: blockIndex(module), target: target, rows: nil,
                    down: d.name, up: "", rank: 0, scale: 1))
                continue
            }
            if d.name.hasSuffix(".lokr_w1") {
                let raw = String(d.name.dropLast(".lokr_w1".count))
                let module = strip(raw)
                let w2 = raw + ".lokr_w2"
                guard let s1 = shapes[d.name], let s2 = shapes[w2], s1.count == 2, s2.count == 2,
                    let (target, rows, shape) = resolve(module, config: config) ?? generic(module, targetShapes)
                else {
                    unmapped.append(module + " (lokr)")
                    continue
                }
                let outRows = rows?.count ?? shape[0]
                guard s1[0] * s2[0] == outRows, s1[1] * s2[1] == shape[1] else {
                    throw EngineError.invalidRequest("LoKr \(module): w1 \(s1) ⊗ w2 \(s2) doesn't fit \(target) \(shape)\(rows.map { " rows \($0)" } ?? "")")
                }
                parts.append(.init(kind: .kronecker, block: blockIndex(module), target: target, rows: rows, down: d.name, up: w2, rank: 0, scale: 1))
                continue
            }
            if d.name.hasSuffix(".hada_w1_a") {
                let raw = String(d.name.dropLast(".hada_w1_a".count))
                let module = strip(raw)
                let names = (w1b: raw + ".hada_w1_b", w2a: raw + ".hada_w2_a", w2b: raw + ".hada_w2_b")
                guard let a1 = shapes[d.name], let b1 = shapes[names.w1b], let a2 = shapes[names.w2a], let b2 = shapes[names.w2b],
                    [a1, b1, a2, b2].allSatisfy({ $0.count == 2 }),
                    let (target, rows, shape) = resolve(module, config: config) ?? generic(module, targetShapes)
                else {
                    unmapped.append(module + " (loha)")
                    continue
                }
                let outRows = rows?.count ?? shape[0]
                guard a1 == [outRows, b1[0]], a2 == [outRows, b2[0]], b1[1] == shape[1], b2[1] == shape[1] else {
                    throw EngineError.invalidRequest("LoHa \(module): w1 \(a1)·\(b1), w2 \(a2)·\(b2) don't fit \(target) \(shape)")
                }
                let rank = b1[0]
                parts.append(.init(
                    kind: .hadamard, block: blockIndex(module), target: target, rows: rows, down: names.w1b, up: d.name,
                    down2: names.w2b, up2: names.w2a, rank: rank, scale: alphas[raw + ".alpha"].map { $0 / Float(rank) } ?? 1))
                continue
            }
            guard let suffix = downSuffixes.first(where: { d.name.hasSuffix($0) }) else { continue }
            let raw = String(d.name.dropLast(suffix.count))
            let module = strip(raw)
            let up = raw + upFor[suffix]!
            guard let downShape = shapes[d.name], let upShape = shapes[up], downShape.count == 2, upShape.count == 2 else {
                unmapped.append(module)
                continue
            }
            guard let (target, rows, shape) = resolve(module, config: config) ?? generic(module, targetShapes) else {
                unmapped.append(module)
                continue
            }
            let rank = downShape[0]
            let outRows = rows?.count ?? shape[0]
            guard upShape == [outRows, rank], downShape[1] == shape[1] else {
                throw EngineError.invalidRequest(
                    "LoRA \(module): up \(upShape) / down \(downShape) don't fit \(target) \(shape)\(rows.map { " rows \($0)" } ?? "")")
            }
            let alpha = alphas[raw + ".alpha"]
            parts.append(.init(
                block: blockIndex(module), target: target, rows: rows, down: d.name, up: up, rank: rank,
                scale: alpha.map { $0 / Float(rank) } ?? 1))
        }
        return MossLoRAPlan(parts: parts.sorted { ($0.target, $0.rows?.lowerBound ?? 0) < ($1.target, $1.rows?.lowerBound ?? 0) }, unmapped: unmapped)
    }

    /// Any other 2-D weight the DiT has (embedders, modulation, text adapter, head) — needs `targetShapes`.
    static func generic(_ module: String, _ targetShapes: [String: [Int]]?) -> (String, Range<Int>?, [Int])? {
        guard let shape = targetShapes?[module + ".weight"], shape.count == 2 else { return nil }
        return (module + ".weight", nil, shape)
    }

    /// `layers.N.…` → N (the 42 single-stream blocks); nil otherwise (text-adapter blocks included).
    static func blockIndex(_ module: String) -> Int? {
        guard module.hasPrefix("layers.") else { return nil }
        return Int(module.dropFirst("layers.".count).prefix { $0.isNumber })
    }

    /// module (prefix-stripped, no suffix) → (canonical weight key, row slice, full weight shape `[out, in]`).
    static func resolve(_ module: String, config c: MossDiTConfig) -> (String, Range<Int>?, [Int])? {
        let parts = module.split(separator: ".").map(String.init)
        // layers.N.proj.<name>
        if parts.count == 4, parts[0] == "layers", let n = Int(parts[1]), (0..<c.layers).contains(n), parts[2] == "proj" {
            let block = "layers.\(n).proj"
            let ranges = c.qkvRowRanges
            switch parts[3] {
            case "q": return ("\(block).qkv.weight", ranges.q, [3 * c.dim, c.dim])
            case "k": return ("\(block).qkv.weight", ranges.k, [3 * c.dim, c.dim])
            case "v": return ("\(block).qkv.weight", ranges.v, [3 * c.dim, c.dim])
            case "qkv": return ("\(block).qkv.weight", nil, [3 * c.dim, c.dim])
            case "o": return ("\(block).o.weight", nil, [c.dim, c.dim])
            case "ffn_up_gate": return ("\(block).ffn_up_gate.weight", nil, [2 * c.ffnDim, c.dim])
            case "ffn_down": return ("\(block).ffn_down.weight", nil, [c.dim, c.ffnDim])
            default: return nil
            }
        }
        return nil
    }
}

extension MossEditLoRAInfo {
    /// Read from a LoRA file's safetensors header (unreadable → nothing known).
    public init(url: URL) {
        let metadata = (try? SafetensorsHeader.readEntries(from: url).metadata) ?? [:]
        self.init(metadata: metadata, filename: url.lastPathComponent)
    }
}
