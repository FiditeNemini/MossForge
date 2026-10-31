import Foundation
import MossForgeCore

// The OpenAI-compatible core (`prompt`, `size`, `n`) and the contained-path rule for LoRAs; the extras are
// MOSS's (variant, cfg, shift, SLG, per-block LoRA strength, JSON layout). See NOTES.md.

/// The JSON body of `/v1/images/generations`, `/v1/moss/generate` and `/v1/moss/jobs`.
///
/// ```json
/// {"prompt": "…" | "layout": {…USERGUIDE layout JSON…}, "negative_prompt": "…",
///  "variant": "sft" | "base" | "turbo", "size": "1024x1024" | "width"/"height", "n": 1,
///  "steps": 50, "guidance_scale": 3, "shift": 5, "seed": 42,
///  "slg": "official" | "off" | {"uncond_skip_layers": [22], "layers": [], "scale": 0, "start": 0, "stop": 1},
///  "loras": [{"path": "lora/kijai/x.safetensors", "strength": -0.5, "blocks": "21-41"}]}
/// ```
/// `GET /v1/moss/options` lists the choices and ranges.
public struct MossGenerationBody: Sendable, Equatable {
    public struct LoRA: Sendable, Equatable {
        /// Relative to the models folder (`lora/…`) or to `<models>/lora`; never absolute or escaping.
        public var path: String
        public var strength: Double
        public var blocks: String?
    }

    public var prompt: String
    public var negativePrompt: String?
    public var variant: MossVariant
    public var width: Int?
    public var height: Int?
    public var n: Int
    public var steps: Int?
    public var cfg: Double?
    public var shift: Double?
    public var seed: UInt64?
    public var slg: SkipLayerGuidance?
    public var loras: [LoRA]
    /// Edit routes: how the edit LoRA sees the source (`edit_layout`; absent = from the LoRA's metadata).
    public var editLayout: MossEditLayout?

    public static let maxImages = 4
    public static let sizeRange = 256...4096
    public static let shiftRange = 0.1...20.0
    public static let cfgRange = 1.0...30.0

    /// Parse and validate a request body; throws `invalidRequest` with a client-facing reason.
    public init(json data: Data) throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw EngineError.invalidRequest("body must be a JSON object with a 'prompt' or a 'layout'")
        }
        func value<T>(_ key: String, _ type: T.Type) throws -> T? {
            guard let v = obj[key], !(v is NSNull) else { return nil }
            guard let t = v as? T else { throw EngineError.invalidRequest("'\(key)' has the wrong type") }
            return t
        }
        func int(_ key: String) throws -> Int? {
            guard let n = try value(key, NSNumber.self) else { return nil }
            guard n.doubleValue == n.doubleValue.rounded() else { throw EngineError.invalidRequest("'\(key)' must be an integer") }
            return n.intValue
        }
        func double(_ key: String) throws -> Double? { try value(key, NSNumber.self)?.doubleValue }

        // Prompt: plain text, or a USERGUIDE layout object passed through verbatim as compact JSON.
        if let layout = obj["layout"] {
            guard JSONSerialization.isValidJSONObject(layout), layout is [String: Any],
                let text = try? JSONSerialization.data(withJSONObject: layout, options: [.sortedKeys, .withoutEscapingSlashes])
            else { throw EngineError.invalidRequest("'layout' must be a JSON object") }
            prompt = String(decoding: text, as: UTF8.self)
        } else {
            prompt = (try value("prompt", String.self) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !prompt.isEmpty else { throw EngineError.invalidRequest("'prompt' (or 'layout') is required") }
        negativePrompt = try value("negative_prompt", String.self)

        let variantName = (try value("variant", String.self) ?? "sft").lowercased()
        guard let v = MossVariant(rawValue: variantName) else {
            throw EngineError.invalidRequest("unknown variant '\(variantName)' (sft | base | turbo)")
        }
        variant = v

        width = try int("width")
        height = try int("height")
        if let size = try value("size", String.self), width == nil || height == nil {
            let parts = size.lowercased().split(separator: "x").map { Int($0) }
            guard parts.count == 2, let sw = parts[0], let sh = parts[1] else {
                throw EngineError.invalidRequest("'size' must look like 1024x1024")
            }
            width = width ?? sw
            height = height ?? sh
        }
        for (name, s) in [("width", width), ("height", height)] {
            guard let s else { continue }
            guard Self.sizeRange.contains(s), s % 16 == 0 else {
                throw EngineError.invalidRequest("\(name) \(s) must be a multiple of 16 in \(Self.sizeRange.lowerBound)…\(Self.sizeRange.upperBound)")
            }
        }

        n = try int("n") ?? 1
        guard (1...Self.maxImages).contains(n) else { throw EngineError.invalidRequest("'n' must be 1…\(Self.maxImages)") }
        steps = try int("steps")
        if let steps, !(1...(v == .turbo ? 8 : 200)).contains(steps) {
            throw EngineError.invalidRequest("'steps' must be 1…\(v == .turbo ? 8 : 200)\(v == .turbo ? " for turbo" : "")")
        }
        cfg = try double("guidance_scale") ?? double("cfg")
        if let cfg, !Self.cfgRange.contains(cfg) { throw EngineError.invalidRequest("'guidance_scale' must be 1…30") }
        if v == .turbo, let cfg, cfg != 1 { throw EngineError.invalidRequest("turbo runs without CFG (guidance_scale 1)") }
        shift = try double("shift")
        if let shift, !Self.shiftRange.contains(shift) { throw EngineError.invalidRequest("'shift' must be 0.1…20") }
        seed = try value("seed", NSNumber.self).map { $0.uint64Value }
        if let name = try value("edit_layout", String.self), name.lowercased() != "auto" {
            guard let layout = MossEditLayout(name: name) else {
                throw EngineError.invalidRequest(
                    "unknown edit_layout '\(name)' (auto | \(MossEditLayout.allCases.map(\.rawValue).joined(separator: " | ")))")
            }
            editLayout = layout
        }

        switch obj["slg"] {
        case nil, is NSNull: slg = nil
        case let name as String:
            switch name.lowercased() {
            case "official": slg = .official
            case "off": slg = .off
            default: throw EngineError.invalidRequest("unknown slg '\(name)' (official | off | {…})")
            }
        case let o as [String: Any]:
            func ints(_ k: String) -> [Int] { (o[k] as? [NSNumber])?.map(\.intValue) ?? [] }
            func num(_ k: String, _ d: Double) -> Double { (o[k] as? NSNumber)?.doubleValue ?? d }
            slg = SkipLayerGuidance(
                uncondSkipLayers: ints("uncond_skip_layers"), layers: ints("layers"), scale: num("scale", 0),
                start: num("start", 0), stop: num("stop", 1))
        default: throw EngineError.invalidRequest("'slg' must be \"official\", \"off\" or an object")
        }

        loras = try (try value("loras", [Any].self) ?? []).map { item in
            guard let o = item as? [String: Any], let path = o["path"] as? String else {
                throw EngineError.invalidRequest("each LoRA needs a 'path'")
            }
            let strength = ((o["strength"] ?? o["scale"]) as? NSNumber)?.doubleValue ?? 1
            let blocks = o["blocks"] as? String
            if let blocks, LoRABlockWeights(parsing: blocks) == nil {
                throw EngineError.invalidRequest("LoRA 'blocks' \(blocks) is not a block spec (e.g. 21-41 or 0-20=0,21-41=1)")
            }
            return LoRA(path: path, strength: strength, blocks: blocks)
        }
    }

    /// A relative `.safetensors` path that can't leave the folder it's resolved against.
    static func isContainedID(_ path: String) -> Bool {
        guard !path.hasPrefix("/"), path.hasSuffix(".safetensors") else { return false }
        return !path.split(separator: "/", omittingEmptySubsequences: false).contains { $0 == ".." || $0 == "." || $0.isEmpty }
    }

    /// The engine request (image `index` of `n` gets `seed + index`), rooted at `modelsDirectory`.
    public func makeRequest(modelsDirectory: URL, index: Int = 0) throws -> GenerationRequest {
        var r = GenerationRequest(
            prompt: prompt, negativePrompt: negativePrompt ?? (variant == .turbo ? "" : MossNegativePrompt.recommended),
            variant: variant, width: width ?? 1024, height: height ?? 1024, steps: steps, cfg: cfg, shift: shift,
            seed: seed.map { $0 &+ UInt64(index) })
        if let slg { r.slg = slg }
        r.loras = try loras.map { l in
            guard Self.isContainedID(l.path) else {
                throw EngineError.invalidRequest("LoRA path '\(l.path)' must be a relative .safetensors path inside the models folder")
            }
            let root = l.path.hasPrefix("lora/") ? modelsDirectory : modelsDirectory.appendingPathComponent("lora")
            let url = root.appendingPathComponent(l.path)
            guard FileManager.default.fileExists(atPath: url.path) else { throw EngineError.invalidRequest("no LoRA at '\(l.path)'") }
            return LoRASpec(url: url, strength: l.strength, blocks: l.blocks.flatMap(LoRABlockWeights.init(parsing:)))
        }
        try r.validate()
        return r
    }
}
