import Foundation

// Sizes are MOSS ratio strings + pixel budgets (`MossDimensions.fit`).

/// A framing for one row of the Character Studio: how much of the character is in shot.
/// The rows run from close-up to full body so a character LoRA learns the face *and* the
/// build, rather than only the head-and-shoulders crop most reference photos give.
public enum CharacterShot: String, Sendable, Codable, CaseIterable, Identifiable {
    case closeUp
    case headAndShoulders
    case halfBody
    case kneeUp
    case fullBody
    case fullBodyAction

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .closeUp: "Close-up"
        case .headAndShoulders: "Head & shoulders"
        case .halfBody: "Waist up"
        case .kneeUp: "Knee up"
        case .fullBody: "Full body"
        case .fullBodyAction: "Full body, action"
        }
    }

    /// The phrase that pins the framing in the prompt.
    public var framing: String {
        switch self {
        case .closeUp: "close-up portrait"
        case .headAndShoulders: "head-and-shoulders portrait"
        case .halfBody: "waist-up shot"
        case .kneeUp: "knee-up shot"
        case .fullBody: "full-body shot"
        case .fullBodyAction: "full-body action shot"
        }
    }

    /// Close shots square, body shots tall, the action row wide (`w:h`) — a mix of aspect buckets
    /// also stops the LoRA tying the character to one canvas shape.
    public var ratio: String {
        switch self {
        case .closeUp, .headAndShoulders: "1:1"
        case .halfBody, .kneeUp, .fullBody: "2:3"
        case .fullBodyAction: "3:2"
        }
    }

    public func dimensions(pixels: Int) -> Dimensions {
        MossDimensions.fit(pixels: pixels, ratio: ratio)
    }
}

/// One seed-prompt row: a framing plus the scenario to put the character in.
public struct CharacterSeed: Sendable, Codable, Equatable, Hashable, Identifiable {
    public var id = UUID()
    public var shot: CharacterShot
    public var prompt: String

    public init(shot: CharacterShot, prompt: String) {
        self.shot = shot
        self.prompt = prompt
    }
}

/// One image to make: the text handed to the prompt enhancer, plus its framing and seed.
public struct CharacterJob: Sendable, Equatable {
    public var shot: CharacterShot
    public var seedIndex: Int
    public var variation: Int
    public var enhanceInput: String
    public var seed: UInt64
}

/// Plans a character-LoRA dataset: each seed prompt is enhanced several times — each time
/// asking for a different camera angle, so the enhancer (greedy, so the same input gives the
/// same output) returns distinct prompts — and each prompt renders with its own seed.
///
/// Count: community guidance for a character LoRA (Flux / Qwen-Image class models) is about
/// 20–25 curated images, 15 at the least, with angle and framing variety mattering more than
/// volume. The default 6 rows × 5 variations = 30 gives a pool to cull down to that.
public enum CharacterDatasetPlan {
    public static let recommendedKeep = 20...25
    public static let defaultVariations = 5
    public static let variationRange = 1...8

    public static let defaultSeeds: [CharacterSeed] = [
        CharacterSeed(shot: .closeUp, prompt: "neutral expression, soft window light, plain background"),
        CharacterSeed(shot: .headAndShoulders, prompt: "smiling, outdoors in a park, natural daylight"),
        CharacterSeed(shot: .halfBody, prompt: "casual clothes, sitting in a cosy café"),
        CharacterSeed(shot: .kneeUp, prompt: "smart outfit, walking down a city street at dusk"),
        CharacterSeed(shot: .fullBody, prompt: "standing relaxed, whole outfit visible, studio backdrop, even light"),
        CharacterSeed(shot: .fullBodyAction, prompt: "mid-stride, dynamic pose, on a windy beach"),
    ]

    /// The camera angle each variation asks for, in order.
    public static let angles = [
        "front view, looking at the camera",
        "three-quarter view from the left",
        "three-quarter view from the right",
        "side profile",
        "seen from a slightly low angle",
        "seen from slightly above",
        "looking away from the camera, candid",
        "turned away, looking back over the shoulder",
    ]

    public static func jobs(
        character: String, seeds: [CharacterSeed], variations: Int, baseSeed: UInt64
    ) -> [CharacterJob] {
        let n = min(max(variations, variationRange.lowerBound), variationRange.upperBound)
        let who = character.trimmingCharacters(in: .whitespacesAndNewlines)
        var out: [CharacterJob] = []
        for (row, seed) in seeds.enumerated() {
            let scene = seed.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !scene.isEmpty else { continue }
            for v in 0..<n {
                let parts = [seed.shot.framing + (who.isEmpty ? "" : " of \(who)"), scene, angles[v]]
                out.append(CharacterJob(
                    shot: seed.shot, seedIndex: row, variation: v,
                    enhanceInput: parts.joined(separator: ", "),
                    seed: jobSeed(base: baseSeed, index: out.count)))
            }
        }
        return out
    }

    /// A 32-bit seed (what other tools reproduce), never 0. An odd multiplier is a bijection
    /// mod 2³², so seeds stay distinct and far from their neighbours.
    static func jobSeed(base: UInt64, index: Int) -> UInt64 {
        let s = UInt32(truncatingIfNeeded: base) &+ UInt32(truncatingIfNeeded: index) &* 0x9E37_79B9
        return UInt64(s == 0 ? 1 : s)
    }

    /// The prompt to render: the enhanced text, with the framing and camera angle re-added
    /// when the enhancer dropped them. A greedy enhancer rewrites "full-body shot" away (which
    /// crops the dataset) and can fold "three-quarter view from the left/right" into one prompt.
    public static func finalPrompt(enhanced: String, shot: CharacterShot, angle: String? = nil) -> String {
        var text = enhanced.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.lowercased().contains(shot.framing) { text = "\(shot.framing), \(text)" }
        if let angle, !text.lowercased().contains(angle.lowercased()) {
            let trimmed = text.hasSuffix(".") ? String(text.dropLast()) : text
            text = "\(trimmed), \(angle)."
        }
        return text
    }

    /// With a reference photo (an edit LoRA sees it as picture 1): restage the same person in the
    /// row's shot and keep their identity. Tested 2026-10-27: a Moss-Image-Edit-layout LoRA restages
    /// (portrait → full-body in a park); an ai-toolkit one keeps the reference's framing.
    public static func referencePrompt(_ prompt: String) -> String {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = p.hasSuffix(".") ? String(p.dropLast()) : p
        return "Show the person from picture 1 in a new photograph: \(body). "
            + "Keep their face, hairstyle, hair colour, skin, freckles and eye colour exactly the same."
    }

    /// A training caption: the trigger word first, then the image's prompt.
    public static func caption(trigger: String, prompt: String) -> String {
        let t = trigger.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? prompt : "\(t), \(prompt)"
    }
}
