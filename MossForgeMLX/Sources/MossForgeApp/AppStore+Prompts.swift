import Foundation
import MossForgeCore
import MossForgeEngine

/// Settings → Prompts → Try it: run one prompt, as edited, on a sample input and show the answer, so
/// prompt tuning doesn't need a trip to Generate or Train. Runs on the shared GPU queue like any job.
struct PromptLab {
    /// The built-in test inputs: an idea for the enhancer, a "X to Y" request for the slider, and for the
    /// captioner an existing caption to correct (its `{{reference}}`; empty = none).
    static let samples: [MossPrompts.Kind: String] = [
        .enhancerRules: "a cosy bookshop window at dusk with a hand-lettered sign saying OPEN LATE",
        .enhancerCaption: "a cosy bookshop window at dusk with a hand-lettered sign saying OPEN LATE",
        .enhancerLayout: "a summer music festival poster: headline SUNSET SOUNDS, three band names, the date 12 July",
        .caption: "",
        .slider: "a summer to winter slider",
    ]
    /// The user's test input per prompt (edited in Try it), starting from the samples.
    var inputs = PromptLab.samples
    /// The captioner's sample image; `nil` = the newest gallery image.
    var image: URL?
    var running: MossPrompts.Kind?
    var results: [MossPrompts.Kind: Result] = [:]

    struct Result {
        var text: String
        var failed: Bool
        var seconds: Double
    }
}

extension AppStore {
    /// The image the captioner's Try it uses.
    var promptLabImage: URL? { promptLab.image ?? gallery.first }

    func canTryPrompt(_ kind: MossPrompts.Kind) -> Bool {
        guard promptLab.running == nil else { return false }
        switch kind {
        case .caption: return engine.capabilities.contains(.captioning) && promptLabImage != nil
        case .slider: return canChat
        case .enhancerRules, .enhancerCaption, .enhancerLayout: return engine.capabilities.contains(.promptEnhancing)
        }
    }

    /// Run `kind` with the current edits; the answer (or the error) lands in `promptLab.results`.
    func tryPrompt(_ kind: MossPrompts.Kind) {
        guard canTryPrompt(kind) else { return }
        let input = promptLab.inputs[kind] ?? ""
        let (engine, queue, prompts, image) = (engine, inferenceQueue, settings.prompts, promptLabImage)
        promptLab.running = kind
        promptLab.results[kind] = nil
        Task {
            let start = Date()
            var result: PromptLab.Result
            do {
                let text: String = try await queue.run {
                    switch kind {
                    case .enhancerRules, .enhancerCaption, .enhancerLayout:
                        let layout = kind == .enhancerLayout
                        let out = try await engine.enhancePrompt(PromptEnhanceRequest(prompt: input, layout: layout, prompts: prompts))
                        return layout ? Self.prettyJSON(out) : out
                    case .caption:
                        let c = try await engine.caption(CaptionRequest(image: image!, reference: input, prompts: prompts))
                        return "Long\n\(c.long)\n\nMedium\n\(c.medium)\n\nShort\n\(c.short)"
                    case .slider:
                        let reply = try await engine.chat(
                            ChatRequest(messages: SliderConceptDraft.messages(for: input, prompts: prompts), maxTokens: 200, temperature: 0.3))
                        return reply + (SliderConceptDraft.parse(reply) == nil ? "\n\n(not four labelled lines — Train would fall back to a guess)" : "")
                    }
                }
                result = .init(text: text, failed: false, seconds: Date().timeIntervalSince(start))
            } catch {
                result = .init(text: "\(error)", failed: true, seconds: Date().timeIntervalSince(start))
            }
            promptLab.results[kind] = result
            promptLab.running = nil
        }
    }

    private nonisolated static func prettyJSON(_ text: String) -> String {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(text.utf8)),
            let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .withoutEscapingSlashes])
        else { return text }
        return String(decoding: data, as: UTF8.self)
    }
}
