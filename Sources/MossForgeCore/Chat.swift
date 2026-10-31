import Foundation

/// One turn of a chat with the on-device language model (Qwen3.5-9B).
public struct ChatMessage: Sendable, Equatable, Codable {
    public enum Role: String, Sendable, Codable { case system, user, assistant }
    public var role: Role
    public var content: String

    public init(_ role: Role, _ content: String) {
        self.role = role
        self.content = content
    }
}

/// A chat completion: the transcript (system first), a token cap and sampling.
public struct ChatRequest: Sendable {
    public var messages: [ChatMessage]
    public var maxTokens: Int
    /// 0 = greedy. Help answers run a little warm (Qwen's non-thinking defaults are 0.7).
    public var temperature: Float
    public var seed: UInt64

    public init(messages: [ChatMessage], maxTokens: Int = 768, temperature: Float = 0.3, seed: UInt64 = 0) {
        self.messages = messages
        self.maxTokens = max(16, min(maxTokens, 4096))
        self.temperature = max(0, temperature)
        self.seed = seed
    }
}

/// Streams the reply so far (the whole text, not a delta) as it is written.
public typealias TextStreamHandler = @Sendable (String) -> Void

/// Qwen3.5's chat template in non-thinking mode (`enable_thinking=False`): ChatML turns, then an
/// assistant prompt with an empty think block. Earlier assistant turns are plain (the template drops
/// their reasoning). Same shape as `MossEnhancerPrompt.chat`.
public enum QwenChatTemplate {
    public static let stopToken = "<|im_end|>"
    static let controlTokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]

    public static func render(_ messages: [ChatMessage]) -> String {
        var s = ""
        for m in messages {
            s += "<|im_start|>\(m.role.rawValue)\n\(sanitize(m.content))<|im_end|>\n"
        }
        return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// Text can't open or close turns: control tokens in content lose their pipes / brackets, so the
    /// tokenizer reads them as plain text (docs, diagnostics and user input are all untrusted here).
    static func sanitize(_ text: String) -> String {
        var t = text
        for token in controlTokens {
            t = t.replacingOccurrences(of: token, with: token.replacingOccurrences(of: "|", with: "").replacingOccurrences(of: "<", with: "‹").replacingOccurrences(of: ">", with: "›"))
        }
        return t
    }
}
