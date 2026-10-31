import Foundation

/// Keyword retrieval over the help chunks (Okapi BM25, heading terms weighted ×3). Small, exact and
/// deterministic — the corpus is a few dozen sections, so no embedding model is needed.
public struct HelpRetriever: Sendable {
    public let chunks: [HelpChunk]
    private let docTerms: [[String: Int]]
    private let lengths: [Double]
    private let averageLength: Double
    private let documentFrequency: [String: Int]

    static let k1 = 1.2
    static let b = 0.75
    static let headingWeight = 3
    /// Developer docs (architecture, technical reference) rank a little below the user guides, so
    /// "how do I…" lands on the how-to and only technical wording reaches the internals.
    static let developerDocWeight = 0.7
    /// Words that ask for the internals ("how does X work under the hood?"): developer docs then
    /// rank a little *above* the guides instead.
    static let technicalIntent: Set<String> = Set(
        ["hood", "internal", "internally", "algorithm", "math", "maths", "formula", "objective", "implement",
         "implemented", "implementation", "architecture", "technical", "technically", "detail", "detailed"].map(stem))
    static let technicalDocWeight = 1.6

    public init(chunks: [HelpChunk]) {
        self.chunks = chunks
        var df: [String: Int] = [:]
        var all: [[String: Int]] = []
        for c in chunks {
            var tf: [String: Int] = [:]
            for t in Self.terms(c.text) { tf[t, default: 0] += 1 }
            for t in Self.terms(c.heading + " " + c.document) { tf[t, default: 0] += Self.headingWeight }
            for t in tf.keys { df[t, default: 0] += 1 }
            all.append(tf)
        }
        docTerms = all
        lengths = all.map { Double($0.values.reduce(0, +)) }
        averageLength = max(1, lengths.reduce(0, +) / Double(max(1, lengths.count)))
        documentFrequency = df
    }

    /// The best chunks for `query`: at most `limit`, and — after the first — only while their text
    /// fits in `budget` characters. Chunks scoring below a fifth of the best are dropped.
    public func search(_ query: String, limit: Int = 4, budget: Int = 6000) -> [HelpChunk] {
        let q = Set(Self.terms(query))
        guard !q.isEmpty else { return [] }
        let developerWeight = q.isDisjoint(with: Self.technicalIntent) ? Self.developerDocWeight : Self.technicalDocWeight
        let n = Double(chunks.count)
        var scored: [(Int, Double)] = []
        for (i, tf) in docTerms.enumerated() {
            var score = 0.0
            for t in q {
                guard let f = tf[t].map(Double.init), let df = documentFrequency[t] else { continue }
                let idf = log(1 + (n - Double(df) + 0.5) / (Double(df) + 0.5))
                score += idf * f * (Self.k1 + 1) / (f + Self.k1 * (1 - Self.b + Self.b * lengths[i] / averageLength))
            }
            if chunks[i].id.hasPrefix("docs/dev/") { score *= developerWeight }
            if score > 0 { scored.append((i, score)) }
        }
        scored.sort { $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0 }
        guard let best = scored.first?.1 else { return [] }
        var out: [HelpChunk] = []
        var used = 0
        for (i, score) in scored.prefix(limit) where score >= best / 5 {
            let c = chunks[i]
            if !out.isEmpty, used + c.text.count > budget { break }
            out.append(c)
            used += c.text.count
        }
        return out
    }

    /// Lower-cased word terms without stop words, lightly stemmed ("upscaling" ≈ "upscale").
    public static func terms(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 1 && !stopWords.contains($0) }
            .map(stem)
            .map { synonyms[$0] ?? $0 }
    }

    /// Words users reach for that the docs spell another way. Written as plain words and
    /// stemmed on both sides, because `terms` looks them up *after* stemming ("image" → "imag",
    /// "picture" → "pictur") — an unstemmed table never matches.
    static let synonyms: [String: String] = Dictionary(uniqueKeysWithValues: [
        ("photo", "image"), ("picture", "image"), ("pic", "image"), ("img", "image"),
        ("upscaler", "upscale"), ("enlarge", "upscale"),
    ].map { (stem($0), stem($1)) })

    static func stem(_ w: String) -> String {
        // British and American spellings meet ("quantised" ≈ "quantized", "normalisation").
        var s = w.replacingOccurrences(of: #"is(e|ed|es|ing|ation|ations)$"#, with: "iz$1", options: .regularExpression)
        if s.hasSuffix("ies"), s.count > 4 {
            s.removeLast(3)
            s += "y"
        } else if s.hasSuffix("s"), s.count > 3, !s.hasSuffix("ss") {
            s.removeLast(1)
        }
        if s.hasSuffix("ing"), s.count > 5 {
            s.removeLast(3)
        } else if s.hasSuffix("ed"), s.count > 4 {
            s.removeLast(2)
        }
        if s.hasSuffix("e"), s.count > 4 { s.removeLast(1) }
        return s
    }

    static let stopWords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from", "how", "i", "if", "in",
        "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "should", "so", "that", "the", "their", "then",
        "there", "this", "to", "use", "was", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
        "we", "our", "get", "want", "would", "could", "best", "way", "any", "some", "about", "make", "need",
    ]
}
