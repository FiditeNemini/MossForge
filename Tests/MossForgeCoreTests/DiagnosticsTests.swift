import Foundation
import Testing

@testable import MossForgeCore

/// Diagnostics & feedback: private by design — redaction, an opt-in bounded log, a report the user
/// reviews in full before it goes anywhere.
@Suite struct DiagnosticsTests {
    let redactor = PrivacyRedactor(context: .init(
        homeDirectory: "/Users/alex", userName: "alex", hostNames: ["Alexs-Mac-Studio.local", "Alexs-Mac-Studio"],
        placeholders: [("/Volumes/Big/models", "<models>"), ("/Users/alex/Pictures/MossForge", "<output>")]))

    @Test func redactsPathsAndNamesToPlaceholders() {
        let text = """
            loaded /Volumes/Big/models/diffusion_models/sft.safetensors
            saved /Users/alex/Pictures/MossForge/moss-1.png and /Users/alex/Desktop/x.png
            other user /Users/sam/Downloads/a.png on Alexs-Mac-Studio.local (Alexs-Mac-Studio)
            Alex ran it; alexander is a different word
            """
        #expect(redactor.redact(text) == """
            loaded <models>/diffusion_models/sft.safetensors
            saved <output>/moss-1.png and ~/Desktop/x.png
            other user /Users/<user>/Downloads/a.png on <host> (<host>)
            <user> ran it; alexander is a different word
            """)
    }

    @Test func redactsContactAndNetworkIdentifiers() {
        let text = "mail a.b+c@example.co.nz from 192.168.1.20 / fe80::1 mac 3c:22:fb:00:aa:01 id 1B4E28BA-2FA1-11D2-883F-0016D3CCA427, local 127.0.0.1"
        #expect(redactor.redact(text) == "mail <email> from <ip> / fe80::1 mac <mac> id <id>, local 127.0.0.1")
    }

    @Test func leavesOrdinaryDiagnosticsAlone() {
        let text = "macOS 26.1 · 64 GB · 1024×1024 · 8 steps · CFG 1 · v0.1.0 (12) · 3.5 GB free"
        #expect(redactor.redact(text) == text)
    }

    @Test func countsWhatItRemoved() {
        #expect(redactor.redactionCount(in: "a@b.io and /Users/alex/x and 10.0.0.2") == 3)
        #expect(redactor.redactionCount(in: "nothing here") == 0)
    }

    // MARK: log

    @Test func logIsBoundedAndClears() {
        var log = DiagnosticsLog(capacity: 3)
        for i in 1...5 { log.append(.info, area: "generate", "job \(i)") }
        #expect(log.entries.map(\.message) == ["job 3", "job 4", "job 5"])
        #expect(log.dropped == 2)
        log.clear()
        #expect(log.entries.isEmpty && log.dropped == 0)
    }

    // MARK: report

    func sampleReport() -> DiagnosticsReport {
        DiagnosticsReport(
            kind: .bug, description: "Generate failed after I picked /Users/alex/Desktop/lora.safetensors",
            sections: [
                .init(id: "app", title: "App", items: [.init("Version", "0.1.0 (12)"), .init("Backend", "mlx")]),
                .init(id: "models", title: "Models", items: [
                    .init("Tokenizer", "missing — expected <models>/text_encoders/qwen3.5-9b-tokenizer", status: .fail),
                ]),
                .init(id: "activity", title: "Recent activity", items: [.init("12:00:01 generate", "error: out of memory", status: .warn)], included: false),
            ],
            createdAt: Date(timeIntervalSince1970: 1_800_000_000))
    }

    @Test func markdownHasOnlyIncludedSectionsAndIsRedacted() {
        let md = sampleReport().markdown(product: .mossForge, redactor: redactor, assistantDraft: "## Summary\nThe tokenizer is missing.")
        #expect(md.hasPrefix("# MossForge feedback — Bug report"))
        #expect(md.contains("~/Desktop/lora.safetensors"))
        #expect(!md.contains("/Users/alex"))
        #expect(md.contains("## Summary\nThe tokenizer is missing."))
        #expect(md.contains("### Models\n- ✗ **Tokenizer**: missing — expected <models>/text_encoders/qwen3.5-9b-tokenizer"))
        #expect(md.contains("- **Version**: 0.1.0 (12)"))
        #expect(!md.contains("Recent activity"))
        #expect(md.contains("Generated on this Mac by MossForge"))
    }

    @Test func problemsAreListedFirstForTheModel() {
        let digest = sampleReport().digest(redactor: redactor)
        #expect(digest.hasPrefix("Problems found:\n- Models › Tokenizer: missing"))
        #expect(!digest.contains("out of memory"))  // excluded section
    }

    @Test func feedbackPromptAsksForAStructuredGroundedDraft() {
        let excerpt = HelpChunk(id: "x", document: "Guide", heading: "Models", text: "Put the tokenizer in text_encoders.")
        let messages = FeedbackPrompt.messages(product: .mossForge, report: sampleReport(), redactor: redactor, excerpts: [excerpt])
        #expect(messages.count == 2)
        let system = messages[0].content
        for heading in ["## Summary", "## What happened", "## Likely cause", "## Suggested next steps"] {
            #expect(system.contains(heading))
        }
        #expect(system.contains("[1] Guide › Models"))
        #expect(system.contains("Never invent"))
        let user = messages[1].content
        #expect(user.contains("Bug report"))
        #expect(user.contains("~/Desktop/lora.safetensors"))
        #expect(user.contains("Problems found:"))
        #expect(!user.contains("/Users/alex"))
    }
}
