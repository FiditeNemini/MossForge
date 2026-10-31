import Foundation
import Testing

@testable import MossForgeCore

/// First-run setup: a Hugging Face repo laid out as `models/` plus `mossforge-manifest.json`
/// (groups, sizes, SHA-256) — the repo is a setting, so the bundle can move accounts.
@Suite struct ModelBundleTests {
    let json = #"""
        {"format": "mossforge-models", "version": 1,
         "groups": [{"id": "essentials", "title": "Essentials", "summary": "s", "default": true, "size": 30},
                    {"id": "sft", "title": "SFT", "summary": "s", "default": false, "size": 7}],
         "files": [{"path": "vae/flux2-vae.safetensors", "group": "essentials", "size": 10, "sha256": "aa"},
                   {"path": "text_encoders/qwen3.5-9b-tokenizer/tokenizer.json", "group": "essentials", "size": 20, "sha256": "bb"},
                   {"path": "diffusion_models/moss_image_sft_bf16.safetensors", "group": "sft", "size": 7, "sha256": "cc"}]}
        """#

    @Test func decodesAndPlansWhatIsMissing() throws {
        let m = try ModelBundleManifest.decode(Data(json.utf8))
        #expect(m.groups.map(\.id) == ["essentials", "sft"])
        #expect(m.defaultGroups == ["essentials"])
        // The VAE is there at the right size; the tokenizer is the wrong size, so it's fetched again.
        let pending = m.pending(groups: ["essentials"], existingSizes: ["vae/flux2-vae.safetensors": 10, "text_encoders/qwen3.5-9b-tokenizer/tokenizer.json": 3])
        #expect(pending.map(\.path) == ["text_encoders/qwen3.5-9b-tokenizer/tokenizer.json"])
        #expect(m.pending(groups: ["essentials", "sft"], existingSizes: [:]).count == 3)
        #expect(m.pending(groups: [], existingSizes: [:]).isEmpty)
    }

    /// The manifest comes off the network: only plain relative paths inside `models/`.
    @Test func refusesPathsOutsideTheModelsFolder() {
        for bad in ["/etc/passwd", "../x.safetensors", "lora/../../x", "", "a//b", "~/x"] {
            let doc = json.replacingOccurrences(of: "vae/flux2-vae.safetensors", with: bad)
            #expect(throws: EngineError.self) { try ModelBundleManifest.decode(Data(doc.utf8)) }
        }
        let wrongFormat = json.replacingOccurrences(of: "mossforge-models", with: "other")
        #expect(throws: EngineError.self) { try ModelBundleManifest.decode(Data(wrongFormat.utf8)) }
    }

    @Test func sourceBuildsResolveURLs() throws {
        let s = ModelBundleSource()
        #expect(s.repo == "FiditeNemini/MossForge" && s.isValid)
        #expect(s.manifestURL.absoluteString == "https://huggingface.co/FiditeNemini/MossForge/resolve/main/mossforge-manifest.json")
        #expect(s.url(for: "text_encoders/qwen3.5-9b-tokenizer/tokenizer.json").absoluteString
            == "https://huggingface.co/FiditeNemini/MossForge/resolve/main/text_encoders/qwen3.5-9b-tokenizer/tokenizer.json")
        #expect(s.url(for: "lora/a b/c.safetensors").absoluteString.hasSuffix("/lora/a%20b/c.safetensors"))
        #expect(ModelBundleSource(repo: " someone/Other ").repo == "someone/Other")
        #expect(!ModelBundleSource(repo: "no-slash").isValid)
        #expect(!ModelBundleSource(repo: "a/b/c").isValid)
        #expect(ModelBundleSource(repo: "https://huggingface.co/someone/Other").repo == "someone/Other")
    }
}
