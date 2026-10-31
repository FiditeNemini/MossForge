import Foundation
import Testing

@testable import MossForgeCore

/// Community edit LoRAs: which reference layout each was trained with (ComfyUI PR #17296's
/// `EDIT_FORMATS`), told from the LoRA's safetensors metadata, and the text encoder's picture prompt.
@Suite struct EditLayoutTests {
    @Test func layoutsMatchComfyUIEditFormats() {
        // ai-toolkit (kv_cache on): isolated references, the encoder sees them.
        #expect(MossEditLayout.aiToolkit.isolatedReferences)
        #expect(!MossEditLayout.aiToolkit.referencesAtTextStart)
        #expect(MossEditLayout.aiToolkit.encoderSeesReferences)
        // ai-toolkit (kv_cache off): references ride along in the sequence.
        #expect(!MossEditLayout.aiToolkitNoCache.isolatedReferences)
        #expect(MossEditLayout.aiToolkitNoCache.encoderSeesReferences)
        // Moss-Image-Edit (chinoll): isolated, at the text's start, text-only encoder.
        #expect(MossEditLayout.mossImageEdit.isolatedReferences)
        #expect(MossEditLayout.mossImageEdit.referencesAtTextStart)
        #expect(!MossEditLayout.mossImageEdit.encoderSeesReferences)
        // MossForge's own Train → Edit: joint, text-only encoder.
        #expect(!MossEditLayout.mossForge.isolatedReferences)
        #expect(!MossEditLayout.mossForge.encoderSeesReferences)
    }

    @Test func detectsTheLayoutFromMetadata() {
        let chinoll = MossEditLoRAInfo(
            metadata: ["adapter_config": #"{"format": "moss-edit-lora", "format_version": 1}"#], filename: "adapter_model.safetensors")
        #expect(chinoll.layout == .mossImageEdit && chinoll.isEdit)

        let aiToolkit = ["software": #"{"name": "ai-toolkit", "version": "0.13.36"}"#, "ss_base_model_version": "moss_image"]
        let armV5 = MossEditLoRAInfo(metadata: aiToolkit, filename: "armV5.safetensors")
        #expect(armV5.layout == .aiToolkit)
        #expect(!armV5.isEdit)  // ai-toolkit doesn't say; the name does when it has "edit"
        #expect(MossEditLoRAInfo(metadata: aiToolkit, filename: "akhaliq-edit-sft-armv5.safetensors").isEdit)

        let ours = MossEditLoRAInfo(metadata: ["produced_by": "MossForge", "mossforge_edit_layout": "mossForge"], filename: "x.safetensors")
        #expect(ours.layout == .mossForge && ours.isEdit)
        let turbo = MossEditLoRAInfo(metadata: [:], filename: "lodestones-8step-sft.safetensors")
        #expect(turbo.layout == nil && !turbo.isEdit)
    }

    /// An explicit choice wins; otherwise an edit LoRA's layout, then any detected one, then MossForge's.
    @Test func resolvesTheLayoutForARequest() {
        let turbo = MossEditLoRAInfo(layout: .aiToolkit, isEdit: false)
        let edit = MossEditLoRAInfo(layout: .mossImageEdit, isEdit: true)
        #expect(MossEditLayout.resolve(explicit: nil, loras: [turbo, edit]) == .mossImageEdit)
        #expect(MossEditLayout.resolve(explicit: .aiToolkitNoCache, loras: [edit]) == .aiToolkitNoCache)
        #expect(MossEditLayout.resolve(explicit: nil, loras: [turbo]) == .aiToolkit)
        #expect(MossEditLayout.resolve(explicit: nil, loras: []) == .mossForge)
    }

    /// ComfyUI `TextEncodeMOSSImageEdit`: "Picture N: <vision>" per reference, then the stripped prompt.
    @Test func picturePromptPrefixesEachReference() {
        let p = MossEditPrompt.instruction("  make it night ", pictureTokens: [2, 1])
        #expect(p == "Picture 1: <|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
            + "Picture 2: <|vision_start|><|image_pad|><|vision_end|>make it night")
        #expect(MossEditPrompt.instruction("make it night", pictureTokens: []) == "make it night")
    }

    /// Raw values, and ComfyUI's `edit_format` names, case-insensitively.
    @Test func layoutsParseFromTheirNames() {
        #expect(MossEditLayout(name: "aiToolkit") == .aiToolkit)
        #expect(MossEditLayout(name: "ai-toolkit") == .aiToolkit)
        #expect(MossEditLayout(name: "AI-Toolkit (kv_cache off)") == .aiToolkitNoCache)
        #expect(MossEditLayout(name: "Moss-Image-Edit") == .mossImageEdit)
        #expect(MossEditLayout(name: "mossforge") == .mossForge)
        #expect(MossEditLayout(name: "auto") == nil)
    }

    /// Edit references break lodestones' data-space loop (2026-10-27: textured noise with every edit
    /// LoRA, while flow Euler edits cleanly), so an edit runs a Turbo the way Edit-HQ stacks it.
    @Test func editsRunATurboOnFlowEuler() {
        var r = GenerationRequest(prompt: "make it night", variant: .base, steps: 8, cfg: 1, shift: 1, sampler: .dataSpaceDMD)
        r.references = [URL(fileURLWithPath: "/tmp/r.png")]
        #expect(r.sampler == .dataSpaceDMD)
        #expect(r.editRunsOnFlow)
        r.moveTurboToFlow()
        #expect(r.sampler == .flowEuler && r.cfg == 1 && r.shift == 5 && r.slg == .off && r.steps == 8)
        #expect(!r.editRunsOnFlow)
        var plain = GenerationRequest(prompt: "a fox", variant: .base, sampler: .dataSpaceDMD)
        #expect(!plain.editRunsOnFlow)  // no references: the data-space loop as before
        plain.references = [URL(fileURLWithPath: "/tmp/r.png")]
        plain.sampler = .flowEuler
        #expect(!plain.editRunsOnFlow)
    }

    /// Generate follows the first reference's aspect.
    @Test func nearestPresetRatio() {
        #expect(MossDimensions.nearestRatio(width: 1000, height: 1000) == "1:1")
        #expect(MossDimensions.nearestRatio(width: 1920, height: 1080) == "16:9")
        #expect(MossDimensions.nearestRatio(width: 800, height: 1200) == "2:3")
        #expect(MossDimensions.nearestRatio(width: 4000, height: 3000) == "4:3")
    }

    @Test func requestTakesUpToFourReferencesAndALayout() throws {
        var r = GenerationRequest(prompt: "make it night")
        r.references = (0..<4).map { URL(fileURLWithPath: "/tmp/r\($0).png") }
        r.editLayout = .aiToolkit
        try r.validate()
        r.references.append(URL(fileURLWithPath: "/tmp/r4.png"))
        #expect(throws: EngineError.self) { try r.validate() }
    }
}
