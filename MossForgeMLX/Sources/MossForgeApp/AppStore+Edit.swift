import Foundation
import MossForgeCore
import MossForgeEngine
import MossForgeMLX

/// Edit LoRAs + reference images (Generate's Edit section, the image menu's Use as Edit Reference).
extension AppStore {
    /// What a LoRA's header says about editing — read once per file.
    func editInfo(_ url: URL) -> MossEditLoRAInfo {
        if let cached = editInfoCache[url] { return cached }
        let info = MossEditLoRAInfo(url: url)
        editInfoCache[url] = info
        return info
    }

    /// Library LoRAs that want reference images (chinoll / Edit-HQ / MossForge Train → Edit, or "edit" in the name).
    var editLoRAs: [URL] { availableLoRAs.filter { editInfo($0).isEdit } }

    /// An edit LoRA is in the stack at a non-zero strength.
    var hasActiveEditLoRA: Bool { draft.loras.contains { $0.strength != 0 && editInfo($0.url).isEdit } }

    /// The layout the backend will use when the draft doesn't pick one.
    var detectedEditLayout: MossEditLayout {
        MossEditLayout.resolve(explicit: nil, loras: draft.loras.filter { $0.strength != 0 }.map { editInfo($0.url) })
    }

    /// Add a reference (≤ 4, no duplicates). The first one sets the output's aspect, and an edit LoRA
    /// joins the stack if there's none yet, so a dropped image is ready to edit.
    func addReference(_ url: URL) {
        guard !draft.references.contains(url), draft.references.count < GenerationRequest.maxReferences else { return }
        if draft.references.isEmpty, draft.exactSize == nil, let size = PixelImage.size(of: url) {
            draft.ratio = MossDimensions.nearestRatio(width: size.width, height: size.height)
        }
        draft.references.append(url)
        if !hasActiveEditLoRA, let lora = editLoRAs.first { draft.loras.append(.init(url: lora)) }
    }
}
