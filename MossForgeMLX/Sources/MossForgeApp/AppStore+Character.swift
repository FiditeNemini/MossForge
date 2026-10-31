import Foundation
import MossForgeCore
import MossForgeEngine

// Character Studio: text-only for now, MOSS pixel budgets,
// gallery file URLs + JSON sidecars, and requests built from Generate's draft (`makeRequest(for:prompt:)`).

/// The Character Studio form: who the character is, the seed-prompt rows, and how big a pool to make.
struct CharacterStudioDraft: Codable, Equatable {
    /// Also the gallery collection the images land in (and the dataset export name).
    var name = ""
    var trigger = ""
    var description = ""
    var seeds = CharacterDatasetPlan.defaultSeeds
    var variations = CharacterDatasetPlan.defaultVariations
    /// Pixel budget per image (each row's ratio is the shot's).
    var pixels = 1_048_576
    /// A photo of the character: every row restages it through an edit LoRA (picture 1).
    var referencePhoto: URL?
    /// That LoRA's strength. Edit LoRAs copy their reference at 1.0 (a seated café photo stayed seated);
    /// 0.7 restaged it as a full-body beach run with the same face and hair (2026-10-27). Optional so
    /// drafts saved before it still decode.
    var referenceStrength: Double?
    static let defaultReferenceStrength = 0.7
    var referenceStrengthValue: Double { referenceStrength ?? Self.defaultReferenceStrength }

    static let pixelChoices = [1_048_576, 1_572_864, 2_097_152]

    var collectionName: String {
        let n = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return n.isEmpty ? "Character" : n
    }

    var plannedCount: Int {
        seeds.filter { !$0.prompt.trimmingCharacters(in: .whitespaces).isEmpty }.count
            * min(max(variations, CharacterDatasetPlan.variationRange.lowerBound), CharacterDatasetPlan.variationRange.upperBound)
    }
}

/// One planned image in the current run.
struct CharacterRow: Identifiable, Equatable {
    enum State: Equatable {
        /// `removed`: rendered, then deleted while culling — never re-rendered.
        case pending, enhancing, ready, generating, done, removed, failed(String)
    }
    let id = UUID()
    var job: CharacterJob
    var prompt = ""
    var state: State = .pending
    var imageURL: URL?
}

extension AppStore {
    /// Write (or rewrite) the prompts: one enhancer pass per planned image.
    func writeCharacterPrompts() { runCharacter(render: false, replan: true) }

    /// Render every row that has no image yet, writing any missing prompts first.
    func generateCharacterDataset() { runCharacter(render: true, replan: characterRows.isEmpty) }

    /// Delete a dataset image while culling: the PNG (and sidecar) go to the Trash and its row is
    /// marked removed, so Generate doesn't render it again.
    func deleteCharacterImage(_ url: URL) {
        trashImage(url)
        for i in characterRows.indices where characterRows[i].imageURL == url { characterRows[i].state = .removed }
    }

    /// The images left in this character's collection — what Export Dataset writes.
    var characterDatasetImages: [URL] {
        let name = character.collectionName
        return gallery.filter { collectionsByImage[$0]?.contains(name) == true }
    }

    func cancelCharacter() {
        guard characterRunning else { return }
        characterStatus = "Cancelling…"
        characterTask?.cancel()
    }

    private func planCharacterRows() {
        let jobs = CharacterDatasetPlan.jobs(
            character: character.description, seeds: character.seeds, variations: character.variations,
            baseSeed: UInt64.random(in: 1...UInt64(UInt32.max)))
        characterRows = jobs.map { CharacterRow(job: $0) }
    }

    private func runCharacter(render: Bool, replan: Bool) {
        guard !characterRunning, !isGenerating else { return }
        if replan { planCharacterRows() }
        guard !characterRows.isEmpty else {
            characterStatus = "Add at least one seed prompt."
            return
        }
        characterRunning = true
        characterProgress = 0
        let started = Date()
        characterTask = Task {
            do {
                try await enhanceCharacterRows()
                if render { try await renderCharacterRows() }
                let made = characterRows.filter { $0.state == .done }.count
                characterStatus = render
                    ? "Done — \(made) images in “\(character.collectionName)” (\(Self.seconds(since: started))). "
                        + "Cull to the best \(CharacterDatasetPlan.recommendedKeep.lowerBound)–"
                        + "\(CharacterDatasetPlan.recommendedKeep.upperBound), then Export Dataset."
                    : "Prompts ready — edit any you like, then Generate."
                note(.info, "character", "\(render ? "\(made) images" : "prompts") in \(Self.seconds(since: started))")
            } catch {
                for i in characterRows.indices where [.enhancing, .generating].contains(characterRows[i].state) {
                    characterRows[i].state = characterRows[i].prompt.isEmpty ? .pending : .ready
                }
                if error is CancellationError || (error as? EngineError).map({ if case .cancelled = $0 { true } else { false } }) == true {
                    characterStatus = "Cancelled."
                } else {
                    characterStatus = "Error: \(error)"
                    note(.error, "character", "\(error)")
                }
            }
            characterProgress = 0
            characterRunning = false
            characterTask = nil
            refreshGallery()
        }
    }

    /// Enhance every row still missing a prompt. The enhancer is greedy, so each variation's input
    /// names a different camera angle — that, not sampling noise, makes them distinct.
    private func enhanceCharacterRows() async throws {
        let todo = characterRows.indices.filter { characterRows[$0].prompt.isEmpty }
        let (engine, queue) = (engine, inferenceQueue)
        for (n, i) in todo.enumerated() {
            try Task.checkCancellation()
            characterStatus = "Writing prompt \(n + 1) of \(todo.count)…"
            characterRows[i].state = .enhancing
            let job = characterRows[i].job
            let request = PromptEnhanceRequest(
                prompt: job.enhanceInput, canvas: job.shot.dimensions(pixels: character.pixels), prompts: settings.prompts)
            let enhanced = try await queue.run {
                if Task.isCancelled { throw EngineError.cancelled }
                return try await engine.enhancePrompt(request, progress: nil)
            }
            characterRows[i].prompt = CharacterDatasetPlan.finalPrompt(
                enhanced: enhanced, shot: job.shot, angle: CharacterDatasetPlan.angles[job.variation])
            characterRows[i].state = .ready
            characterProgress = Double(n + 1) / Double(todo.count)
        }
    }

    private func renderCharacterRows() async throws {
        let todo = characterRows.indices.filter { ![.done, .removed].contains(characterRows[$0].state) }
        let (engine, queue, output, name) = (engine, inferenceQueue, settings.outputDirectory, character.collectionName)
        for (n, i) in todo.enumerated() {
            try Task.checkCancellation()
            characterStatus = "Generating \(n + 1) of \(todo.count)…"
            characterRows[i].state = .generating
            let row = characterRows[i]
            let reference = character.referencePhoto
            let request = try makeRequest(
                for: characterDraft(for: row), prompt: reference == nil ? row.prompt : CharacterDatasetPlan.referencePrompt(row.prompt))
            let total = Double(todo.count)
            let result = try await queue.run {
                if Task.isCancelled { throw EngineError.cancelled }
                return try await engine.generate(request) { p in
                    Task { @MainActor in self.characterProgress = (Double(n) + p.fraction) / total }
                }
            }
            let url = try Self.save(result, request: request, in: output)
            var meta = GallerySidecar.read(for: url)
            meta = GallerySidecar.setting(GallerySidecar.collections(in: meta).union([name]), in: meta)
            meta["character"] = ["name": name, "trigger": character.trigger, "shot": row.job.shot.rawValue] as [String: Any]
            if reference != nil {
                // The caption a dataset export uses stays the row's prompt; the edit instruction is kept beside it.
                meta["edit_prompt"] = request.prompt
                meta["prompt"] = row.prompt
            }
            try? GallerySidecar.write(meta, for: url)
            characterRows[i].imageURL = url
            characterRows[i].state = .done
            refreshGallery()
        }
    }

    /// Generate's draft (model, Turbo, sampling, LoRAs) re-aimed at one dataset image: its prompt,
    /// the shot's size and the row's fixed seed. No layout or masks; the reference photo, when set, is
    /// the only reference, with an edit LoRA that restages (Moss-Image-Edit layout preferred).
    private func characterDraft(for row: CharacterRow) -> GenerateDraft {
        var d = draft
        d.prompt = row.prompt
        d.useLayout = false
        d.references = character.referencePhoto.map { [$0] } ?? []
        d.editMask = nil
        d.editLayout = nil
        if character.referencePhoto != nil, let lora = characterEditLoRA {
            d.loras.removeAll { editInfo($0.url).isEdit }
            d.loras.append(.init(url: lora, strength: character.referenceStrengthValue))
        }
        d.exactSize = row.job.shot.dimensions(pixels: character.pixels)
        d.seed = row.job.seed
        d.lockSeed = true
        return d
    }
}

extension AppStore {
    /// The edit LoRA Character Studio's reference photo runs through: one that restages
    /// (Moss-Image-Edit layout), else any edit LoRA.
    var characterEditLoRA: URL? {
        editLoRAs.first { editInfo($0).layout == .mossImageEdit } ?? editLoRAs.first
    }
}

extension AppStore {
    private static let characterKey = "mossforge.character.v1"

    static func loadCharacter() -> CharacterStudioDraft {
        guard let data = UserDefaults.standard.data(forKey: characterKey),
            let decoded = try? JSONDecoder().decode(CharacterStudioDraft.self, from: data)
        else { return CharacterStudioDraft() }
        return decoded
    }

    static func persistCharacter(_ value: CharacterStudioDraft) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        UserDefaults.standard.set(data, forKey: characterKey)
    }
}
