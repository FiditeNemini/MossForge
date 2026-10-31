import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// Deleting a checkpoint is the one destructive thing the Models screen can do, so the
/// guard rails live in the engine (testable) rather than in the view.
@Suite struct ModelFileOpsTests {
    private func withScratch(_ body: (URL) throws -> Void) throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("kf-modelfileops-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        try body(root)
    }

    @discardableResult
    private func touch(_ url: URL) throws -> URL {
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        try Data("weights".utf8).write(to: url)
        return url
    }

    /// A path outside the configured models directory is never deletable, however it got
    /// selected — the screen must not be able to trash arbitrary files on disk.
    @Test func refusesFilesOutsideTheModelsDirectory() throws {
        try withScratch { root in
            let outside = root.appendingPathComponent("outside/important.safetensors")
            let elsewhere = try touch(outside)
            let modelsDir = root.appendingPathComponent("models", isDirectory: true)
            try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
            #expect(throws: ModelFileError.self) {
                try ModelFileOps.validateDeletable(elsewhere, root: modelsDir)
            }
            // …and the file is still there.
            #expect(FileManager.default.fileExists(atPath: elsewhere.path))
        }
    }

    /// Only weight files are deletable — a stray document in `models/` isn't the app's to remove.
    @Test func refusesNonWeightFiles() throws {
        try withScratch { root in
            let notes = try touch(root.appendingPathComponent("notes.txt"))
            #expect(throws: ModelFileError.self) {
                try ModelFileOps.validateDeletable(notes, root: root)
            }
        }
    }

    @Test func refusesFilesThatAreNoLongerThere() throws {
        try withScratch { root in
            let ghost = root.appendingPathComponent("gone.safetensors")
            #expect(throws: ModelFileError.self) {
                try ModelFileOps.validateDeletable(ghost, root: root)
            }
        }
    }

    /// Nested layouts (`models/lora/…`, `models/krea2/turbo/…`) are the normal case.
    @Test func acceptsWeightsNestedUnderTheModelsDirectory() throws {
        try withScratch { root in
            let lora = try touch(root.appendingPathComponent("lora/style.safetensors"))
            #expect(throws: Never.self) { try ModelFileOps.validateDeletable(lora, root: root) }
        }
    }

    /// The `.mfmeta.json` sidecar carries the trigger words / weight band; it
    /// travels with the model so a delete doesn't leave an orphan behind.
    @Test func sidecarTravelsWithTheModel() throws {
        try withScratch { root in
            let model = root.appendingPathComponent("style.safetensors")
            #expect(ModelFileOps.companions(of: model) == [ModelFileOps.sidecarURL(for: model)])
        }
    }

    /// The delete is a *move to Trash*, not an unlink — recoverable by the user, and it takes
    /// the sidecar with it.
    @Test func movesTheModelAndItsSidecarToTheTrash() throws {
        try withScratch { root in
            let model = try touch(root.appendingPathComponent("lora/style.safetensors"))
            let sidecar = try touch(ModelFileOps.sidecarURL(for: model))
            let moved = try ModelFileOps.moveToTrash(model, root: root)
            #expect(moved.count == 2)
            #expect(!FileManager.default.fileExists(atPath: model.path))
            #expect(!FileManager.default.fileExists(atPath: sidecar.path))
        }
    }
}
