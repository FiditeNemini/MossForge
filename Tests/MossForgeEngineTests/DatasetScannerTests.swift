import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// The dataset scanner turns a directory of image + caption-sidecar files into training
/// pairs — the directory-based alternative to a Gallery collection, so datasets built by
/// other tools/models can train a LoRA here directly.
@Suite struct DatasetScannerTests {
    /// Make a temp directory with the given files (empty content unless a caption is given).
    private func fixture(_ files: [(name: String, content: String?)]) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mf-dataset-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in files {
            try (f.content ?? "").write(
                to: dir.appendingPathComponent(f.name), atomically: true, encoding: .utf8)
        }
        return dir
    }

    @Test func pairsImagesWithTxtSidecarsSortedByName() throws {
        let dir = try fixture([
            ("b.png", nil), ("b.txt", "a red fox in snow"),
            ("a.jpg", nil), ("a.txt", "a blue jay"),
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let samples = DatasetScanner.scan(directory: dir)
        #expect(samples.count == 2)
        // Sorted by filename → a before b.
        #expect(samples[0].image.lastPathComponent == "a.jpg")
        #expect(samples[0].caption == "a blue jay")
        #expect(samples[1].image.lastPathComponent == "b.png")
        #expect(samples[1].caption == "a red fox in snow")
    }

    @Test func fallsBackToHumanisedStemWhenNoSidecar() throws {
        let dir = try fixture([("orange_tabby-cat.webp", nil)])
        defer { try? FileManager.default.removeItem(at: dir) }

        let samples = DatasetScanner.scan(directory: dir)
        #expect(samples.count == 1)
        #expect(samples[0].caption == "orange tabby cat")
    }

    @Test func ignoresNonImageFilesAndBlankCaptions() throws {
        let dir = try fixture([
            ("notes.md", "ignore me"),
            ("cat.png", nil), ("cat.txt", "   \n  "),  // blank caption → stem fallback
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let samples = DatasetScanner.scan(directory: dir)
        #expect(samples.count == 1)  // only cat.png
        #expect(samples[0].image.lastPathComponent == "cat.png")
        #expect(samples[0].caption == "cat")  // blank sidecar ignored
    }

    @Test func missingDirectoryYieldsNoSamples() {
        let dir = URL(fileURLWithPath: "/no/such/mf-dataset-\(UUID().uuidString)")
        #expect(DatasetScanner.scan(directory: dir).isEmpty)
    }
}
