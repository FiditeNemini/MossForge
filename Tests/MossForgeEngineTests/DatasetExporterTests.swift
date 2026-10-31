import Foundation
import Testing

import MossForgeCore
@testable import MossForgeEngine

/// Gallery → "Export dataset…" / "Export selected…": writes `name.png` (+ `name.txt`) into a
/// folder in the kohya / ai-toolkit layout. Names must be filesystem-safe, unique, and never
/// overwrite what is already in the folder.
@Suite struct DatasetExporterTests {
    private func tempDir() throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mf-export-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func item(_ name: String, caption: String = "", prompt: String = "") -> DatasetExportItem {
        DatasetExportItem(
            source: URL(fileURLWithPath: "/src/\(UUID().uuidString).png"),
            nameHint: name,
            caption: caption,
            prompt: prompt
        )
    }

    @Test func slugIsFilesystemSafe() {
        #expect(DatasetExporter.slug("A red fox / in snow: 2!") == "a-red-fox-in-snow-2")
        #expect(DatasetExporter.slug("   ") == "image")
        #expect(DatasetExporter.slug("../../etc/passwd") == "etc-passwd")
        #expect(DatasetExporter.slug(String(repeating: "x", count: 200)).count == 60)
    }

    @Test func namesAreUniqueCaseInsensitively() {
        let plan = DatasetExporter.plan(
            [item("Fox"), item("fox"), item("FOX"), item("jay")],
            existing: []
        )
        #expect(plan.map(\.stem) == ["fox", "fox-2", "fox-3", "jay"])
    }

    @Test func avoidsExistingFilesInTheFolder() {
        let plan = DatasetExporter.plan([item("fox"), item("jay")], existing: ["Fox.png", "jay.txt"])
        #expect(plan.map(\.stem) == ["fox-2", "jay-2"])
    }

    @Test func keepsTheSourceExtension() {
        let jpg = DatasetExportItem(
            source: URL(fileURLWithPath: "/src/a.JPG"),
            nameHint: "a",
            caption: "",
            prompt: ""
        )
        #expect(DatasetExporter.plan([jpg], existing: [])[0].imageName == "a.jpg")
    }

    @Test func captionFallsBackToPromptAndTakesTrigger() {
        #expect(DatasetExporter.captionText(caption: "  a fox ", prompt: "p", trigger: "") == "a fox")
        #expect(DatasetExporter.captionText(caption: "", prompt: "a jay", trigger: "") == "a jay")
        #expect(DatasetExporter.captionText(caption: "a fox", prompt: "", trigger: "ohwx") == "ohwx, a fox")
        // Already prefixed → not doubled.
        #expect(
            DatasetExporter.captionText(caption: "ohwx, a fox", prompt: "", trigger: "ohwx") == "ohwx, a fox"
        )
        #expect(DatasetExporter.captionText(caption: "", prompt: "", trigger: "ohwx") == "ohwx")
    }

    @Test func exportWritesPairsWithoutOverwriting() throws {
        let src = try tempDir()
        let dst = try tempDir()
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: dst)
        }
        let png = src.appendingPathComponent("x.png")
        try Data([1, 2, 3]).write(to: png)
        try "keep me".write(to: dst.appendingPathComponent("fox.txt"), atomically: true, encoding: .utf8)

        let items = [
            DatasetExportItem(source: png, nameHint: "fox", caption: "a fox", prompt: ""),
            DatasetExportItem(source: png, nameHint: "fox", caption: "", prompt: "a fox again"),
        ]
        let result = try DatasetExporter.export(items, to: dst, captions: true, trigger: "ohwx")
        #expect(result.written.map(\.stem) == ["fox-2", "fox-3"])
        #expect(try String(contentsOf: dst.appendingPathComponent("fox.txt"), encoding: .utf8) == "keep me")
        #expect(
            try String(contentsOf: dst.appendingPathComponent("fox-2.txt"), encoding: .utf8) == "ohwx, a fox"
        )
        #expect(
            try String(contentsOf: dst.appendingPathComponent("fox-3.txt"), encoding: .utf8)
                == "ohwx, a fox again"
        )
        #expect(try Data(contentsOf: dst.appendingPathComponent("fox-2.png")) == Data([1, 2, 3]))
    }

    @Test func imagesOnlyExportSkipsCaptionsAndMissingSources() throws {
        let src = try tempDir()
        let dst = try tempDir()
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: dst)
        }
        let png = src.appendingPathComponent("x.png")
        try Data([9]).write(to: png)
        let missing = src.appendingPathComponent("gone.png")
        let result = try DatasetExporter.export(
            [
                DatasetExportItem(source: png, nameHint: "a", caption: "c", prompt: ""),
                DatasetExportItem(source: missing, nameHint: "b", caption: "c", prompt: ""),
            ],
            to: dst,
            captions: false
        )
        #expect(result.written.map(\.stem) == ["a"])
        #expect(result.skipped == [missing])
        let names = try FileManager.default.contentsOfDirectory(atPath: dst.path).sorted()
        #expect(names == ["a.png"])
    }
}

/// A gallery image → export item: the sidecar caption at the chosen length, else the prompt.
@Suite struct GalleryDatasetTests {
    @Test func usesTheCaptionThenThePrompt() {
        let png = URL(fileURLWithPath: "/out/moss-1.png")
        let captioned = GallerySidecar.adding(MossCaption(long: "L", medium: "M", short: "S"), to: ["prompt": "a fox"])
        let item = GalleryDataset.item(image: png, meta: captioned, detail: .medium)
        #expect(item.caption == "M" && item.prompt == "a fox" && item.nameHint == "a fox")
        let layout = GalleryDataset.item(image: png, meta: ["prompt": #"{"items":[]}"#], detail: .long)
        #expect(layout.caption.isEmpty && layout.nameHint == "moss-1")  // JSON prompts don't make file names
    }
}
