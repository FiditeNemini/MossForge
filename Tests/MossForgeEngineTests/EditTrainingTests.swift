import Foundation
import Testing

@testable import MossForgeCore
@testable import MossForgeEngine

/// Experimental edit training (the community / Ostris AI Toolkit method): the source image rides along
/// as clean reference tokens on the RoPE axis MOSS leaves at 0, and datasets pair targets with controls.
@Suite struct EditTrainingTests {
    /// Reference k: `r = k`, the image tokens' `t` (= text length), its own zero-centred h/w grid.
    @Test func referenceTokensUseTheirOwnRIndex() {
        let p = MossRoPEIndex.referencePositions(index: 1, gridHeight: 2, gridWidth: 3, sequencePosition: 7)
        #expect(p.count == 6)
        #expect(p[0] == [1, 7, -0.5, -1])
        #expect(p[5] == [1, 7, 0.5, 1])
        let a = MossRoPEIndex.referenceAngles(index: 2, gridHeight: 1, gridWidth: 1, sequencePosition: 4, headDim: 128, base: 10000)
        #expect(a.count == 4 * 16)
        #expect(a[0] == 2)  // r = 2 × band 0 (1)
        #expect(a[16] == 4)  // t = 4 × band 0
    }

    private func folder(_ files: [String: String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("edit-\(UUID())")
        for (path, text) in files {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(text.utf8).write(to: url)
        }
        return root
    }

    /// AI Toolkit's layout: targets + instructions in the folder, sources with the same name in `control/`.
    @Test func pairsTargetsWithControlsByName() throws {
        let root = try folder([
            "a.png": "", "a.txt": "make the sky purple", "b.jpg": "", "b.txt": "add a hat", "c.png": "",
            "control/a.png": "", "control/b.png": "", "control/z.png": "",
        ])
        defer { try? FileManager.default.removeItem(at: root) }
        let scan = EditDatasetScanner.scan(directory: root)
        #expect(scan.pairs.map(\.target.lastPathComponent) == ["a.png", "b.jpg"])
        #expect(scan.pairs.map(\.control.lastPathComponent) == ["a.png", "b.png"])
        #expect(scan.pairs.map(\.instruction) == ["make the sky purple", "add a hat"])
        #expect(scan.missingControl.map(\.lastPathComponent) == ["c.png"])
        #expect(EditDatasetScanner.controlFolder(for: root).lastPathComponent == "control")
    }

    @Test func editRequestsNeedPairs() throws {
        var r = TrainingRequest(baseModel: URL(fileURLWithPath: "/m"), dataset: URL(fileURLWithPath: "/d"), output: URL(fileURLWithPath: "/o"))
        r.mode = .edit
        #expect(throws: EngineError.self) { try r.validate(sampleCount: 0) }
        try r.validate(sampleCount: 3)
        #expect(TrainingMode.edit.displayName.contains("Edit"))
    }
}
