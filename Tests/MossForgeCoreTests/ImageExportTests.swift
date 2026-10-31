import Foundation
import Testing

@testable import MossForgeCore

/// Gallery / Generate → Save a Copy: never overwrites, Finder-style " 2" suffixes.
@Suite struct ImageExportTests {
    let folder = URL(fileURLWithPath: "/tmp/out")

    @Test func keepsTheNameWhenFree() {
        let src = URL(fileURLWithPath: "/pics/moss-1.png")
        #expect(ImageExport.destination(for: src, in: folder, taken: []).lastPathComponent == "moss-1.png")
    }

    @Test func numbersAroundExistingFilesAndEachOther() {
        let a = URL(fileURLWithPath: "/pics/a.png")
        let b = URL(fileURLWithPath: "/other/a.png")
        let plan = ImageExport.destinations(for: [a, b], in: folder, existing: ["a.png", "a 2.png"])
        #expect(plan.map(\.lastPathComponent) == ["a 3.png", "a 4.png"])
    }

    @Test func takenNamesAreCaseInsensitiveLikeAPFS() {
        let src = URL(fileURLWithPath: "/pics/Fox.PNG")
        #expect(ImageExport.destination(for: src, in: folder, taken: ["fox.png"]).lastPathComponent == "Fox 2.PNG")
    }

    @Test func suggestedNameUsesThePromptWhenThereIsOne() {
        #expect(ImageExport.suggestedName(prompt: "A red fox, in snow at dawn!", seed: 42) == "a-red-fox-in-snow-at-dawn-42.png")
        #expect(ImageExport.suggestedName(prompt: "", seed: 7) == "moss-7.png")
        #expect(ImageExport.suggestedName(prompt: "{\"summary\":{}}", seed: 7) == "moss-layout-7.png")
        let long = ImageExport.suggestedName(prompt: String(repeating: "word ", count: 40), seed: 1)
        #expect(long.count <= 60)
    }
}
