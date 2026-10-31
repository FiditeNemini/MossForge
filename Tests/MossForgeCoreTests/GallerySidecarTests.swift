import Foundation
import Testing

@testable import MossForgeCore

/// The gallery's JSON sidecar (`<image>.json`): the generation request, plus what the app adds later —
/// a caption (Gallery → Caption) and, for an upscaled copy, where it came from.
@Suite("GallerySidecar — gallery JSON metadata")
struct GallerySidecarTests {
    @Test func sidecarSitsBesideTheImage() {
        let png = URL(fileURLWithPath: "/out/moss-20261020T0900Z-42.png")
        #expect(GallerySidecar.url(for: png).path == "/out/moss-20261020T0900Z-42.json")
    }

    @Test func captionIsMergedWithoutLosingTheRequest() throws {
        let meta: [String: Any] = ["prompt": "a lighthouse", "seed": 42]
        let caption = MossCaption(long: "A tall lighthouse.", medium: "A lighthouse.", short: "Lighthouse.")
        let merged = GallerySidecar.adding(caption, to: meta)
        #expect(merged["prompt"] as? String == "a lighthouse" && merged["seed"] as? Int == 42)
        let c = try #require(merged["caption"] as? [String: Any])
        #expect(c["long"] as? String == "A tall lighthouse." && c["short"] as? String == "Lighthouse.")
        #expect(GallerySidecar.caption(in: merged) == caption)
    }

    @Test func upscaledCopyKeepsTheRequestAndRecordsItsSource() {
        let source = URL(fileURLWithPath: "/out/a.png")
        let meta: [String: Any] = ["prompt": "p", "width": 1024, "height": 1024, "seed": 7]
        let request = UpscaleRequest(image: source, scale: 4, method: .esrgan, esrganModel: "RealESRGAN_x4plus.safetensors")
        let result = UpscaleResult(output: URL(fileURLWithPath: "/out/a_x4.png"), width: 4096, height: 4096)
        let up = GallerySidecar.upscaled(meta, request: request, result: result)
        #expect(up["prompt"] as? String == "p" && up["seed"] as? Int == 7)
        #expect(up["width"] as? Int == 4096 && up["height"] as? Int == 4096)
        let u = up["upscale"] as? [String: Any]
        #expect(u?["from"] as? String == "a.png" && u?["method"] as? String == "esrgan" && u?["scale"] as? Double == 4)
        #expect(u?["model"] as? String == "RealESRGAN_x4plus.safetensors")
    }

    @Test func roundTripsThroughDisk() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sidecar-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let png = dir.appendingPathComponent("x.png")
        #expect(GallerySidecar.read(for: png).isEmpty)  // missing sidecar → empty, not an error
        try GallerySidecar.write(["prompt": "hi"], for: png)
        #expect(GallerySidecar.read(for: png)["prompt"] as? String == "hi")
    }
}

@Suite("GallerySidecar — collections")
struct GalleryCollectionsTests {
    @Test func collectionsAreATagListInTheSidecar() {
        var meta: [String: Any] = ["prompt": "p"]
        #expect(GallerySidecar.collections(in: meta).isEmpty)
        meta = GallerySidecar.setting(["foxes", "posters"], in: meta)
        #expect(GallerySidecar.collections(in: meta) == ["foxes", "posters"])
        #expect(meta["collections"] as? [String] == ["foxes", "posters"])  // stored sorted
        meta = GallerySidecar.setting([], in: meta)
        #expect(meta["collections"] == nil && meta["prompt"] as? String == "p")
    }
}

@Suite("FolderCaptioning — caption a folder of images as name.txt")
struct FolderCaptioningTests {
    @Test func pendingSkipsCaptionedImagesUnlessOverwriting() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("fc-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for n in ["a.png", "b.JPG", "c.webp", "notes.md", "b.txt"] {
            FileManager.default.createFile(atPath: dir.appendingPathComponent(n).path, contents: Data([1]))
        }
        #expect(FolderCaptioning.pending(in: dir, overwrite: false).map(\.lastPathComponent) == ["a.png", "c.webp"])
        #expect(FolderCaptioning.pending(in: dir, overwrite: true).map(\.lastPathComponent) == ["a.png", "b.JPG", "c.webp"])
        try FolderCaptioning.write("a fox", trigger: "mossfox", for: dir.appendingPathComponent("a.png"))
        #expect(try String(contentsOf: dir.appendingPathComponent("a.txt"), encoding: .utf8) == "mossfox, a fox")
    }
}
