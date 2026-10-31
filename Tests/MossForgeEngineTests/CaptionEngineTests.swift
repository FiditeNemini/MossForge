import CoreGraphics
import Foundation
import ImageIO
import MossForgeCore
import Testing

@testable import MossForgeEngine

/// Pure geometry for the Qwen3.5 vision path: `smart_resize` and the 3-D M-RoPE positions,
/// transcribed from transformers 5.8 `qwen2_vl.smart_resize` and `Qwen3_5Model.get_rope_index`.
@Suite struct QwenVisionGeometryTests {
    @Test func smartResizeSnapsToMultiplesOf32WithinThePixelBudget() {
        // Already on the grid.
        #expect(QwenVisionGeometry.smartResize(height: 512, width: 512, maxPixels: 1024 * 1024) == (512, 512))
        // Round half to even: 528/32 = 16.5 → 16.
        #expect(QwenVisionGeometry.smartResize(height: 528, width: 1000, maxPixels: 1024 * 1024) == (512, 992))
        // Too big: scaled down by √(hw/max), floored to the grid.
        #expect(QwenVisionGeometry.smartResize(height: 2048, width: 4096, maxPixels: 1024 * 1024) == (704, 1440))
        // Too small for Qwen3.5's 256² minimum: scaled up, ceiled.
        #expect(QwenVisionGeometry.smartResize(height: 100, width: 200) == (192, 384))
    }

    @Test func gridAndTokenCounts() {
        let g = QwenVisionGeometry.grid(height: 512, width: 768)
        #expect((g.t, g.h, g.w) == (1, 32, 48))
        #expect(g.patches == 1536 && g.tokens == 384)
    }

    /// Text runs 0…n−1 on all axes; image tokens get (start, start+row, start+col); text resumes at
    /// start + max(h, w)/merge.
    @Test func mropePositionsForTextImageText() {
        let pad = 9
        // 2 text, a 2×3-token image (grid 4×6 patches), 2 text.
        let ids = [1, 2] + Array(repeating: pad, count: 6) + [3, 4]
        let p = QwenMRoPEIndex.positions(ids: ids, imageTokenID: pad, grids: [.init(t: 1, h: 4, w: 6)])
        #expect(p.ids[0] == [0, 1, 2, 2, 2, 2, 2, 2, 5, 6])
        #expect(p.ids[1] == [0, 1, 2, 2, 2, 3, 3, 3, 5, 6])
        #expect(p.ids[2] == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6])
        #expect(p.next == 7)
    }

    @Test func mropePositionsForTextOnlyAreSequential() {
        let p = QwenMRoPEIndex.positions(ids: [5, 6, 7], imageTokenID: 9, grids: [])
        #expect(p.ids == [[0, 1, 2], [0, 1, 2], [0, 1, 2]] && p.next == 3)
    }
}

/// Captioning is an engine capability: the stub answers without weights.
@Suite struct CaptionCapabilityTests {
    static func png(width: Int = 64, height: Int = 48) -> Data {
        let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(out as CFMutableData, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        CGImageDestinationFinalize(dest)
        return out as Data
    }

    @Test func everyBackendAdvertisesCaptioning() {
        #expect(EngineCapabilities.all.contains(.captioning))
        #expect(MossEngine().capabilities.names.contains("caption"))
    }

    @Test func stubCaptionNamesTheFile() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mf-cap-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.png().write(to: url)
        let c = try await MossEngine().caption(CaptionRequest(image: url, detail: .short))
        #expect(c.short.contains(url.lastPathComponent))
        #expect(!c.long.isEmpty && !c.medium.isEmpty)
        await #expect(throws: EngineError.self) {
            _ = try await MossEngine().caption(CaptionRequest(image: URL(fileURLWithPath: "/nope.png")))
        }
    }
}

@Suite struct CaptionRouteTests {
    let router = MossServerRouter(
        engine: MossEngine(), modelsDirectory: FileManager.default.temporaryDirectory
            .appendingPathComponent("mf-cap-router-\(UUID().uuidString)"))

    func post(_ body: [String: Any]) async throws -> HTTPResponse {
        await router.handle(HTTPRequest(
            method: "POST", path: "/v1/moss/caption", body: try JSONSerialization.data(withJSONObject: body)))
    }

    func json(_ r: HTTPResponse) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: r.body) as? [String: Any])
    }

    @Test func captionsAnUploadedImage() async throws {
        let r = try await post(["image": CaptionCapabilityTests.png().base64EncodedString(), "detail": "medium"])
        #expect(r.status == 200)
        let obj = try json(r)
        #expect(obj["caption"] as? String == obj["medium"] as? String)
        #expect((obj["long"] as? String)?.isEmpty == false && obj["short"] != nil)
        #expect(obj["watermark"] as? [Any] != nil && obj["borders"] as? [String] != nil)
    }

    @Test func captionsAnEarlierJobsImage() async throws {
        let gen = await router.handle(HTTPRequest(
            method: "POST", path: "/v1/images/generations",
            body: try JSONSerialization.data(withJSONObject: ["prompt": "x", "size": "256x256"])))
        let id = try #require(try json(gen)["job"] as? String)
        #expect(try await post(["job": id]).status == 200)
    }

    @Test func badCaptionRequestsAre4xx() async throws {
        #expect(try await post([:]).status == 400)
        #expect(try await post(["image": "aGk=", "job": "x"]).status == 400)
        #expect(try await post(["image": "bm90IGFuIGltYWdl"]).status == 400)  // not an image
        #expect(try await post(["image": CaptionCapabilityTests.png().base64EncodedString(), "detail": "huge"]).status == 400)
        #expect(try await post(["job": "nope"]).status == 404)
        #expect(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/caption")).status == 405)
    }
}
