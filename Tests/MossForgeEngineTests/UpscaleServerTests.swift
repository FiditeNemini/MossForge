import CoreGraphics
import Foundation
import ImageIO
import MossForgeCore
import Testing
import UniformTypeIdentifiers

@testable import MossForgeEngine

// The upscale body and route suites (`/v1/moss/*`).

@Suite struct UpscaleBodyTests {
    let models = URL(fileURLWithPath: "/models")

    func body(_ json: String) throws -> UpscaleBody {
        try JSONDecoder().decode(UpscaleBody.self, from: Data(json.utf8))
    }

    @Test func fieldsMapOntoTheUpscaleRequest() throws {
        let b = try body("""
            {"image":"aGk=","scale":4,"method":"esrgan","esrgan_model":"4x-UltraSharp.safetensors",
             "softening":"light","color_fix":"adain","seed":9,"tile_overlap":128}
            """)
        let req = try b.makeRequest(input: URL(fileURLWithPath: "/tmp/in.png"), output: URL(fileURLWithPath: "/tmp/o.png"))
        #expect(req.scale == 4)
        #expect(req.method == .esrgan)
        #expect(req.esrganModel == "4x-UltraSharp.safetensors")
        #expect(req.settings.soften == .light)
        #expect(req.settings.colorFix == .adain)
        #expect(req.settings.seed == 9)
        #expect(req.settings.overlap == 128)
        #expect(req.output?.path == "/tmp/o.png")
        // Defaults: ×2 VOSR2 with the Forge-measured settings.
        let d = try body(#"{"image":"aGk="}"#).makeRequest(input: URL(fileURLWithPath: "/i.png"), output: nil)
        #expect((d.scale, d.method, d.settings) == (2, .vosr, UpscaleSettings()))
    }

    @Test func decodesBase64AndDataURLs() throws {
        #expect(try body(#"{"image":"aGk="}"#).imageData() == Data("hi".utf8))
        #expect(try body(#"{"image":"data:image/png;base64,aGk="}"#).imageData() == Data("hi".utf8))
        #expect(throws: EngineError.self) { try body(#"{"image":"not base64!"}"#).imageData() }
    }

    @Test func rejectsBadChoicesAndSources() throws {
        let input = URL(fileURLWithPath: "/i.png")
        for json in [
            #"{}"#, #"{"image":"aGk=","job":"abc"}"#,
            #"{"image":"aGk=","scale":0.5}"#, #"{"image":"aGk=","scale":9}"#,
            #"{"image":"aGk=","method":"waifu"}"#, #"{"image":"aGk=","softening":"max"}"#,
            #"{"image":"aGk=","color_fix":"lut"}"#, #"{"image":"aGk=","tile_overlap":10}"#,
            #"{"image":"aGk=","esrgan_model":"../x.safetensors"}"#,
            #"{"image":"aGk=","esrgan_model":"/etc/x.safetensors"}"#,
        ] {
            #expect(throws: EngineError.self, "\(json)") {
                let b = try body(json)
                try b.validateSource()
                _ = try b.makeRequest(input: input, output: nil)
            }
        }
    }
}

@Suite struct MossServerUpscaleRouteTests {
    let router = MossServerRouter(
        engine: MossEngine(), modelsDirectory: FileManager.default.temporaryDirectory
            .appendingPathComponent("mf-up-router-\(UUID().uuidString)"))

    func json(_ r: HTTPResponse) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: r.body) as? [String: Any])
    }

    static func png(width: Int, height: Int) -> Data {
        let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(out as CFMutableData, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        CGImageDestinationFinalize(dest)
        return out as Data
    }

    func post(_ path: String, _ body: [String: Any]) async throws -> HTTPResponse {
        await router.handle(HTTPRequest(
            method: "POST", path: path, body: try JSONSerialization.data(withJSONObject: body)))
    }

    func waitDone(_ id: String) async throws -> [String: Any] {
        for _ in 0..<200 {
            let obj = try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/jobs/\(id)")))
            if obj["state"] as? String == "done" || obj["state"] as? String == "failed" { return obj }
            try await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("job \(id) never finished")
        return [:]
    }

    @Test func upscaleAnswersThePNGWithItsSize() async throws {
        let input = Self.png(width: 96, height: 64)
        let r = try await post("/v1/moss/upscale", ["image": input.base64EncodedString(), "scale": 2, "seed": 3])
        #expect(r.status == 200)
        #expect(r.contentType == "image/png")
        #expect(r.headers["X-MossForge-Size"] == "192x128")
        #expect(r.headers["X-MossForge-Seed"] == "3")
        #expect(r.body == input)  // the stub copies its input
        let status = try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/status")))
        let job = try #require((status["jobs"] as? [[String: Any]])?.first)
        #expect(job["kind"] as? String == "upscale")
        #expect((job["width"] as? Int, job["height"] as? Int) == (192, 128))
    }

    @Test func upscaleJobsRunQueuedAndCanChainOnAnEarlierJob() async throws {
        let submit = try await post(
            "/v1/moss/upscale/jobs", ["image": Self.png(width: 64, height: 64).base64EncodedString(), "scale": 4])
        #expect(submit.status == 202)
        let id = try #require(try json(submit)["id"] as? String)
        let done = try await waitDone(id)
        #expect(done["state"] as? String == "done")
        #expect(done["kind"] as? String == "upscale")
        let image = await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/jobs/\(id)/image"))
        #expect(image.status == 200 && image.contentType == "image/png")

        // Upscale an earlier job's output by id — no re-upload.
        let again = try await post("/v1/moss/upscale", ["job": id, "scale": 2, "method": "esrgan"])
        #expect(again.status == 200)
        #expect(again.headers["X-MossForge-Size"] == "128x128")
    }

    @Test func badUpscaleInputsAre400() async throws {
        #expect(try await post("/v1/moss/upscale", ["image": "bm90IGFuIGltYWdl"]).status == 400)  // not an image
        #expect(try await post("/v1/moss/upscale", ["job": "nope"]).status == 404)
        #expect(try await post("/v1/moss/upscale", ["image": "aGk=", "method": "x"]).status == 400)
        // Output too large to be a sane request.
        let big = Self.png(width: 4096, height: 64).base64EncodedString()
        #expect(try await post("/v1/moss/upscale", ["image": big, "scale": 4]).status == 400)
        // A job index past its images.
        let gen = try await post("/v1/images/generations", ["prompt": "x", "size": "256x256"])
        let genID = try #require(try json(gen)["job"] as? String)
        #expect(try await post("/v1/moss/upscale", ["job": genID, "index": 3]).status == 400)
        #expect(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/upscale")).status == 405)
    }

    /// "Upscale what you just made": a generation job's PNG chains straight into an upscale.
    @Test func generatedImagesChainIntoAnUpscale() async throws {
        let gen = try await post("/v1/images/generations", ["prompt": "x", "size": "256x256"])
        let genID = try #require(try json(gen)["job"] as? String)
        let up = try await post("/v1/moss/upscale", ["job": genID, "scale": 2, "method": "esrgan"])
        #expect(up.status == 200)
        #expect(up.headers["X-MossForge-Size"] == "512x512")
    }

    @Test func optionsListTheUpscaleChoices() async throws {
        let r = await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/options"))
        #expect(r.status == 200)
        let obj = try json(r)
        #expect((obj["capabilities"] as? [String])?.contains("upscale") == true)
        let up = try #require(obj["upscale"] as? [String: Any])
        #expect(up["methods"] as? [String] == ["vosr", "esrgan"])
        #expect(up["esrgan_models"] as? [String] == [])
        #expect(up["max_output_side"] as? Int == UpscaleBody.maxOutputSide)
    }
}
