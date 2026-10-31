import CoreGraphics
import Foundation
import ImageIO
import MossForgeCore
import Testing

@testable import MossForgeEngine

// MOSS has no edit
// task presets, so an edit is the generation body + one source image (a reference on the RoPE r axis)
// + the client's edit LoRA, which is required.

@Suite struct MossEditBodyTests {
    let models = FileManager.default.temporaryDirectory.appendingPathComponent("mf-edit-body-\(UUID().uuidString)")

    init() throws {
        let lora = models.appendingPathComponent("lora")
        try FileManager.default.createDirectory(at: lora, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: lora.appendingPathComponent("edit.safetensors").path, contents: Data())
    }

    func request(_ json: String, size: (Int, Int) = (1000, 750)) throws -> GenerationRequest {
        let data = Data(json.utf8)
        let gen = try MossGenerationBody(json: data)
        let edit = try JSONDecoder().decode(MossEditBody.self, from: data)
        return try edit.makeRequest(
            body: gen, modelsDirectory: models, reference: URL(fileURLWithPath: "/tmp/ref0.png"),
            sourceWidth: size.0, sourceHeight: size.1)
    }

    @Test func outputSizeFollowsTheSourceSnappedAndCapped() {
        #expect(MossEditBody.fittedSize(width: 1000, height: 750) == (1008, 752))
        #expect(MossEditBody.fittedSize(width: 3000, height: 2000) == (2048, 1360))
        #expect(MossEditBody.fittedSize(width: 40, height: 30) == (512, 384))
    }

    @Test func theSourceBecomesTheReferenceAtItsOwnSize() throws {
        let r = try request(#"{"prompt":"make it night","loras":[{"path":"edit.safetensors"}],"seed":5,"variant":"turbo"}"#)
        #expect(r.references.map(\.lastPathComponent) == ["ref0.png"])
        #expect(r.loras.map(\.url.lastPathComponent) == ["edit.safetensors"])
        #expect((r.width, r.height) == (1008, 752))
        #expect(r.prompt == "make it night")
        #expect(r.seed == 5)
        #expect(r.variant == .turbo)
        #expect(r.edit == nil)  // not an Edit in Layout mask
    }

    /// `edit_layout` picks how the LoRA sees the source; absent or "auto" leaves it to the metadata.
    @Test func editLayoutIsPassedThrough() throws {
        let r = try request(#"{"prompt":"make it night","loras":[{"path":"edit.safetensors"}],"edit_layout":"Moss-Image-Edit"}"#)
        #expect(r.editLayout == .mossImageEdit)
        #expect(try request(#"{"prompt":"x","loras":[{"path":"edit.safetensors"}],"edit_layout":"auto"}"#).editLayout == nil)
        #expect(throws: EngineError.self) { try request(#"{"prompt":"x","loras":[{"path":"edit.safetensors"}],"edit_layout":"nope"}"#) }
    }

        @Test func aClientSizeWins() throws {
        let r = try request(#"{"prompt":"x","loras":[{"path":"edit.safetensors"}],"size":"512x768"}"#)
        #expect((r.width, r.height) == (512, 768))
    }

    @Test func anEditNeedsAnActiveLoRA() throws {
        for json in [#"{"prompt":"x"}"#, #"{"prompt":"x","loras":[{"path":"edit.safetensors","strength":0}]}"#] {
            #expect {
                _ = try request(json)
            } throws: { error in
                guard case EngineError.invalidRequest(let m) = error else { return false }
                return m.contains("edit LoRA")
            }
        }
    }

    @Test func sourcesMustBeExactlyOne() throws {
        func body(_ json: String) throws -> MossEditBody { try JSONDecoder().decode(MossEditBody.self, from: Data(json.utf8)) }
        #expect(throws: EngineError.self) { try body(#"{"prompt":"x"}"#).validateSource() }
        #expect(throws: EngineError.self) { try body(#"{"prompt":"x","image":"aGk=","job":"j"}"#).validateSource() }
        try body(#"{"prompt":"x","job":"j"}"#).validateSource()
        try body(#"{"prompt":"x","image":"aGk="}"#).validateSource()
    }
}

@Suite struct MossEditRouteTests {
    let models = FileManager.default.temporaryDirectory.appendingPathComponent("mf-edit-route-\(UUID().uuidString)")
    let router: MossServerRouter

    init() throws {
        let lora = models.appendingPathComponent("lora")
        try FileManager.default.createDirectory(at: lora, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: lora.appendingPathComponent("edit.safetensors").path, contents: Data())
        router = MossServerRouter(engine: MossEngine(), modelsDirectory: models)
    }

    func json(_ r: HTTPResponse) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: r.body) as? [String: Any])
    }

    func post(_ path: String, _ body: [String: Any]) async throws -> HTTPResponse {
        await router.handle(HTTPRequest(method: "POST", path: path, body: try JSONSerialization.data(withJSONObject: body)))
    }

    let lora: [[String: Any]] = [["path": "edit.safetensors"]]

    @Test func editRunsAsAnEditJobAtTheSourceSize() async throws {
        let png = MossServerUpscaleRouteTests.png(width: 96, height: 64).base64EncodedString()
        let r = try await post("/v1/moss/edit", ["prompt": "make it night", "image": png, "seed": 5, "loras": lora])
        #expect(r.status == 200)
        #expect(r.contentType == "image/png")
        #expect(r.headers["X-MossForge-Size"] == "512x336")  // 96×64 scaled up
        let status = try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/status")))
        let job = try #require((status["jobs"] as? [[String: Any]])?.first)
        #expect(job["kind"] as? String == "edit")
        #expect(job["state"] as? String == "done")
    }

    @Test func editsChainOnAnEarlierJob() async throws {
        let gen = try await post("/v1/moss/generate", ["prompt": "a fox", "size": "256x256", "seed": 2])
        let genID = try #require(gen.headers["X-MossForge-Job"])
        let submit = try await post("/v1/moss/edit/jobs", ["prompt": "make it night", "job": genID, "loras": lora])
        #expect(submit.status == 202)
        let id = try #require(try json(submit)["id"] as? String)
        for _ in 0..<200 {
            let obj = try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/jobs/\(id)")))
            if obj["state"] as? String == "done" {
                #expect(obj["kind"] as? String == "edit")
                #expect((obj["width"] as? Int, obj["height"] as? Int) == (512, 512))
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("edit job never finished")
    }

    @Test func badInputsAre4xx() async throws {
        let png = MossServerUpscaleRouteTests.png(width: 64, height: 64).base64EncodedString()
        #expect(try await post("/v1/moss/edit", ["prompt": "x", "loras": lora]).status == 400)  // no image
        #expect(try await post("/v1/moss/edit", ["prompt": "x", "image": "bm90IGFuIGltYWdl", "loras": lora]).status == 400)
        #expect(try await post("/v1/moss/edit", ["prompt": "x", "job": "nope", "loras": lora]).status == 404)
        #expect(try await post("/v1/moss/edit", ["image": png, "loras": lora]).status == 400)  // no instruction
        #expect(try await post("/v1/moss/edit", ["prompt": "x", "image": png]).status == 400)  // no edit LoRA
        #expect(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/edit")).status == 405)
    }

    @Test func optionsDescribeTheEditRoute() async throws {
        let obj = try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/options")))
        let edit = try #require(obj["edit"] as? [String: Any])
        #expect(edit["needs_lora"] as? Bool == true)
        #expect(edit["fitted_side_range"] as? [Int] == [512, 2048])
    }
}
