import Foundation
import MossForgeCore
import Testing

@testable import MossForgeEngine

// HTTP parser, inference queue, job store, MOSS body and router suites.

@Suite struct HTTPRequestParserTests {
    @Test func waitsForTheFullHeaderAndBody() {
        #expect(HTTPRequestParser.parse(Data("GET /health HTTP/1.1\r\nHost: x\r\n".utf8)) == .incomplete)
        let partial = "POST /v1/moss/generate HTTP/1.1\r\nContent-Length: 10\r\n\r\n{\"pro"
        #expect(HTTPRequestParser.parse(Data(partial.utf8)) == .incomplete)
    }

    @Test func parsesMethodPathQueryHeadersAndBody() throws {
        let raw = "POST /v1/images/generations?debug=1&x=a%20b HTTP/1.1\r\n"
            + "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}trailing"
        guard case .complete(let req) = HTTPRequestParser.parse(Data(raw.utf8)) else {
            Issue.record("expected a complete request"); return
        }
        #expect(req.method == "POST")
        #expect(req.path == "/v1/images/generations")
        #expect(req.query == ["debug": "1", "x": "a b"])
        #expect(req.headers["content-type"] == "application/json")
        #expect(req.body == Data("{}".utf8))  // exactly Content-Length bytes
    }

    @Test func rejectsGarbageAndOversizedHeaders() {
        #expect(HTTPRequestParser.parse(Data("NONSENSE\r\n\r\n".utf8)) == .invalid)
        let huge = "GET / HTTP/1.1\r\nX: " + String(repeating: "a", count: 70_000)
        #expect(HTTPRequestParser.parse(Data(huge.utf8)) == .invalid)
        let tooBig = "POST / HTTP/1.1\r\nContent-Length: 999999999999\r\n\r\n"
        #expect(HTTPRequestParser.parse(Data(tooBig.utf8)) == .invalid)
    }
}

@Suite struct InferenceQueueTests {
    /// Tracks how many jobs are inside the queue at once.
    actor Gauge {
        var current = 0, peak = 0
        func enter() { current += 1; peak = max(peak, current) }
        func leave() { current -= 1 }
    }

    @Test func runsOneJobAtATimeInArrivalOrder() async throws {
        // One GPU, one model resident: concurrent HTTP requests must not denoise in parallel.
        let queue = InferenceQueue()
        let gauge = Gauge()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<6 {
                group.addTask {
                    try await queue.run {
                        await gauge.enter()
                        try await Task.sleep(for: .milliseconds(5))
                        await gauge.leave()
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(await gauge.peak == 1)
        #expect(await queue.pending == 0)
    }

    @Test func aFailedJobDoesNotWedgeTheQueue() async throws {
        let queue = InferenceQueue()
        await #expect(throws: EngineError.self) {
            try await queue.run { () -> Int in throw EngineError.cancelled }
        }
        #expect(try await queue.run { 42 } == 42)
    }

    actor Counter {
        var n = 0
        func bump() { n += 1 }
    }

    @Test func releasesModelsOnceAfterGoingIdle() async throws {
        // A resident server shouldn't pin ~40 GB forever: after the idle timeout the queue
        // asks the backend to drop its models — once, not on every tick.
        let queue = InferenceQueue()
        let released = Counter()
        await queue.setIdleRelease(after: .milliseconds(40)) { await released.bump() }
        _ = try await queue.run { 1 }
        try await Task.sleep(for: .milliseconds(250))
        #expect(await released.n == 1)
        #expect(await queue.pending == 0)
    }

    @Test func aNewJobPostponesTheRelease() async throws {
        let queue = InferenceQueue()
        let released = Counter()
        await queue.setIdleRelease(after: .milliseconds(150)) { await released.bump() }
        _ = try await queue.run { 1 }
        try await Task.sleep(for: .milliseconds(60))
        _ = try await queue.run { 2 }  // resets the idle clock
        try await Task.sleep(for: .milliseconds(60))
        #expect(await released.n == 0)
        try await Task.sleep(for: .milliseconds(300))
        #expect(await released.n == 1)
    }

    @Test func noReleaseBeforeAnyJobOrWithoutATimeout() async throws {
        let queue = InferenceQueue()
        let released = Counter()
        await queue.setIdleRelease(after: .milliseconds(20)) { await released.bump() }
        try await Task.sleep(for: .milliseconds(100))
        #expect(await released.n == 0)  // nothing has run → nothing resident to drop
        await queue.setIdleRelease(after: nil) { await released.bump() }
        _ = try await queue.run { 2 }
        try await Task.sleep(for: .milliseconds(100))
        #expect(await released.n == 0)
    }

    @Test func enablingTheTimeoutAfterAJobStillReleases() async throws {
        // Turning the setting on later should still free what an earlier job loaded.
        let queue = InferenceQueue()
        let released = Counter()
        _ = try await queue.run { 1 }
        await queue.setIdleRelease(after: .milliseconds(20)) { await released.bump() }
        try await Task.sleep(for: .milliseconds(150))
        #expect(await released.n == 1)
    }

    @Test func stubEngineReleaseIsANoOp() async {
        await MossEngine().releaseResidentModels()
    }
}

@Suite struct GenerationJobStoreTests {
    @Test func jobsMoveQueuedRunningDoneWithStageAndSteps() async throws {
        let store = GenerationJobStore()
        let job = await store.create(width: 512, height: 512, steps: 8)
        #expect(job.state == .queued)
        #expect(job.stage == "queued")

        await store.start(job.id)
        await store.progress(job.id, GenerationProgress(step: 0, totalSteps: 8, message: "Loading models…"))
        #expect(await store.job(job.id)?.stage == "Loading models…")
        await store.progress(job.id, GenerationProgress(step: 3, totalSteps: 8, message: "denoise 3/8"))
        // A late, out-of-order update must not move the job backwards.
        await store.progress(job.id, GenerationProgress(step: 2, totalSteps: 8, message: "denoise 2/8"))
        var now = try #require(await store.job(job.id))
        #expect((now.state, now.step, now.totalSteps, now.stage) == (.running, 3, 8, "denoise 3/8"))

        // After the last step the backend decodes + saves without further progress.
        await store.progress(job.id, GenerationProgress(step: 8, totalSteps: 8, message: "denoise 8/8"))
        #expect(await store.job(job.id)?.stage == "decoding")

        await store.finish(job.id, images: [.init(seed: 5, width: 512, height: 512, png: Data([1, 2]))])
        now = try #require(await store.job(job.id))
        #expect(now.state == .done && now.images.count == 1)
        // Progress after completion is ignored.
        await store.progress(job.id, GenerationProgress(step: 1, totalSteps: 8, message: "denoise 1/8"))
        #expect(await store.job(job.id)?.state == .done)
    }

    @Test func keepsOnlyTheMostRecentFinishedJobs() async {
        let store = GenerationJobStore(keep: 3)
        var ids: [String] = []
        for _ in 0..<5 {
            let j = await store.create(width: 64, height: 64, steps: 1)
            await store.finish(j.id, images: [])
            ids.append(j.id)
        }
        #expect(await store.job(ids[0]) == nil)
        #expect(await store.recent().map(\.id) == Array(ids.suffix(3).reversed()))  // newest first
    }
}


@Suite("MossGenerationBody — OpenAI core + MOSS extras")
struct MossGenerationBodyTests {
    let models = FileManager.default.temporaryDirectory.appendingPathComponent("moss-body-\(UUID().uuidString)")

    func body(_ json: String) throws -> MossGenerationBody { try MossGenerationBody(json: Data(json.utf8)) }

    @Test func openAISizeAndMossExtrasMapOntoTheRequest() throws {
        let b = try body(#"{"prompt":"a fox","size":"768x512","n":2,"seed":7,"variant":"base","steps":30,"guidance_scale":4.5,"shift":3,"slg":"off","negative_prompt":"blurry"}"#)
        let r0 = try b.makeRequest(modelsDirectory: models)
        let r1 = try b.makeRequest(modelsDirectory: models, index: 1)
        #expect((r0.width, r0.height, r0.variant, r0.steps, r0.cfg, r0.shift) == (768, 512, .base, 30, 4.5, 3))
        #expect(r0.slg == .off && r0.negativePrompt == "blurry")
        #expect(r0.seed == 7 && r1.seed == 8)  // image i gets seed + i
    }

    @Test func defaultsAreSFTWithTheRecommendedNegativeAndOfficialSLG() throws {
        let r = try body(#"{"prompt":"a fox"}"#).makeRequest(modelsDirectory: models)
        #expect(r.variant == .sft && r.width == 1024 && r.steps == 50 && r.cfg == 3)
        #expect(r.negativePrompt == MossNegativePrompt.recommended && r.slg == .official)
    }

    @Test func turboIsFourStepsNoCFGNoNegative() throws {
        let r = try body(#"{"prompt":"a fox","variant":"turbo"}"#).makeRequest(modelsDirectory: models)
        #expect(r.steps == 4 && r.cfg == 1 && r.shift == 5 && r.sampler == .flowEuler && r.negativePrompt == "")
        #expect(throws: EngineError.self) { try body(#"{"prompt":"x","variant":"turbo","steps":20}"#) }
        #expect(throws: EngineError.self) { try body(#"{"prompt":"x","variant":"turbo","guidance_scale":3}"#) }
    }

    @Test func layoutObjectBecomesTheCompactJSONPrompt() throws {
        let b = try body(#"{"layout":{"summary":{"caption":"a poster"},"items":[],"bbox_format":"yxyx"}}"#)
        let obj = try #require(JSONSerialization.jsonObject(with: Data(b.prompt.utf8)) as? [String: Any])
        #expect((obj["summary"] as? [String: Any])?["caption"] as? String == "a poster")
        #expect(!b.prompt.contains("\n"))
    }

    @Test func slgObjectAndLoRAsWithBlockWeights() throws {
        try FileManager.default.createDirectory(at: models.appendingPathComponent("lora/kijai"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: models.appendingPathComponent("lora/kijai/k.safetensors").path, contents: Data())
        let r = try body(#"{"prompt":"x","slg":{"layers":[24],"scale":3,"start":0.01},"loras":[{"path":"kijai/k.safetensors","strength":-0.5,"blocks":"21-41"}]}"#)
            .makeRequest(modelsDirectory: models)
        #expect(r.slg == .standard(layers: [24], scale: 3, start: 0.01, stop: 1))
        #expect(r.loras.count == 1 && r.loras[0].strength(forBlock: 30) == -0.5 && r.loras[0].strength(forBlock: 3) == 0)
        #expect(r.loras[0].url.path.hasSuffix("lora/kijai/k.safetensors"))
    }

    @Test func rejectsBadInputAndEscapingPaths() throws {
        for bad in [
            #"{"nope":1}"#, #"{"prompt":"  "}"#, #"{"prompt":"x","size":"big"}"#, #"{"prompt":"x","width":1000}"#,
            #"{"prompt":"x","variant":"raw"}"#, #"{"prompt":"x","n":9}"#, #"{"prompt":"x","slg":"maybe"}"#,
            #"{"prompt":"x","loras":[{"path":"a.safetensors","blocks":"z"}]}"#, "[]",
        ] {
            #expect(throws: EngineError.self, "\(bad)") { try body(bad) }
        }
        for escape in ["/etc/x.safetensors", "../x.safetensors", "lora/../../x.safetensors", "missing.safetensors"] {
            #expect(throws: EngineError.self, "\(escape)") {
                try body(#"{"prompt":"x","loras":[{"path":"\#(escape)"}]}"#).makeRequest(modelsDirectory: models)
            }
        }
    }
}

@Suite("MossServerRouter — routes on the stub backend")
struct MossServerRouterTests {
    let router = MossServerRouter(
        engine: MossEngine(), modelsDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("moss-router-\(UUID().uuidString)"))

    func json(_ r: HTTPResponse) throws -> [String: Any] { try #require(JSONSerialization.jsonObject(with: r.body) as? [String: Any]) }

    func waitDone(_ id: String) async throws -> [String: Any] {
        for _ in 0..<400 {
            let obj = try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/jobs/\(id)")))
            if ["done", "failed"].contains(obj["state"] as? String) { return obj }
            try await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("job \(id) never finished")
        return [:]
    }

    @Test func healthModelsAndOptions() async throws {
        #expect(try json(await router.handle(HTTPRequest(method: "GET", path: "/health")))["backend"] as? String == "stub")
        #expect(try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/models")))["object"] as? String == "list")
        let gen = try #require(try json(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/options")))["generate"] as? [String: Any])
        #expect(gen["variants"] as? [String] == ["sft", "turbo", "base"])
        #expect((gen["variant_defaults"] as? [String: [String: Any]])?["turbo"]?["steps"] as? Int == 4)
    }

    @Test func imagesGenerationsAnswersInOpenAIShapeWithPixels() async throws {
        let r = await router.handle(HTTPRequest(
            method: "POST", path: "/v1/images/generations", body: Data(#"{"prompt":"a fox","seed":9,"size":"512x512","n":2,"variant":"turbo"}"#.utf8)))
        #expect(r.status == 200)
        let data = try #require(try json(r)["data"] as? [[String: Any]])
        #expect(data.map { $0["seed"] as? Int } == [9, 10])
        #expect(data.allSatisfy { ($0["b64_json"] as? String).flatMap { Data(base64Encoded: $0) }?.starts(with: [0x89, 0x50]) == true })
    }

    @Test func generateAnswersRawPNGWithHeaders() async throws {
        let r = await router.handle(HTTPRequest(method: "POST", path: "/v1/moss/generate", body: Data(#"{"prompt":"x","seed":3,"size":"256x256"}"#.utf8)))
        #expect(r.status == 200 && r.body.starts(with: [0x89, 0x50, 0x4E, 0x47]))
        #expect(r.headers["X-MossForge-Seed"] == "3" && r.headers["X-MossForge-Size"] == "256x256")
    }

    @Test func jobsRunInTheBackgroundAndServeTheirImage() async throws {
        let submit = await router.handle(HTTPRequest(method: "POST", path: "/v1/moss/jobs", body: Data(#"{"prompt":"a fox","seed":4,"size":"256x256","variant":"turbo"}"#.utf8)))
        #expect(submit.status == 202)
        let id = try #require(try json(submit)["id"] as? String)
        let done = try await waitDone(id)
        #expect(done["state"] as? String == "done" && done["total_steps"] as? Int == 4)
        let image = await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/jobs/\(id)/image"))
        #expect(image.status == 200 && image.body.starts(with: [0x89, 0x50]))
    }

    @Test func enhanceUsesTheBackendAndLayoutReturnsJSON() async throws {
        let r = await router.handle(HTTPRequest(method: "POST", path: "/v1/moss/enhance", body: Data(#"{"prompt":"summer sale","layout":true}"#.utf8)))
        #expect(r.status == 200)
        let text = try #require(try json(r)["prompt"] as? String)
        #expect((try? JSONSerialization.jsonObject(with: Data(text.utf8))) is [String: Any])
    }

    @Test func errorsAndUnknownRoutes() async throws {
        let bad = await router.handle(HTTPRequest(method: "POST", path: "/v1/images/generations", body: Data(#"{"nope":1}"#.utf8)))
        let err = try json(bad)["error"] as? [String: Any]
        #expect(bad.status == 400 && err?["message"] is String)
        #expect(await router.handle(HTTPRequest(method: "GET", path: "/nope")).status == 404)
        #expect(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/generate")).status == 405)
        #expect(await router.handle(HTTPRequest(method: "DELETE", path: "/v1/moss/jobs/x")).status == 405)
        #expect(await router.handle(HTTPRequest(method: "GET", path: "/v1/moss/jobs/nope")).status == 404)
    }
}

@Suite(.serialized) struct MossHTTPServerTests {
    @Test func servesHealthOverLoopback() async throws {
        let server = MossHTTPServer(router: MossServerRouter(engine: MossEngine(), modelsDirectory: FileManager.default.temporaryDirectory))
        let port = try await server.start(port: 0, policy: NetworkPolicy())  // 0 = ephemeral, loopback only
        defer { server.stop() }
        let url = try #require(URL(string: "http://127.0.0.1:\(port)/health"))
        let (data, response) = try await URLSession.shared.data(from: url)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        #expect((try JSONSerialization.jsonObject(with: data) as? [String: Any])?["status"] as? String == "ok")
    }
}
