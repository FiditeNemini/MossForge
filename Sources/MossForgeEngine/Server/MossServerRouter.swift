import Foundation
import ImageIO
import MossForgeCore

// Queue/job/response machinery; MOSS routes (`/v1/moss/*`), MOSS options and model list. See NOTES.md.

/// The resident server's routes, independent of the transport (`MossHTTPServer`, the app's Server
/// screen later). Generation runs through one `InferenceQueue` — one model resident, one GPU — and
/// every generation is a tracked job.
///
/// - `GET  /health` — status, backend, queue depth.
/// - `GET  /v1/status` — queue depth + recent jobs with their live stage (newest first).
/// - `GET  /v1/models` — OpenAI model list of the checkpoints under the models folder.
/// - `POST /v1/images/generations` — OpenAI images API (`data[].b64_json`) + MOSS extras.
/// - `POST /v1/moss/generate` — same body, answers the first image as raw PNG.
/// - `POST /v1/moss/jobs` — same body, answers `202 {"id"}` at once; the job runs queued.
/// - `GET  /v1/moss/jobs/{id}` — that job's status; `…/{id}/image[?index=n]` — its PNG.
/// - `POST /v1/moss/enhance` — `{"prompt", "layout": bool, "width", "height"}` → `{"prompt"}`.
/// - `POST /v1/moss/upscale` — `UpscaleBody` (base64 image or an earlier job's id), answers the PNG;
///   `POST /v1/moss/upscale/jobs` — same body, answers `202 {"id"}` (shared job routes).
/// - `POST /v1/moss/edit` — an edit-LoRA edit: `MossGenerationBody` fields (prompt = the instruction,
///   `loras` must hold the edit LoRA) + `MossEditBody` (base64 image or a job id), answers the PNG;
///   `POST /v1/moss/edit/jobs` — same body, answers `202 {"id"}`.
/// - `POST /v1/moss/caption` — `CaptionBody` (base64 image or a job id) → the recaption JSON
///   (`caption`, `long`, `medium`, `short`, `watermark`, `borders`).
/// - `GET  /v1/moss/options` — the tunable choices and ranges.
/// - `GET  /v1/moss/loras` — the LoRAs under `<models>/lora` with their rank.
public struct MossServerRouter: Sendable {
    public let engine: MossEngine
    public let modelsDirectory: URL
    public let queue: InferenceQueue
    public let jobs: GenerationJobStore
    /// Job start/finish lines for the server log (the transport wires this to its `onLog`).
    public let events = EventSink()

    public init(
        engine: MossEngine, modelsDirectory: URL, queue: InferenceQueue = InferenceQueue(),
        jobs: GenerationJobStore = GenerationJobStore()
    ) {
        self.engine = engine
        self.modelsDirectory = modelsDirectory
        self.queue = queue
        self.jobs = jobs
    }

    static let routes: [String: String] = [
        "/health": "GET", "/v1/status": "GET", "/v1/models": "GET",
        "/v1/images/generations": "POST", "/v1/moss/generate": "POST", "/v1/moss/jobs": "POST",
        "/v1/moss/enhance": "POST", "/v1/moss/options": "GET", "/v1/moss/loras": "GET",
        "/v1/moss/upscale": "POST", "/v1/moss/upscale/jobs": "POST", "/v1/moss/caption": "POST",
        "/v1/moss/edit": "POST", "/v1/moss/edit/jobs": "POST",
    ]
    public static let jobPrefix = "/v1/moss/jobs/"

    public func handle(_ request: HTTPRequest) async -> HTTPResponse {
        if request.path.hasPrefix(Self.jobPrefix) {
            guard request.method == "GET" else { return .error(405, message: "\(Self.jobPrefix)… accepts GET only") }
            return await jobRoute(request)
        }
        guard let method = Self.routes[request.path] else { return .error(404, message: "not found") }
        guard method == request.method else { return .error(405, message: "\(request.path) accepts \(method) only") }
        switch request.path {
        case "/health":
            return .json(["status": "ok", "backend": engine.backendName, "queue": await queue.pending])
        case "/v1/status":
            let recent = await jobs.recent()
            var payload: [String: Any] = [
                "backend": engine.backendName, "queue": await queue.pending, "jobs": recent.map { $0.summary() },
            ]
            if let current = recent.first(where: { $0.state == .running }) { payload["current"] = current.summary() }
            return .json(payload)
        case "/v1/models":
            return .json(["object": "list", "data": modelList()])
        case "/v1/moss/options":
            return .json(options())
        case "/v1/moss/loras":
            return .json(["object": "list", "data": loraList()])
        case "/v1/moss/enhance":
            return await enhance(request)
        case "/v1/moss/caption":
            return await caption(request)
        case "/v1/moss/upscale", "/v1/moss/upscale/jobs":
            switch await parseUpscale(request) {
            case .failure(let response): return response
            case .success(let up):
                let job = await jobs.create(width: up.width, height: up.height, steps: 1, kind: .upscale)
                if request.path == "/v1/moss/upscale/jobs" {
                    Task { await runUpscale(job.id, up) }
                    return .json(["id": job.id, "state": "queued", "queue": await queue.pending + 1], status: 202)
                }
                await runUpscale(job.id, up)
                return await respond(job.id, rawPNG: true)
            }
        case "/v1/moss/edit", "/v1/moss/edit/jobs":
            switch await parseEdit(request) {
            case .failure(let response): return response
            case .success(let ed):
                let job = await jobs.create(width: ed.request.width, height: ed.request.height, steps: ed.request.steps, kind: .edit)
                if request.path == "/v1/moss/edit/jobs" {
                    Task { await runEdit(job.id, ed) }
                    return .json(["id": job.id, "state": "queued", "queue": await queue.pending + 1], status: 202)
                }
                await runEdit(job.id, ed)
                return await respond(job.id, rawPNG: true)
            }
        case "/v1/moss/jobs":
            switch parse(request) {
            case .failure(let response): return response
            case .success(let body):
                let job = await jobs.create(width: body.width ?? 1024, height: body.height ?? 1024, steps: body.steps ?? body.variant.defaults.steps)
                Task { await run(job.id, body) }
                return .json(["id": job.id, "state": "queued", "queue": await queue.pending + 1], status: 202)
            }
        default:
            switch parse(request) {
            case .failure(let response): return response
            case .success(let body):
                let job = await jobs.create(width: body.width ?? 1024, height: body.height ?? 1024, steps: body.steps ?? body.variant.defaults.steps)
                await run(job.id, body)
                return await respond(job.id, rawPNG: request.path == "/v1/moss/generate")
            }
        }
    }

    private enum Parsed { case success(MossGenerationBody), failure(HTTPResponse) }

    private func parse(_ http: HTTPRequest) -> Parsed {
        do {
            let body = try MossGenerationBody(json: http.body)
            _ = try body.makeRequest(modelsDirectory: modelsDirectory)  // validate before queueing
            return .success(body)
        } catch let EngineError.invalidRequest(reason) {
            return .failure(.error(400, message: reason))
        } catch {
            return .failure(.error(400, message: "\(error)"))
        }
    }

    /// Run one request (n images, seeds `seed+i`) through the shared queue.
    private func run(_ id: String, _ body: MossGenerationBody) async {
        let (engine, jobs, models) = (engine, jobs, modelsDirectory)
        let label = "\(body.variant.rawValue) \(body.width ?? 1024)×\(body.height ?? 1024) · \(body.steps ?? body.variant.defaults.steps) steps × \(body.n)"
        await track(id, label) {
            var images: [GenerationJobStore.Image] = []
            for index in 0..<body.n {
                let request = try body.makeRequest(modelsDirectory: models, index: index)
                let result = try await engine.generate(request) { p in Task { await jobs.progress(id, p) } }
                images.append(.init(seed: result.seed, width: result.width, height: result.height, png: result.imagePNG))
            }
            return images
        }
    }

    private func enhance(_ http: HTTPRequest) async -> HTTPResponse {
        guard let obj = (try? JSONSerialization.jsonObject(with: http.body)) as? [String: Any],
            let prompt = obj["prompt"] as? String, !prompt.trimmingCharacters(in: .whitespaces).isEmpty
        else { return .error(400, message: "body must be a JSON object with a 'prompt'") }
        let canvas = Dimensions(width: obj["width"] as? Int ?? 1024, height: obj["height"] as? Int ?? 1024)
        do {
            let text = try await engine.enhancePrompt(
                PromptEnhanceRequest(prompt: prompt, layout: obj["layout"] as? Bool ?? false, canvas: canvas), progress: nil)
            return .json(["prompt": text])
        } catch EngineError.notImplemented(let what) {
            return .error(501, message: "\(what) is not available on the \(engine.backendName) backend")
        } catch {
            return .error(500, message: "\(error)")
        }
    }

    /// A parsed upscale: the engine request, its scratch folder (removed after the run) and
    /// the output size.
    struct UpscaleJob: Sendable {
        let request: UpscaleRequest
        let scratch: URL
        let width: Int
        let height: Int
    }

    private enum UpscaleParsed { case success(UpscaleJob), failure(HTTPResponse) }

    private func parseUpscale(_ http: HTTPRequest) async -> UpscaleParsed {
        let body: UpscaleBody
        do {
            body = try JSONDecoder().decode(UpscaleBody.self, from: http.body)
        } catch {
            return .failure(.error(400, message: "body must be a JSON object with 'image' (base64) or 'job'"))
        }
        do {
            try body.validateSource()
            let input: Data
            switch await sourceImage(base64: body.image, job: body.job, index: body.index) {
            case .failure(let e): return .failure(e.response)
            case .success(let data): input = data
            }
            let (w, h) = try Self.pixelSize(input, field: "image")

            let scratch = FileManager.default.temporaryDirectory
                .appendingPathComponent("mossforge-upscale-\(UUID().uuidString)")
            let name = input.starts(with: [0xFF, 0xD8]) ? "input.jpg" : "input.png"
            let request = try body.makeRequest(
                input: scratch.appendingPathComponent(name), output: scratch.appendingPathComponent("output.png"))
            let (tw, th) = UpscaleTiling.targetSize(width: w, height: h, scale: request.scale)
            guard max(tw, th) <= UpscaleBody.maxOutputSide else {
                throw EngineError.invalidRequest(
                    "\(w)×\(h) ×\(request.scale) would be \(tw)×\(th); the server's limit is \(UpscaleBody.maxOutputSide) px per side")
            }
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            try input.write(to: request.image)
            return .success(UpscaleJob(request: request, scratch: scratch, width: tw, height: th))
        } catch let EngineError.invalidRequest(reason) {
            return .failure(.error(400, message: reason))
        } catch {
            return .failure(.error(500, message: "could not stage the upscale input: \(error)"))
        }
    }

    private func runUpscale(_ id: String, _ up: UpscaleJob) async {
        defer { try? FileManager.default.removeItem(at: up.scratch) }
        let (engine, jobs) = (engine, jobs)
        let r = up.request
        await track(id, "upscale ×\(r.scale) \(r.method.rawValue) → \(up.width)×\(up.height)") {
            let result = try await engine.upscale(r) { p in Task { await jobs.progress(id, p) } }
            return [.init(
                seed: r.settings.seed, width: result.width, height: result.height,
                png: try? Data(contentsOf: result.output))]
        }
    }

    /// Pixel size of an encoded PNG/JPEG; `invalidRequest` naming `field` when it isn't one.
    static func pixelSize(_ data: Data, field: String) throws -> (Int, Int) {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
            let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
            let w = props[kCGImagePropertyPixelWidth] as? Int, let h = props[kCGImagePropertyPixelHeight] as? Int
        else { throw EngineError.invalidRequest("'\(field)' is not a decodable PNG/JPEG") }
        return (w, h)
    }

    /// A parsed edit: the engine request and the scratch folder holding its reference (removed after the run).
    struct EditJob: Sendable {
        let request: GenerationRequest
        let scratch: URL
    }

    private enum EditParsed { case success(EditJob), failure(HTTPResponse) }

    private func parseEdit(_ http: HTTPRequest) async -> EditParsed {
        let gen: MossGenerationBody, body: MossEditBody
        do {
            gen = try MossGenerationBody(json: http.body)
            body = try JSONDecoder().decode(MossEditBody.self, from: http.body)
            try body.validateSource()
        } catch let EngineError.invalidRequest(reason) {
            return .failure(.error(400, message: reason))
        } catch {
            return .failure(.error(400, message: "body must be a JSON object with 'prompt', 'loras' and 'image' (base64) or 'job'"))
        }
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("mossforge-edit-\(UUID().uuidString)")
        do {
            let input: Data
            switch await sourceImage(base64: body.image, job: body.job, index: body.index) {
            case .failure(let e): return .failure(e.response)
            case .success(let data): input = data
            }
            let (w, h) = try Self.pixelSize(input, field: "image")
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            let reference = scratch.appendingPathComponent(input.starts(with: [0xFF, 0xD8]) ? "ref0.jpg" : "ref0.png")
            try input.write(to: reference)
            let request = try body.makeRequest(
                body: gen, modelsDirectory: modelsDirectory, reference: reference, sourceWidth: w, sourceHeight: h)
            return .success(EditJob(request: request, scratch: scratch))
        } catch {
            try? FileManager.default.removeItem(at: scratch)
            if case EngineError.invalidRequest(let reason) = error { return .failure(.error(400, message: reason)) }
            return .failure(.error(500, message: "could not stage the edit input: \(error)"))
        }
    }

    private func runEdit(_ id: String, _ ed: EditJob) async {
        defer { try? FileManager.default.removeItem(at: ed.scratch) }
        let (engine, jobs) = (engine, jobs)
        let r = ed.request
        await track(id, "edit \(r.variant.rawValue) \(r.width)×\(r.height) · \(r.steps) steps") {
            let result = try await engine.generate(r) { p in Task { await jobs.progress(id, p) } }
            return [.init(seed: result.seed, width: result.width, height: result.height, png: result.imagePNG)]
        }
    }

    /// The bytes of a request's input image: base64 `image` (via `UpscaleBody.imageData`) or a
    /// finished job's PNG. Exactly one source must be given (checked by the caller's body).
    private func sourceImage(base64: String?, job: String?, index: Int?) async -> Result<Data, HTTPResponseError> {
        if let id = job, !id.isEmpty {
            guard let found = await jobs.job(id) else { return .failure(.init(.error(404, message: "no such job"))) }
            guard found.state == .done else {
                return .failure(.init(.error(409, message: "job \(id) is \(found.state.rawValue) (\(found.stage))")))
            }
            let i = index ?? 0
            guard found.images.indices.contains(i), let png = found.images[i].png else {
                return .failure(.init(.error(400, message: "job \(id) has no image \(i)")))
            }
            return .success(png)
        }
        var text = base64 ?? ""
        if text.hasPrefix("data:"), let comma = text.firstIndex(of: ",") { text = String(text[text.index(after: comma)...]) }
        text.removeAll { $0.isWhitespace }
        guard let data = Data(base64Encoded: text), !data.isEmpty else {
            return .failure(.init(.error(400, message: "'image' must be base64 (or a data: URL)")))
        }
        return .success(data)
    }

    /// `POST /v1/moss/caption`: `CaptionBody` → the recaption JSON, with `caption` = the requested detail.
    private func caption(_ http: HTTPRequest) async -> HTTPResponse {
        let body: CaptionBody
        do {
            body = try JSONDecoder().decode(CaptionBody.self, from: http.body)
            try body.validate()
        } catch let EngineError.invalidRequest(reason) {
            return .error(400, message: reason)
        } catch {
            return .error(400, message: "body must be a JSON object with 'image' (base64) or 'job'")
        }
        let input: Data
        switch await sourceImage(base64: body.image, job: body.job, index: body.index) {
        case .failure(let e): return e.response
        case .success(let data): input = data
        }
        guard let src = CGImageSourceCreateWithData(input as CFData, nil), CGImageSourceGetCount(src) > 0,
            CGImageSourceCopyPropertiesAtIndex(src, 0, nil) != nil
        else { return .error(400, message: "'image' is not a decodable PNG/JPEG") }
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("mossforge-caption-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let (engine, events) = (engine, events)
        do {
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            let url = scratch.appendingPathComponent(input.starts(with: [0xFF, 0xD8]) ? "input.jpg" : "input.png")
            try input.write(to: url)
            let request = try body.makeRequest(image: url)
            let result = try await queue.run {
                events.emit("caption started · \(request.detail.rawValue)")
                return try await engine.caption(request, progress: nil)
            }
            events.emit("caption done")
            return .json([
                "caption": result.text(request.detail), "long": result.long, "medium": result.medium, "short": result.short,
                "watermark": result.watermarks.map { ["box_2d": $0.box, "label": $0.label] as [String: Any] },
                "borders": result.borders,
            ])
        } catch let EngineError.invalidRequest(reason) {
            return .error(400, message: reason)
        } catch EngineError.notImplemented(let what) {
            return .error(501, message: "\(what) is not available on the \(engine.backendName) backend")
        } catch {
            return .error(500, message: "\(error)")
        }
    }

    /// Run one job's work through the shared queue, recording start/finish/failure.
    private func track(_ id: String, _ label: String, _ work: @escaping @Sendable () async throws -> [GenerationJobStore.Image]) async {
        let (jobs, events) = (jobs, events)
        do {
            let images = try await queue.run {
                await jobs.start(id)
                events.emit("job \(id) started · \(label)")
                return try await work()
            }
            await jobs.finish(id, images: images)
            let secs = await jobs.job(id)?.summary()["elapsed_s"] ?? 0
            events.emit("job \(id) done · \(secs)s")
        } catch let EngineError.invalidRequest(reason) {
            await jobs.fail(id, error: reason, invalidRequest: true)
            events.emit("job \(id) rejected · \(reason)")
        } catch {
            await jobs.fail(id, error: "\(error)")
            events.emit("job \(id) failed · \(error)")
        }
    }

    /// The synchronous routes' reply for a finished job.
    private func respond(_ id: String, rawPNG: Bool) async -> HTTPResponse {
        guard let job = await jobs.job(id) else { return .error(500, message: "job \(id) vanished") }
        if job.state == .failed { return .error(job.invalidRequest ? 400 : 500, message: job.error ?? "generation failed") }
        if rawPNG, let first = job.images.first, let png = first.png {
            return .png(png, headers: [
                "X-MossForge-Seed": "\(first.seed)", "X-MossForge-Job": id, "X-MossForge-Size": "\(first.width)x\(first.height)",
            ])
        }
        let data: [[String: Any]] = job.images.map { image in
            var entry: [String: Any] = ["seed": image.seed, "width": image.width, "height": image.height]
            if let png = image.png { entry["b64_json"] = png.base64EncodedString() }
            return entry
        }
        return .json(["created": Int(Date().timeIntervalSince1970), "data": data, "job": id])
    }

    /// `GET /v1/moss/jobs/{id}` and `GET /v1/moss/jobs/{id}/image[?index=n]`.
    private func jobRoute(_ request: HTTPRequest) async -> HTTPResponse {
        let parts = request.path.dropFirst(Self.jobPrefix.count).split(separator: "/").map(String.init)
        guard let id = parts.first, parts.count <= 2, parts.count == 1 || parts[1] == "image", let job = await jobs.job(id)
        else { return .error(404, message: "no such job") }
        guard parts.count == 2 else { return .json(job.summary()) }
        switch job.state {
        case .queued, .running:
            return .error(409, message: "job \(id) is \(job.state.rawValue) (\(job.stage))")
        case .failed:
            return .error(job.invalidRequest ? 400 : 500, message: job.error ?? "generation failed")
        case .done:
            let index = request.query["index"].flatMap(Int.init) ?? 0
            guard job.images.indices.contains(index), let png = job.images[index].png else {
                return .error(404, message: "job \(id) has no image \(index)")
            }
            return .png(png, headers: ["X-MossForge-Seed": "\(job.images[index].seed)", "X-MossForge-Job": id])
        }
    }

    /// What `MossGenerationBody` accepts, with the per-variant defaults.
    func options() -> [String: Any] {
        func defaults(_ v: MossVariant) -> [String: Any] {
            let d = v.defaults
            return ["steps": d.steps, "guidance_scale": d.cfg, "shift": d.shift, "sampler": d.sampler.rawValue,
                    "uses_negative_prompt": d.cfg != 1]
        }
        let variants: [MossVariant] = [.sft, .turbo, .base]  // the body's default first
        let up = UpscaleSettings()
        return [
            "backend": engine.backendName,
            "capabilities": engine.capabilities.names,
            "generate": [
                "variants": variants.map(\.rawValue),
                "variant_defaults": Dictionary(uniqueKeysWithValues: variants.map { ($0.rawValue, defaults($0)) }),
                "size_range": [MossGenerationBody.sizeRange.lowerBound, MossGenerationBody.sizeRange.upperBound],
                "size_multiple": 16,
                "steps_range": [1, 200], "turbo_steps_range": [1, 8],
                "guidance_range": [MossGenerationBody.cfgRange.lowerBound, MossGenerationBody.cfgRange.upperBound],
                "shift_range": [MossGenerationBody.shiftRange.lowerBound, MossGenerationBody.shiftRange.upperBound],
                "max_images": MossGenerationBody.maxImages,
                "slg": ["official", "off", "{uncond_skip_layers, layers, scale, start, stop}"],
                "slg_default": ["uncond_skip_layers": [22], "layers": [Int](), "scale": 0.0, "start": 0.0, "stop": 1.0] as [String: Any],
                "lora_blocks": "\"21-41\" (only those) or \"0-20=0,21-41=0.5,*=1,other=0\"",
                "recommended_negative": MossNegativePrompt.recommended,
                "layout": "pass a USERGUIDE layout object as 'layout' (abs_centralized yxyx boxes)",
            ] as [String: Any],
            "edit": [
                "needs_lora": true,
                "max_sources": 1,
                "fitted_side_range": [MossEditBody.minFittedSide, MossEditBody.maxFittedSide],
                "note": "POST /v1/moss/edit: generation fields + 'image' (base64) or 'job'; 'loras' must hold an edit LoRA",
            ] as [String: Any],
            "upscale": [
                "methods": UpscaleMethod.allCases.map(\.rawValue),
                "esrgan_models": UpscaleModels.esrganModels(in: modelsDirectory),
                "vosr_installed": UpscaleModels.missingVOSR(in: modelsDirectory).isEmpty,
                "softening": UpscaleSoftening.allCases.map(\.rawValue),
                "color_fix": UpscaleColorFix.allCases.map(\.rawValue),
                "scale_range": [UpscaleRequest.scaleRange.lowerBound, UpscaleRequest.scaleRange.upperBound],
                "tile_overlap_range": [UpscaleSettings.overlapRange.lowerBound, UpscaleSettings.overlapRange.upperBound],
                "max_output_side": UpscaleBody.maxOutputSide,
                "defaults": [
                    "scale": 2, "method": UpscaleMethod.vosr.rawValue, "softening": up.soften.rawValue,
                    "color_fix": up.colorFix.rawValue, "seed": up.seed, "tile_overlap": up.overlap,
                ] as [String: Any],
            ] as [String: Any],
        ]
    }

    /// LoRAs under `<models>/lora`, by models-relative id, with their rank.
    func loraList() -> [[String: Any]] {
        let root = modelsDirectory.standardizedFileURL.path
        let dir = modelsDirectory.appendingPathComponent("lora")
        let urls = (FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)?.allObjects as? [URL]) ?? []
        return urls.filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }.compactMap { url in
            guard case .lora(let rank) = (try? MossInspector.inspect(url)) ?? .unknown else { return nil }
            let path = url.standardizedFileURL.path
            return ["id": path.hasPrefix(root + "/") ? String(path.dropFirst(root.count + 1)) : path,
                    "name": url.deletingPathExtension().lastPathComponent, "rank": rank]
        }
    }

    /// Every classifiable `.safetensors` under the models folder.
    func modelList() -> [[String: Any]] {
        let root = modelsDirectory.standardizedFileURL.path
        guard let walker = FileManager.default.enumerator(at: modelsDirectory, includingPropertiesForKeys: nil) else { return [] }
        var out: [[String: Any]] = []
        for case let url as URL in walker where url.pathExtension == "safetensors" {
            guard let kind = try? MossInspector.inspect(url) else { continue }
            let label: String
            switch kind {
            case .dit(_, let precision): label = "dit/\(precision.rawValue)"
            case .textEncoder: label = "text_encoder"
            case .vae: label = "vae"
            case .lora: label = "lora"
            case .dmdSampler: label = "dmd_sampler"
            case .unknown: continue
            }
            let path = url.standardizedFileURL.path
            out.append(["id": path.hasPrefix(root + "/") ? String(path.dropFirst(root.count + 1)) : path,
                        "object": "model", "owned_by": "local", "kind": label])
        }
        return out.sorted { ($0["id"] as? String ?? "") < ($1["id"] as? String ?? "") }
    }
}

/// A thread-safe, settable log callback shared between a router and its transport.
public final class EventSink: @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable (String) -> Void)?

    public init() {}

    public func setHandler(_ handler: (@Sendable (String) -> Void)?) { lock.withLock { self.handler = handler } }
    public func emit(_ line: String) { lock.withLock { handler }?(line) }
}

/// A ready error reply, carried through `Result` by the routes' input helpers.
struct HTTPResponseError: Error {
    let response: HTTPResponse
    init(_ response: HTTPResponse) { self.response = response }
}
