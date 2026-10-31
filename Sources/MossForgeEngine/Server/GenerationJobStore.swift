import Foundation
import MossForgeCore

/// Every generation the resident server runs — synchronous requests and submitted jobs alike —
/// with its live stage, so clients can see where a request is (queued → loading/encoding →
/// denoise n/N → decoding → done) instead of waiting blind.
public actor GenerationJobStore {
    public enum State: String, Sendable { case queued, running, done, failed }

    public struct Image: Sendable, Equatable {
        public var seed: UInt64
        public var width: Int
        public var height: Int
        /// `nil` when the backend renders no pixels (the stub).
        public var png: Data?

        public init(seed: UInt64, width: Int, height: Int, png: Data?) {
            self.seed = seed
            self.width = width
            self.height = height
            self.png = png
        }
    }

    /// What the job does: a generation, an upscale or an edit (they share the queue and the job routes).
    public enum Kind: String, Sendable { case generate, upscale, edit }

    public struct Job: Sendable, Equatable, Identifiable {
        public let id: String
        public var kind: Kind = .generate
        public var state: State
        /// Human-readable stage: "queued", the backend's progress message, or "decoding".
        public var stage: String
        public var step: Int
        public var totalSteps: Int
        public var width: Int
        public var height: Int
        public let created: Date
        public var started: Date?
        public var finished: Date?
        public var images: [Image] = []
        public var error: String?
        /// The prompt actually generated, when the server rewrote it (`enhance`).
        public var prompt: String?
        /// The failure was the request's fault (answer 400, not 500).
        public var invalidRequest = false

        /// JSON for `/v1/moss/jobs/{id}` and `/v1/status` (no image bytes).
        public func summary(now: Date = Date()) -> [String: Any] {
            var out: [String: Any] = [
                "id": id, "kind": kind.rawValue, "state": state.rawValue, "stage": stage, "step": step,
                "total_steps": totalSteps, "width": width, "height": height,
                "elapsed_s": ((finished ?? now).timeIntervalSince(started ?? created) * 10).rounded() / 10,
                "images": images.count,
            ]
            if let started { out["waited_s"] = (started.timeIntervalSince(created) * 10).rounded() / 10 }
            if let seed = images.first?.seed { out["seed"] = seed }
            if let error { out["error"] = error }
            if let prompt { out["prompt"] = prompt }
            return out
        }
    }

    private var jobs: [String: Job] = [:]
    private var order: [String] = []  // oldest first
    private let keep: Int

    public init(keep: Int = 32) {
        self.keep = max(1, keep)
    }

    public func create(width: Int, height: Int, steps: Int, kind: Kind = .generate) -> Job {
        let id = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12)).lowercased()
        let job = Job(
            id: id, kind: kind, state: .queued, stage: "queued", step: 0, totalSteps: steps,
            width: width, height: height, created: Date())
        jobs[id] = job
        order.append(id)
        prune()
        return job
    }

    public func start(_ id: String) {
        guard var job = jobs[id], job.state == .queued else { return }
        job.state = .running
        job.stage = "starting"
        job.started = Date()
        jobs[id] = job
    }

    /// Apply a backend progress update. Updates arrive via unordered tasks, so a stale one
    /// (lower step) or one after completion is ignored.
    public func progress(_ id: String, _ p: GenerationProgress) {
        guard var job = jobs[id], job.state == .running, p.step >= job.step else { return }
        job.step = p.step
        if p.totalSteps > 0 { job.totalSteps = p.totalSteps }
        // The last denoise step is followed by VAE decode + PNG save, which report nothing.
        job.stage = p.totalSteps > 0 && p.step >= p.totalSteps ? "decoding" : p.message
        jobs[id] = job
    }

    /// Record the rewritten prompt (prompt enhance) the job is generating from.
    public func setPrompt(_ id: String, _ prompt: String) {
        jobs[id]?.prompt = prompt
    }

    public func finish(_ id: String, images: [Image]) {
        guard var job = jobs[id] else { return }
        job.state = .done
        job.stage = "done"
        job.images = images
        job.finished = Date()
        if job.started == nil { job.started = job.finished }
        jobs[id] = job
        prune()
    }

    public func fail(_ id: String, error: String, invalidRequest: Bool = false) {
        guard var job = jobs[id] else { return }
        job.state = .failed
        job.stage = "failed"
        job.error = error
        job.invalidRequest = invalidRequest
        job.finished = Date()
        jobs[id] = job
        prune()
    }

    public func job(_ id: String) -> Job? { jobs[id] }

    /// All tracked jobs, newest first.
    public func recent() -> [Job] { order.reversed().compactMap { jobs[$0] } }

    /// Drop the oldest *finished* jobs beyond `keep` (queued/running ones are never dropped).
    private func prune() {
        var excess = order.count - keep
        guard excess > 0 else { return }
        order.removeAll { id in
            guard excess > 0, let j = jobs[id], j.state == .done || j.state == .failed else { return false }
            excess -= 1
            jobs[id] = nil
            return true
        }
    }
}
