import Foundation

/// Single-flight FIFO for GPU work. One model is resident and one GPU does the denoising, so
/// concurrent server requests take turns instead of contending for memory and compute.
///
/// Optionally it frees that resident model when nothing has run for a while
/// (`setIdleRelease`): the release runs once per idle stretch, through the queue itself, so
/// a request that arrives mid-release waits for it rather than racing it.
public actor InferenceQueue {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    /// Jobs running or waiting.
    public private(set) var pending = 0

    private var idleTimeout: Duration?
    private var onIdle: (@Sendable () async -> Void)?
    /// Bumped by every job, so a timer armed before it knows it is stale.
    private var epoch = 0
    /// True when a job has run since the last release (something may be resident).
    private var hasResident = false
    private var idleTimer: Task<Void, Never>?

    public init() {}

    /// Release resident models after `timeout` with no jobs (`nil` turns it off). If a job
    /// already ran, the clock starts now.
    public func setIdleRelease(after timeout: Duration?, release: @escaping @Sendable () async -> Void) {
        idleTimeout = timeout
        onIdle = release
        armIdleTimer()
    }

    public func run<T: Sendable>(_ job: @Sendable () async throws -> T) async throws -> T {
        try await runSlot(markResident: true, job)
    }

    private func runSlot<T: Sendable>(markResident: Bool, _ job: @Sendable () async throws -> T) async throws -> T {
        pending += 1
        if markResident {
            epoch += 1
            idleTimer?.cancel()
            idleTimer = nil
        }
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
        defer {
            pending -= 1
            if markResident { hasResident = true }
            // Hand the slot straight to the next waiter (FIFO), or free it.
            if waiters.isEmpty {
                busy = false
                armIdleTimer()
            } else {
                waiters.removeFirst().resume()
            }
        }
        return try await job()
    }

    private func armIdleTimer() {
        idleTimer?.cancel()
        idleTimer = nil
        guard let timeout = idleTimeout, onIdle != nil, hasResident, pending == 0 else { return }
        let armedAt = epoch
        idleTimer = Task { [weak self] in
            try? await Task.sleep(for: timeout)
            guard !Task.isCancelled else { return }
            await self?.fireIdle(armedAt: armedAt)
        }
    }

    private func fireIdle(armedAt: Int) async {
        guard armedAt == epoch, hasResident, pending == 0, let release = onIdle else { return }
        hasResident = false
        idleTimer = nil
        _ = try? await runSlot(markResident: false) { await release() }
    }
}
