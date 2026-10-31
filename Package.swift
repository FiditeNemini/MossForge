// swift-tools-version: 6.0
import PackageDescription

// Root package: Core + Engine + CLI. Deliberately dependency-free so it builds instantly and
// offline with the StubBackend. The MLX backend and the SwiftUI app live in the sibling
// `MossForgeMLX` package (the composition root that links Metal), so `swift build`/`swift test`
// stay fast and MLX-free. See docs/dev/architecture.md.
let package = Package(
    name: "MossForge",
    platforms: [
        // macOS first; SwiftUI views are written to port to iOS later.
        .macOS(.v14)
    ],
    products: [
        .library(name: "MossForgeCore", targets: ["MossForgeCore"]),
        .library(name: "MossForgeEngine", targets: ["MossForgeEngine"]),
        .executable(name: "mossforge", targets: ["MossForgeCLI"]),
    ],
    targets: [
        // Pure, Sendable value types shared by every layer. No deps.
        .target(name: "MossForgeCore"),

        // The engine seam: capability protocols + EngineBackend + MossEngine facade + StubBackend,
        // plus the pure-logic model tooling (safetensors, quant metadata, MossInspector, RoPE index)
        // that the MLX backend uses and that is unit-tested here.
        .target(name: "MossForgeEngine", dependencies: ["MossForgeCore"]),

        // `mossforge` — a thin async command router over the engine (StubBackend today).
        .executableTarget(name: "MossForgeCLI", dependencies: ["MossForgeCore", "MossForgeEngine"]),

        .testTarget(name: "MossForgeCoreTests", dependencies: ["MossForgeCore"]),
        .testTarget(name: "MossForgeEngineTests", dependencies: ["MossForgeEngine", "MossForgeCore"]),
    ]
)
