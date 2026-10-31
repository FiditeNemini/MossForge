// swift-tools-version: 6.0
import PackageDescription

// Composition-root package. Depends on the root package by path —
// named `MOSS-Image` explicitly, since a path package's identity is otherwise its folder name and
// a git worktree's folder isn't `MOSS-Image` — and pulls in mlx-swift (Metal)
// and swift-transformers (the Qwen3.5 tokenizer). Keeping it separate keeps the root package's
// `swift build`/`swift test` fast and MLX-free.
let package = Package(
    name: "MossForgeMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MossForgeMLX", targets: ["MossForgeMLX"]),
        .executable(name: "moss-mlx", targets: ["MossForgeMLXCLI"]),
        .executable(name: "MossForgeApp", targets: ["MossForgeApp"]),
    ],
    dependencies: [
        .package(name: "MOSS-Image", path: ".."),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),  // 0.33 needs Sendable fixes
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // Live layout tags (`.tweakable`): a no-op stand-in in Packages/Tweakbench.
        .package(name: "Tweakbench", path: "Packages/Tweakbench"),
    ],
    targets: [
        // The real backend lands here phase by phase: MossDiT (SiTU-GLU, shared modulation,
        // 4D RoPE), Qwen3.5 hybrid encoder, FLUX.2 VAE, flow-Euler + data-space DMD samplers.
        .target(
            name: "MossForgeMLX",
            dependencies: [
                .product(name: "MossForgeCore", package: "MOSS-Image"),
                .product(name: "MossForgeEngine", package: "MOSS-Image"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // `moss-mlx` — real generation + the numerics harness (`*-parity`). No xctest target:
        // the SwiftPM test runner can't load mlx-swift's metallib, so GPU code is verified by
        // running this executable.
        .executableTarget(name: "MossForgeMLXCLI", dependencies: ["MossForgeMLX"]),
        // The macOS SwiftUI app — the composition root. Injects MossBackend when MOSS weights are
        // present, else falls back to the canned StubBackend.
        .executableTarget(
            name: "MossForgeApp",
            dependencies: [
                "MossForgeMLX",
                .product(name: "MossForgeCore", package: "MOSS-Image"),
                .product(name: "MossForgeEngine", package: "MOSS-Image"),
                .product(name: "Tweakbench", package: "Tweakbench"),
            ]
        ),
    ]
)
