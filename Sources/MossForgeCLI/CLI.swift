import Foundation
import MossForgeCore
import MossForgeEngine

/// `mossforge` — a thin async router over `MossEngine` (StubBackend; the MLX-linked `moss-mlx`
/// in MossForgeMLX drives real generation).
@main
struct MossForgeCLI {
    static let usage = """
        mossforge \(MossForge.version)
          capabilities                          active backend + supported ops
          inspect <file.safetensors>…           classify checkpoints from their headers
          schedule [--steps N] [--shift S] [--paper]  official snapped σ grid (or the paper's)
          generate -p <prompt> [-o out.png] [--variant base|sft|turbo] [--size WxH] [--seed N]
          enhance -p <prompt> [--layout]        stub prompt / layout enhancement
          help-search <question>                the help docs sections Help → Ask would use
          help-corpus [--check]                 regenerate the app's embedded docs (run from the repo root)
          models list|download [--repo owner/name] [--groups essentials,sft,…] [--models dir]
                                                the model bundle (first-run setup; default \(ModelBundleSource.defaultRepo))
          serve [--port 7860] [--models dir] [--lan]  resident HTTP server on this backend (stub here;
                                                the MLX one is `moss-mlx serve`)
        """

    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else { print(usage); return }
        let command = args.removeFirst()
        let engine = MossEngine()
        do {
            switch command {
            case "capabilities":
                print("backend: \(engine.backendName)")
                print("ops: \(engine.capabilities.names.joined(separator: ", "))")
            case "inspect":
                for path in args { print("\(path): \(describe(try await engine.inspectModel(URL(fileURLWithPath: path))))") }
            case "schedule":
                let steps = Int(value(of: "--steps", in: args) ?? "") ?? 50
                let shift = Double(value(of: "--shift", in: args) ?? "") ?? 5
                let sigmas = args.contains("--paper")
                    ? MossSchedule.sigmas(steps: steps, shift: shift) : MossSchedule.pipelineSigmas(steps: steps, shift: shift)
                for (k, s) in sigmas.enumerated() {
                    print(String(format: "%3d  σ=%.6f  t=%.2f", k, s, s * 1000))
                }
            case "generate":
                var request = GenerationRequest(
                    prompt: value(of: "-p", in: args) ?? "",
                    variant: MossVariant(rawValue: value(of: "--variant", in: args) ?? "sft") ?? .sft,
                    seed: UInt64(value(of: "--seed", in: args) ?? ""))
                if let size = value(of: "--size", in: args) {
                    let wh = size.split(separator: "x").compactMap { Int($0) }
                    if wh.count == 2 { (request.width, request.height) = (wh[0], wh[1]) }
                }
                let result = try await engine.generate(request) { p in
                    FileHandle.standardError.write(Data("\r\(p.step)/\(p.totalSteps)".utf8))
                }
                let out = URL(fileURLWithPath: value(of: "-o", in: args) ?? "mossforge.png")
                try result.imagePNG.write(to: out)
                print("\nwrote \(out.path) (\(result.width)×\(result.height), seed \(result.seed), backend \(engine.backendName))")
            case "models":
                try await models(args)
            case "serve":
                let port = UInt16(value(of: "--port", in: args) ?? "7860") ?? 7860
                let models = URL(fileURLWithPath: value(of: "--models", in: args) ?? "models")
                let server = MossHTTPServer(router: MossServerRouter(engine: engine, modelsDirectory: models))
                setvbuf(stdout, nil, _IOLBF, 0)
                server.onLog = { print($0) }
                try await server.start(port: port, policy: NetworkPolicy(egressAllowed: args.contains("--lan")))
                while true { try await Task.sleep(for: .seconds(3600)) }
            case "help-search":
                for c in HelpRetriever(chunks: HelpCorpus.chunks).search(args.joined(separator: " ")) {
                    print("── \(c.title)\n\(c.text)\n")
                }
            case "help-corpus":
                // Rewrite the generated Swift file from the manifest's docs (run from the repo root) — or,
                // with --check, fail if it's stale.
                let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                let out = root.appendingPathComponent("Sources/MossForgeCore/HelpCorpusDocuments.swift")
                do {
                    print(try HelpCorpus.regenerate(root: root, output: out, check: args.contains("--check")))
                } catch HelpCorpusError.stale {
                    FileHandle.standardError.write(Data("help corpus is stale — run `swift run mossforge help-corpus`\n".utf8))
                    exit(1)
                }
            case "enhance":
                print(try await engine.enhancePrompt(PromptEnhanceRequest(prompt: value(of: "-p", in: args) ?? "", layout: args.contains("--layout"))))
            default:
                print(usage)
                exit(2)
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    static func value(of flag: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func describe(_ kind: MossModelKind) -> String {
        switch kind {
        case .dit(let c, let p):
            "MOSS-Image DiT \(p.rawValue) — dim \(c.dim), \(c.layers) layers, \(c.heads)×\(c.headDim) heads, ffn \(c.ffnDim), latent \(c.latentChannels), qkv \(c.qkvLayout.rawValue)"
        case .textEncoder(let layers, let full):
            "Qwen3.5 text encoder — \(layers) layers, full attention at \(full)"
        case .vae(let ch): "FLUX.2 VAE — \(ch) latent channels"
        case .lora(let rank): "DiT LoRA — rank \(rank)"
        case .dmdSampler(let steps): "Turbo data-space sampler — \(steps) step codes"
        case .unknown: "unknown"
        }
    }

    /// `models list` — the bundle's groups and what's already here; `models download` — fetch the
    /// chosen groups (default: the bundle's defaults) into `--models`. Running it is the consent, so
    /// egress is allowed here; the app asks via Settings → Privacy.
    static func models(_ args: [String]) async throws {
        let source = ModelBundleSource(repo: value(of: "--repo", in: args) ?? ModelBundleSource.defaultRepo)
        guard source.isValid else { throw EngineError.invalidRequest("--repo must be owner/name (got '\(source.repo)')") }
        let folder = URL(fileURLWithPath: value(of: "--models", in: args) ?? "models")
        let downloader = ModelBundleDownloader(source: source, models: folder, policy: NetworkPolicy(egressAllowed: true))
        let manifest = try await downloader.fetchManifest()
        let have = downloader.existingSizes(manifest)
        func gb(_ n: Int64) -> String { String(format: "%.1f GB", Double(n) / 1e9) }
        switch args.first {
        case "list", nil:
            print("\(source.repo) → \(folder.path)")
            for g in manifest.groups {
                let missing = manifest.pending(groups: [g.id], existingSizes: have)
                let state = missing.isEmpty ? "installed" : "\(missing.count) file(s), \(gb(missing.reduce(0) { $0 + $1.size })) to fetch"
                print("  \(g.default ? "*" : " ") \(g.id.padding(toLength: 12, withPad: " ", startingAt: 0)) \(gb(g.size).padding(toLength: 8, withPad: " ", startingAt: 0)) \(g.title) — \(state)")
            }
            print("  (* = downloaded by default)")
        case "download":
            let groups = value(of: "--groups", in: args).map { Set($0.split(separator: ",").map(String.init)) } ?? manifest.defaultGroups
            let unknown = groups.subtracting(manifest.groups.map(\.id))
            guard unknown.isEmpty else { throw EngineError.invalidRequest("unknown group(s) \(unknown.sorted()) — see `mossforge models list`") }
            let written = try await downloader.download(groups: groups, manifest: manifest) { p in
                FileHandle.standardError.write(Data(String(
                    format: "\r[%d/%d] %5.1f%%  %5.1f MB/s  %@", p.fileIndex + 1, p.fileCount, p.fraction * 100, p.bytesPerSecond / 1e6, p.file).utf8))
            }
            print("\n\(written.isEmpty ? "already installed" : "downloaded \(written.count) file(s)") into \(folder.path)")
        default:
            throw EngineError.invalidRequest("models list | download")
        }
    }
}
