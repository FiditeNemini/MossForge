# Contributing to MossForge

Thanks for helping. A few rules keep the codebase easy to work in.

## Layout

```text
Package.swift          Root SwiftPM package: Core + Engine + CLI (MLX-free, builds and tests fast)
Sources/
  MossForgeCore/       Sendable value types: requests, schedules, dimensions, layout prompts, errors, …
  MossForgeEngine/     Capability protocols, EngineBackend, MossEngine, StubBackend; safetensors,
                       quantized decode, model inspection, the HTTP server
  MossForgeCLI/        the `mossforge` CLI
Tests/                 Swift Testing suites (Core + Engine)
MossForgeMLX/          MLX package: the real backend, the `moss-mlx` CLI and the SwiftUI app
mcp/                   MCP bridge (Python, uv) to the resident server
Scripts/               validate, run/build the app, release, torch parity references (*_ref.py)
docs/{user,dev}/       how to use it / how it works
```

## Rules

1. **Keep the engine seam clean.** Every capability goes through `EngineBackend` / `MossEngine`
   (`MossForgeEngine`). The app, CLI and server depend only on that surface; swapping `StubBackend` for
   the MLX backend is one line at construction.
2. **Test first.** Add the narrowest failing Swift Testing case, make it pass, rerun `swift test`.
3. **Validate before you open a PR:** `Scripts/validate.sh` (build, test, advisory format lint) must pass.
4. **Never commit weights.** `models/` is gitignored; weights are multi-GB and user-downloaded.
5. **Ground model code in the real model.** Dimensions, key layouts and sampler math come from the
   safetensors headers and the upstream MOSS-Image code, not guesses
   ([docs/dev/technical-reference.md](docs/dev/technical-reference.md)). GPU numerics are checked with a
   `moss-mlx *-parity` command against a standalone torch reference in `Scripts/*_ref.py`.
6. **Respect licences.** GPL code (e.g. ComfyUI) may be read for semantics but never copied. Record any
   adapted third-party code and its licence in [NOTES.md](NOTES.md).
7. **Keep docs in sync.** Update `docs/user` / `docs/dev` in the same change as the behaviour, and
   regenerate the in-app Help corpus with `swift run mossforge help-corpus` when a user doc changes.
8. **Look at UI changes.** Rebuild and run the app (`Scripts/run-app.sh`) before saying a change works.
   Keep SwiftUI views portable to iOS where practical, and tag new screens and key controls with
   `.tweakable("page.region.element")`.

GPU code has no XCTest target (the SwiftPM test runner can't load mlx-swift's metallib), so it is verified
through `moss-mlx` commands rather than `swift test`.

## Commits

Conventional Commits (`feat(app):`, `fix(train):`, `docs:`, …). Explain what changed and why in the body,
and say what you tested (unit tests, a GPU run, the built app) and what you didn't.
