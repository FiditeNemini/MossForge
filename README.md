# MossForge

A native macOS studio for **[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)**, OpenMOSS's
Apache-2.0 5B text-to-image model with long bilingual text rendering and JSON layout control. It runs
entirely on your Mac through [MLX](https://github.com/ml-explore/mlx-swift), with no cloud inference.

- **Generate** from a prompt or a box-by-box **layout** (exact visible text where you place it), with SFT,
  Base, 4-step and 8-step **Turbo**, LoRAs (per-block strength, LoKr, LoHa) and edit LoRAs with reference images.
- **Enhance** a short idea into a detailed prompt or a layout, **caption** images and **upscale** them
  (VOSR2, ESRGAN, SPAN), all with on-device models. The prompts behind these are editable in Settings → Prompts.
- **Train** LoRAs, LoKr, concept sliders and (experimentally) edit LoRAs, and build character datasets in
  Character Studio.
- **Model tools:** convert, quantize (int8·ConvRot, fp8, nvfp4), fuse LoRAs, merge and extract LoRAs.
- A resident **HTTP server** (`/v1/moss/*`), an **MCP bridge** (`mcp/`) and the `moss-mlx` CLI.
- Built-in **Help** that answers from the bundled docs, and opt-in, local-only diagnostics.

Requirements: an Apple-silicon Mac on macOS 14 or later, Xcode 16 or later (Swift 6), and roughly 16 GB of
disk for the essential models. 64 GB of memory or more is comfortable; less works with Settings → Memory →
Save memory.

## Build and run

```sh
swift build && swift test                     # Core + Engine + CLI (fast, MLX-free)
Scripts/run-app.sh                            # build the SwiftUI app and launch it
cd MossForgeMLX && swift run -c release moss-mlx smoke
```

`Scripts/build-app.sh --open` builds the Xcode app (needs `brew install xcodegen`). Signing and notarised
releases are described in [docs/dev/release.md](docs/dev/release.md). Put your own team and bundle id in
`Config/Signing.local.xcconfig` (see the `.example` file); without it, Debug builds are ad-hoc signed.

## Models

Weights are not in this repository. In the app, **Models → Get Models** downloads a checksummed bundle
from Hugging Face (default `FiditeNemini/MossForge`, changeable in Settings). Or put the files under
`models/` yourself in the ComfyUI layout. See [docs/user/index.md](docs/user/index.md).

## Documentation

- Using it: [docs/user/index.md](docs/user/index.md), [settings guide](docs/user/settings-guide.md),
  [training](docs/user/training.md), [editing](docs/user/editing.md), [captioning](docs/user/captioning.md),
  [upscaling](docs/user/upscaling.md), [help and feedback](docs/user/help-and-feedback.md)
- How it works: [docs/dev/architecture.md](docs/dev/architecture.md),
  [docs/dev/technical-reference.md](docs/dev/technical-reference.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Licences and provenance: [docs/dev/licensing.md](docs/dev/licensing.md), [NOTES.md](NOTES.md)

The upstream model code is useful reading alongside the Swift port. It is not a build dependency:

```sh
git clone https://github.com/OpenMOSS/MOSS-Image Reference/MOSS-Image
```

## Licence

MossForge's source is [Apache-2.0](LICENSE). Model weights are separate and keep their own licences (see
NOTES.md). They are never committed here.
