# Upscaling

Two engines, behind the same `upscale` capability (CLI, server, MCP):

| | VOSR2 (`vosr`, default) | ESRGAN family (`esrgan`) |
|---|---|---|
| What | one-step generative SR (1.4B DiT + DINOv2-L) | RRDBNet (Real-ESRGAN, UltraSharp, Remacri) or SPAN (ClearReality) |
| Look | rebuilds texture: skin, fabric, foliage, lettering | sharp and fast; invents little |
| Speed (M2 Ultra) | 512→1024 ≈ 6 s (9 tiles, weights cached) | well under a second |

## Weights

```
models/vosr/VOSR2/checkpoints/ema_model.safetensors      # hf CSWRY/VOSR (Apache-2.0)
models/vosr/dinov2-large/model.safetensors               # hf facebook/dinov2-large (Apache-2.0)
models/vosr/qwen_image_vae.safetensors                   # Qwen-Image VAE (Apache-2.0) — VOSR2's latent space
models/esrgan/*.safetensors                              # e.g. RealESRGAN_x2plus (BSD-3-Clause)
```

VOSR2 works in the Qwen-Image VAE's 16-channel latent space, so it needs that VAE. MOSS's FLUX.2 VAE won't do.
ESRGAN files must be safetensors with the upstream key names. Convert `.pth` files to
safetensors first. With no `--esrgan-model`, RealESRGAN ×2plus is used if installed, otherwise the first
file by name.

## CLI

```sh
cd MossForgeMLX
swift run -c release moss-mlx upscale photo.png --scale 2                          # VOSR2 → photo_x2_vosr.png
swift run -c release moss-mlx upscale photo.png --scale 4 --method esrgan --esrgan-model 4x-UltraSharp.safetensors
```

VOSR knobs (defaults measured in Forge): `--seed 42` (fixed seed = repeatable), `--tile 512`, `--overlap 256`
(raise it if seams show), `--soften medium` (`off|light|medium|strong`; removes sampler grain that VOSR would
otherwise read as texture, so go stronger for smooth skin), `--color wavelet` (`adain|none`), `--batch 4`
(tiles per DiT call). The output is rounded down to a multiple of 8 px.

## Server and MCP

`POST /v1/moss/upscale` returns the PNG (`X-MossForge-Size`); `POST /v1/moss/upscale/jobs` returns
`202 {"id"}` and then follows the usual job routes. Body: `image` (base64 or `data:` URL) **or** `job` (an
earlier job's id, plus an optional `index`), along with `scale` (1–8), `method`, `esrgan_model`, `softening`,
`color_fix`, `seed` and `tile_overlap`. Outputs are capped at 8192 px per side. The server never reads file
paths. `GET /v1/moss/options` → `upscale` lists the installed ESRGAN files and whether VOSR2 is installed.

The MCP tool `upscale_image` takes either `job_id` (for example, right after `generate_image`) or an
`image_path` inside `MOSSFORGE_IMAGE_DIRS` (default `~/Pictures`) or the bridge's output folder. It saves the
PNG and returns its path, not the image itself.

Memory: VOSR2 is about 7 GB in f32. It unloads a resident DiT before it runs, and generating unloads VOSR2.

## App

Gallery → select an image → **Upscale…**: method, scale (×2/×3/×4) and the VOSR knobs (softening, colour match,
seed, overlap) or the ESRGAN model. The result is written to the output folder as `<name>_x<scale>[_vosr].png`
with a JSON sidecar copied from the source plus an `upscale` record (`from`, `method`, `scale`, …).
