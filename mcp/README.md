# mossforge-mcp

An MCP server that lets MCP clients (LM Studio, Claude Desktop, …) generate images with MossForge
(MOSS-Image on Apple silicon). It is a thin bridge: the **resident server** owns the models and the GPU,
and this forwards tool calls to it over loopback HTTP.

1. Start the resident server:
   ```sh
   cd MossForgeMLX && swift run -c release moss-mlx serve --models ../models   # :7860, loopback
   ```
   (`--keep-encoder` keeps the 16 GB Qwen3.5 encoder resident between requests; `--lan` exposes it on the
   network; `--idle-unload N` frees models after N idle minutes, default 10.)
2. Point your MCP client at the bridge:
   ```json
   { "mcpServers": { "mossforge": {
       "command": "uv",
       "args": ["run", "--directory", "/path/to/MOSS-Image/mcp", "mossforge-mcp"],
       "env": { "MOSSFORGE_URL": "http://127.0.0.1:7860" } } } }
   ```

## Tools

| Tool | What it does |
|---|---|
| `generate_image` | Generate (prompt **or** JSON `layout`). Returns the image inline plus its saved path, seed and job id. |
| `get_generation` | Collect a generation or upscale that outlasted the wait window. |
| `caption_image` | Official MOSS recaption (long/medium/short, verbatim text, watermarks) of an earlier job or a local image, on-device Qwen3.5-9B. |
| `edit_image` | Edit an earlier job's image (`job_id`) or a local PNG/JPEG (`image_path`) with an instruction and an **edit LoRA** (`loras`, required — experimental: the source runs as a clean reference token block; `edit_layout` overrides the layout read from the LoRA's metadata — `auto`, `mossForge`, `ai-toolkit`, `ai-toolkit (kv_cache off)`, `Moss-Image-Edit`). Returns the edited image inline plus its saved path (`mossforge-edit-…`). |
| `upscale_image` | VOSR2 (generative) or ESRGAN upscale of an earlier job (`job_id`) or a local PNG/JPEG (`image_path`); returns the saved path. |
| `enhance_prompt` | Expand a short idea into a detailed prompt, or (`layout: true`) a JSON layout prompt. |
| `list_models` | Checkpoints, encoder, VAE, LoRAs, DMD sampler (`id`, `kind`). |
| `list_loras` | LoRAs with `rank`. |
| `generation_options` | Variants with defaults, ranges, SLG and LoRA block syntax. |
| `mossforge_status` | Queue depth, the running job's stage, recent jobs. |

`generate_image` parameters (all optional, but exactly one of `prompt` / `layout`):

| Parameter | Values | Default |
|---|---|---|
| `variant` | `turbo` (8 steps, no CFG, ~1 min at 1024²) · `sft` (50 steps, CFG 3, ~5 min) · `base` | `turbo` |
| `layout` | USERGUIDE layout object — `abs_centralized` `yxyx` boxes, origin at the canvas centre | — |
| `aspect_ratio` + `long_side` | `"16:9"`, … · 256–4096 | `1:1` · `MOSSFORGE_DEFAULT_SIZE` |
| `width`, `height` | 256–4096, multiples of 16 | — |
| `steps` | Turbo 1–8; SFT/Base 1–200 | 8 / 50 |
| `guidance_scale`, `negative_prompt`, `shift` | SFT/Base only | 3 · recommended tags · 5 |
| `slg` | `official` (skip block 22 in the negative branch) · `off` | `official` |
| `seed` | ≥ 0 | random |
| `loras` | `[{"path": "lora/x.safetensors", "strength": 0.8, "blocks": "21-41"}]` | — |

`blocks`: `"21-41"` applies the LoRA only to those DiT blocks; `"0-20=0,21-41=0.5,other=0"` weights
ranges (unlisted blocks stay 1). Kijai's SFT banding fix = his SFT LoRA at `-0.5`, blocks `21-41`.

| Env var | Default |
|---|---|
| `MOSSFORGE_URL` | `http://127.0.0.1:7860` |
| `MOSSFORGE_OUTPUT_DIR` | `~/Pictures/MossForge/mcp` |
| `MOSSFORGE_DEFAULT_SIZE` | `1024` (the long side) |
| `MOSSFORGE_WAIT_SECONDS` | `45` (per tool call) |
| `MOSSFORGE_IMAGE_DIRS` | `~/Pictures` (folders `upscale_image` / `caption_image` / `edit_image` may read; `os.pathsep`-separated; the output dir always counts) |

Develop: `uv run pytest` (the end-to-end test needs `swift build` in the repo root — it runs the stub-backed
`mossforge serve`), `uv run ruff check`, `uv run mypy`.
