# Captioning

MossForge captions images on device with **Qwen3.5-9B**, the same file the text encoder uses
(`models/text_encoders/qwen3.5_9b_int8_convrot.safetensors`). Its vision tower (27 blocks) encodes the image,
and the language model answers MOSS-Image's **official recaption prompt** (USERGUIDE, "Image Captioning")
with a JSON object:

```json
{"caption": {"long": "…", "medium": "…", "short": "…"},
 "watermark": [{"box_2d": [y_min, x_min, y_max, x_max], "label": "…"}], "borders": ["none"]}
```

`long` describes composition, subjects, colours, lighting and style. It keeps visible text **verbatim**,
including its layout, which is what MOSS-Image training captions need. Watermark boxes are normalised 0–1000.

## CLI

```sh
cd MossForgeMLX
swift run -c release moss-mlx caption photo.png                     # prints the long caption
swift run -c release moss-mlx caption photo.png --detail short --json
swift run -c release moss-mlx caption photo.png --reference "an old caption to correct"
```

`--max-pixels` sets the vision budget (default 1024² ≈ 1024 image tokens; min 256², max 2048²). On an
M2 Ultra a 1024² image takes about 17 s, including loading the model. The model stays loaded for the next
image until idle unload releases it (under Settings → Memory → *Save memory*, a generation releases it too). Sampling uses Qwen's non-thinking settings
(temperature 0.7, top-p 0.8, top-k 20, `--seed`). If an answer doesn't parse, it retries once.

## Server and MCP

`POST /v1/moss/caption` with `image` (base64 or `data:` URL) **or** `job` (plus an optional `index`),
`detail` (long | medium | short), `reference`, `max_pixels` and `seed` returns
`{"caption", "long", "medium", "short", "watermark", "borders"}`, where `caption` is the requested detail.
The request runs through the same GPU queue as generations.

The MCP tool `caption_image` takes either `job_id` or an `image_path` inside `MOSSFORGE_IMAGE_DIRS`, and
returns the caption followed by the full JSON.

## App

Gallery → select an image → **Caption** (pick Long / Medium / Short to display). Several images, a collection or
all of them: **Caption…** in the multi-select pane, or the **Dataset** menu above the grid — one after another,
skipping already-captioned images unless *Re-caption* is on. **Dataset → Caption a folder of images…** writes
kohya `name.txt` files (optional trigger word) beside every image in a folder. The full answer is saved in the
image's JSON sidecar under `caption`; a non-layout prompt from the sidecar is passed as the reference.
