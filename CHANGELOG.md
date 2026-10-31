# Changelog

Signed, notarised releases of MossForge. Per-change detail is in the commit bodies (`git log`).

## 0.2.13 — 2026-11-01

- **Settings → Prompts:** Settings now has General and Prompts tabs. Prompts shows the instructions MossForge gives
  the on-device Qwen3.5 model — the prompt enhancer's rules and its caption and layout answer formats, the
  captioner and the slider designer — and lets you edit them, with an *edited* mark, Reset per prompt and Reset all.
  Edits save as you type and apply in the app (the server and CLIs keep the built-ins); a prompt you never touched
  picks up later improvements.
- **Try it:** each prompt has an editable Test input and a Try it button that runs the prompt, as edited, and shows
  the answer, how long it took and a Copy button — tune a prompt without leaving Settings. The captioner's test
  uses the newest gallery image (or one you choose) and an optional existing caption.

## 0.2.12 — 2026-10-31

- **Turbo is now akhaliq 4-step:** after a side-by-side comparison on stock SFT and finetunes, the SFT Turbo is SFT +
  akhaliq's 4-step LoRA (4 steps, CFG 1) everywhere — the app's default, `moss-mlx generate --variant turbo`, the
  server and MCP `"turbo"`. It spelled text correctly, kept blacks clean and upscaled best. lodestones' SFT
  data-space Turbo is retired (it washed images out and added stray text); lodestones `lora_latest` at 0.7 remains
  as the **8 steps — lodestones** profile. Base keeps lodestones' Base 8-step Turbo.
- **Get Models is much faster and resumable:** big files download in 32 MB pieces over 8 connections at once
  (14 → 54 MB/s on our test line), the progress line shows speed and time left, and **Cancel**, a dropped
  connection or quitting keeps what's downloaded — the button then reads **Resume**. Each piece is retried
  before giving up.
- **New model bundle:** Essentials now contains SFT with the akhaliq 4-step Turbo built in (5.4 GB, stock sampler);
  the optional groups are SFT, Base, the akhaliq and lodestones `lora_latest` LoRAs, and the Base Turbo.
- **Settings → Memory** (Automatic / Keep models loaded / Save memory): on Macs with enough memory the text encoder
  and the other models stay loaded between jobs — images come ~2.5 s faster on a 192 GB Mac, and Enhance,
  Caption, Help and VOSR2 no longer unload the image model. 64 GB Macs keep the memory-saving behaviour.
- **One shared Qwen3.5:** the text encoder, the enhancer, the captioner and Help now share one copy of the model
  instead of loading their own (about 30 GB less with everything loaded), so Layout from prompt → Generate loads
  nothing again.
- **Fix:** Layout from prompt no longer fails when the model leaves an optional field blank or labels a box with
  an unexpected category.

## 0.2.11 — 2026-10-30

- **Extract LoRA:** Models → Tools → **Extract LoRA…** turns a finetune or merge into a LoRA against its parent
  (Base or SFT, picked from the model's family and changeable), at a chosen rank or an energy/threshold cutoff. The
  LoRA lands in `lora/extracted/` and applies to the parent like any other LoRA — a few hundred MB instead of a
  10 GB model. It's an approximation: on DirtyMoss a rank-64 LoRA keeps the style and layout but not every detail,
  so compare before deleting a finetune. Also `moss-mlx extract-lora`.

## 0.2.10 — 2026-10-30

- **LoRA strength range:** each LoRA can have its own strength range, set in Models → **Slider range** (−2 to 2
  unless changed). Generate's strength slider spans it, so LoRAs made for big weights (e.g. −10 to 10) are usable;
  sliders trained in MossForge bring their recommended range with them. Type in the box beside the slider to go
  past either end.
- **Fix:** a LoRA's **Default strength** from the Models screen is now used when you add it in Generate (it always
  started at 1.0).

## 0.2.9 — 2026-10-28

- **Internal:** the app's own padding, frames, fonts and the sidebar column width are declared on its
  Tweakbench tags. Release builds lay out exactly as before; developer builds
  can now override those values directly.

## 0.2.8 — 2026-10-28

- **Fix:** diffusion models saved by Forge's checkpoint merger (every tensor under `model.diffusion_model.`) load
  and convert/fuse/merge like any DiT, instead of failing with "not a MOSS-Image DiT".

## 0.2.7 — 2026-10-28

- **Character Studio:** the reference photo now restages through chinoll's Moss-Image-Edit LoRA at a new
  **Strength** slider (default 0.7), so rows no longer copy the reference pose. The model library now recognises
  bare `lora_A`, LoHa and LoKr edit LoRAs (they were hidden as "unknown").
- **MCP:** `edit_image` takes `edit_layout` (`auto`, `mossForge`, `ai-toolkit`, `ai-toolkit (kv_cache off)`,
  `Moss-Image-Edit`), like the HTTP edit route.

## 0.2.6 — 2026-10-27

- **Edit LoRAs:** community edit LoRAs work — chinoll/Moss-Image-Edit and akhaliq/MOSS-Image-Edit-HQ (armV5) —
  with ComfyUI's four reference layouts, read from the LoRA (or picked in Generate → Edit → Layout).
- **Generate → Edit:** up to four drag-and-drop reference wells ("Picture 1…4"); **Mask…** paints where Picture 1
  may change and keeps everything else exactly. With Turbo on, edits run its LoRA on flow Euler, CFG 1.
- **Character Studio:** an optional reference photo, restaged in every image through an edit LoRA.
- **LoHa LoRAs** load (comfyanonymous's 4 MP hires LoRA); 2048² images no longer crash while decoding.
- **Get Models:** download the model bundle (default FiditeNemini/MossForge, set in Settings → Folders) from
  Models or Generate's first-run banner; `mossforge models list | download`.
- Seed shown as a plain number; Share and Save are labelled buttons.

## 0.2.5 — 2026-10-27

- **Fix:** a fused Turbo's `…-dataspace.safetensors` adapter is no longer listed as a diffusion model
  (picking it failed with "not a MOSS-Image DiT").

## 0.2.4 — 2026-10-27

- **Generate → Turbo:** new choice *None — model is already 8-step, CFG 1* for a few-step model
  without a recipe file: 8 steps, CFG 1, no Turbo LoRA.

## 0.2.3 — 2026-10-27

- **Fix:** `moss-mlx convert | fuse | merge` now go through the engine like the app, so a Turbo LoRA
  fused on the command line keeps its settings file and data-space file and runs as an 8-step
  Turbo model with `generate --model`.
- Themed sidebar hover re-checked in the Xcode-built app (no change needed).

## 0.2.2 — 2026-10-25

- Help, Feedback and the theme layer reworked; the Help composer buttons no longer stack in Tweakbench builds.
