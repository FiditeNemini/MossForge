# Editing images

Two ways to change an image: **Edit in Layout** (works today on any image made in MossForge — change part of it,
keep the rest pixel for pixel) and **edit LoRAs** (instruction edits like "make the sky purple" — community LoRAs
such as chinoll's or Edit-HQ, or your own trained on before/after pairs; see training.md).

## Edit in Layout

Right-click an image in the **Gallery** or the result on **Generate** → **Edit in Layout…** (or select it in the
Gallery and press **Edit…**). MossForge restores everything that made it — the prompt as a layout, the **seed**,
size, model, Turbo, steps, CFG, shift, negative prompt and LoRAs — and shows the image under the layout canvas.

Then change only what should change:

- **Edit a box**: change a text box's text ("WINTER" → "SUMMER") or an object box's caption.
- **Add a box**: draw it over the area to redraw and describe what goes there. For an image made from a plain
  prompt (no layout), this is how you pick the area — the prompt becomes the layout's caption.
- **Move or delete a box**: the old and new places are redrawn.

The banner shows how much of the image will be redrawn. Press **Redraw changes**: only the changed boxes (plus a
one-cell margin, so the edit blends in) are generated again; everything else stays **exactly** as it was. The
result is saved as a new image (the sidecar records `edited_from`), and it becomes the starting point for the next
edit, so you can make several changes in turn. **Stop editing** returns to normal generation.

Tips:

- Make the box cover the whole thing you're changing — a box over half a word redraws half a word. The image under
  the canvas helps.
- Update the overall caption too when the change affects the whole picture's description (e.g. "winter poster" →
  "summer poster").
- The same seed keeps the redrawn area close to the original in style; for a different take, unlock the seed.

How it works: the original is encoded by the VAE; at every sampling step the area outside the changed boxes is set
back to the original, re-noised to that step with the image's own noise, so the model sees the real surroundings
while it redraws the inside. At the end the original pixels are pasted back outside the box with a soft edge. It
works with every sampler (Turbo included) and costs the same as a normal generation.

## Edit LoRAs and reference images

An **edit LoRA** turns MOSS-Image into an instruction editor: give it the picture and say what to change
("make the snake bright red, keep everything else the same"). Community ones that work in MossForge
(all for **SFT**, put them in `models/lora/`):

| LoRA | Layout | Notes |
|---|---|---|
| [chinoll/Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit) (`adapter_model.safetensors`; Comfy-Org's `moss_image_edit_lora_rank_64_bf16` is the same) | Moss-Image-Edit | Tightest containment in our tests |
| [akhaliq/MOSS-Image-Edit-HQ](https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ) `weights/armV5/armV5.safetensors` | ai-toolkit | Also sees the references through the text encoder |
| Your own (Train → Edit) | MossForge | |

In Generate, the **Edit** section has the reference wells: drop an image on a well (or click it), up to four
("Picture 1…4" — with several, name them in the instruction: "put the hat from picture 2 on the person in
picture 1"). The first image sets the output's aspect, and an installed edit LoRA joins the stack if none is
there. Any image's right-click → **Use as Edit Reference** does the same from the Gallery or the preview.

**Layout** says how the LoRA was trained to see its references; **Auto** reads it from the LoRA file (the
choices match ComfyUI's *Text Encode MOSS-Image Edit* `edit_format`): *ai-toolkit* (references attend only to
each other, computed once per image; the text encoder also gets "Picture N:" + the image), *ai-toolkit (kv_cache
off)*, *Moss-Image-Edit* (references at the start of the text, text-only encoder) and *MossForge*. The wrong layout
still edits, just less cleanly — or not at all.

**Mask… (protect the rest).** A prompt like "don't change anything else" doesn't stop a LoRA from repainting
what touches the change (both LoRAs turned the snake's coils purple when asked for purple bark). Under Picture 1,
**Mask…** opens a painter: brush where the image may change (Erase, Invert — paint what to protect, then invert —
and Clear), **Use Mask**. Everything outside the mask is pinned to the original at every step and pasted back
pixel for pixel at the end (Edit in Layout's mechanism), on the 16 px token grid with a one-cell feather. The
mask belongs to that Picture 1 and follows size changes. CLI: `--edit picture1.png --edit-mask mask.png` (white =
change) with `--reference picture1.png`.

**Several references.** Up to four. Name each in the instruction ("the snake from picture 2 around the shoulders
of the woman in picture 1"). Compositing is these LoRAs' weak spot — they were trained on single-image edits —
so picture 2 tends to ghost into picture 1 (double exposure), at 4 steps and at 20 steps CFG 3 alike. Mask the
area the new thing should go: the rest of picture 1 then stays exact.

**Speed vs quality.** With Turbo on, an edit runs the Turbo **LoRA** on flow Euler at CFG 1 (the Edit-HQ recipe:
4 steps with akhaliq, 8 with lodestones `lora_latest` — ~11 s at 512² here). A data-space Turbo (lodestones Base
8-step) can't carry references, so an edit runs its LoRA on flow Euler instead.
A model with a **built-in** (fused) Turbo can't edit — pick the SFT model with a Turbo LoRA. Turn Turbo off for the
LoRAs' recommended quality pass: 50 steps, CFG 3 (raise to 3.5–4 if an edit lands too weakly).

Over HTTP: `POST /v1/moss/edit` (raw PNG) or `/v1/moss/edit/jobs` (→ `202 {"id"}`) takes the generation body
(`prompt` = the instruction, `variant`, `seed`, `steps`, …) plus `image` (base64) **or** `job` (+ `index`, an
earlier server job's image — never a file path). `loras` must hold the edit LoRA; a request without one is refused.
The output follows the source's size and aspect (long side kept within 512–2048, snapped to 16 px) unless you give
`size` / `width` / `height`; `edit_layout` (`auto`, `aiToolkit`, `ai-toolkit`, `Moss-Image-Edit`, …) picks the
layout. MCP: `edit_image` (same `edit_layout` argument). The server and `moss-mlx generate --reference` give the identical
image for the same request.

From the command line: `moss-mlx generate --prompt "make the sky purple" --reference before.png --lora edit.safetensors@1`
(`--reference` up to four times, `--edit-layout aiToolkit|aiToolkitNoCache|mossImageEdit|mossForge`),
and `moss-mlx generate --edit original.png --edit-box y0,x0,y1,x1 --seed N --prompt "…"` for an Edit-in-Layout style
redraw (boxes in centred pixel coordinates, like layout JSON).
