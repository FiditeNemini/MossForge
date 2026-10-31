# Training LoRAs

The **Train** screen (or `moss-mlx train`) trains an adapter on a MOSS DiT, on device. Three modes sit at the
top: **From images** (a style, subject or concept from your pictures), **Slider** (a concept you can dial up and
down, no images needed) and **Edit (experimental)** (teach the model an image edit from before/after pairs).
Results are saved to `models/lora/trained/<name>.safetensors` with a sidecar, and appear in Generate's LoRA menu.

## From images

Choose a folder of images, each with an optional `name.txt` caption (**Caption N missing…** writes them on device
first — images without one train on their file name). Pick the DiT (**Base** by default, as MOSS recommends; LoRAs
trained on SFT only suit SFT), a name and an optional **trigger word** (prefixed to every caption as
`trigger, caption`; use it in prompts to call the LoRA up).

**LoRA type** (From images): *Character* (one person, consistent from close-up to full body — attention + FFN at
rank 32, ~80 steps per image), *Concept* (an object, outfit, pose or action — attention only at rank 16, ~60 steps
per image), *Style* (a look applied to any subject — attention + FFN at rank 32, the longest runs) or *Custom* (set
everything by hand). A type fills rank, layers, learning rate (1e-4), steps scaled to the dataset (clamped to a band,
rounded to 50; re-scaled when you pick a new folder unless you've changed the steps) and about four checkpoints,
shows the dataset size it suits and a **captioning tip** — the part most often got wrong. Change anything and it
says *Modified* with a **Reset**. These are community starting points for Flux-class DiTs, not yet measured on
MOSS-Image. With a type, the **Preset** stays the run length: *Quick test* (512², a tenth of the steps), *Standard*
(1024², the type's steps), *Detailed* (1024², half as many again).

Presets (Custom): *Quick test* (512², rank 8, 200 steps), *Standard* (1024², rank 16, 1000 steps), *Detailed* (1024², rank 32,
attention + FFN, 2000 steps). **Train: Attention** learns composition and subject; **Attention + FFN** also learns
texture and detail (concepts, typography) at a larger file. Checkpoints every N steps go to `<name>-checkpoints/`;
**Stop & save** keeps what's trained so far. **Try in Generate** picks a Turbo profile for the LoRA's family and
adds the LoRA. The loss chart is noisy by design (each step draws a random noise level) — watch the average line.

## Test images at every checkpoint

Under **Test images**, give up to three prompts — or press **Suggest from captions** and Qwen3.5 reads your
captions and writes three: one close to the training images, one in a new scene, one harder test. They are
rendered **before training**, at **every save** and at the **end**, with the **same seed** each time, so each row
in the bottom half of the progress pane differs only by what the LoRA has learned (drag the splitter between the loss chart
and the images to resize them). Click an image to inspect it: **Fit**, **100%**, **200%** or **400%** (actual pixels,
scrollable), its pixel size, **Show in Finder**, and the same prompt at every checkpoint along the bottom; right-click a
thumbnail to share or save. Defaults: 20 steps at 512² (a few seconds per image on an M2 Ultra —
the panel shows an estimate). The trigger word is added automatically. Images are saved in
`<name>-checkpoints/samples/`. Watch for the point where the subject is learned but the prompt still works —
later steps can start copying the training images (overfitting).

## Adapter type: LoRA or LoKr

**LoRA** (low-rank, tune with *Rank*) is the classic choice. **LoKr** (LyCORIS Kronecker, `ΔW = w1 ⊗ w2`) is often
more expressive per parameter; *Kronecker factor* sizes the split (*Auto* = balanced, 4/8/16 = smaller first
factor). Both load in Generate, the server and Models → Tools → Fuse, and LoKr files from other LyCORIS trainers
load too when they use MOSS's module names and full factors (`lokr_w1` / `lokr_w2`; the decomposed
`lokr_w1_a/b` form isn't supported yet).

## Caption dropout

The share of steps that train with an **empty caption** (0–30%, default off). 5–10% helps a LoRA keep working
with prompts that don't look like its captions, and makes a style LoRA apply more evenly. Leave it off for
subject LoRAs that should only appear with their trigger word.

## Save memory (gradient checkpointing)

On by default. Each DiT block is recomputed in the backward pass instead of keeping its activations: results are
identical, memory roughly halves (1024², attention + FFN, rank 16: 101.7 → 51.3 GB peak), and on large runs it is
usually **faster** too (62 → 9.8 s/step in that test, because training no longer runs into memory pressure).
Turning it off only helps small runs on Macs with plenty of memory.

## Sliders

A Concept Slider learns a *direction* — young ↔ old, summer ↔ winter, photo ↔ painting — that you apply at any
weight, negative or positive. No images are needed: each step renders its own latent with the frozen model.

1. Describe it ("a young to old slider") and press **Suggest**: Qwen3.5 drafts the name, **Target** (the neutral
   subject, e.g. "a portrait of a person"), **Positive (+)** and **Negative (−)** descriptors. For "X to Y", X is −
   and Y is +. Review them — write descriptors only; the target is added for you.
2. **Preserve** (optional): attributes the slider must not drag along, e.g. `male, female` for an age slider.
3. **Range** is the weight span it is trained for (must contain 0, default −1…1). Sliders usually need a little
   more push on the + side, so the saved recommended range doubles the positive end (−1…2).
4. **Strength η** (default 1) is how far the far end goes. η 1 keeps the rest of the image steady; raise it only if
   +2 is still too subtle. **Latent steps** (default 10) is the schedule each training latent is drawn from.

Defaults: 300 steps at learning rate 1e-4, attention only. Test images for a slider are one prompt (or the target)
at the range's −end, 0 and +end, so you watch the direction appear. In Generate, add the slider like any LoRA and
set its strength anywhere in the range.

## Edit training (experimental)

Teaches the model an **instruction edit** from before/after pairs, using the community method (as in Ostris's AI
Toolkit and early MOSS edit LoRAs): the source image is fed to the DiT as clean **reference tokens** on the RoPE
axis MOSS leaves unused, and the LoRA learns to produce the edited target.

Folder layout (AI Toolkit's): the **edited targets** with `name.txt` **instructions** ("make the sky purple") in the
folder, and the **sources** with the same file names in a `control/` subfolder. Targets without a matching source
are skipped (the panel says how many). Test images use the first pair's source as the reference.

To use the result: in Generate, add the edit LoRA, then **Add…** a reference image (or right-click any image →
**Use as Edit Reference**) and write the instruction as the prompt. Up to three references.

Honest expectations: this is experimental. Even untrained, reference tokens make the model follow the source's
composition closely; learning a specific edit reliably needs **hundreds to thousands of pairs** (the public proof of
concept used 3,000 MagicBrush pairs and got some edits working). For changing part of an image you made in
MossForge, **Edit in Layout** (see editing.md) works today without any training.

## How it trains

The technical report's recipe where it applies to LoRAs: frozen Qwen3.5 encoder (captions encoded once), cached VAE
latents per aspect bucket, `z = (1−σ)x₀ + σε` with σ from the final-stage distribution (truncated logit-normal
μ −1.09, s 0.95, cutoff 0.95 + 1% uniform), target velocity `ε − x₀`, per-token MSE, AdamW (0.9, 0.95, wd 0.01),
gradient clip 1.0, 50-step warmup, adapters on the fused `proj.qkv / proj.o` (+ `ffn_up_gate / ffn_down`) with
α = rank. Measured: 512² ≈ 1.5 s/step on an M2 Ultra.

## Command line

```sh
swift run -c release moss-mlx train ~/datasets/foxes --rank 8 --steps 200 --res 512 --trigger mossfox \
  --sample "mossfox in a forest" --sample "mossfox as a watercolour"          # test images at checkpoints
moss-mlx train ~/datasets/foxes --lokr --lokr-factor 8 --caption-dropout 0.05  # LoKr + caption dropout
moss-mlx train --positive "old, grey hair" --negative "young, smooth skin" \
  --target "a portrait of a person" --preserve "male, female" --steps 300 --out ../models/lora/trained/age.safetensors
moss-mlx train ~/datasets/edits --edit-pairs --steps 1000                         # edit pairs (targets + control/)
moss-mlx generate --prompt "make the sky purple" --reference before.png --lora edit.safetensors@1
```

`--no-checkpointing` turns gradient checkpointing off; the command prints the training loop's peak memory.
