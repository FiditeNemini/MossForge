# Technical reference: how MossForge runs MOSS-Image

How the models are built and how MossForge applies them: the network shapes, the sampler maths,
Turbo, LoRA formats, quantised weights, layouts and editing, upscaling, captioning and training.
Everything here is read from the real weights or the official code (where the paper and the code
disagree, the code wins, because it matches the weights), and the MLX ports are checked
numerically against PyTorch references (`moss-mlx *-parity`). For which settings to use, see the
user guide and the settings guide; this page is for the *why* and *how*.

## Pipeline overview

A text-to-image run is four stages:

1. **Encode the prompt.** The prompt (plain text, tags or layout JSON) is wrapped in Qwen3.5's
   chat template, tokenised (at most 4096 tokens, no padding) and run through the **Qwen3.5-9B**
   language model. The last layer's hidden states, after the final norm, are the conditioning:
   `[tokens, 4096]`.
2. **Denoise.** The latent starts as Gaussian noise in the DiT's token space,
   `[128, H/16, W/16]` (32 VAE channels × a 2×2 patch per token, so one token per 16×16 pixels).
   The DiT predicts a **velocity** at each step and a flow-matching Euler sampler integrates it
   from σ = 1 (noise) to σ = 0 (image), with classifier-free guidance. Turbo replaces this loop
   with an 8-step data-space loop (below).
3. **Un-normalise.** The DiT works in a batch-normalised latent space; each of the 128 patch
   channels is mapped back with `z · √(var + 1e-4) + mean` (the VAE file's `bn` statistics), then
   the 2×2 patches are unfolded back to 32 channels.
4. **Decode** with the FLUX.2 VAE to RGB.

All of it runs in MLX on the GPU. There is no cloud step.

## The MOSS DiT

A **single-stream** diffusion transformer of 5.08 billion parameters, trained from scratch.
Base and SFT are the same architecture (379 tensors, identical names and shapes); SFT is the
post-trained model, Base the raw pretrain (MOSS recommends it for LoRA training).

### Blocks: attention, feed-forward and modulation

- **42 blocks**, hidden size **3072**, **24 heads × 128**.
- **Attention**: one fused `proj.qkv` projection (rows q | k | v), per-head **q/k RMSNorm**, 4D RoPE,
  plain full scaled-dot-product attention over the whole sequence, output projection `proj.o`.
  No bias on block linears.
- **SiTU-GLU feed-forward** (hidden 8192): `ffn_up_gate` produces `hidden` and `gate` (hidden
  first), and `SiTU(h, g) = (4·tanh(g/4)·σ(g)) ⊙ (25·tanh(h/25))` — a soft-capped SiLU-style gate
  times a soft-capped value — then `ffn_down`.
- **RMSNorm** with eps 1e-6 and a plain `w` scale (not Qwen's `1 + w`).
- **One shared time modulation.** A sinusoidal embedding of `timestep = 1000·σ` (256 dims,
  `[cos | sin]`, float64 phases) goes through Linear → SiLU → Linear, and a *single*
  `shared_block_modulation` maps it to six vectors used by **all 42 blocks**: shift, scale and gate
  around attention and around the FFN: `x += gate · attn(norm(x) · (1 + scale) + shift)`, likewise
  for the FFN. (Most DiTs give every block its own modulation; MOSS shares one.) The output head
  has its own shift/scale (`video_head_modulation`). There is no SiLU before either modulation.
- **Input and output**: `video_embedder` (Linear 128 → 3072) embeds patches; the head applies the
  modulated final norm and `final_linear_video` (3072 → 128).
- **One sequence**: image tokens **first**, then text tokens; they attend to each other in every
  block (that is what "single-stream" means). Only the image part of the output is kept.

The output is a velocity `v = ε − x₀` (flow matching: `z_σ = (1 − σ)·x₀ + σ·ε`).

### Fused vs split qkv

The Comfy-Org release files ship `layers.N.proj.qkv` fused `[9216, 3072]`; the official diffusers
`transformer/` folder and most LoRAs use split `proj.q`, `proj.k`, `proj.v`. MossForge's in-memory
layout is **fused** (one matmul): split files are concatenated q | k | v when they load, and
split-named LoRA deltas land on the matching row ranges (rows 0–3071 q, 3072–6143 k, 6144–9215 v,
checked bit-exact against the official split weights).

### Position encoding (4D RoPE)

Each token has four coordinates **(r, t, h, w)**: r a reference index, t a sequence position, h and
w the row and column on the patch grid. The 128 head dims are split into four axes of 32, each
with the same 16 frequencies `1 / 10000^(j/16)`, applied rotate-half style.

- Text token k: `(0, k, 0, 0)` — ordinary 1D positions.
- Image token (i, j): `(0, L_text, i − (H−1)/2, j − (W−1)/2)` — the grid is **zero-centred**
  (half-integer for even grids) and every image token sits at the same t, "after the text".

Because r = 0 for both text and the image being made, the r subspace is unrotated: in practice the
first 32 of 128 dims carry no position (partial RoPE). The axis order is (r, t, h, w) in code; the
paper's prose says (r, t, w, h) — the code matches the weights. Edit references use r (below).

### The text adapter

`text_embedder` turns the 4096-d Qwen3.5 states into 3072-d DiT tokens: a projection, then two
small transformer blocks (pre-norm, **no** time modulation, fused qkv with bias, q/k norm, RoPE on
the text positions, SiTU FFN with separate gate and up projections).

## The Qwen3.5-9B text encoder

The encoder is the language model of **Qwen3.5-9B** (frozen; MOSS was trained with it frozen):
32 layers, hidden 4096, SwiGLU MLP 12288, vocabulary 248,320. It is a **hybrid** model:

- Layers **3, 7, 11, …, 31** (every fourth) are gated **full attention**: 16 query heads × 256 with
  an output gate (`out · σ(gate)`), 4 key/value heads, `(1 + w)` q/k RMSNorm, rotary embedding on
  the first 64 of 256 dims with θ 10⁷, causal.
- The other 24 are **Gated DeltaNet** linear attention: a causal depthwise conv (kernel 4) + SiLU,
  l2-normalised q/k, a learned decay `g = exp(−exp(A_log) · softplus(a + dt_bias))` and write
  strength `β = σ(b)` driving a delta-rule recurrent state; a gated RMSNorm on the output. MossForge
  runs the recurrence in a Metal kernel (from mlx-lm).

**How it becomes conditioning.** The prompt is wrapped as
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`, tokenised as one string,
cut at 4096 tokens, and run through all 32 layers. The hidden states **after the final `(1 + w)`
RMSNorm** (`layer = "last"`) go to the DiT's text adapter — one layer, not a stack of taps. The
unconditional branch encodes the negative prompt (empty by default) the same way.

The same file is a full VLM (vision tower, LM head): MossForge also uses it to caption images,
enhance prompts and layouts, and answer Help questions. Its states are sensitive to precision: bf16
weights drift measurably from F32 on long repetitive prompts (natural prompts stay close). The app
runs it in bf16 by default (half the memory); F32 is the parity reference.

## The FLUX.2 VAE

The VAE is FLUX.2's: a 2D convolutional autoencoder (GroupNorm 32, mid-block attention) with
**32 latent channels** and **×8** downsampling.

- **Encode**: pixels → `quant_conv` → posterior **mean** (no sampling) `[32, H/8, W/8]`.
- **Patchify** 2×2: `c (i pi) (j pj) → (c pi pj) i j`, i.e. channel = `c·4 + dy·2 + dx` → `[128, H/16, W/16]`.
- **Latent batch norm**: subtract the file's `bn.running_mean` and divide by
  `√(bn.running_var + 1e-4)` per channel, no affine. This is the space the DiT sees and the noise
  is drawn in.
- **Decode** is the exact inverse, then clamp to [−1, 1].
- **Large images** (above 1024², i.e. > 16 384 mid-block tokens): the mid-block attention runs in
  2048-query chunks and each decoder level is evaluated on its own, so no Metal command buffer runs
  long enough for macOS's GPU watchdog ("Impacting Interactivity"), which killed 2048² decodes before
  2026-10-27. Chunking queries is exact; ≤ 1024² takes the unchanged path.

Image sizes are multiples of 16 px (one DiT token per 16×16 pixels); 1024² is 4096 image tokens.

## Flow sampling: the σ grid, shift, CFG and SLG

**The shifted σ grid.** With shift s (default **5**) and N steps, the official pipeline takes
`v = linspace(1, s/(1000 + s − 1), N)` and maps it through `σ = s·v / (1 + (s − 1)·v)`, which spends
more steps at high noise. Each σ is then **snapped to the training grid** k/1000 (ties down),
duplicates are merged, and a final 0 is appended. The model receives `timestep = 1000·σ`. (The
paper's / ComfyUI's grid, `σ_k = s·u_k/(1 + (s−1)·u_k)` with `u_k = 1 − k/N`, differs in the tail:
50 steps end …0.115, 0.024, 0 officially vs …0.093, 0.) MossForge uses the official grid.

**Euler**: `x ← x + (σ_next − σ)·v`, in F32 whatever the weights' dtype.

**Classifier-free guidance**: `v = u + cfg·(c − u)`, c the prompt's velocity, u the negative
prompt's (default CFG **3**, range 3–6).

**Skip-layer guidance (SLG).** By default the unconditional pass runs with **block 22 skipped**
("uncond skip"): the guidance then pushes away from a slightly degraded model as well as from the
negative prompt, which sharpens detail. Optional standard SLG adds `scale·(c − c_skip)`, where
`c_skip` is the conditional pass with the chosen blocks skipped, inside a start–stop window of the
schedule. Turning SLG off gives a more photographic look but detail can collapse.

**Packed passes.** The conditional, unconditional and SLG passes of a step run as one forward: each
branch's rows are concatenated so every matmul runs once, and attention runs per branch (no
padding, no mask). It is bit-identical to separate forwards and keeps the GPU busier.

## Turbo: the few-step samplers (data-space DMD and flow)

**lodestones' 8-step DMD Turbo** is a data-space distribution-matching distillation of SFT (and
separately of Base). It ships as a rank-128 LoRA on every block's q, k, v, o and FFN linears, plus
**fp32 replacement tensors** for the time embedding and the modulation biases (a t = 1 time vector
is folded into the biases — valid only together), plus a `dataspace` file with a 128×128 mixing
matrix `Q` and 8 step codes. Its sampler is **not** flow Euler:

```
z ~ N(0, 1);  x₀ = 0
for r in 0..<8:   x = z + Q·x₀ ;  v = DiT(x, step_codes[r], text) ;  x₀ = z − v
decode x₀
```

CFG is 1 (one forward per step), and any prefix of the 8 steps works. The distillation delta is
smaller than bf16 rounding, so the LoRA is applied **unmerged in F32** and the time embedding and
modulation biases stay F32; without the replacement tensors the loop yields noise. The step codes
are the official 8-step shift-5 timesteps.

**Flow-route Turbo LoRAs** (akhaliq's 4- and 2-step, lodestones' Base LoRA used alone) need no
special sampler: the LoRA on the stock flow-Euler sampler at CFG 1, shift 5, with the 4- or 2-step
official grid. Each LoRA's sidecar records its route, steps and family (SFT or Base).

**Exporting a Turbo checkpoint**: merge the Turbo LoRA into bf16/fp32 weights **before**
quantising. Quantising first, or merging onto int8, loses the effect, because the delta is below
the quantisation step.

## LoRA and LoKr: formats and key mapping

A LoRA changes a weight by a low-rank product, `W' = W + strength · (α/rank) · up · down`. **LoKr**
(LyCORIS) uses a Kronecker product, `ΔW = strength · (w1 ⊗ w2)`, which reaches a higher effective
rank for the same number of parameters; MossForge applies it as `w1 · X · w2ᵀ` on reshaped
activations rather than building the dense ΔW.

`MossLoRAKeyMap` reads the naming schemes found in the wild and maps them onto canonical names
(`layers.N.proj.qkv`, `proj.o`, `proj.ffn_up_gate`, `proj.ffn_down`, plus embedders, modulation,
text adapter and head):

- **kohya** (`diffusion_model.` prefix, `lora_down` / `lora_up` / `alpha`) — lodestones' Turbo,
  MossForge's own trained files.
- **diffusers / PEFT** (`transformer.`, `lora_A` / `lora_B`).
- **ComfyUI / Kijai** full deltas: `.diff` (a weight delta, e.g. norms) and `.diff_b` (a bias delta).
- **moss-edit-lora** (chinoll, Edit-HQ's converted file): bare `layers.N.proj.q.lora_A` / `lora_B` (no
  `.weight`), alphas in the `adapter_config` metadata JSON's `modules`.
- **LoKr**: `lokr_w1` / `lokr_w2`.
- **LoHa**: `hada_w1_a/b`, `hada_w2_a/b`, `alpha` — `ΔW = (α/r)·(w1_a·w1_b) ⊙ (w2_a·w2_b)` (comfyanonymous's
  hires LoRA, fused qkv). It has no low-rank form, so attach merges it in F32 onto a saved copy of the
  weight, stored back in the working dtype (detach restores the copy).

Split `proj.q/k/v` updates become three row slices of the fused qkv; fused-qkv updates apply to
all rows. Keys with no DiT counterpart are **reported**, never guessed or silently dropped.

**Per-block strength** (`file@strength:blocks`) scales a LoRA on chosen blocks only — e.g. Kijai's
SFT-minus-Base LoRA on blocks 21+ at −0.5 removes SFT's vertical banding.

At generation LoRAs are applied **unmerged** (extra low-rank matmuls in F32); **Fuse** in Models →
Tools folds them into a new checkpoint with fp32 maths.

## Quantised weights: int8·cr, fp8, nvfp4, int4·cr

MossForge reads and writes the formats the MOSS ecosystem uses, and **dequantises to the working
dtype as it loads**:

- **int8·convrot** (Comfy-Org's release format, comfy-kitchen ConvRot): per-row int8 with a per-row
  F32 `weight_scale`, plus a **Hadamard rotation** folded into the weights in groups (256 for the DiT
  and the language model; 64 and 16 in the vision tower) to spread outliers. ConvRot uses a
  *regular* Hadamard (built from a 4×4 base whose rows each sum to 2); it is its own inverse, so
  loading computes `W = (int8 · scale) @ H`. The parameters live in a small per-tensor JSON blob,
  `<layer>.comfy_quant` — these files have no global quantisation metadata. In the DiT only the 168
  block linears are int8; everything else stays bf16. Dequantised weights match bf16 at cosine
  ≥ 0.99995.
- **fp8** (E4M3 / E5M2) with a per-tensor scale, **int4·convrot**, and **nvfp4** (4-bit E2M1 values
  with an E4M3 scale per block of 16 and a global scale) are also decoded.

**Low-bit is for size, not speed, on Apple silicon.** A diffusion step pushes thousands of tokens
through every weight, so it is compute-bound; MLX's int8 quantised matmul measured about 17%
*slower* than fp16 at 8k tokens. MossForge therefore unpacks once at load and runs dense: an
int8·cr checkpoint generates **as fast as bf16, not faster**. Quantisation saves disk, download
and loading memory. The DiT computes in **fp16** by default — on M1/M2 (no bf16 ALUs) fp16 matmul
and attention are about 20% faster, and fp16's longer mantissa keeps the velocity ~9× closer to the
F32 reference than bf16; MOSS's weights fit fp16's range. Norms, modulation, RoPE and the time MLP
stay F32.

**Export** (convert / fuse / merge) writes the Comfy-Org layout so ComfyUI and Forge read the
result: the 168 block linears quantised with their `_scale` and `comfy_quant` blob, the rest bf16,
and a Turbo's step embedder and modulation biases kept fp32. **Merge** is `(1 − α)·A + α·B`, or
`A + α·(B − C)` to transplant what B learned on top of C into A.

## Layouts and editing

**Layout prompts.** A layout is JSON through the same text pathway — MOSS was trained on it, so
there is no extra network: a summary (meta tags, caption) and items, each with a label, caption,
optional exact text and a `bbox`. Boxes are `[y1, x1, y2, x2]` in **`abs_centralized`**
coordinates: pixels with the origin at the canvas centre. Zero-centred absolute coordinates
generalise across resolutions better than a 0–1000 grid (paper App. K). Explicit `\n` line breaks in
text items help rendering. The canvas edits a draft and sends the compiled JSON as the prompt; the
on-device enhancer writes drafts in pixel `xyxy` from the top-left and the official compiler (ported
from `compile_prompt.py`) converts them.

**Edit in Layout** (regional redraw). MossForge diffs the original layout against the edited one
(changed, added and removed boxes, plus a one-token margin) into a token mask. The original image is
VAE-encoded, and after every step the latent **outside** the mask is reset to the original noised
to the next σ, `(1 − σ')·x₀ + σ'·ε` with the seed's own noise (for data-space Turbo, the x₀ estimate
is pinned instead). Finally the original pixels are pasted back through the upsampled mask, so
everything outside the edited boxes is pixel-identical.

**Reference images / edit LoRAs** (community method). The released model has no image input, but
RoPE's r axis is free: reference k is VAE-encoded to **clean** tokens at `(r = k, h, w)` (its own
centred grid, unscaled frequencies) and conditioned at **timestep 0**. Output and loss cover only the
image rows; with no references the forward is bit-identical to plain generation. How the references
join is the LoRA's training **layout** (`MossEditLayout`, = ComfyUI PR #17296's `EDIT_FORMATS`):

| Layout | Attention | References' t | Encoder sees them |
|---|---|---|---|
| `mossForge` (Train → Edit) | in the sequence (per-row modulation) | the image's | no |
| `aiToolkitNoCache` (`index_timestep_zero`) | in the sequence | the image's | yes |
| `aiToolkit` (`…_isolated`, Edit-HQ) | **isolated**: only each other; per-block K/V cached once | the image's (both 0) | yes |
| `mossImageEdit` (`…_text_start`, chinoll) | isolated | the text's start (0) | no |

Isolated references run once through all 42 blocks (LoRAs on) and each block's post-RoPE K/V is
appended to every branch's keys/values (`MossDiT.isolatedReferenceKV`). They are cached at t = 0, so
an ai-toolkit branch shifts its own image + text t by −L_text (`MossRoPEIndex.referenceSequence`;
relative attention inside the branch is unchanged). "Encoder sees them": the prompt becomes
`Picture 1: <|vision_start|><|image_pad|>…<|vision_end|>…` + instruction, each reference area-scaled to
≤ 384² then `smart_resize` (×32, ≥ 3136 px), Qwen3.5 vision tokens spliced in with M-RoPE positions —
for the negative prompt too. The layout comes from the request, else the LoRA's metadata
(`mossforge_edit_layout`, `adapter_config.format == "moss-edit-lora"`, ai-toolkit's `software`), else `mossForge`.

**Edits and Turbo.** lodestones' data-space loop (`x = z + Q·x₀`) turns isolated-reference edits into
textured noise (2026-10-27, both community LoRAs, while flow Euler edits cleanly), so an edit with a Turbo
runs its LoRA alone on flow Euler, ≤ 8 steps, CFG 1, shift 5, no SLG (`GenerationRequest.moveTurboToFlow`)
— Edit-HQ's stack. A fused Turbo has the loop's time embedding baked in (flow on it renders flat colour),
so edits on one are refused.

**Masked edits.** Generate → Edit → Mask… (or `--edit-mask`) sets `GenerationRequest.edit` with Picture 1 as
the source next to `references`, so the Edit-in-Layout pin (`MossPipeline.editPin`: re-noised original outside
the mask each step, feathered pixel paste-back) bounds the edit LoRA. `EditMask.resampled/paint/inverted` are the
helpers; the mask is resampled when the output size changes. Character Studio's reference photo is one more
reference per row through `AppStore.characterEditLoRA` (Moss-Image-Edit layout first) with
`CharacterDatasetPlan.referencePrompt`.

## Upscaling: VOSR2, ESRGAN and SPAN

- **ESRGAN family** (RRDBNet ×2/×4: Real-ESRGAN, UltraSharp, Remacri) and **SPAN** ×4
  (ClearReality): deterministic CNN upscalers, ported to MLX bit-exact against the PyTorch
  originals. Sharp and fast; they invent little.
- **VOSR2** is a 1.4B **one-step generative** super-resolution model. The image is first
  softened, resized bicubically to the output size and encoded with the **Qwen-Image VAE** (16 channels — VOSR2's own latent space, not
  MOSS's FLUX.2 one); for each square latent tile a DINOv2-L encoder reads the matching pixel crop
  (layer-17 tokens) and a LightningDiT generates the detail in a single step from the low-quality
  latent plus noise. Tiles overlap and are cross-faded; four tiles go through each model call.
  *Input softening* removes sampler grain first (which VOSR would read as texture), and *colour
  match* (wavelet or AdaIN) takes the low-frequency colour from the input.

## Captioning (the vision tower)

Qwen3.5-9B's vision tower is Qwen3-VL's design **without DeepStack**: 27 blocks, hidden 1152, 16
heads × 72, 16-px patches (two frames for a still image), 2D rotary positions plus an interpolated
48×48 learned position grid; every 2×2 patch group is merged into one 4096-d token. Preprocessing
resizes to a multiple of 32 within a pixel budget (minimum 256²) and normalises with mean = std = 0.5.
The image tokens replace `<|image_pad|>` placeholders, and the language model uses 3-axis
**M-RoPE** positions (an image spans rows and columns; text after it resumes at start + max(h, w)/2).
Captions use the official recaption prompt (long / medium / short captions, watermarks, borders),
decoded greedily with a KV cache so each new token costs one forward step.

## LoRA training: the objectives

Training is native Swift/MLX: automatic differentiation over the adapter weights only, with the DiT,
VAE and text encoder frozen. Adapters start as a no-op (`up` = 0) so training departs smoothly from
the base. Optimiser: AdamW with β (0.9, 0.95) and weight decay 0.01 (the MOSS recipe), global-norm
gradient clipping at 1.0; α = rank.

**From images.** Each image is VAE-encoded once at its own aspect (scaled to the pixel budget, never
up, multiples of 16) and each caption encoded once. Every step draws σ from **MOSS's final
pretraining distribution** (report eq. 4): `σ = 1 − sigmoid(Z)`, `Z ~ N(−1.09, 0.95²)`, cut off at
0.95 and rescaled onto the 1000-step grid, mixed with 1% uniform — weighted towards mid-to-low noise.
It then builds `z_σ = (1 − σ)·x₀ + σ·ε` and minimises the mean squared error between the predicted
velocity and `ε − x₀` over the image tokens (no σ weighting). Because σ and ε are random each step,
per-step loss is noisy; the moving average is the signal.

**Caption dropout** trains a share of steps (5–10% typical) on the empty caption's context, so the
LoRA also works with prompts unlike its captions.

**Gradient checkpointing** wraps each DiT block so its activations are discarded in the forward pass
and recomputed in the backward pass (MLX Python's `mx.checkpoint`; the block's adapters are inputs so
their gradients flow). Same results; at 1024² it roughly halves peak memory (101.7 → 51.3 GB measured)
and is usually *faster*, because without it the activations push the GPU into memory pressure.

**Slider (Concept Sliders).** No images. Each step makes its own latent by partly denoising noise
with the frozen model and the target prompt (k of the *latent steps*, CFG flow sampler), then
computes the frozen model's velocity for the target, `target + positive` and `target + negative`
prompts (one packed three-branch forward). The adapter, applied at a random weight m from the
trained range, is trained towards `v(target) + m · η/peak · (v(positive) − v(negative))`. Training
across the whole range makes the slider linear in its weight; **η** sets how far the full-strength
end pushes; *Preserve* adds prompt variants with an attribute appended to all three prompts, so the
learned direction doesn't depend on it.

**Edit (experimental).** Pairs of a source (in `control/`) and an edited target with an instruction:
the source rides along as reference tokens (above) and the loss is the usual velocity loss on the
target's rows.

**Layers.** *Attention* trains `proj.qkv` and `proj.o` in every block (composition, subject, style);
*Attention + FFN* also trains `ffn_up_gate` and `ffn_down` (texture, detail, typography). **LoKr**
sizes its two Kronecker factors by the *factor* setting (Auto = balanced).

**Test images** are rendered inside the trainer with the adapters as they stand (unmerged), on the
official sampler at a fixed seed per prompt, so rows differ only by the LoRA. The saved file is a
kohya-style safetensors (`lora_down` / `lora_up` or `lokr_w1` / `lokr_w2`, alpha = rank) with the
training settings in its metadata, and it loads back through the same key map as any other LoRA.

## Performance on Apple silicon

The ceiling is **matrix-multiply throughput**: a 1024² CFG step is about 97 TFLOP. MossForge packs
the CFG/SLG passes into one forward, fuses the elementwise work (modulation, q/k norm + RoPE, SiTU,
gated residual) into single kernels, uses MLX's fused attention (never materialising the attention
matrix), and computes the DiT in fp16 — 1024² × 50 steps went from 311 to 225 s on an M2 Ultra. The
real levers beyond that are fewer steps (Turbo: 8 single-pass steps instead of 50 two-pass ones)
and fewer pixels. Quantised weights don't speed it up (see above).
