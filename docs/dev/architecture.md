# Architecture (developer)

## Packages

```
MOSS-Image/ (root package "MossForge", MLX-free)
  MossForgeCore     value types — no deps
  MossForgeEngine   capability protocols, EngineBackend, MossEngine, StubBackend, pure model tooling
  MossForgeCLI      `mossforge`
MOSS-Image/MossForgeMLX/ (sibling package; depends on the root by path, identity "MOSS-Image")
  MossForgeMLX      MLX backend (Phase 1+)
  MossForgeMLXCLI   `moss-mlx` — generation + `*-parity` numerics harness
```

Three front-ends (app, CLI, server) talk only to `MossEngine`, which wraps an `EngineBackend`.
`StubBackend` makes everything run without weights; `MossBackend` (MLX) is injected at the
composition root. Pure logic the MLX code needs (header parsing, key maps, RoPE coordinates,
schedules, sizing) lives in Core/Engine so it is unit-tested; GPU numerics are verified by
running `moss-mlx` (the xctest runner cannot load mlx-swift's metallib).

## Engine surface today

| Protocol | Stub | MLX |
|---|---|---|
| `ImageGenerating.generate` | gradient PNG + per-step progress | Phase 1 |
| `PromptEnhancing.enhancePrompt` | canned prose / layout JSON | Phase 5 |
| `ModelInspecting.inspectModel` | real (header-only, shared default impl) | same |
| `ImageUpscaling`, `ImageCaptioning`, `ModelTooling`, `LoRATraining` | canned / not implemented | Phases 5–7 |
| `TextChatting.chat` (Help → Ask, feedback drafts) | not implemented (no `.chat` capability) | `MossAssistant`: Qwen3.5 ChatML, non-thinking, streamed |
| `EngineBackend.diagnostics()` | `[]` | GPU, MLX memory, loaded models |

Model tooling (convert / fuse / merge) runs through `MossEngine.runModelTool` from both the app and
`moss-mlx convert|fuse|merge`: the engine writes the output's `.mfmeta.json` sidecar (a fused Turbo
becomes a built-in recipe) and copies a data-space Turbo's `dataspace` file beside the output, so
`moss-mlx generate --model <fused>` runs it at its recipe.

**Extract LoRA** (`ModelToolOperation.extractLoRA(parent:rank:cutoff:)`, `moss-mlx extract-lora`) is the
fourth tool: input = a finetune / merge, `parent` = the model it came from (default
`MossLoRAExtraction.defaultParent`: the official `moss_image_<family>_*` file of the source's sidecar family,
bf16/fp32 first). The pure half is `MossLoRAExtraction` (Engine): `plan` pairs the two files' logical keys
(`WeightLoadPlan.logicalShapes`, prefix-stripped, qkv-fused) into 2-D weights → low-rank, 1-D weights → `.diff`,
biases → `.diff_b`, and reports keys only one side has or whose shapes differ (the official file's
`rope.bands`). Every 2-D weight in the DiT runs through the LoRA-aware `linear` / `project`, so all of them
(block projections, text adapter, embedders, modulation, head) are factorised; norms and biases can't be,
and at ~0.3 M values in all they're stored whole rather than dropped. Names are the trainer's kohya keys
(`MossLoRAOutput.keys`), alpha = rank (scale 1); `LoRARankCutoff` (Core) picks each layer's rank (fixed,
energy fraction of `‖ΔW‖²`, or σ threshold). The MLX half (`MossLoRAExtractor.swift`) streams one tensor of
each model at a time through `WeightLoader.tensor` (any precision → fp32), inside an `autoreleasepool` per
layer (FileHandle reads are autoreleased `NSData`; without it both models' bytes pile up, 23 GB peak vs
3.4 GB), and factorises with a randomized SVD (oversample 16, 2 power iterations; QR / SVD of the small core
on the CPU stream, so no n×n `Vᵀ`). √σ goes into both factors so bf16 doesn't underflow. The engine writes
the sidecar: the parent's family, strength 1, notes naming source, parent and rank; the header carries
`produced_by` / `op` / `source` / `parent` / `rank` / `cutoff`. `moss-mlx extract-selftest` checks the
numerics end to end (a known rank-6 difference → extract → fuse back through `MossLoRAKeyMap`).

## Model facts

See `docs/dev/technical-reference.md` (DiT, encoder, VAE, RoPE, sampler, quant formats, Turbo/DMD,
LoRAs). Pure-Swift pieces already grounded:

- `MossDiTConfig` — header → dims, fused/split qkv detection, qkv row split.
- `MossRoPEIndex` — (r,t,h,w) coordinates, image-first order, bands.
- `MossSchedule` — `σ = s·u/(1+(s−1)u)`, timestep = 1000σ.
- `MossDimensions` — pixel budget × ratio → multiples of 16 (gradio parity).
- `LayoutPrompt` — abs_centralized yxyx JSON.
- `ComfyQuantBlob` — per-tensor `comfy_quant` decode params.
- `MossInspector` — classify DiT / Qwen3.5 encoder / FLUX.2 VAE / LoRA / DMD sampler.
- `MossWeightPlan` — header → per-tensor dequant op (passthrough / fp8 / nvfp4 / int8·cr /
  int4·cr), ConvRot group size from `_quantization_metadata` **or** per-tensor `comfy_quant`
  blobs, prefix scoping, split `proj.{q,k,v}` → fused `proj.qkv`, geometry `validate`.

## Weight loading

`WeightLoader.load(url:prefix:working:)` (MossForgeMLX) executes a `MossWeightPlan`: bf16/F32
files go straight through `MLX.loadArrays`; quantized files are dequantized per tensor on the
GPU (int8 codes × per-row scale, then `W_rot @ H` with the regular Hadamard of the blob's group
size — 256 for the DiT/LM, 64 and 16 in the Qwen3.5 vision tower). The **canonical in-memory
DiT layout is fused `layers.N.proj.qkv`** (one matmul, the Comfy-Org file layout); split files are
concatenated q|k|v at load, and split-named LoRAs map onto `MossDiTConfig.qkvRowRanges`.
The Qwen3.5 file is a whole VLM: load the text path with `prefix: "model.language_model."`.

Checks: `moss-mlx load-parity <bf16> <int8>` (unquantized tensors bit-identical, dequantized
cosine ≥ 0.98, plus a no-un-rotation control that must collapse) and `moss-mlx load <file>`
(everything materializes and is finite).

## VAE + latent bridge

`FluxVAE` (MossForgeMLX) = FLUX.2 VAE, ported from ComfyUI's LDM modules (transcribed in
`Scripts/vae_ref.py`): `encode` pixels [0,1] → posterior mean `[32,H/8,W/8]` → 2×2 patchify
`(c pi pj)` → BatchNorm (file's `bn.running_*`, eps 1e-4) → DiT tokens `[128,H/16,W/16]`;
`decode` is the inverse. NCHW at the API, NHWC inside; norms + attention in F32, convs in the
working dtype (F32 default; bf16 costs ~0.1 dB). `MossLatentFormat` (Engine) is the unit-tested CPU
reference for the index maps; `FluxVAEConfig` derives the geometry from the header.

Checks: `moss-mlx vae-parity <vae> MossForgeMLX/Fixtures/vae` (fixture regenerated by
`python3 -I Scripts/vae_ref.py models/vae/flux2-vae.safetensors MossForgeMLX/Fixtures/vae` — it is
gitignored by the `*.safetensors` weights guard) and `moss-mlx vae-roundtrip <vae> <image> --out x.png`.

## Text encoder

`Qwen35Encoder` (MossForgeMLX) = the Qwen3.5-9B text stack loaded from the Comfy-Org int8·cr VLM file
with `prefix: "model.language_model."` (vision tower, lm_head, MTP never materialised). Layers 3,7,…,31
are gated full attention (q|gate per head, (1+w) q/k RMSNorm, split-half RoPE θ 1e7 on 64/256 dims,
causal GQA 16/4 SDPA, ·σ(gate)); the rest are Gated DeltaNet (causal depthwise conv k=4 + SiLU,
l2-norm q/k, decay `exp(−exp(A_log)·softplus(a+dt_bias))`, β = σ(b), recurrence in the
`GatedDelta` Metal kernel, gated RMSNorm). Output = final (1+w) RMSNorm states `[T, 4096]` F32.
`MossTokenizer` wraps swift-transformers with the MOSS chat template + 4096 cap (tokenizer files in
`models/text_encoders/qwen3.5-9b-tokenizer/`, from Qwen/Qwen3.5-9B, Apache-2.0).

Checks: `moss-mlx encoder-parity <te> <tokenizerDir> MossForgeMLX/Fixtures/encoder` (fixtures from
`Scripts/encoder_ref.py`, ~90 s on CPU) and `moss-mlx encoder-diverge <te> <fixture>` (bf16 vs f32 per layer).
Precision: f32 is the parity gate; bf16 drifts on long repetitive prompts (see the class doc).

## DiT

`MossDiT` (MossForgeMLX) ports `comfyui/moss-image/model.py`: residual stream, norms, modulation, the
time MLP and RoPE in F32; block/adapter matmuls in the working dtype. Canonical weights are fused
`proj.qkv` (rows q|k|v — checked bit-exact against OpenMOSS-Team/MOSS-Image-SFT `transformer/`).
RoPE angles come from `MossRoPEIndex.angles` (Engine, tested); the timestep sinusoid from
`MossTimestepEmbedding` (float64 phases). `callAsFunction(latent, timestep: 1000σ, context:)` → velocity;
`skipLayers` is the SLG hook for step 5.

Speed (the sampler is matmul-bound — ~97 TFLOP per 1024² CFG step):
- **Packed branches.** `velocities(latent, timestep:, branches:)` runs cond, uncond (with its skipped
  blocks) and standard-SLG passes as one forward: each branch's `[image | text]` rows are concatenated,
  matmuls/norms/modulation run once over all rows, attention runs per branch (one SDPA each — no padding
  or mask, so nothing leaks). Bit-identical to separate forwards (`dit-bench` reports the diff). A
  block a branch skips runs on the other branches' rows only and is spliced back.
- **Fused elementwise.** `FusedOps` compiles modulate (RMSNorm·(1+scale)+shift), q/k RMSNorm + RoPE,
  SiTU and the gated residual into single kernels; matmuls return the working dtype and the F32 math
  happens inside the kernels, so the big F32 intermediates never hit memory. Same math, same numbers.
- **fp16 working dtype** (default; `--dit-dtype bf16` to override). M1/M2 have no bf16 ALUs: fp16
  GEMM/SDPA are ~20% faster there, and fp16's mantissa puts the velocity ~9× closer to the f32
  reference (`dit-parity --dtype all`: 8.2e-4 vs 7.1e-3). Weights fit (max |w| 12.75).
- Measured and rejected: pre-transposed weights (no change — MLX GEMM takes `Wᵀ` views for free);
  int8 `quantizedMatmul` (~17% slower than fp16 at 8k tokens — compute-bound; it only saves memory).

Checks: `moss-mlx dit-parity <dit> MossForgeMLX/Fixtures/dit` (fixture from `Scripts/dit_ref.py`, which
uses the encoder fixture's real context — regenerate encoder fixtures first) and `moss-mlx dit-denoise`
(VAE → noise → DiT with a real caption → x̂₀; reference-independent end-to-end check).

## Generation + end-to-end check

`MossPipeline` (MossForgeMLX) follows OpenMOSS `pipeline_moss_image.py`: `MossSamplerPlan` (Core) decides
σ, timestep and which passes run per step; guidance and Euler are F32. `MossBackend` is the
`EngineBackend` (`MossEngine(backend: MossBackend(models:))`). `moss-mlx e2e-parity <models>
MossForgeMLX/Fixtures/e2e` runs the whole pipeline at fixed noise against `Scripts/e2e_ref.py`
(~90 s on CPU to regenerate; the refs are importable modules now): all-F32 must match at ≥ 50 dB;
the bf16 DiT is gated on composition only (pooled PSNR), because bf16 rounding sends few-step CFG-3
trajectories to different fine detail — expected, the official bf16 pipeline does the same.

**Residency.** `MossResidencyPolicy` (Core, pure) turns Settings → Memory (`MossResidencyMode`: auto /
keepLoaded / saveMemory) and Metal's `recommendedMaxWorkingSetSize` into two flags: `keepEncoderResident`
(≥ 64 GiB: the pipeline keeps the text encoder between generations) and `modelsCoexist` (≥ 96 GiB: no
evictions between the DiT, Qwen3.5 users and VOSR2). `MossBackend(residency:)` reads them; nil = auto.
`SharedQwen35` (MossForgeMLX) is the one Qwen3.5-9B behind the pipeline's encoder, `MossEnhancer`,
`MossCaptioner` and `MossAssistant` — the transformer once, `lm_head` and the vision tower on demand;
`trim()` (after a job when models don't coexist) keeps only what `keepEncoderResident` asks for,
`releaseAll()` is idle unload / training / model tools. Standalone pipelines and users (parity commands)
make a private one. Checked by A/B against the pre-sharing binary through `moss-mlx serve`: enhance,
generate and caption outputs byte-identical in auto and saveMemory.

Testing policy: after the one 50-step CFG-3 baseline, generation tests use the 8-step CFG-1 Turbo
LoRA (Phase 2) or short runs; parity fixtures stay tiny.

## Server

`MossForgeEngine/Server/`: `MossHTTPServer` (Network.framework listener) → `MossServerRouter` (routes,
`MossGenerationBody` validation, one `InferenceQueue`, `GenerationJobStore`) → `MossEngine` → any backend. The
same router serves the stub (`mossforge serve`, used by the MCP end-to-end test) and MLX (`moss-mlx serve`).
`/v1/moss/edit[/jobs]` parses the same JSON twice — as `MossGenerationBody` and as `MossEditBody` (one source:
base64 `image` or a finished `job`, never a path) — stages the source in a scratch folder as
`GenerationRequest.references[0]`, sizes the output from the source unless the client gave one, refuses a request
with no active LoRA, and runs as job kind `edit`. `pixelSize` and `sourceImage` are shared with upscale/caption.
`mcp/` is the Python MCP bridge that submits jobs and polls them with progress (`edit_image` → the edit route).

## App

`MossForgeMLX/Sources/MossForgeApp` (executable target, the composition root): `AppStore` builds
`MossEngine(backend: MossBackend(models:))` when an SFT/Base DiT + encoder + tokenizer + VAE are present, else the stub.
In-app generation and the Server screen share one `InferenceQueue`. The layout canvas edits a `LayoutDraft` (Core,
tested) and the request carries `layoutPrompt().jsonString()` as the prompt. `AppStore.makeRequest(for:prompt:)` turns
any `GenerateDraft` into a request (model, Turbo, sampling, LoRAs, references); Generate and Character Studio both use
it. Character Studio (`AppStore+Character.swift`) plans rows with `CharacterDatasetPlan` (Core, tested), enhances each
through the shared queue, renders a copy of the Generate draft per row (shot ratio, fixed 32-bit seed) and tags the
saved images into a collection; the prompt is the export caption. Train's LoRA type is `LoRATrainingProfile` (Core,
tested) on `TrainDraft`, beside the Quick / Standard / Detailed run-length presets. Dev launch: `Scripts/run-app.sh` (an
unbundled `swift run` binary can't be driven by accessibility tooling, and MLX needs `mlx-swift_Cmlx.bundle` next to
the executable).

## Help + diagnostics

The help corpus is the curated `HelpCorpus.manifest` (user docs + architecture + licensing + `technical-reference.md`), embedded as Swift by
`swift run mossforge help-corpus` (`HelpCorpusDocuments.swift`; a test keeps it in sync and checks it for personal
details). `HelpCorpus.chunks` splits it at `##`/`###` and at bold-lead paragraphs (`**Title.**`); `HelpRetriever` is BM25 with
heading terms ×3 (no embedding model — a few dozen sections), British/American spellings unified and a small stemmed
synonym table (photo/picture → image). `docs/dev/` chunks score ×0.7 unless the question has technical intent
("under the hood", "algorithm", "maths"…, then ×1.6), so how-to questions land on the guides. Check
`mossforge help-search` on common how-to questions after adding a dev doc. `HelpAssistantPrompt` puts the top excerpts, numbered, in the system prompt with
"answer only from these"; `QwenChatTemplate` renders ChatML and neutralises control tokens in any content.
Diagnostics: `DiagnosticsCollector` (Engine) builds sections from local checks only; `PrivacyRedactor` (Core)
replaces folders, account/host names, e-mail, IPv4, MAC addresses and UUIDs, before the model sees the digest and
again on the final Markdown. `DiagnosticsLog` is memory-only, bounded, and written only while enabled.
`moss-mlx ask "…"` and `mossforge help-search "…"` exercise the same path from a terminal.

## Training extras, sliders and edits

`MossLoRATrainer` shares one loop across modes: **dataset** (flow-match on cached latents), **slider** (Concept
Sliders on MOSS's flow sampler: partly denoise a latent k ∈ 1..<N steps with the frozen DiT + CFG, then train
the LoRA at a random weight m to move v(target) by m·η/peak·(v(pos) − v(neg)); base velocities via one packed
3-branch forward) and **edit** (reference tokens, below). Checkpoint **test images** render inside the trainer with the
live adapters (official sampler, fixed seed per prompt) and reach the app as `GenerationProgress.files`.
**LoKr**: `TrainableLoRA`/`LoRAUpdate` carry a Kronecker flag; `MossDiT.kroneckerApply` computes `x·(w1⊗w2)ᵀ` as
`w1 · X · w2ᵀ` on `x` reshaped `[N, b, d]` (never the dense ΔW; `train-smoke` checks it against the dense product);
`MossLoRAKeyMap` parses `lokr_w1/lokr_w2` (fused or split qkv rows); fuse folds the dense product.
**Gradient checkpointing**: `MossDiT.checkpointBlocks` wraps each block in an MLX `CustomFunction` whose VJP re-runs
the block via `vjp` (the block's adapters are inputs so their gradients flow) — MLX Python's `mx.checkpoint`.
**Caption dropout** swaps in the empty-caption context.

**Edit references** (community method — Ostris AI Toolkit / ComfyUI-MOSS-Image-Ostris-Edit, thearkorbit PoC):
`MossDiT.references` latents are embedded with `video_embedder` and appended after each branch's text, RoPE
`(r = k, t = textLength, h, w)` (`MossRoPEIndex.referencePositions`; r is unused by the released model), and
conditioned at **timestep 0**: modulation becomes per-row `[6, rows, dim]` (step mod for image+text, t=0 mod for
references). Output and loss cover only the image rows. With no references the forward is bit-identical to before.
Not yet: Qwen3.5 vision tokens for the references ("Picture N:"), and Ostris's cached reference K/V pass.

**Edit in Layout** (`GenerationRequest.edit`): `LayoutEditMask` diffs the original vs edited layout (changed, added,
removed items) into a token mask (+1-cell margin). The pipeline VAE-encodes the original and after each step sets
the unmasked latent to `(1−σ')·x₀ + σ'·ε` (ε = the seed's noise; data-space DMD: pins the x₀ estimate), then pastes
the original pixels back through the bilinear-upsampled mask. Verified: pixels outside the mask are identical.

