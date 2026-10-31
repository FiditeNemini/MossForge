# NOTES — provenance & licences

## Licences

| Thing | Licence | Notes |
|---|---|---|
| MossForge source (this repo) | Apache-2.0 | `LICENSE` |
| MOSS-Image **code** ([OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)) | Apache-2.0 | incl. the ComfyUI node (`comfyui/moss-image/LICENSE`, Apache-2.0) |
| MOSS-Image **weights** (OpenMOSS-Team/MOSS-Image-Base, -SFT; Comfy-Org repacks) | Apache-2.0 | Commercial use, redistribution and fine-tunes are permitted under Apache-2.0 terms (keep the licence + NOTICE, state changes). Never committed here — size, not licence. |
| Qwen3.5-9B text encoder | Apache-2.0 (Qwen) | verify on the HF card before bundling |
| FLUX.2 VAE (`models/vae/flux2-vae.safetensors`) | **Apache-2.0** | BFL's `flux2` README, section "FLUX.2 autoencoder": *released under Apache 2.0* at `black-forest-labs/FLUX.2-dev/ae.safetensors`. The autoencoder's licence is separate from the [dev] transformer's non-commercial licence; [klein] 4B (Apache-2.0) also ships it. Bundle with the Apache-2.0 notice. |
| Community: lodestones DMD, toxicdog INT8 Turbo, akhaliq turbo | Apache-2.0 | see `docs/dev/technical-reference.md` |
| Community: Kijai/MOSS-image_experimental LoRAs | no licence tag | treat as all-rights-reserved until clarified |

## Reference code (read-only, not a build dependency)

- [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) — clone it to `Reference/MOSS-Image` to
  read alongside the port. Used for: DiT forward (`comfyui/moss-image/model.py`), config derivation
  (`loaders.py`), prompt template (`text_encoder.py`), size presets (`gradio_presets.py`), inference
  defaults + layout schema (`USERGUIDE.md`).

## Adapted from the reference (independent Swift)

| MossForge file | Adapted from | What |
|---|---|---|
| `MossForgeEngine/MossDiTConfig.swift` | `comfyui/moss-image/loaders.py::model_config` | header → config derivation |
| `MossForgeEngine/MossRoPEIndex.swift` | `comfyui/moss-image/model.py::_RoPE` | 4D coordinates + bands; `angles` = the `_rotate` table |
| `MossForgeMLX/.../MossDiT.swift`, `Scripts/dit_ref.py` | `comfyui/moss-image/model.py` (Apache-2.0) | DiT forward; the torch ref is a near-verbatim transcription with plain torch ops |
| `MossForgeCore/MossDimensions.swift` | `gradio_presets.py::fit_size` | pixel-budget sizing |
| `MossForgeCore/MossVariant.swift` (`MossPromptTemplate`) | `comfyui/moss-image/text_encoder.py` | chat template, 4096 cap |
| `MossForgeCore/MossSchedule.swift` | Technical report App. J | shifted σ grid |
| `MossForgeCore/LayoutPrompt.swift` | `USERGUIDE.md` "Layout Prompts" | JSON schema |
| `MossForgeCore/MossPromptCompiler.swift` (+ tests) | `skills/moss-image-prompt-enhancer/scripts/compile_prompt.py` (+ `test_compile_prompt.py`) | port; golden outputs from the Python reference |
| `MossForgeCore/MossEnhancerPrompt.swift` | `skills/moss-image-prompt-enhancer/SKILL.md`, `references/layout.md` | condensed into the system prompt |

## Ported models (independent Swift/MLX)

- **Upscalers:** VOSR2 ([cswry/VOSR](https://github.com/cswry/VOSR), Apache-2.0) with DINOv2-L
  (Apache-2.0) and the Qwen-Image / Wan VAE (Apache-2.0); Real-ESRGAN and SPAN architectures (verified against
  [spandrel](https://github.com/chaiNNer-org/spandrel) outputs). PIL bicubic and OpenCV filters are written
  from the published algorithms.
- **Captioner / vision tower:** Qwen3.5-9B's vision model and interleaved M-RoPE, transcribed from
  transformers 5.8 (`Qwen3_5VisionModel`, `Qwen3_5Model.get_rope_index`, `apply_interleaved_mrope`;
  Apache-2.0, read for semantics). The recaption prompt is quoted verbatim from the MOSS-Image `USERGUIDE.md`
  (Apache-2.0).
- **Concept Sliders:** the training objective from Gandikota et al., "Concept Sliders: LoRA Adaptors for
  Precise Control in Diffusion Models" (2023), on MOSS's flow sampler.
- **LoKr / LoHa:** the LyCORIS adapter formats (Kronecker and Hadamard products).

## Algorithm references (no code copied)

| MossForge file | Reference | Notes |
|---|---|---|
| `MossForgeMLX/.../FluxVAE.swift`, `Sources/MossForgeEngine/MossLatentFormat.swift`, `Scripts/vae_ref.py` | ComfyUI `comfy/ldm/modules/diffusionmodules/model.py` (LDM Encoder/Decoder, originally CompVis latent-diffusion, MIT) and `comfy/ldm/models/autoencoder.py` (`batch_norm_latent`) | ComfyUI is **GPL-3.0**: it was read for the algorithm and constants only (eps 1e-4, `(c pi pj)` order, posterior mean, pad (0,1,0,1)); the Swift and the torch reference are independent implementations. Keep it that way — don't paste ComfyUI code here. |
| `MossForgeMLX/.../Qwen35Encoder.swift`, `Sources/MossForgeEngine/Qwen35TextConfig.swift`, `Scripts/encoder_ref.py` | ComfyUI `comfy/text_encoders/qwen35.py` + `llama.py`, `comfy/sd1_clip.py`; Qwen/Qwen3.5-9B `config.json` | Read for semantics only (GPL-3.0): (1+w) RMSNorms, gated q, partial split-half RoPE θ 1e7, chunked delta rule (torch ref), `layer="last"` = final-norm output, no BOS/padding. Independent implementations. |
| `MossForgeMLX/.../MossLoRAExtractor.swift`, `Sources/MossForgeEngine/MossLoRAExtraction.swift` | Halko, Martinsson & Tropp 2011, "Finding structure with randomness" (randomized range finder + power iterations); kohya-ss `networks/extract_lora_from_models.py` for the idea of LoRA-from-diff (alpha = rank, √σ split is our choice) | Algorithm only. Independent Swift/MLX. |


### Community methods (algorithm only, no code copied)

- **Edit LoRAs / reference tokens** — the clean-reference-tokens-on-the-unused-RoPE-axis, timestep-0 method as
  described by Ostris (AI Toolkit MOSS edit training; `ostris/ComfyUI-MOSS-Image-Ostris-Edit`, MIT, README only) and
  `thearkorbit/moss-image-edit-lora-poc` (Apache-2.0, model card). Dataset layout = AI Toolkit's (`control/`).
  Implemented independently in `MossDiT.references`, `MossRoPEIndex.referencePositions`, `EditDatasetScanner`.
- **Edit layouts** (2026-10-27) — ComfyUI PR #17296 "Support MOSS-Image edit LoRAs" (`comfy/ldm/moss_image/model.py`,
  `comfy_extras/nodes_moss_image.py`; ComfyUI is **GPL-3.0**, so read for semantics only, nothing copied): the four
  `EDIT_FORMATS`, isolated references with per-block cached K/V, the `f_offset` shift, "Picture N:" + ≤ 384² vision
  prompt. Written independently as `MossEditLayout`, `MossDiT.isolatedReferenceKV`, `MossRoPEIndex.referenceSequence`,
  `MossPipeline.referencePictures`. chinoll/Moss-Image-Edit (weights Apache-2.0; its project code is GPLv3 and was
  not used) and akhaliq/MOSS-Image-Edit-HQ (Apache-2.0) model cards for formats and settings.
- **LyCORIS LoHa** — `ΔW = (w1_a·w1_b) ⊙ (w2_a·w2_b)·α/r` from the LyCORIS format (comfyanonymous's hires LoRA
  headers), in `MossLoRAKeyMap` (`.hadamard`) and `MossDiT.attachLoRA` / `MossModelTools.fuseLoRA`.
- **Gradient checkpointing** — MLX Python's `mx.checkpoint` idea (custom function whose VJP recomputes), written
  against mlx-swift's `CustomFunction`/`vjp`.

## Copied from MIT-licensed code

| MossForge file | Source | Licence |
|---|---|---|
| `MossForgeMLX/Sources/MossForgeMLX/GatedDelta.swift` (Metal kernel body) | mlx-lm `mlx_lm/models/gated_delta.py` `_make_gated_delta_kernel` (scalar gate, no mask) | MIT, Copyright © 2025 Apple Inc. — keep the attribution comment |


## Downloaded (gitignored, under `models/`)

- `models/lora/kijai/moss_image_sft_lora_blocks_21_to_49_rank_512_bf16.safetensors` from
  https://huggingface.co/Kijai/MOSS-image_experimental (2026-10-18). **No licence tag — local testing only;
  never bundle or redistribute** until Kijai clarifies.

- `models/lora/lodestones-dmd/{time_embedding,dataspace}.safetensors, README.md` from
  https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd (2026-10-18), **Apache-2.0**.
  `Scripts/turbo_ref.py` followed that README + the demo Space's sampler loop (algorithm only). The SFT
  data-space files, `turbo_ref.py` and `turbo-parity` were removed 2026-10-31 (Turbo = akhaliq 4-step); the
  SFT repo's `lora_latest.safetensors` stays as `lora/lodestones/lodestones-8step-sft-latest.safetensors`.

- `models/text_encoders/qwen3.5-9b-tokenizer/{tokenizer.json,tokenizer_config.json,config.json,LICENSE}` from
  https://huggingface.co/Qwen/Qwen3.5-9B (2026-10-18). Licence: **Apache-2.0** (LICENSE file checked).

- `models/vosr/{VOSR2,dinov2-large,qwen_image_vae.safetensors}` and `models/esrgan/`: VOSR2 (hf CSWRY/VOSR), DINOv2-L (facebook/dinov2-large) and the
  Qwen-Image VAE are **Apache-2.0**; RealESRGAN ×2plus is **BSD-3-Clause**; other ESRGAN/SPAN files
  (UltraSharp, Remacri, ClearReality) carry their authors' licences — check before bundling.

