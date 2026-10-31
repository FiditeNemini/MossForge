# Licensing

- **MossForge source**: Apache-2.0 (`LICENSE`).
- **MOSS-Image code and weights** (OpenMOSS-Team): **Apache-2.0**. There are no field-of-use, revenue or redistribution restrictions. Obligations when we
  redistribute or bundle weights: include the Apache-2.0 licence and any NOTICE, and mark modified
  files (e.g. our own quantized exports) as changed.
- **Components to verify before bundling in a shipped product**: the Qwen3.5-9B text encoder
  (Qwen, expected Apache-2.0) and the FLUX.2 VAE file (taken from FLUX.2-klein-4B — confirm that
  exact file's licence; FLUX.2 [dev] itself is non-commercial).
- **Community weights**: lodestones / toxicdog / akhaliq Turbo work is Apache-2.0; Kijai's LoRAs carry
  no licence tag — don't redistribute.
- **Upscalers** (user-downloaded): VOSR2, DINOv2-L and the Qwen-Image VAE are Apache-2.0; RealESRGAN ×2plus is
  BSD-3-Clause; community ESRGAN/SPAN models (UltraSharp, Remacri, ClearReality) — check each before bundling.
- Weights are never committed (`models/` is gitignored) — for size, and so users fetch them from source.

Per-file provenance: `NOTES.md`.
