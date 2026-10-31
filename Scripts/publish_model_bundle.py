#!/usr/bin/env python3
"""Publish MossForge's model bundle: stage `models/` files listed in Scripts/model-bundle.tsv (group<TAB>path) as
APFS clones, write mossforge-manifest.json (sizes, SHA-256, groups) + README.md, and upload to a Hugging Face model
repo — what Models → Get Models… downloads (Settings → Folders → Model bundle). Needs `huggingface_hub` and a token.

  python3 -I Scripts/publish_model_bundle.py --repo <owner>/MossForge [--models models] [--stage /tmp/bundle] [--dry-run] [--prune]

--prune deletes repo files no longer listed (after the upload, in one commit; --dry-run just lists them). Use a fresh
--stage folder: everything in it is uploaded.
"""
import argparse, os, shutil, subprocess, sys

def stage(models, listing, out):
    os.makedirs(out, exist_ok=True)
    for line in open(listing):
        if not line.strip(): continue
        _, p = line.rstrip("\n").split("\t")
        dst = os.path.join(out, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            subprocess.run(["cp", "-c", os.path.join(models, p), dst], check=True)  # APFS clone: no extra disk

def write_manifest(root, listing):
    import json, hashlib
    groups = {
     "essentials": ("Essentials", "Text encoder, tokenizer, VAE and the 4-step Turbo SFT model (akhaliq, built in) — generate straight away.", True),
     "sft": ("MOSS-Image SFT (bf16)", "The full-quality 50-step model; also what the Turbo LoRAs and edit LoRAs run on.", False),
     "base": ("MOSS-Image Base (bf16)", "The pre-SFT model: more varied, better for training and some LoRAs.", False),
     "turbo-4step": ("Turbo LoRA for SFT (akhaliq, 4-step)", "MossForge's Turbo: stock Euler, 4 steps, CFG 1 — for the bf16 SFT model and finetunes.", False),
     "turbo-8step": ("8-step LoRA for SFT (lodestones lora_latest)", "The 8-step alternative at strength 0.7, stock Euler.", False),
     "turbo-base": ("Turbo LoRA for Base (lodestones, 8-step)", "Data-space DMD adapter bundle for the Base model.", False),
    }
    files = []
    for line in open(listing):
        g, p = line.rstrip("\n").split("\t")
        h = hashlib.sha256()
        with open(os.path.join(root, p), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
        files.append({"path": p, "group": g, "size": os.path.getsize(os.path.join(root, p)), "sha256": h.hexdigest()})
        print("hashed", p, flush=True)
    manifest = {"format": "mossforge-models", "version": 1,
      "groups": [{"id": k, "title": v[0], "summary": v[1], "default": v[2],
                  "size": sum(f["size"] for f in files if f["group"] == k)} for k, v in groups.items()],
      "files": files}
    json.dump(manifest, open(os.path.join(root, "mossforge-manifest.json"), "w"), indent=2)
    gb = lambda n: f"{n/1e9:.1f} GB"
    rows = "\n".join(f"| `{g['id']}` | {g['title']} | {gb(g['size'])} | {'yes' if g['default'] else 'optional'} |" for g in manifest["groups"])
    readme = f"""---
license: apache-2.0
tags:
- moss-image
- mossforge
- mlx
base_model:
- OpenMOSS-Team/MOSS-Image-SFT
- OpenMOSS-Team/MOSS-Image-Base
---

# MossForge model bundle

Everything [MossForge](https://huggingface.co/FiditeNemini) (a native macOS studio for
[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)) needs, laid out exactly as its
`models/` folder. MossForge's first-run setup reads `mossforge-manifest.json`, downloads the groups
you pick into place and checks every file's SHA-256. You can add more weights and LoRAs to `models/`
later, and MossForge picks them up.

This is a mirror of third-party weights, not new models. Every file keeps its original licence (all Apache-2.0).

| Group | Contents | Size | Default |
|---|---|---|---|
{rows}

## Layout

```
models/
├── diffusion_models/   DiTs (+ .mfmeta.json sidecars)
├── text_encoders/      qwen3.5_9b_int8_convrot.safetensors, qwen3.5-9b-tokenizer/
├── vae/                flux2-vae.safetensors
└── lora/<author>/      LoRAs with their supporting files and sidecars
```

## Provenance and licences

| File(s) | Source | Licence |
|---|---|---|
| `diffusion_models/moss_image_{{sft,base}}_bf16` | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) repack of [OpenMOSS-Team/MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | Apache-2.0 |
| `text_encoders/qwen3.5_9b_int8_convrot` | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) (Qwen3.5-9B, int8 ConvRot) | Apache-2.0 |
| `text_encoders/qwen3.5-9b-tokenizer/` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache-2.0 (`LICENSE` included) |
| `vae/flux2-vae` | FLUX.2 VAE (`ae.safetensors`) | Apache-2.0 |
| `lora/lodestones/lodestones-8step-sft-latest` | [lodestones/MOSS-Image-SFT-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) `lora_latest.safetensors` | Apache-2.0 |
| `lora/lodestones/lodestones-8step-base*` | [lodestones/MOSS-Image-Base-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd) | Apache-2.0 |
| `lora/akhaliq/akhaliq-4step-sft` | [akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) v2.0 | Apache-2.0 |
| `diffusion_models/moss_image_sft_akhaliq-4step-sft_int8_convrot*` | Made by MossForge: the akhaliq 4-step LoRA (v2.0) fused into SFT bf16 in fp32, then int8 ConvRot; runs on stock Euler, 4 steps, CFG 1 | Apache-2.0 |

MOSS-Image is by the OpenMOSS Team. This bundle is unofficial and not affiliated with or endorsed by OpenMOSS.
"""
    open(os.path.join(root, "README.md"), "w").write(readme)
    print("manifest + README written", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--repo", required=True)
    a.add_argument("--models", default="models")
    a.add_argument("--list", default=os.path.join(os.path.dirname(__file__), "model-bundle.tsv"))
    a.add_argument("--stage", default="/tmp/mossforge-bundle")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--prune", action="store_true", help="delete repo files that are no longer listed")
    args = a.parse_args()
    stage(args.models, args.list, args.stage)
    write_manifest(args.stage, args.list)
    from huggingface_hub import HfApi
    api = HfApi()
    keep = {l.rstrip("\n").split("\t")[1] for l in open(args.list) if l.strip()} | {"README.md", "mossforge-manifest.json", ".gitattributes"}
    stale = []
    if args.prune and api.repo_exists(args.repo):
        stale = sorted(f for f in api.list_repo_files(args.repo) if f not in keep)
        for f in stale: print("stale:", f)
    if args.dry_run: sys.exit(0)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_large_folder(repo_id=args.repo, folder_path=args.stage, repo_type="model", num_workers=4)
    if stale:
        api.delete_files(args.repo, delete_patterns=stale, commit_message="Remove files no longer in the bundle")
        print("removed", len(stale), "stale file(s)")
    print("published", args.repo)
