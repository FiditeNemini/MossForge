#!/usr/bin/env python3
# End-to-end torch reference for MOSS-Image generation (TODO Phase 1 step 6), composed from the
# verified standalone references: encoder_ref (Qwen3.5 text stack), dit_ref (DiT forward + block
# skipping), vae_ref (BN-denormalise + unpatchify + VAE decode). The sampling loop follows OpenMOSS
# `pipeline_moss_image.py` exactly: the snapped training-grid σ schedule, fp32 noise in token space,
# per step `p = u + cfg·(c − u)` with the negative branch run with block 22 skipped, Euler update,
# then decode with clamp(−1, 1). Small and short on purpose (256×384, 6 steps) — the point is the
# composition, not image quality.
#
#   python3 -I Scripts/e2e_ref.py models/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
#       models/text_encoders/qwen3.5-9b-tokenizer/tokenizer.json \
#       models/diffusion_models/moss_image_sft_bf16.safetensors models/vae/flux2-vae.safetensors \
#       MossForgeMLX/Fixtures/e2e
# then: cd MossForgeMLX && swift run -c release moss-mlx e2e-parity ../models Fixtures/e2e
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling refs (-I drops the script dir)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import dit_ref  # noqa: E402
import encoder_ref  # noqa: E402
import vae_ref  # noqa: E402

TE, TOK, DIT, VAE, OUT = sys.argv[1:6]
GH, GW, STEPS, SHIFT, CFG, UNCOND_SKIP = 16, 24, 6, 5.0, 3.0, {22}
PROMPT = "A lighthouse on a rocky cliff at sunset, poster with the title \"NORTH LIGHT\" in bold letters."
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
torch.set_grad_enabled(False)


def pipeline_sigmas(n, shift, total=1000):  # MOSSImageScheduler.set_timesteps, verbatim math
    def grid(count, s):
        lower = s / (total + s - 1)
        v = torch.linspace(1, lower, count)
        return s * v / (1 + (s - 1) * v)

    sigmas = grid(n, shift)
    training = grid(total, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.cat([torch.unique_consecutive(training[nearest]), torch.zeros(1)])


start = time.time()
tok = Tokenizer.from_file(TOK)
encoder_ref.load(TE)
ids = {name: tok.encode(encoder_ref.TEMPLATE.format(p), add_special_tokens=False).ids[:4096] for name, p in (("pos", PROMPT), ("neg", NEGATIVE))}
ctx = {name: encoder_ref.encode(i, taps=())[0] for name, i in ids.items()}
print(f"encoded {[len(i) for i in ids.values()]} tokens ({time.time() - start:.1f}s)")

dit_ref.load(DIT)
sigmas = pipeline_sigmas(STEPS, SHIFT)
noise = torch.randn(1, 128, GH, GW, generator=torch.Generator().manual_seed(1234))
x = noise.clone()
for i in range(len(sigmas) - 1):
    t = float(sigmas[i]) * 1000
    cond = dit_ref.forward(x, t, ctx["pos"][None], {})
    uncond = dit_ref.forward(x, t, ctx["neg"][None], {}, skip=UNCOND_SKIP)
    x = x + (sigmas[i + 1] - sigmas[i]) * (uncond + CFG * (cond - uncond))
print(f"sampled {len(sigmas) - 1} steps σ {[round(float(s), 3) for s in sigmas]} ({time.time() - start:.1f}s)")

vae_ref.load(VAE)
image = vae_ref.decode_tokens(x)
os.makedirs(OUT, exist_ok=True)
save_file(
    {
        "noise": noise.contiguous(),
        "sigmas": sigmas.contiguous(),
        "ids_pos": torch.tensor(ids["pos"], dtype=torch.int32),
        "ids_neg": torch.tensor(ids["neg"], dtype=torch.int32),
        "context_pos": ctx["pos"].contiguous(),
        "context_neg": ctx["neg"].contiguous(),
        "latent": x.contiguous(),
        "image": image.contiguous(),
    },
    os.path.join(OUT, "e2e.safetensors"),
)
with open(os.path.join(OUT, "e2e.json"), "w", encoding="utf-8") as fh:
    import json

    json.dump({"prompt": PROMPT, "negative": NEGATIVE, "steps": STEPS, "shift": SHIFT, "cfg": CFG,
               "uncond_skip": sorted(UNCOND_SKIP), "width": GW * 16, "height": GH * 16}, fh, ensure_ascii=False, indent=1)
print(f"saved {OUT}/e2e.safetensors: image {list(image.shape)} ({time.time() - start:.1f}s)")
