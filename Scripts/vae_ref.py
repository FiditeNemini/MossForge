#!/usr/bin/env python3
# Standalone torch reference for the FLUX.2 VAE (LDM AutoencoderKL, BFL key names) plus the MOSS
# latent bridge, used to numerically verify the MLX port. Pure torch + safetensors (no comfy /
# diffusers / transformers import) — the algorithm is ComfyUI's, transcribed:
#   comfy/ldm/modules/diffusionmodules/model.py  Encoder / Decoder / ResnetBlock / AttnBlock /
#                                                Upsample (nearest ×2 + conv) / Downsample (pad 0,1,0,1 + s2 conv)
#   comfy/ldm/models/autoencoder.py              quant_conv → posterior mode → 2×2 patchify → batch_norm (eps 1e-4)
#   comfy/sd.py VAE                              pixels [0,1] → ×2−1 ; decode → (x+1)/2 clamp
#
# Regenerate the committed fixture (float32, CPU, deterministic):
#   python3 -I Scripts/vae_ref.py models/vae/flux2-vae.safetensors MossForgeMLX/Fixtures/vae
# then verify the Swift port:
#   cd MossForgeMLX && swift run -c release moss-mlx vae-parity ../models/vae/flux2-vae.safetensors Fixtures/vae
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sd = {}


def load(path):
    sd.clear()
    sd.update({k: v.float() for k, v in load_file(path).items() if v.is_floating_point()})
torch.set_grad_enabled(False)


def gn(x, p):  # GroupNorm(32, eps=1e-6, affine)
    return F.group_norm(x, 32, sd[p + ".weight"], sd[p + ".bias"], eps=1e-6)


def conv(x, p, stride=1, padding=1):
    return F.conv2d(x, sd[p + ".weight"], sd[p + ".bias"], stride=stride, padding=padding)


def resblock(x, p):
    h = conv(F.silu(gn(x, p + ".norm1")), p + ".conv1")
    h = conv(F.silu(gn(h, p + ".norm2")), p + ".conv2")
    if p + ".nin_shortcut.weight" in sd:
        x = conv(x, p + ".nin_shortcut", padding=0)
    return x + h


def attn(x, p):  # single head over h·w, scale c^-0.5
    b, c, h, w = x.shape
    y = gn(x, p + ".norm")
    q, k, v = (conv(y, f"{p}.{n}", padding=0).reshape(b, 1, c, h * w).transpose(-1, -2) for n in "qkv")
    o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w)
    return x + conv(o, p + ".proj_out", padding=0)


def levels(prefix):
    n = 0
    while f"{prefix}.{n}.block.0.conv1.weight" in sd:
        n += 1
    return n


def blocks(prefix):
    n = 0
    while f"{prefix}.block.{n}.conv1.weight" in sd:
        n += 1
    return n


def encode(x):
    h = conv(x, "encoder.conv_in")
    L = levels("encoder.down")
    for i in range(L):
        for b in range(blocks(f"encoder.down.{i}")):
            h = resblock(h, f"encoder.down.{i}.block.{b}")
        if i != L - 1:
            h = conv(F.pad(h, (0, 1, 0, 1)), f"encoder.down.{i}.downsample.conv", stride=2, padding=0)
    h = resblock(h, "encoder.mid.block_1")
    h = attn(h, "encoder.mid.attn_1")
    h = resblock(h, "encoder.mid.block_2")
    h = conv(F.silu(gn(h, "encoder.norm_out")), "encoder.conv_out")
    moments = conv(h, "encoder.quant_conv", padding=0)
    return moments[:, : moments.shape[1] // 2]  # DiagonalGaussian mode = mean


def decode(z):
    h = conv(z, "decoder.post_quant_conv", padding=0)
    h = conv(h, "decoder.conv_in")
    h = resblock(h, "decoder.mid.block_1")
    h = attn(h, "decoder.mid.attn_1")
    h = resblock(h, "decoder.mid.block_2")
    for i in reversed(range(levels("decoder.up"))):
        for b in range(blocks(f"decoder.up.{i}")):
            h = resblock(h, f"decoder.up.{i}.block.{b}")
        if i != 0:
            h = conv(F.interpolate(h, scale_factor=2.0, mode="nearest"), f"decoder.up.{i}.upsample.conv")
    return conv(F.silu(gn(h, "decoder.norm_out")), "decoder.conv_out")


def patchify(z):  # "... c (i pi) (j pj) -> ... (c pi pj) i j"
    b, c, h, w = z.shape
    return z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify(z):  # "... (c pi pj) i j -> ... c (i pi) (j pj)"
    b, c4, h, w = z.shape
    return z.reshape(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c4 // 4, h * 2, w * 2)


EPS = 1e-4


def bn():
    return sd["bn.running_mean"].view(1, -1, 1, 1), sd["bn.running_var"].view(1, -1, 1, 1)


def decode_tokens(tokens):
    """Normalised DiT tokens [1,128,h,w] → pixels [1,3,16h,16w] in [0,1] (pipeline: clamp(−1,1) then (x+1)/2)."""
    mean, var = bn()
    return (decode(unpatchify(tokens * torch.sqrt(var + EPS) + mean)).clamp(-1, 1) + 1) / 2



def main():
    VAE, OUT = sys.argv[1], sys.argv[2]
    H, W = (int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "128x192").split("x"))
    load(VAE)
    mean, var = bn()
    # A deterministic, structured test image in [0,1]: gradients + rings + a little noise, so both
    # low and high frequencies are exercised. Non-square on purpose (catches h/w swaps).
    g = torch.Generator().manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    r = (xx**2 + yy**2).sqrt()
    image = torch.stack([(xx + 1) / 2, (torch.sin(r * 12) + 1) / 2, (yy + 1) / 2])[None]
    image = (image * 0.9 + 0.1 * torch.rand(image.shape, generator=g)).clamp(0, 1)

    x = image * 2 - 1
    z = encode(x)  # [1, 32, H/8, W/8]
    tokens = (patchify(z) - mean) / torch.sqrt(var + EPS)  # [1, 128, H/16, W/16] — what the DiT sees
    z_back = unpatchify(tokens * torch.sqrt(var + EPS) + mean)
    decoded = ((decode(z_back) + 1) / 2).clamp(0, 1)  # [1, 3, H, W]

    # Decoder-only check from an independent latent: unit-normal tokens (the DiT's output domain).
    noise_tokens = torch.randn(tokens.shape, generator=g)
    noise_decoded = decode(unpatchify(noise_tokens * torch.sqrt(var + EPS) + mean))  # raw [-1,1]-ish, unclamped

    os.makedirs(OUT, exist_ok=True)
    save_file(
        {
            "image": image.contiguous(),
            "vae_mean": z.contiguous(),
            "tokens": tokens.contiguous(),
            "decoded": decoded.contiguous(),
            "noise_tokens": noise_tokens.contiguous(),
            "noise_decoded": noise_decoded.contiguous(),
        },
        os.path.join(OUT, "vae.safetensors"),
    )
    psnr = -10 * math.log10(F.mse_loss(decoded, image).item())
    print(f"saved {OUT}/vae.safetensors  image {list(image.shape)} tokens {list(tokens.shape)}  round-trip PSNR {psnr:.2f} dB")
    print(f"tokens mean {tokens.mean():.4f} std {tokens.std():.4f}  (BN-normalised ⇒ ≈0 / ≈1 on natural images)")


if __name__ == "__main__":
    main()
