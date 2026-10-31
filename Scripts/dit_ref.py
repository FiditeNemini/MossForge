#!/usr/bin/env python3
# Standalone torch reference for the MOSS-Image DiT forward, used to verify the MLX port.
# Transcribed from Reference/MOSS-Image/comfyui/moss-image/model.py (Apache-2.0, OpenMOSS) with the
# ComfyUI `operations` swapped for plain torch modules and `optimized_attention` for SDPA. The
# Comfy-Org file's fused `proj.qkv` is split q|k|v (verified bit-exact against the official split
# transformer, 2026-10-18). Pure torch + safetensors; F32 on CPU.
#
#   python3 -I Scripts/dit_ref.py models/diffusion_models/moss_image_sft_bf16.safetensors \
#       MossForgeMLX/Fixtures/encoder/short.safetensors MossForgeMLX/Fixtures/dit
# then: cd MossForgeMLX && swift run -c release moss-mlx dit-parity <dit> Fixtures/dit
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

GH, GW = 16, 24  # token grid (256×384 px), non-square on purpose
SIGMA = 0.7
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 8)

sd = {}
LORA = {}  # module ("layers.N.proj.q") -> list of (down [r,in], up [out,r], scale), applied unmerged in F32


def attach_lora(path, strength=1.0):
    """kohya `diffusion_model.<m>.lora_down/lora_up/alpha` → y += strength·(alpha/r)·up(down(x)), unmerged."""
    t = load_file(path)
    for key in t:
        if key.endswith(".lora_down.weight"):
            raw = key[: -len(".lora_down.weight")]
            down, up = t[key].float(), t[raw + ".lora_up.weight"].float()
            alpha = float(t[raw + ".alpha"]) if raw + ".alpha" in t else down.shape[0]
            LORA.setdefault(raw.removeprefix("diffusion_model."), []).append((down, up, strength * alpha / down.shape[0]))


def override(path):
    """Replace parameters with F32 tensors (lodestones time_embedding.safetensors)."""
    for k, v in load_file(path).items():
        assert sd[k].shape == v.shape, k
        sd[k] = v.float()


def load(path):
    """Load the DiT as F32, splitting the Comfy-Org fused qkv into q|k|v."""
    sd.clear()
    sd.update({k: v.float() for k, v in load_file(path).items()})
    for i in range(42):
        qkv = sd.pop(f"layers.{i}.proj.qkv.weight", None)
        if qkv is not None:
            sd[f"layers.{i}.proj.q.weight"], sd[f"layers.{i}.proj.k.weight"], sd[f"layers.{i}.proj.v.weight"] = qkv.chunk(3, 0)
HEADS, DIM = 24, 3072


def lin(x, p, bias=True):
    y = F.linear(x, sd[p + ".weight"], sd.get(p + ".bias") if bias else None)
    for down, up, scale in LORA.get(p, ()):
        y = y + scale * F.linear(F.linear(x, down), up)
    return y


def rms(x, p):
    return F.rms_norm(x, (x.shape[-1],), sd[p + ".weight"], 1e-6)


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q, k, v, rope):
    shape = (*q.shape[:2], HEADS, q.shape[-1] // HEADS)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)


def text_block(x, rope, p):
    q, k, v = lin(rms(x, p + ".norm1"), p + ".qkv").chunk(3, -1)
    q = rms(q.unflatten(-1, (HEADS, -1)), p + ".q_norm").flatten(2)
    k = rms(k.unflatten(-1, (HEADS, -1)), p + ".k_norm").flatten(2)
    x = x + lin(attention(q, k, v, rope), p + ".proj")
    y = rms(x, p + ".norm2")
    return x + lin(situ(lin(y, p + ".ffn_up"), lin(y, p + ".ffn_gate")), p + ".ffn_down")


def block(x, rope, mod, p):
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.unbind(1)
    y = rms(x, p + ".norm1") * (1 + scale_a[:, None]) + shift_a[:, None]
    q = rms(lin(y, p + ".proj.q", False).unflatten(-1, (HEADS, -1)), p + ".q_norm").flatten(2)
    k = rms(lin(y, p + ".proj.k", False).unflatten(-1, (HEADS, -1)), p + ".k_norm").flatten(2)
    x = x + lin(attention(q, k, lin(y, p + ".proj.v", False), rope), p + ".proj.o", False) * gate_a[:, None]
    y = rms(x, p + ".norm2") * (1 + scale_f[:, None]) + shift_f[:, None]
    hidden, gate = lin(y, p + ".proj.ffn_up_gate", False).chunk(2, -1)
    return x + lin(situ(hidden, gate), p + ".proj.ffn_down", False) * gate_f[:, None]


def rope_table(height, width, text_length, head_dim=128, base=10000.0):
    n = head_dim // 8
    bands = 1 / (base ** (torch.arange(n, dtype=torch.float32) / n))
    h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4)
    text[:, 1] = torch.arange(text_length)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def forward(latent, timestep, context, taps, skip=()):
    _, _, height, width = latent.shape
    half = 128
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = torch.tensor([timestep], dtype=torch.float64)[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    t = lin(F.silu(lin(embedding, "time_embedding.0")), "time_embedding.2")
    mod = lin(t, "shared_block_modulation").view(1, 6, -1)
    head = lin(t, "video_head_modulation").view(1, 2, -1)
    video = lin(latent.flatten(2).transpose(1, 2), "video_embedder")
    rope = rope_table(height, width, context.shape[1])
    text = lin(context, "text_embedder.proj")
    for b in range(2):
        text = text_block(text, rope[:, height * width :], f"text_embedder.blocks.{b}")
    taps["text"] = text[0]
    x = torch.cat((video, text), 1)
    for i in range(42):
        if i in skip:  # pipeline_moss_image.py: skipped blocks are left out entirely
            continue
        x = block(x, rope, mod, f"layers.{i}")
        if i in (0, 21):
            taps[f"layer{i}"] = x[0]
    x = rms(x[:, : height * width], "final_norm_video")
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = lin(x, "final_linear_video", False)
    taps["mod"] = mod[0]
    return x.transpose(1, 2).reshape(1, -1, height, width)




def main():
    DIT, CONTEXT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    load(DIT)
    context = load_file(CONTEXT)["hidden"].float()[None]  # real Qwen3.5 states [1, 33, 4096]
    g = torch.Generator().manual_seed(0)
    latent = torch.randn(1, 128, GH, GW, generator=g)
    timestep = 1000 * SIGMA
    start = time.time()
    taps = {}
    velocity = forward(latent, timestep, context, taps)
    os.makedirs(OUT, exist_ok=True)
    save_file(
        {
            "latent": latent.contiguous(),
            "context": context[0].contiguous(),
            "timestep": torch.tensor([timestep]),
            "velocity": velocity.contiguous(),
            **{k: v.contiguous() for k, v in taps.items()},
        },
        os.path.join(OUT, "dit.safetensors"),
    )
    print(
        f"saved {OUT}/dit.safetensors: grid {GH}x{GW}, text {context.shape[1]}, σ {SIGMA}; "
        f"velocity std {velocity.std():.4f} ({time.time() - start:.1f}s)"
    )


if __name__ == "__main__":
    main()
