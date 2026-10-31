#!/usr/bin/env python3
# Torch reference for the MossForge captioner: Qwen3.5-9B's vision tower + multimodal greedy decode, built on
# encoder_ref.py (pure torch + safetensors + tokenizers + PIL; no transformers import). Transcribed from
# transformers 5.8 (read for semantics):
#   qwen2_vl/image_processing_qwen2_vl.py  smart_resize, rescale/normalize (mean = std = 0.5 per
#                                           Qwen/Qwen3.5-9B preprocessor_config.json), patch layout
#                                           (block-row, block-col, row, col) × (c, t, py, px)
#   qwen3_5/modeling_qwen3_5.py           Qwen3_5VisionModel (Conv3d patch embed, bilinear pos-embed
#                                           interpolation, 2-D vision RoPE θ 1e4, 27 LayerNorm blocks with
#                                           gelu-tanh MLP, merger LN → fc1 → GELU → fc2), get_rope_index
#                                           (image (s, s+row, s+col); text resumes at s + max(h, w)/2),
#                                           interleaved M-RoPE section [11, 11, 10]
# The resize is PIL bicubic (the slow processor's), which MossForge's PILResample reproduces exactly; the
# fast processor's torchvision antialiased bicubic differs by about a level. Each decode step recomputes the
# whole sequence (no cache), so this independently checks MossForge's cached multimodal prefill + decode.
#
#   python3 -I Scripts/caption_ref.py models/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
#       models/text_encoders/qwen3.5-9b-tokenizer/tokenizer.json docs/assets/app-layout-turbo-1024.png \
#       MossForgeMLX/Fixtures/caption [steps=8] [max_pixels=65536]
# then: cd MossForgeMLX && swift run -c release moss-mlx caption-parity ../models Fixtures/caption
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import encoder_ref as E  # noqa: E402

torch.set_grad_enabled(False)
V = "/model.visual."
VH, VHEADS, VDEPTH, PATCH, TPATCH, MERGE, GRID_SIDE = 1152, 16, 27, 16, 2, 2, 48
VHD = VH // VHEADS
MROPE = [11, 11, 10]
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")  # caption.json paths are repo-relative


def smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=1024 * 1024):
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def preprocess(path, max_pixels):
    img = Image.open(path).convert("RGB")
    rh, rw = smart_resize(img.height, img.width, max_pixels=max_pixels)
    img = img.resize((rw, rh), Image.BICUBIC)
    x = torch.tensor(list(img.getdata()), dtype=torch.uint8).view(rh, rw, 3).permute(2, 0, 1).float()
    x = (x / 255 - 0.5) / 0.5
    gh, gw = rh // PATCH, rw // PATCH
    p = x.reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH).permute(1, 4, 2, 5, 0, 3, 6)
    p = p.unsqueeze(5).expand(-1, -1, -1, -1, -1, TPATCH, -1, -1).reshape(gh * gw, 3 * TPATCH * PATCH * PATCH)
    return p.contiguous(), (1, gh, gw)


def vision(patches, grid):
    t, gh, gw = grid
    h = patches @ E.W(V + "patch_embed.proj.weight").reshape(VH, -1).T + E.W(V + "patch_embed.proj.bias")
    # bilinear pos-embed interpolation (fast_pos_embed_interpolate), then merge-block order
    hi, wi = torch.linspace(0, GRID_SIDE - 1, gh), torch.linspace(0, GRID_SIDE - 1, gw)
    hf, wf = hi.int(), wi.int()
    hc, wc = (hf + 1).clip(max=GRID_SIDE - 1), (wf + 1).clip(max=GRID_SIDE - 1)
    dh, dw = hi - hf, wi - wf
    pe = E.W(V + "pos_embed.weight")
    idx = [(hf[:, None] * GRID_SIDE + wf[None]), (hf[:, None] * GRID_SIDE + wc[None]),
           (hc[:, None] * GRID_SIDE + wf[None]), (hc[:, None] * GRID_SIDE + wc[None])]
    wts = [(1 - dh)[:, None] * (1 - dw)[None], (1 - dh)[:, None] * dw[None], dh[:, None] * (1 - dw)[None], dh[:, None] * dw[None]]
    pos = sum(pe[i.flatten()] * w.flatten()[:, None] for i, w in zip(idx, wts))
    pos = pos.view(gh // MERGE, MERGE, gw // MERGE, MERGE, VH).permute(0, 2, 1, 3, 4).reshape(-1, VH)
    h = h + pos
    # 2-D rope (rot_pos_emb): row/col in merge-block order, VHD/4 freqs each
    inv = 1.0 / (10000 ** (torch.arange(0, VHD // 2, 2).float() / (VHD // 2)))
    br, bc, ir, ic = torch.meshgrid(torch.arange(gh // MERGE), torch.arange(gw // MERGE), torch.arange(MERGE), torch.arange(MERGE), indexing="ij")
    rows, cols = (br * MERGE + ir).flatten(), (bc * MERGE + ic).flatten()
    rot = torch.cat([rows[:, None] * inv[None], cols[:, None] * inv[None]], -1)
    emb = torch.cat([rot, rot], -1)
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]

    def rh(x):
        a, b = x[..., : VHD // 2], x[..., VHD // 2 :]
        return torch.cat([-b, a], -1)

    S = h.shape[0]
    for i in range(VDEPTH):
        p = f"{V}blocks.{i}."
        x = F.layer_norm(h, (VH,), E.W(p + "norm1.weight"), E.W(p + "norm1.bias"), 1e-6)
        qkv = (x @ E.W(p + "attn.qkv.weight").T + E.W(p + "attn.qkv.bias")).view(S, 3, VHEADS, VHD).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = q * cos + rh(q) * sin, k * cos + rh(k) * sin
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        o = o[0].transpose(0, 1).reshape(S, VH)
        h = h + o @ E.W(p + "attn.proj.weight").T + E.W(p + "attn.proj.bias")
        x = F.layer_norm(h, (VH,), E.W(p + "norm2.weight"), E.W(p + "norm2.bias"), 1e-6)
        x = F.gelu(x @ E.W(p + "mlp.linear_fc1.weight").T + E.W(p + "mlp.linear_fc1.bias"), approximate="tanh")
        h = h + x @ E.W(p + "mlp.linear_fc2.weight").T + E.W(p + "mlp.linear_fc2.bias")
    last = h
    m = F.layer_norm(h, (VH,), E.W(V + "merger.norm.weight"), E.W(V + "merger.norm.bias"), 1e-6).view(-1, VH * 4)
    m = F.gelu(m @ E.W(V + "merger.linear_fc1.weight").T + E.W(V + "merger.linear_fc1.bias"))
    return last, m @ E.W(V + "merger.linear_fc2.weight").T + E.W(V + "merger.linear_fc2.bias")


def positions(ids, pad, grid):
    t_, h_, w_ = [], [], []
    pos, i = 0, 0
    _, gh, gw = grid
    while i < len(ids):
        if ids[i] == pad:
            for r in range(gh // MERGE):
                for c in range(gw // MERGE):
                    t_.append(pos), h_.append(pos + r), w_.append(pos + c)
            i += (gh // MERGE) * (gw // MERGE)
            pos += max(gh, gw) // MERGE
        else:
            t_.append(pos), h_.append(pos), w_.append(pos)
            pos, i = pos + 1, i + 1
    return torch.tensor([t_, h_, w_])


def lm(x, pos):
    """Multimodal Qwen3.5 text stack (encoder_ref layers) with interleaved M-RoPE."""
    inv = E.THETA ** (-torch.arange(0, E.ROT, 2).float() / E.ROT)
    freqs = pos.float()[:, :, None] * inv[None, None]  # [3, T, 32]
    ft = freqs[0].clone()
    for dim, off in ((1, 1), (2, 2)):
        ft[:, off : MROPE[dim] * 3 : 3] = freqs[dim, :, off : MROPE[dim] * 3 : 3]
    emb = torch.cat([ft, ft], -1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    x = x[None]
    for i in range(E.LAYERS):
        p = f"layers.{i}."
        h = E.rms(x, E.W(p + "input_layernorm.weight"))
        x = x + (E.attention(h, p + "self_attn.", cos, sin) if i in E.FULL else E.deltanet(h, p + "linear_attn."))
        h = E.rms(x, E.W(p + "post_attention_layernorm.weight"))
        x = x + (F.silu(h @ E.W(p + "mlp.gate_proj.weight").T) * (h @ E.W(p + "mlp.up_proj.weight").T)) @ E.W(
            p + "mlp.down_proj.weight"
        ).T
    return E.rms(x, E.W("norm.weight"))[0]


def main():
    TE, TOK, IMG, OUT = sys.argv[1:5]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    max_pixels = int(sys.argv[6]) if len(sys.argv) > 6 else 65536
    E.load(TE)
    E.MEMO = {}
    tok = Tokenizer.from_file(TOK)
    start = time.time()
    patches, grid = preprocess(IMG, max_pixels)
    last, image_tokens = vision(patches, grid)
    n = image_tokens.shape[0]
    print(f"vision: grid {grid}, {n} tokens, {time.time() - start:.0f}s", flush=True)

    guide = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Reference", "MOSS-Image", "USERGUIDE.md")
    with open(guide, encoding="utf-8") as fh:
        text = fh.read()
    body = text.split("#### Caption Prompt\n\n```text\n", 1)[1].split("\n```", 1)[0]
    instruction = body.replace("{{reference}}", "(none)")
    chat = ("<|im_start|>user\n" + instruction + "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = tok.encode(chat, add_special_tokens=False).ids
    pad = tok.token_to_id("<|image_pad|>")
    assert ids.count(pad) == n
    prompt_len = len(ids)
    head = E.W("/lm_head.weight")
    first_logits = None
    for step in range(steps):
        x = E.W("embed_tokens.weight", rows=torch.tensor(ids))
        s = ids.index(pad)
        x = torch.cat([x[:s], image_tokens, x[s + n :]])
        logits = lm(x, positions(ids, pad, grid))[-1] @ head.T
        if first_logits is None:
            first_logits = logits.clone()
        ids.append(int(logits.argmax()))
        print(f"step {step}: {tok.decode(ids[prompt_len:])!r} ({time.time() - start:.0f}s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    save_file({
        "patches": patches, "vision_last": last.contiguous(), "image_tokens": image_tokens.contiguous(),
        "prompt_ids": torch.tensor(ids[:prompt_len], dtype=torch.int32),
        "generated_ids": torch.tensor(ids[prompt_len:], dtype=torch.int32), "first_logits": first_logits.contiguous(),
    }, os.path.join(OUT, "caption.safetensors"))
    with open(os.path.join(OUT, "caption.json"), "w", encoding="utf-8") as fh:
        json.dump({"image": os.path.relpath(os.path.abspath(IMG), REPO), "max_pixels": max_pixels, "grid": list(grid),
                   "text": tok.decode(ids[prompt_len:])}, fh, ensure_ascii=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
