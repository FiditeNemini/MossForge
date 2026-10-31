#!/usr/bin/env python3
# Standalone torch reference for the MOSS-Image text encoder: Qwen3.5-9B text stack, final-norm
# hidden states for the chat-wrapped prompt. Pure torch + safetensors + `tokenizers` (no comfy /
# transformers import). The algorithm is ComfyUI's, transcribed (read for semantics only — GPL):
#   comfy/text_encoders/qwen35.py   GatedDeltaNet (causal depthwise conv k=4 + SiLU, l2-norm q/k,
#                                   q·Dk^-½, CHUNKED gated delta rule, gated RMSNorm), GatedAttention
#                                   (q|gate per head, (1+w) q/k RMSNorm, partial split-half RoPE θ=1e7
#                                   on 64/256 dims, GQA 16/4, causal, ·σ(gate)), SwiGLU MLP, (1+w) norms
#   comfy/text_encoders/llama.py    Llama2_.forward: causal mask, final norm → layer="last"
#   comfyui/moss-image/text_encoder.py  template "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n",
#                                   no BOS/EOS/padding, truncate to 4096 tokens
# Weights: the Comfy-Org int8·convrot file, dequantized here as W = (i8 · scale[out,1]) @ H_g
# (regular-H4 Hadamard, g from the per-tensor comfy_quant blob).
#
#   python3 -I Scripts/encoder_ref.py models/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
#       models/text_encoders/qwen3.5-9b-tokenizer/tokenizer.json MossForgeMLX/Fixtures/encoder
# then: cd MossForgeMLX && swift run -c release moss-mlx encoder-parity <te> <tokenizerDir> Fixtures/encoder
import json
import math
import os
import struct
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

PREFIX = "model.language_model."
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 8)

# --- weights -------------------------------------------------------------------------------
header, f = None, None
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def load(te):
    """Open the Comfy-Org int8·cr Qwen3.5 file (weights are dequantized lazily per use)."""
    global header, f
    with open(te, "rb") as fh:
        header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    f = safe_open(te, "pt")


def hadamard(g):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    m = h4
    while m.shape[0] < g:
        m = torch.kron(m, h4)
    return (m / math.sqrt(g)).float()


def group_size(name):
    blob = name[: -len(".weight")] + ".comfy_quant"
    if blob not in header:
        return 0
    return json.loads(bytes(f.get_tensor(blob).tolist()).decode()).get("convrot_groupsize", 0)


MEMO = None  # set to {} to keep dequantized weights (generation recomputes the whole sequence per step)


def W(name, rows=None):
    """Dequantized F32 weight (optionally only some rows, for the embedding gather)."""
    if MEMO is not None and rows is None:
        if name not in MEMO:
            MEMO[name] = _W(name)
        return MEMO[name]
    return _W(name, rows)


def _W(name, rows=None):
    full = name[1:] if name.startswith("/") else PREFIX + name  # "/lm_head.weight" = outside the LM prefix
    t = f.get_tensor(full)
    if t.dtype != torch.int8:
        return (t if rows is None else t[rows]).float()
    s = f.get_tensor(full + "_scale").float()
    if rows is not None:
        t, s = t[rows], s[rows]
    w = t.float() * s
    g = group_size(full)
    if g > 1:
        o, i = w.shape
        w = (w.reshape(o, i // g, g) @ hadamard(g)).reshape(o, i)
    return w


# --- config (Qwen3.5-9B) -----------------------------------------------------------------------
HIDDEN, LAYERS, EPS = 4096, 32, 1e-6
HEADS, KV, HD, ROT, THETA = 16, 4, 256, 64, 1e7
KH, VH, KD, VD, CONV = 16, 32, 128, 128, 4
FULL = {i for i in range(LAYERS) if (i + 1) % 4 == 0}


def rms(x, w, add=True):
    return F.rms_norm(x, (x.shape[-1],), (w + 1) if add else w, EPS)


def chunk_gated_delta_rule(q, k, v, g, beta, chunk=64):
    # Transcribed from comfy qwen35.torch_chunk_gated_delta_rule (initial_state None).
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    q = q * Dk**-0.5
    v_beta, k_beta = v * beta[..., None], k * beta[..., None]
    q, k, v, k_beta, v_beta = (x.reshape(B, H, -1, chunk, x.shape[-1]) for x in (q, k, v, k_beta, v_beta))
    g = g.reshape(B, H, -1, chunk).cumsum(-1)
    mask = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), 0)
    decay = ((g[..., :, None] - g[..., None, :]).tril().exp()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(mask, 0)
    for i in range(1, chunk):
        row, sub = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
    attn = attn + torch.eye(chunk)
    v = attn @ v_beta
    k_cum = attn @ (k_beta * g.exp()[..., None])
    state = torch.zeros(B, H, Dk, Dv)
    out = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), 1)
    for i in range(v.shape[2]):
        qi, ki, vi = q[:, :, i], k[:, :, i], v[:, :, i]
        a = (qi @ ki.transpose(-1, -2) * decay[:, :, i]).masked_fill(mask, 0)
        v_new = vi - k_cum[:, :, i] @ state
        out[:, :, i] = (qi * g[:, :, i, :, None].exp()) @ state + a @ v_new
        state = state * g[:, :, i, -1, None, None].exp() + (
            ki * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    return out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2)


def deltanet(x, p):
    B, T, _ = x.shape
    proj = x @ W(p + "in_proj_qkv.weight").T
    z = x @ W(p + "in_proj_z.weight").T
    b = x @ W(p + "in_proj_b.weight").T
    a = x @ W(p + "in_proj_a.weight").T
    conv_w = W(p + "conv1d.weight")  # [C, 1, K]
    mixed = F.silu(F.conv1d(proj.transpose(1, 2), conv_w, groups=conv_w.shape[0], padding=CONV - 1)[:, :, :T])
    q, k, v = mixed.transpose(1, 2).split([KH * KD, KH * KD, VH * VD], -1)
    beta = b.sigmoid()
    g = -W(p + "A_log").exp() * F.softplus(a + W(p + "dt_bias"))
    q = q.reshape(B, T, KH, KD).repeat_interleave(VH // KH, 2)
    k = k.reshape(B, T, KH, KD).repeat_interleave(VH // KH, 2)
    v = v.reshape(B, T, VH, VD)
    o = chunk_gated_delta_rule(q, k, v, g, beta)
    o = rms(o.reshape(-1, VD), W(p + "norm.weight"), add=False) * F.silu(z.reshape(-1, VD))
    return o.reshape(B, T, -1) @ W(p + "out_proj.weight").T


def rope(x, cos, sin):  # split-half on the first ROT dims
    xr, xp = x[..., :ROT], x[..., ROT:]
    a, b = xr[..., : ROT // 2], xr[..., ROT // 2 :]
    return torch.cat([xr * cos + torch.cat([-b, a], -1) * sin, xp], -1)


def attention(x, p, cos, sin):
    B, T, _ = x.shape
    qg = (x @ W(p + "q_proj.weight").T).view(B, T, HEADS, 2 * HD)
    q, gate = qg[..., :HD], qg[..., HD:].reshape(B, T, -1)
    k = (x @ W(p + "k_proj.weight").T).view(B, T, KV, HD)
    v = (x @ W(p + "v_proj.weight").T).view(B, T, KV, HD).transpose(1, 2)
    q = rope(rms(q, W(p + "q_norm.weight")).transpose(1, 2), cos, sin)
    k = rope(rms(k, W(p + "k_norm.weight")).transpose(1, 2), cos, sin)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    o = o.transpose(1, 2).reshape(B, T, -1) * gate.sigmoid()
    return o @ W(p + "o_proj.weight").T


def encode(ids, taps=(0, 3)):
    T = len(ids)
    x = W("embed_tokens.weight", rows=torch.tensor(ids))[None]
    inv = THETA ** (-torch.arange(0, ROT, 2).float() / ROT)
    freqs = torch.arange(T).float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    tapped = {}
    for i in range(LAYERS):
        p = f"layers.{i}."
        h = rms(x, W(p + "input_layernorm.weight"))
        x = x + (attention(h, p + "self_attn.", cos, sin) if i in FULL else deltanet(h, p + "linear_attn."))
        h = rms(x, W(p + "post_attention_layernorm.weight"))
        x = x + (F.silu(h @ W(p + "mlp.gate_proj.weight").T) * (h @ W(p + "mlp.up_proj.weight").T)) @ W(
            p + "mlp.down_proj.weight"
        ).T
        if i in taps:
            tapped[f"layer{i}"] = x[0].clone()
    return rms(x, W("norm.weight"))[0], tapped


# --- prompts --------------------------------------------------------------------------------


def main():
    TE, TOK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    load(TE)
    tok = Tokenizer.from_file(TOK)
    short ="A red fox in fresh snow at dawn, poster with the title \"EXPLORE\" in bold serif letters."
    layout = json.dumps(
        {
            "summary": {"meta": ["poster"], "canvas": "single", "caption": "a travel poster, 中文标题 与 English text"},
            "items": [
                {"label": "text", "caption": "headline", "text": "EXPLORE\nTHE NORTH", "bbox": [-400, -300, -200, 300]},
                {"label": "object", "caption": "a red fox sitting on a rock", "bbox": [-100, -250, 400, 250]},
            ],
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        },
        ensure_ascii=False,
    )
    paragraphs = []
    n = 0
    while len(tok.encode(TEMPLATE.format("\n\n".join(paragraphs)), add_special_tokens=False).ids) < 4300:
        paragraphs.append(
            f"Section {n}: a detailed scene description number {n * 37 % 101}, with lanterns, rivers and "
            f"mountains; 第{n}段，灯笼、河流与山脉。Prices ${n}.{n % 10}9, ratio {n}:{n + 3}. " + (layout if n % 5 == 0 else "")
        )
        n += 1
    long = "\n\n".join(paragraphs)  # > 4096 tokens once wrapped ⇒ exercises truncation

    os.makedirs(OUT, exist_ok=True)
    for name, prompt in (("short", short), ("long", long)):
        text = TEMPLATE.format(prompt)
        ids = tok.encode(text, add_special_tokens=False).ids
        assert ids == tok.encode(text).ids, "tokenizer post-processor adds tokens"
        ids = ids[:4096]
        start = time.time()
        hidden, taps = encode(ids, taps=(0, 3) if name == "short" else ())
        tensors = {"ids": torch.tensor(ids, dtype=torch.int32), "hidden": hidden.contiguous(), **taps}
        save_file(tensors, os.path.join(OUT, f"{name}.safetensors"))
        with open(os.path.join(OUT, f"{name}.txt"), "w", encoding="utf-8") as fh:
            fh.write(prompt)
        print(
            f"{name}: {len(ids)} tokens, hidden {list(hidden.shape)} mean {hidden.mean():.4f} std {hidden.std():.4f} "
            f"({time.time() - start:.1f}s)"
        )


if __name__ == "__main__":
    main()
