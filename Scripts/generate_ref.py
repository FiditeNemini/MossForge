#!/usr/bin/env python3
# Torch reference for Qwen3.5-9B greedy generation (the on-device prompt enhancer's decoder), built on
# encoder_ref.py: every step recomputes the WHOLE sequence (no cache) and takes argmax of lm_head over
# the final-norm state of the last token — an independent check of MossForge's cached prefill/decode
# (KV cache on attention layers, conv + recurrent state on DeltaNet layers). Short on purpose.
#
#   python3 -I Scripts/generate_ref.py models/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
#       models/text_encoders/qwen3.5-9b-tokenizer/tokenizer.json MossForgeMLX/Fixtures/generate [16]
# then: cd MossForgeMLX && swift run -c release moss-mlx generate-parity ../models Fixtures/generate
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import encoder_ref  # noqa: E402

TE, TOK, OUT = sys.argv[1:4]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
CHAT = ("<|im_start|>user\nWrite a one-sentence image caption for a red fox sitting in fresh snow at dawn.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
torch.set_grad_enabled(False)

tok = Tokenizer.from_file(TOK)
encoder_ref.load(TE)
encoder_ref.MEMO = {}
head = encoder_ref.W("/lm_head.weight")
ids = tok.encode(CHAT, add_special_tokens=False).ids
prompt_len = len(ids)
start = time.time()
first_logits = None
for step in range(STEPS):
    hidden, _ = encoder_ref.encode(ids, taps=())
    logits = hidden[-1] @ head.T
    if first_logits is None:
        first_logits = logits.clone()
    ids.append(int(logits.argmax()))
    print(f"step {step}: {tok.decode(ids[prompt_len:])!r} ({time.time() - start:.0f}s)", flush=True)
os.makedirs(OUT, exist_ok=True)
save_file({"prompt_ids": torch.tensor(ids[:prompt_len], dtype=torch.int32),
           "generated_ids": torch.tensor(ids[prompt_len:], dtype=torch.int32),
           "first_logits": first_logits.contiguous()}, os.path.join(OUT, "generate.safetensors"))
with open(os.path.join(OUT, "generate.json"), "w", encoding="utf-8") as fh:
    json.dump({"chat": CHAT, "text": tok.decode(ids[prompt_len:])}, fh, ensure_ascii=False)
print("saved", OUT)
