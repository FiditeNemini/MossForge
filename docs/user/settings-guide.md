# Choosing settings

What each Generate control does, what to start with, and why. Numbers come from the MOSS-Image user
guide and from measurements on an M2 Ultra; your Mac may be slower or faster.

## Quick answer

- **Just want a good image fast:** Profile *Fast — 4 steps* (the default): SFT + the akhaliq 4-step Turbo LoRA,
  CFG 1, 1 MP. About 6–7 s at 512², well under a minute at 1024².
- **An 8-step alternative:** *8 steps — lodestones* (SFT + lodestones `lora_latest` at 0.7, flow sampler). On
  text-heavy prompts akhaliq 4-step matched or beat it in our comparison, and it's faster.
- **Previews while you iterate on a prompt:** *Draft — fastest* (akhaliq 2-step, exactly 2 steps), then switch up
  with **Lock seed** on to keep the composition.
- **Maximum fidelity, slow:** *Full quality — 50 steps* (Turbo off): 50 steps, CFG 3, shift 5, block 22 skipped in
  the negative branch, the recommended negative prompt. About 3¾ minutes for 1024².

## Model: SFT or Base

**SFT** is the post-trained model: better looking images and better instruction following — use it for generating.
**Base** is the raw pretrained model; MOSS recommends it as the starting point for **training LoRAs**. A LoRA trained
on Base suits Base; one trained on SFT only suits SFT. The Model picker shows each DiT's family, and the Turbo and
LoRA menus only offer files marked for that family.

## Turbo on or off

A Turbo is a few-step distillation LoRA. It runs at **CFG 1**, so there is no negative prompt and no CFG, shift or
SLG to tune — the distillation already baked guidance in. That is why Turbo is several times faster (1024²: about
40–66 s against 225 s): fewer steps, and one model pass per step instead of two (conditional + unconditional).

| Turbo | Family | Steps | Notes |
|---|---|---|---|
| akhaliq 2-step | SFT | exactly 2 | rough drafts |
| akhaliq 4-step | SFT | 4 | the default: sharp, correct text; 6–8 steps above 1024² |
| lodestones lora_latest | SFT | 8 | the 8-step alternative (strength 0.7, flow sampler) |
| lodestones 8-step Base | Base | 8 | Turbo for the Base family |

Turn Turbo **off** for the most faithful text rendering and fine detail, or to use a negative prompt.

A model with a Turbo **fused in** (Models → Tools → Fuse) shows *Turbo: built in* and needs no LoRA. For a
few-step model that came without MossForge's recipe file, pick **None — model is already 8-step, CFG 1**: it
runs the model at 8 steps and CFG 1 with no Turbo LoRA stacked on top. Don't pick it for a normal SFT or Base
model — those need 50 steps or a Turbo LoRA.

## Steps

With Turbo, use the recipe's steps (the slider defaults to it); more steps don't help a distilled model and the
2-step LoRA needs exactly 2. Without Turbo, use **50 or more** — the official default is 50; fewer steps lose detail
and text accuracy.

## CFG (guidance)

Only without Turbo. **3** is the official default; the useful range is **3–6**. Raise it for **short prompts**
(the model has less to go on and benefits from a stronger push), keep it at 3 for long, detailed captions — very high
CFG tends to over-saturate and stiffen images.

## Shift

Only without Turbo. **5** (the official value — keep it). Shift spends more of the sampling steps at high noise,
where the overall composition is decided; the model was tuned and evaluated with 5.

## Skip layer guidance (block 22)

Only without Turbo. On by default, as in the official pipeline: the negative (unconditional) pass skips DiT block 22.
Leave it on unless you want to compare against plain CFG.

## Negative prompt

Only without Turbo. The recommended one is `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`. Keep it
short — it is a list of things to steer away from, not a second description.

## Size

**Pixels** sets the budget (1, 1.5 or 2 MP are the trained sizes; 0.25–4 MP work but extrapolate) and **Ratio**
the shape; the app picks the nearest dimensions that are multiples of 16. Turbo LoRAs are trained at **1 MP**, so
stay there with Turbo. Time grows a little faster than the pixel count; memory too. For a large print, generate at
1 MP and **Upscale…** in the Gallery instead.

## Seed

Each image gets a random seed, shown under the preview and saved in its sidecar. **Lock seed** reuses it, so you can
change one thing (a word, the steps, a LoRA's strength) and see only that change.

## Prompts and text

Write in natural language (English works best; Chinese is supported) or tags. Put visible text **literally, in
quotes**, and keep its line breaks — MOSS renders the text you give it. **Enhance** rewrites a short prompt into
the detailed caption style MOSS was trained on (Qwen3.5, on device); **Layout from prompt** drafts text and object
boxes on the layout canvas. Long, specific captions generally beat short ones.

### Editing the model's instructions (Settings → Prompts)

**Settings → Prompts** shows the instructions MossForge gives the on-device Qwen3.5 model, and lets you edit them:
the prompt enhancer's rules, its caption and layout answer formats (`{width}` / `{height}` become the canvas),
the captioner (`{{reference}}` becomes an existing caption, or `(none)`) and the concept designer behind
Suggest on the Train screen. Edits save as you type and apply to the app — the local server and the CLIs keep the built-in prompts.
An edited prompt is marked *edited*; **Reset** restores one, **Reset all** every one, and a prompt left blank uses
the built-in. Only your edits are kept, so a prompt you never touched picks up improvements in later versions.

To tune one, edit it and press **Try it**: it runs the prompt on the **Test input** under the editor and shows the
answer and how long it took. The test input starts as a sample — an idea to enhance, or an "X to Y" request —
and you can change it to whatever you're testing (**Reset input** brings the sample back). For the captioner, choose
the image (the newest gallery image by default) and optionally type an existing caption to stand in for
`{{reference}}`. The answer is an enhanced caption, the layout JSON, the three captions, or four labelled lines. The
enhancer and captioner answers must stay valid JSON in the shape the format asks for; if not, Try it shows the error
(after one automatic retry). Try it needs the MLX backend and queues behind any running GPU job.

## Layout

Use **Layout** when where things go matters (posters, slides, labels). Draw a **text box** for each piece of
text and type its exact text; draw **object boxes** for subjects and give each a short caption. Keep boxes inside
the canvas and avoid heavy overlaps. The overall caption still describes the whole image.

## LoRAs and strength

Add a LoRA from the menu (only ones for the model's family are listed); strength **1** is its trained effect,
0.5–0.8 is gentler, negative values push away from it. The **blocks** field limits a LoRA to some DiT blocks
(`21-41`) or weights them (`0-20=0,21-41=0.5`). Known recipe: Kijai's SFT LoRA on blocks 21–49 at **−0.5** reduces
the faint vertical banding SFT can show.

A LoRA starts at its **default strength**, and its strength control spans the **range** set for it in the Models
inspector (−2 to 2 unless changed). Widen it for LoRAs made for bigger weights, e.g. −10 to 10; ones made on the
Train screen carry their recommended range. Type in the box beside the control to go past either end.

## Memory and speed

The DiT (about 10 GB in bf16, 5.4 GB in int8·convrot) and VAE stay loaded between images. **Settings → Memory**
decides what else stays:

- **Automatic** (the default) chooses from your Mac's GPU memory: on Macs with about 96 GB or more the text
  encoder (Qwen3.5-9B, about 16 GB) stays loaded too, saving roughly 2.5 s per image; on 128 GB Macs and up
  Enhance, Caption, Help and VOSR2 also stay loaded beside the DiT, so switching between them reloads nothing.
  A 64 GB Mac keeps the *Save memory* rules.
- **Keep models loaded** does all of that whatever the Mac — only pick it if you have the memory.
- **Save memory** loads the text encoder per image and keeps one big model at a time: Enhance, Caption and Help
  and VOSR2 unload the DiT first, and the next image reloads it.

The text encoder, Enhance, Caption and Help are one Qwen3.5 model loaded once and shared (about 19 GB with its
vision part), so going from *Layout from prompt* to Generate, or from Caption to Help, doesn't load it again.

The line under the picker says what your choice means on this Mac. **Unload models after N idle minutes** gives all
of it back when you stop, whichever you pick. On M1/M2 Macs (no bf16 hardware) the DiT runs in fp16, which is also
the more accurate choice.

**If generation is slow:** use a Turbo profile (4 or 8 steps) and stay at 1 MP; the first image after launch — or,
under *Save memory*, after Enhance, Caption, Help or an upscale — is slower because models have to load again;
pick *Keep models loaded* and set idle unload to *Never* if you have the memory; and remember the app runs one GPU job at a time, so a caption batch, a
training run or a request to the local server makes Generate wait in the queue. Lots of memory pressure (other big
apps open) also slows every step.
