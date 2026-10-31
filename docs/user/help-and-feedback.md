# Help, sharing and feedback

## Sharing and saving images

Right-click an image — the result on **Generate** (in Prompt or Layout mode) or any image in the **Gallery** — for:

- **Share…** — the macOS share menu (Mail, Messages, AirDrop, Photos, Notes, …). Nothing is shared until you pick a
  service.
- **Save a Copy…** — save the PNG anywhere under a readable name (the prompt's first words + seed). With several
  Gallery images selected, **Save N Copies…** asks for a folder; existing files are never replaced — copies get a number.
- **Copy Image** — paste into another app (an image editor gets the pixels, Finder gets the file).

Both menus also have **Edit in Layout…** (redraw part of the image — see editing.md) and **Use as Edit
Reference** (for edit LoRAs). Generate's right-click menu also has **Show in Gallery** and **Reveal in Finder**; the share and save buttons sit next
to the seed under the preview, and in the Gallery detail pane beside Reveal in Finder. Shared and saved copies are the
plain PNG — the prompt and settings stay in the Gallery's JSON sidecar, which is not included.

## Ask (the Help assistant)

**Help → Ask** answers questions about MossForge — how to do something, how it works, which settings to use and
why — using the documentation built into the app. It runs on your Mac with the same Qwen3.5 model as Enhance and
Caption; your questions never leave the computer.

- Every answer lists its **sources** (the documentation sections it was given); click one to read it. Numbers like
  [1] in the answer point at them.
- It only knows what the documentation says. If something isn't covered, it should say so rather than guess — check
  the sources when an answer matters.
- Follow-up questions ("and for Base?") use the conversation so far. **New chat** starts over; chats are not saved.
- The first question loads Qwen3.5 (about 18 GB; the first answer takes roughly 10–15 s on an M2 Ultra); the next
  ones are faster. Generating an image unloads it again.
- "How do I…" questions are answered from the user guides; ask about the internals ("how does the Turbo sampler
  work under the hood?", "what is the slider training objective?") and it draws on the **technical reference**
  (the DiT, encoder, VAE, samplers, LoRA formats, quantisation, editing, upscaling and training maths).
- Without the language model (models missing), Ask still finds and shows the most relevant documentation sections.

## Diagnostics and feedback (off by default)

Diagnostics are **off** until you turn them on in **Settings → Diagnostics & feedback**. While off, MossForge records
nothing and the Feedback tab only explains how to enable it. Turn them on when you want to report a problem or an idea.

**Private by design:**

- **Nothing is ever sent by MossForge.** A report is text you read, edit and then share yourself (copy, save as a
  Markdown file, or the share menu) — or don't.
- **What is collected:** the app version and backend; macOS version, chip, memory and free disk space; which model
  components are present (names, sizes, kinds) and whether they look right; non-personal settings (theme, LAN access,
  server port, idle unload); the GPU's state (device, memory in use, which models are loaded); and, while diagnostics
  are on, a short in-memory log of job outcomes and errors (for example "generate failed: out of memory").
- **What is never collected:** your prompts, images, captions, gallery file names, or anything outside MossForge's
  own folders.
- **Redaction:** folder paths become `<models>`, `<output>` or `~`; your account and computer names, e-mail and IP
  addresses, hardware addresses and unique IDs become placeholders. This runs before the assistant sees anything and
  again on the final text, including what you type. The preview shows how many details were replaced.
- **Your choice of sections:** each section has a switch. The activity log and model file names start **off**.
- **Memory only:** the activity log lives in memory, is limited to the last 200 events and disappears when you quit or
  press **Clear log**, or when you turn diagnostics off.

**Writing a report** (Help → Feedback):

1. Pick what it is (bug, performance, idea, question) and describe it in your own words.
2. **Run diagnostics** collects the sections above and shows each check as ✓ (fine), ⚠︎ (worth a look) or ✗
   (a problem, such as a missing tokenizer).
3. **Draft with AI** (optional) has Qwen3.5 turn your description and the redacted diagnostics into a clear report —
   summary, what happened, likely cause, suggested next steps — citing the documentation when there's a known fix.
   It runs on device; it is told never to invent details, but read the draft before using it.
4. Edit the text, then **Copy**, **Save…** or **Share…**.
