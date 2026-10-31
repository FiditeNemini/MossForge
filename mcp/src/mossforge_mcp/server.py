"""``mossforge-mcp`` — exposes MOSS-Image generation to MCP clients (LM Studio, Claude Desktop, …).

Submit-a-job / poll-with-progress / hand-back-the-id design; MOSS tools and parameters (layout JSON,
variants, SLG, per-block LoRA strength, enhance_prompt), ``upscale_image``,
``caption_image``, and ``edit_image`` (an edit LoRA, no task presets).

It forwards to the resident server (``moss-mlx serve``), which owns the models and the GPU.

Environment:
    MOSSFORGE_URL           server URL (default http://127.0.0.1:7860)
    MOSSFORGE_OUTPUT_DIR    where PNGs are saved (default ~/Pictures/MossForge/mcp)
    MOSSFORGE_DEFAULT_SIZE  the long side when the caller gives no size (default 1024)
    MOSSFORGE_WAIT_SECONDS  how long one tool call waits before handing back a job id (default 45)
    MOSSFORGE_IMAGE_DIRS    folders ``upscale_image`` / ``caption_image`` / ``edit_image`` may read
                            ``image_path`` from, separated by ``os.pathsep`` (default ~/Pictures;
                            the output dir always counts)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, Image, MCPServer
from pydantic import BaseModel, Field

from .client import (
    DEFAULT_URL,
    MossForgeClient,
    MossForgeError,
    caption_body,
    edit_body,
    generation_body,
    upscale_body,
)

mcp = MCPServer("mossforge")

Reporter = Callable[[float, float | None, str | None], Awaitable[None]]


def _client() -> MossForgeClient:
    return MossForgeClient(os.environ.get("MOSSFORGE_URL", DEFAULT_URL))


def _output_dir() -> Path:
    default = Path.home() / "Pictures" / "MossForge" / "mcp"
    return Path(os.environ.get("MOSSFORGE_OUTPUT_DIR", default)).expanduser()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def size_for(aspect_ratio: str, long_side: int) -> tuple[int, int]:
    """Width × height for ``"W:H"`` with the longer side ``long_side``, both snapped to 16 (≥ 256)."""
    try:
        aw, ah = (float(p) for p in aspect_ratio.split(":"))
    except ValueError:
        raise ValueError(f"aspect_ratio must look like 16:9, got {aspect_ratio!r}") from None
    if aw <= 0 or ah <= 0:
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio!r}")

    def snap(v: float) -> int:
        return max(256, min(4096, round(v / 16) * 16))

    if aw >= ah:
        return snap(long_side), snap(long_side * ah / aw)
    return snap(long_side * aw / ah), snap(long_side)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # base64 of this stays under the server's 64 MB body limit


def _image_dirs() -> list[Path]:
    raw = os.environ.get("MOSSFORGE_IMAGE_DIRS")
    dirs = [Path(d).expanduser() for d in raw.split(os.pathsep) if d] if raw else [Path.home() / "Pictures"]
    return [_output_dir(), *dirs]


def read_allowed_image(path: str) -> bytes:
    """The bytes of a PNG/JPEG inside one of the allowed folders (``MOSSFORGE_IMAGE_DIRS`` +
    the output dir). The calling model chooses the path, so nothing else is readable."""
    resolved = Path(path).expanduser().resolve()
    roots = [d.resolve() for d in _image_dirs()]
    if not any(resolved.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(r) for r in roots)
        raise MossForgeError(f"{path} is outside the folders upscale_image may read ({allowed})")
    if resolved.suffix.lower() not in IMAGE_SUFFIXES or not resolved.is_file():
        raise MossForgeError(f"{path} is not a .png/.jpg file")
    if resolved.stat().st_size > MAX_UPLOAD_BYTES:
        raise MossForgeError(f"{path} is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return resolved.read_bytes()


async def wait_for_job(
    client: MossForgeClient, job_id: str, wait_seconds: float, report: Reporter, poll_seconds: float = 1.5
) -> dict[str, Any]:
    """Poll a job until it finishes or ``wait_seconds`` pass, reporting stage/step as progress."""
    deadline = time.monotonic() + wait_seconds
    last_stage = None
    while True:
        status = await asyncio.to_thread(client.job, job_id)
        if status["state"] in ("done", "failed"):
            return status
        stage = f"{status['stage']} · {status['elapsed_s']}s"
        if stage != last_stage:
            await report(float(status.get("step", 0)), float(status.get("total_steps") or 0) or None, stage)
            last_stage = stage
        if time.monotonic() >= deadline:
            return status
        await asyncio.sleep(poll_seconds)


async def _finish(client: MossForgeClient, job_id: str, status: dict[str, Any]) -> list[Any]:
    upscale = status.get("kind") == "upscale"
    if status["state"] == "failed":
        return [f"MossForge error — {status.get('error', 'generation failed')}"]
    if status["state"] != "done":
        verb = "Upscaling" if upscale else "Still generating"
        return [
            f"{verb} (job {job_id}: {status['stage']}, step {status.get('step', 0)}/"
            f"{status.get('total_steps', '?')}, {status.get('elapsed_s', 0)}s so far). "
            f'Call get_generation with job_id="{job_id}" to collect the image.'
        ]
    result = await asyncio.to_thread(client.job_image, job_id)
    if result.png is None:
        return [f"No image rendered ({result.note or 'no image'}); seed {result.seed}."]
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    # One file per job: a retried get_generation reuses it instead of writing another copy.
    existing = sorted(out.glob(f"mossforge-*-{job_id}.png"))
    prefix = {"upscale": "mossforge-upscale", "edit": "mossforge-edit"}.get(
        str(status.get("kind")), "mossforge"
    )
    path = existing[0] if existing else out / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{job_id}.png"
    if not existing:
        path.write_bytes(result.png)
    caption = (
        f"Saved to {path} ({status.get('width')}×{status.get('height')}, seed {result.seed}, "
        f"{status.get('elapsed_s')}s; job {job_id})."
    )
    if upscale:
        # Upscales are large (4K+ PNGs): hand back the path, not megabytes of base64.
        return [caption + " Not shown inline because of its size."]
    return [Image(data=result.png, format="png"), caption]


@mcp.tool()
def mossforge_status() -> dict[str, Any]:
    """Where the MossForge server is: backend, queue depth, the running job's stage and recent jobs."""
    return _client().status()


@mcp.tool()
def list_models() -> list[dict[str, Any]]:
    """The local checkpoints the server can see (id = path relative to its models folder,
    kind = dit/<precision> | text_encoder | vae | lora | dmd_sampler)."""
    return _client().models()


@mcp.tool()
def list_loras() -> list[dict[str, Any]]:
    """The LoRAs under the models folder: id (pass as a ``loras`` entry's path) and rank."""
    return _client().loras()


@mcp.tool()
def generation_options() -> dict[str, Any]:
    """What generate_image accepts on this server: variants with their defaults (steps, guidance,
    shift), size/step/guidance ranges, SLG and LoRA block syntax — and what upscale_image accepts
    (methods, installed ESRGAN files, softening, colour fix, defaults). Call it once when unsure."""
    return _client().options()


@mcp.tool()
def enhance_prompt(
    prompt: Annotated[str, Field(description="A short idea to expand.")],
    layout: Annotated[
        bool, Field(description="Return a JSON layout prompt (boxes for text and objects) instead of prose.")
    ] = False,
    width: Annotated[
        int, Field(ge=256, le=4096, multiple_of=16, description="Canvas width for layout boxes.")
    ] = 1024,
    height: Annotated[int, Field(ge=256, le=4096, multiple_of=16, description="Canvas height.")] = 1024,
) -> str:
    """Expand a short request into a detailed MOSS-Image prompt, or (layout=true) a JSON layout prompt
    to pass to generate_image as ``layout``."""
    try:
        return _client().enhance(prompt, layout=layout, width=width, height=height)
    except MossForgeError as error:
        return f"MossForge error — {error}"


class LoRA(BaseModel):
    path: str = Field(
        description='A list_loras id ("lora/style.safetensors") or a filename under models/lora.'
    )
    strength: float = Field(default=1.0, ge=-5.0, le=10.0, description="Strength; negative subtracts.")
    blocks: str | None = Field(
        default=None,
        description='Per-block strength over the 42 DiT blocks: "21-41" applies only there; '
        '"0-20=0,21-41=0.5,other=0" weights ranges (unlisted blocks 1). E.g. Kijai\'s SFT fix: '
        "the SFT LoRA at strength -0.5, blocks 21-41.",
    )


Size = Annotated[int | None, Field(ge=256, le=4096, multiple_of=16)]


# structured_output=False: these return MCP content (an Image + text).
@mcp.tool(structured_output=False)
async def generate_image(
    ctx: Context,
    prompt: Annotated[
        str | None,
        Field(
            description="What to draw. MOSS-Image renders dense text well — write visible text literally, "
            "in quotes, with explicit line breaks. English or Chinese. Omit when giving layout."
        ),
    ] = None,
    layout: Annotated[
        dict[str, Any] | None,
        Field(
            description='A JSON layout prompt instead of prose: {"summary": {"meta": [...], "canvas": '
            '"single", "caption": "..."}, "items": [{"label": "text"|"object", "caption": "...", '
            '"text": "...", "bbox": [y1, x1, y2, x2]}], "bbox_coord_space": "abs_centralized", '
            '"bbox_format": "yxyx"} — boxes in pixels with the origin at the canvas centre.'
        ),
    ] = None,
    variant: Annotated[
        Literal["turbo", "sft", "base"] | None,
        Field(
            description='"turbo" (default here): SFT + akhaliq 4-step, no CFG — ~25 s at 1024². "sft": the '
            'release model, 50 steps + CFG 3 (~5 min), honours negative_prompt. "base": pretrain, for LoRAs.'
        ),
    ] = None,
    aspect_ratio: Annotated[
        str | None,
        Field(description='"W:H", e.g. "1:1", "3:2", "16:9"; ignored when width and height are given.'),
    ] = None,
    long_side: Annotated[
        int | None, Field(ge=256, le=4096, description="Long side in px for aspect_ratio (default 1024).")
    ] = None,
    width: Annotated[Size, Field(description="Exact width, a multiple of 16.")] = None,
    height: Annotated[Size, Field(description="Exact height, a multiple of 16.")] = None,
    steps: Annotated[
        int | None,
        Field(ge=1, le=200, description="Turbo 1–8 (default 4; 6–8 above 1024²); SFT/Base ≥ 50 recommended."),
    ] = None,
    guidance_scale: Annotated[
        float | None, Field(ge=1, le=30, description="CFG, SFT/Base only (default 3; 3–6 useful).")
    ] = None,
    negative_prompt: Annotated[str | None, Field(description="What to avoid. SFT/Base only.")] = None,
    shift: Annotated[
        float | None, Field(ge=0.1, le=20, description="Flow shift (default 5). SFT/Base.")
    ] = None,
    slg: Annotated[
        Literal["official", "off"] | None,
        Field(
            description='"official" (default): skip block 22 in the negative branch; "off": more photoreal.'
        ),
    ] = None,
    seed: Annotated[
        int | None, Field(ge=0, description="Fix for reproducible results; omit for random.")
    ] = None,
    loras: Annotated[list[LoRA] | None, Field(description="LoRAs to apply (see list_loras).")] = None,
) -> list[Any]:
    """Generate an image locally with MOSS-Image on the MossForge server.

    Returns the image plus the path it was saved to, its seed and job id. If generation outlasts the
    wait window, returns a job id instead — then call get_generation.
    """
    if (prompt is None) == (layout is None):
        return ["MossForge error — give exactly one of prompt or layout."]
    if width is None or height is None:
        try:
            w, h = size_for(aspect_ratio or "1:1", long_side or _env_int("MOSSFORGE_DEFAULT_SIZE", 1024))
        except ValueError as error:
            return [f"MossForge error — {error}"]
        width, height = width or w, height or h
    body = generation_body(
        prompt,
        layout=layout,
        variant=variant or "turbo",
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        shift=shift,
        slg=slg,
        loras=[lora.model_dump(exclude_none=True) for lora in loras] if loras else None,
    )
    return await _submit_and_wait(lambda client: client.submit(body), ctx)


async def _submit_and_wait(submit: Callable[[MossForgeClient], str], ctx: Context) -> list[Any]:
    client = _client()
    try:
        job_id = await asyncio.to_thread(submit, client)
        status = await wait_for_job(
            client, job_id, _env_int("MOSSFORGE_WAIT_SECONDS", 45), ctx.report_progress
        )
        return await _finish(client, job_id, status)
    except MossForgeError as error:
        return [f"MossForge error — {error}"]


@mcp.tool(structured_output=False)
async def upscale_image(
    ctx: Context,
    job_id: Annotated[
        str | None,
        Field(description="Upscale the image of an earlier generate_image / upscale_image job (its job id)."),
    ] = None,
    image_path: Annotated[
        str | None,
        Field(
            description="Or a local .png/.jpg — only inside MOSSFORGE_IMAGE_DIRS (default ~/Pictures) "
            "or the bridge's output folder."
        ),
    ] = None,
    index: Annotated[int | None, Field(ge=0, le=3, description="Which image of job_id (default 0).")] = None,
    scale: Annotated[
        float, Field(ge=1, le=8, description="Size factor (default 2). Output is capped at 8192 px per side.")
    ] = 2,
    method: Annotated[
        Literal["vosr", "esrgan"],
        Field(
            description='"vosr" (default): VOSR2 generative upscaler — rebuilds real texture (skin, '
            'fabric, foliage); slow (about a minute for 1024→2048). "esrgan": Real-ESRGAN family — seconds, '
            "sharp, invents little."
        ),
    ] = "vosr",
    esrgan_model: Annotated[
        str | None,
        Field(
            description="ESRGAN filename from generation_options' upscale.esrgan_models "
            "(default: the server's)."
        ),
    ] = None,
    softening: Annotated[
        Literal["off", "light", "medium", "strong"] | None,
        Field(
            description="VOSR pre-filter that removes sampler grain VOSR would misread as texture "
            "(default medium). Raise for smooth skin/portraits, lower for already-clean photos."
        ),
    ] = None,
    color_fix: Annotated[
        Literal["wavelet", "adain", "none"] | None,
        Field(description="How colours are matched back to the input (default wavelet)."),
    ] = None,
    seed: Annotated[
        int | None, Field(ge=0, description="VOSR noise seed (default 42; fixed = repeatable).")
    ] = None,
    tile_overlap: Annotated[
        int | None,
        Field(ge=64, le=384, description="VOSR tile overlap in px (default 256); raise if tile seams show."),
    ] = None,
) -> list[Any]:
    """Enlarge an image with the MossForge server's upscalers (VOSR2 or ESRGAN).

    Give job_id (best — e.g. right after generate_image) or image_path. The result is saved
    as a PNG and its path returned (not inline: upscales are large). VOSR usually outlasts
    the wait window — then call get_generation with the returned job id.
    """
    if (job_id is None) == (image_path is None):
        return ["MossForge error — give exactly one of job_id or image_path."]
    try:
        image = await asyncio.to_thread(read_allowed_image, image_path) if image_path else None
    except MossForgeError as error:
        return [f"MossForge error — {error}"]
    body = upscale_body(
        image_png=image,
        job_id=job_id,
        index=index,
        scale=scale,
        method=method,
        esrgan_model=esrgan_model,
        softening=softening,
        color_fix=color_fix,
        seed=seed,
        tile_overlap=tile_overlap,
    )
    return await _submit_and_wait(lambda client: client.submit_upscale(body), ctx)


@mcp.tool(structured_output=False)
async def edit_image(
    instruction: Annotated[
        str,
        Field(
            description='What to change, as an instruction: "make it night", "give her a red scarf". '
            "Not a description of the whole image."
        ),
    ],
    loras: Annotated[
        list[LoRA],
        Field(
            description="The edit LoRA (required — e.g. one trained in MossForge's Train → Edit; see "
            "list_loras), plus any style LoRAs."
        ),
    ],
    ctx: Context,
    job_id: Annotated[
        str | None,
        Field(description="Edit the image of an earlier generate_image / edit_image job (its id)."),
    ] = None,
    image_path: Annotated[
        str | None,
        Field(
            description="Or a local .png/.jpg inside MOSSFORGE_IMAGE_DIRS (default ~/Pictures) "
            "or the bridge's output folder."
        ),
    ] = None,
    index: Annotated[int | None, Field(ge=0, le=3, description="Which image of job_id (default 0).")] = None,
    variant: Annotated[
        Literal["turbo", "sft", "base"] | None,
        Field(
            description='"turbo" (default here, 4 steps, no CFG), or "sft" / "base" — match the '
            "LoRA's family."
        ),
    ] = None,
    width: Annotated[Size, Field(description="Output width (default: follows the source image).")] = None,
    height: Annotated[Size, Field(description="Output height (default: follows the source image).")] = None,
    steps: Annotated[
        int | None, Field(ge=1, le=200, description="Default: the variant's (4 on Turbo).")
    ] = None,
    seed: Annotated[int | None, Field(ge=0, description="Fix for reproducible results.")] = None,
    edit_layout: Annotated[
        Literal["auto", "mossForge", "ai-toolkit", "ai-toolkit (kv_cache off)", "Moss-Image-Edit"] | None,
        Field(
            description='How the edit LoRA sees the source. "auto" (default) reads the LoRA\'s metadata; '
            '"Moss-Image-Edit" for chinoll\'s LoRA, "ai-toolkit" for ai-toolkit LoRAs (e.g. armV5), '
            '"mossForge" for LoRAs trained in MossForge.'
        ),
    ] = None,
) -> list[Any]:
    """Edit an image with a MOSS-Image edit LoRA on the MossForge server (experimental: the source
    image is a clean reference the LoRA learned to edit). Give job_id (best — right after
    generate_image) or image_path, an instruction and the edit LoRA. The edited image is returned
    and saved."""
    if (job_id is None) == (image_path is None):
        return ["MossForge error — give exactly one of job_id or image_path."]
    try:
        image = await asyncio.to_thread(read_allowed_image, image_path) if image_path else None
        body = edit_body(
            instruction,
            image_png=image,
            job_id=job_id,
            index=index,
            variant=variant or "turbo",
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            loras=[lora.model_dump(exclude_none=True) for lora in loras],
            edit_layout=edit_layout,
        )
    except (MossForgeError, ValueError) as error:
        return [f"MossForge error — {error}"]
    return await _submit_and_wait(lambda client: client.submit_edit(body), ctx)


@mcp.tool(structured_output=False)
async def caption_image(
    job_id: Annotated[
        str | None, Field(description="Caption the image of an earlier generate_image / upscale_image job.")
    ] = None,
    image_path: Annotated[
        str | None,
        Field(
            description="Or a local .png/.jpg — only inside MOSSFORGE_IMAGE_DIRS (default ~/Pictures) "
            "or the bridge's output folder."
        ),
    ] = None,
    detail: Annotated[
        Literal["long", "medium", "short"],
        Field(description="Which caption to lead with (all three are returned); long = training detail."),
    ] = "long",
    reference: Annotated[
        str | None, Field(description="An existing caption for the model to check and correct.")
    ] = None,
) -> list[Any]:
    """Describe an image with MossForge's on-device VLM (Qwen3.5-9B) using MOSS-Image's official
    recaption prompt: long / medium / short captions (visible text preserved verbatim), watermark
    boxes and borders. Useful for dataset captions or to turn an image into a MOSS prompt.
    Takes ~15–40 s (the model loads on first use)."""
    if (job_id is None) == (image_path is None):
        return ["MossForge error — give exactly one of job_id or image_path."]
    try:
        image = await asyncio.to_thread(read_allowed_image, image_path) if image_path else None
        body = caption_body(image_png=image, job_id=job_id, detail=detail, reference=reference)
        result = await asyncio.to_thread(_client().caption, body)
    except MossForgeError as error:
        return [f"MossForge error — {error}"]
    return [f"{result.get('caption', '')}\n\n" + json.dumps(result, ensure_ascii=False, indent=2)]


@mcp.tool(structured_output=False)
async def get_generation(job_id: str, ctx: Context) -> list[Any]:
    """Collect a generation or upscale that was still running when its tool returned."""
    client = _client()
    try:
        status = await wait_for_job(
            client, job_id, _env_int("MOSSFORGE_WAIT_SECONDS", 45), ctx.report_progress
        )
        return await _finish(client, job_id, status)
    except MossForgeError as error:
        return [f"MossForge error — {error}"]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
