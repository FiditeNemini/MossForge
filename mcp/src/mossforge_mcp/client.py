"""A thin, typed client for the MossForge resident server's HTTP API.

MOSS routes (`/v1/moss/*`) and fields (variant sft|base|turbo, SLG, per-block LoRA strength, layout).

The server is local-first (loopback unless the user enables LAN access), so the default base URL is
``http://127.0.0.1:7860``. Generation is serialized server-side, so a request may wait behind others.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:7860"
GENERATE_TIMEOUT = 900.0


class MossForgeError(RuntimeError):
    """The server answered with an error (its OpenAI-shaped ``error.message``)."""


@dataclass(frozen=True)
class GeneratedImage:
    png: bytes | None
    seed: int | None
    note: str | None = None


def generation_body(
    prompt: str | None = None,
    *,
    layout: dict[str, Any] | None = None,
    variant: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    guidance_scale: float | None = None,
    shift: float | None = None,
    slg: str | dict[str, Any] | None = None,
    loras: list[dict[str, Any]] | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    """The JSON body for ``/v1/moss/generate`` and ``/v1/moss/jobs``, omitting unset fields (server
    defaults apply). Exactly one of ``prompt`` / ``layout`` (a USERGUIDE layout object)."""
    if (prompt is None) == (layout is None):
        raise ValueError("give exactly one of prompt or layout")
    body: dict[str, Any] = {"prompt": prompt} if prompt is not None else {"layout": layout}
    optional: dict[str, Any] = {
        "variant": variant,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "shift": shift,
        "slg": slg,
        "loras": loras,
        "n": n,
    }
    body.update({k: v for k, v in optional.items() if v not in (None, [])})
    return body


def upscale_body(
    *,
    image_png: bytes | None = None,
    job_id: str | None = None,
    index: int | None = None,
    scale: float | None = None,
    method: str | None = None,
    esrgan_model: str | None = None,
    softening: str | None = None,
    color_fix: str | None = None,
    seed: int | None = None,
    tile_overlap: int | None = None,
) -> dict[str, Any]:
    """The JSON body for ``/v1/moss/upscale[/jobs]``: one source (image bytes or a job id)."""
    if (image_png is None) == (job_id is None):
        raise ValueError("give exactly one of image_png or job_id")
    body: dict[str, Any] = (
        {"image": base64.b64encode(image_png).decode("ascii")} if image_png is not None else {"job": job_id}
    )
    optional: dict[str, Any] = {
        "index": index,
        "scale": scale,
        "method": method,
        "esrgan_model": esrgan_model,
        "softening": softening,
        "color_fix": color_fix,
        "seed": seed,
        "tile_overlap": tile_overlap,
    }
    body.update({k: v for k, v in optional.items() if v not in (None, "")})
    return body


def caption_body(
    *,
    image_png: bytes | None = None,
    job_id: str | None = None,
    index: int | None = None,
    detail: str | None = None,
    reference: str | None = None,
    max_pixels: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """The JSON body for ``/v1/moss/caption``: one source (image bytes or a job id)."""
    if (image_png is None) == (job_id is None):
        raise ValueError("give exactly one of image_png or job_id")
    body: dict[str, Any] = (
        {"image": base64.b64encode(image_png).decode("ascii")} if image_png is not None else {"job": job_id}
    )
    optional: dict[str, Any] = {
        "index": index,
        "detail": detail,
        "reference": reference,
        "max_pixels": max_pixels,
        "seed": seed,
    }
    body.update({k: v for k, v in optional.items() if v not in (None, "")})
    return body


def edit_body(
    instruction: str,
    *,
    image_png: bytes | None = None,
    job_id: str | None = None,
    index: int | None = None,
    edit_layout: str | None = None,
    **generation: Any,
) -> dict[str, Any]:
    """The JSON body for ``/v1/moss/edit[/jobs]``: one image source (bytes or a job id) plus any
    ``generation_body`` field (loras — which must hold the edit LoRA — seed, steps, size…).
    ``edit_layout`` picks how the LoRA sees the source; ``None`` / ``"auto"`` = from its metadata."""
    if (image_png is None) == (job_id is None):
        raise ValueError("give exactly one of image_png or job_id")
    body = generation_body(instruction, **generation)
    if image_png is not None:
        body["image"] = base64.b64encode(image_png).decode("ascii")
    layout = None if edit_layout is None or edit_layout.lower() == "auto" else edit_layout
    optional: dict[str, Any] = {"job": job_id, "index": index, "edit_layout": layout}
    body.update({k: v for k, v in optional.items() if v not in (None, "")})
    return body


class MossForgeClient:
    def __init__(self, base_url: str = DEFAULT_URL, http: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(timeout=httpx.Timeout(GENERATE_TIMEOUT, connect=5.0))

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code >= 400:
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.text or f"HTTP {response.status_code}"
            raise MossForgeError(f"{response.status_code}: {message}")
        return response

    def _get(self, path: str) -> Any:
        return self._check(self._http.get(f"{self.base_url}{path}")).json()

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = self._get("/health")
        return result

    def models(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = self._get("/v1/models").get("data", [])
        return data

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = self._get("/v1/status")
        return result

    def options(self) -> dict[str, Any]:
        result: dict[str, Any] = self._get("/v1/moss/options")
        return result

    def loras(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = self._get("/v1/moss/loras").get("data", [])
        return data

    def enhance(self, prompt: str, *, layout: bool = False, width: int = 1024, height: int = 1024) -> str:
        body = {"prompt": prompt, "layout": layout, "width": width, "height": height}
        response = self._check(self._http.post(f"{self.base_url}/v1/moss/enhance", json=body))
        text: str = response.json()["prompt"]
        return text

    def submit(self, body: dict[str, Any]) -> str:
        """Queue a generation; returns its job id at once (``POST /v1/moss/jobs``)."""
        response = self._check(self._http.post(f"{self.base_url}/v1/moss/jobs", json=body, timeout=30.0))
        job_id: str = response.json()["id"]
        return job_id

    def submit_upscale(self, body: dict[str, Any]) -> str:
        """Queue an upscale; returns its job id at once (``POST /v1/moss/upscale/jobs``)."""
        response = self._check(
            self._http.post(f"{self.base_url}/v1/moss/upscale/jobs", json=body, timeout=120.0)
        )
        job_id: str = response.json()["id"]
        return job_id

    def submit_edit(self, body: dict[str, Any]) -> str:
        """Queue an edit-LoRA edit; returns its job id at once (``POST /v1/moss/edit/jobs``)."""
        response = self._check(
            self._http.post(f"{self.base_url}/v1/moss/edit/jobs", json=body, timeout=120.0)
        )
        job_id: str = response.json()["id"]
        return job_id

    def caption(self, body: dict[str, Any]) -> dict[str, Any]:
        """Caption an image (``POST /v1/moss/caption``); waits for the queue and the VLM."""
        response = self._check(self._http.post(f"{self.base_url}/v1/moss/caption", json=body))
        result: dict[str, Any] = response.json()
        return result

    def job(self, job_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._get(f"/v1/moss/jobs/{job_id}")
        return result

    def job_image(self, job_id: str, index: int = 0) -> GeneratedImage:
        response = self._http.get(f"{self.base_url}/v1/moss/jobs/{job_id}/image", params={"index": index})
        if response.status_code == 404:
            return GeneratedImage(png=None, seed=self.job(job_id).get("seed"), note="no image")
        response = self._check(response)
        seed = response.headers.get("x-mossforge-seed")
        return GeneratedImage(png=response.content, seed=int(seed) if seed else None)

    def generate(self, body: dict[str, Any]) -> GeneratedImage:
        """Synchronous ``/v1/moss/generate`` (raw PNG)."""
        response = self._check(self._http.post(f"{self.base_url}/v1/moss/generate", json=body))
        if response.headers.get("content-type", "").startswith("image/png"):
            seed = response.headers.get("x-mossforge-seed")
            return GeneratedImage(png=response.content, seed=int(seed) if seed else None)
        payload = response.json()
        first = (payload.get("data") or [{}])[0]
        return GeneratedImage(png=None, seed=first.get("seed"), note=payload.get("note"))
