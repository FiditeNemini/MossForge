"""Bridge tests: mock-transport client and
job-flow tests, the real-MCP-tool-layer regression, and an end-to-end run against `mossforge serve`
(stub backend — real routes and validation, gradient PNGs, no GPU)."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from mossforge_mcp import server
from mossforge_mcp.client import (
    MossForgeClient,
    MossForgeError,
    caption_body,
    edit_body,
    generation_body,
    upscale_body,
)

PNG = b"\x89PNG\r\n\x1a\nfake"


def mock_client(handler) -> MossForgeClient:  # type: ignore[no-untyped-def]
    return MossForgeClient("http://moss.test", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_generation_body_omits_unset_fields_and_needs_one_source() -> None:
    assert generation_body("a fox") == {"prompt": "a fox"}
    body = generation_body("a fox", width=512, seed=0, loras=[{"path": "k.safetensors", "blocks": "21-41"}])
    assert body == {
        "prompt": "a fox",
        "width": 512,
        "seed": 0,
        "loras": [{"path": "k.safetensors", "blocks": "21-41"}],
    }
    assert generation_body(layout={"items": []}, variant="turbo") == {
        "layout": {"items": []},
        "variant": "turbo",
    }
    with pytest.raises(ValueError):
        generation_body()
    with pytest.raises(ValueError):
        generation_body("x", layout={})


def test_generate_returns_png_and_seed_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=PNG, headers={"content-type": "image/png", "X-MossForge-Seed": "42"}
        )

    result = mock_client(handler).generate({"prompt": "a fox"})
    assert seen == {"path": "/v1/moss/generate", "body": {"prompt": "a fox"}}
    assert (result.png, result.seed) == (PNG, 42)


def test_server_errors_surface_their_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "'prompt' (or 'layout') is required", "type": "x"}}
        )

    with pytest.raises(MossForgeError, match="400: 'prompt'"):
        mock_client(handler).generate({})


class FakeJobServer:
    """A mock resident server whose job reports `running` for `running_polls` polls."""

    def __init__(self, running_polls: int, kind: str = "generate") -> None:
        self.polls = 0
        self.running_polls = running_polls
        self.kind = kind
        self.submitted: dict[str, object] = {}
        self.submitted_to = ""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in ("/v1/moss/jobs", "/v1/moss/upscale/jobs", "/v1/moss/edit/jobs"):
            self.submitted = json.loads(request.content)
            self.submitted_to = path
            return httpx.Response(202, json={"id": "j1", "state": "queued", "queue": 1})
        if path == "/v1/moss/jobs/j1":
            self.polls += 1
            if self.polls <= self.running_polls:
                stage = "encoding prompt" if self.polls == 1 else f"code {self.polls}"
                return httpx.Response(
                    200,
                    json={
                        "id": "j1",
                        "state": "running",
                        "stage": stage,
                        "step": self.polls,
                        "total_steps": 8,
                        "elapsed_s": float(self.polls),
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "j1",
                    "kind": self.kind,
                    "state": "done",
                    "stage": "done",
                    "step": 8,
                    "total_steps": 8,
                    "elapsed_s": 9.0,
                    "seed": 5,
                    "width": 1024,
                    "height": 1024,
                },
            )
        if path == "/v1/moss/jobs/j1/image":
            return httpx.Response(
                200, content=PNG, headers={"content-type": "image/png", "X-MossForge-Seed": "5"}
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})


async def collect(
    reports: list[tuple[float, float | None, str | None]], p: float, t: float | None, m: str | None
) -> None:
    reports.append((p, t, m))


def test_job_flow_reports_progress_then_saves_the_png(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeJobServer(running_polls=3)
    client = mock_client(fake)
    monkeypatch.setenv("MOSSFORGE_OUTPUT_DIR", str(tmp_path))
    reports: list[tuple[float, float | None, str | None]] = []

    async def go() -> list[object]:
        job_id = client.submit(generation_body("a fox", variant="turbo"))
        status = await server.wait_for_job(
            client, job_id, 10, lambda p, t, m: collect(reports, p, t, m), poll_seconds=0
        )
        return await server._finish(client, job_id, status)

    image, caption = asyncio.run(go())
    assert fake.submitted == {"prompt": "a fox", "variant": "turbo"}
    assert [m for _, _, m in reports][0].startswith("encoding prompt")
    saved = list(tmp_path.glob("mossforge-*-j1.png"))
    assert len(saved) == 1 and saved[0].read_bytes() == PNG
    assert str(saved[0]) in caption and image.data == PNG


def test_tools_return_image_content_through_the_real_mcp_tool_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeJobServer(running_polls=0)
    monkeypatch.setattr(server, "_client", lambda: mock_client(fake))
    monkeypatch.setenv("MOSSFORGE_OUTPUT_DIR", str(tmp_path))
    layout = {
        "summary": {"caption": "a poster"},
        "items": [],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    for name, args in [
        ("get_generation", {"job_id": "j1"}),
        ("generate_image", {"prompt": "a fox"}),
        (
            "generate_image",
            {
                "layout": layout,
                "aspect_ratio": "16:9",
                "loras": [{"path": "k.safetensors", "strength": -0.5, "blocks": "21-41"}],
            },
        ),
    ]:
        result = asyncio.run(server.mcp.call_tool(name, args))
        assert [block.type for block in result.content] == ["image", "text"], name
    # The tool defaults to Turbo and sizes from the aspect ratio; LoRA blocks pass through.
    assert fake.submitted["variant"] == "turbo"
    assert (fake.submitted["width"], fake.submitted["height"]) == (1024, 576)
    assert fake.submitted["layout"] == layout and "prompt" not in fake.submitted
    assert fake.submitted["loras"] == [{"path": "k.safetensors", "strength": -0.5, "blocks": "21-41"}]


def test_prompt_xor_layout_is_enforced_by_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_client", lambda: mock_client(FakeJobServer(running_polls=0)))
    result = asyncio.run(server.mcp.call_tool("generate_image", {}))
    assert "exactly one of prompt or layout" in result.content[0].text


def test_a_long_job_hands_back_its_id_instead_of_timing_out() -> None:
    client = mock_client(FakeJobServer(running_polls=10_000))

    async def go() -> list[object]:
        status = await server.wait_for_job(client, "j1", 0, lambda p, t, m: collect([], p, t, m))
        return await server._finish(client, "j1", status)

    (message,) = asyncio.run(go())
    assert 'get_generation with job_id="j1"' in message


def test_upscale_body_needs_exactly_one_source() -> None:
    assert upscale_body(job_id="j1", scale=4, method="esrgan") == {
        "job": "j1",
        "scale": 4,
        "method": "esrgan",
    }
    assert upscale_body(image_png=b"hi") == {"image": "aGk="}
    for kwargs in ({}, {"image_png": b"x", "job_id": "j"}):
        with pytest.raises(ValueError, match="exactly one"):
            upscale_body(**kwargs)  # type: ignore[arg-type]


def test_image_paths_are_confined_to_the_allowed_folders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    allowed, outside = tmp_path / "pics", tmp_path / "secret"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("MOSSFORGE_IMAGE_DIRS", str(allowed))
    monkeypatch.setenv("MOSSFORGE_OUTPUT_DIR", str(tmp_path / "out"))
    (allowed / "a.png").write_bytes(PNG)
    (outside / "b.png").write_bytes(PNG)
    (allowed / "notes.txt").write_text("x")
    (allowed / "link.png").symlink_to(outside / "b.png")
    assert server.read_allowed_image(str(allowed / "a.png")) == PNG
    for path, message in [
        (outside / "b.png", "outside"),
        (allowed / ".." / "secret" / "b.png", "outside"),
        (allowed / "link.png", "outside"),  # a symlink can't smuggle a file out
        (allowed / "notes.txt", "not a .png"),
        (allowed / "missing.png", "not a .png"),
    ]:
        with pytest.raises(MossForgeError, match=message):
            server.read_allowed_image(str(path))


def test_upscale_tool_returns_the_saved_path_not_inline_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeJobServer(running_polls=0, kind="upscale")
    monkeypatch.setattr(server, "_client", lambda: mock_client(fake))
    monkeypatch.setenv("MOSSFORGE_OUTPUT_DIR", str(tmp_path))
    result = asyncio.run(
        server.mcp.call_tool("upscale_image", {"job_id": "g7", "scale": 4, "softening": "strong"})
    )
    assert fake.submitted_to == "/v1/moss/upscale/jobs"
    assert fake.submitted == {"job": "g7", "scale": 4, "method": "vosr", "softening": "strong"}
    assert [block.type for block in result.content] == ["text"]
    saved = list(tmp_path.glob("mossforge-upscale-*-j1.png"))
    assert len(saved) == 1 and str(saved[0]) in result.content[0].text
    assert "1024×1024" in result.content[0].text

    both = asyncio.run(server.mcp.call_tool("upscale_image", {"job_id": "a", "image_path": "/x.png"}))
    assert "exactly one" in both.content[0].text


def test_aspect_ratio_sizes_snap_to_16() -> None:
    assert server.size_for("1:1", 1024) == (1024, 1024)
    assert server.size_for("16:9", 1024) == (1024, 576)
    assert server.size_for("9:16", 1536) == (864, 1536)
    with pytest.raises(ValueError):
        server.size_for("wide", 1024)


# --- end to end against the real `mossforge serve` (stub backend) ----------------------------

MOSSFORGE = Path(__file__).resolve().parents[2] / ".build" / "debug" / "mossforge"


@pytest.fixture
def live_server() -> Iterator[str]:
    if not MOSSFORGE.exists():
        pytest.skip("run `swift build` in the repo root first")
    proc = subprocess.Popen(  # noqa: S603 — our own build product
        [str(MOSSFORGE), "serve", "--port", "0", "--models", "/nonexistent"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        match = re.search(r"listening on 127\.0\.0\.1:(\d+)", line)
        assert match, f"unexpected first line: {line!r}"
        yield f"http://127.0.0.1:{match.group(1)}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_end_to_end_against_mossforge_serve(
    live_server: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = MossForgeClient(live_server)
    assert client.health()["backend"] == "stub"
    assert client.options()["generate"]["variant_defaults"]["turbo"]["steps"] == 4
    result = client.generate(generation_body("a fox", width=256, height=256, seed=3, variant="turbo"))
    assert result.png is not None and result.png.startswith(b"\x89PNG") and result.seed == 3
    with pytest.raises(MossForgeError, match="multiple of 16"):
        client.generate(generation_body("a fox", width=500))
    with pytest.raises(MossForgeError, match="turbo runs without CFG"):
        client.generate(generation_body("a fox", variant="turbo", guidance_scale=3))
    assert json.loads(client.enhance("summer sale", layout=True))["items"]

    # The job flow the tools use: submit → poll (progress) → save. (Driven directly: progress
    # reporting needs a live MCP request context, which the mocked tool-layer test covers.)
    monkeypatch.setenv("MOSSFORGE_OUTPUT_DIR", str(tmp_path))

    async def go() -> list[object]:
        job_id = client.submit(generation_body("a fox", width=256, height=256, seed=9, variant="turbo"))
        status = await server.wait_for_job(client, job_id, 10, lambda p, t, m: collect([], p, t, m), 0.05)
        return await server._finish(client, job_id, status)

    image, caption = asyncio.run(go())
    assert str(tmp_path) in str(caption)
    assert client.status()["jobs"][0]["seed"] == 9

    # Edits are validated before queueing: an edit with no edit LoRA is refused.
    gen_job = client.status()["jobs"][0]["id"]
    with pytest.raises(MossForgeError, match="edit LoRA"):
        client.submit_edit(edit_body("make it night", job_id=gen_job))


def test_caption_body_needs_exactly_one_source() -> None:
    assert caption_body(job_id="j1", detail="short") == {"job": "j1", "detail": "short"}
    assert caption_body(image_png=b"hi", reference="a fox") == {"image": "aGk=", "reference": "a fox"}
    for kwargs in ({}, {"image_png": b"x", "job_id": "j"}):
        with pytest.raises(ValueError, match="exactly one"):
            caption_body(**kwargs)  # type: ignore[arg-type]


def test_caption_tool_returns_the_caption_and_its_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "caption": "Long.",
                "long": "Long.",
                "medium": "Med.",
                "short": "Short.",
                "watermark": [],
                "borders": ["none"],
            },
        )

    monkeypatch.setattr(server, "_client", lambda: mock_client(handler))
    monkeypatch.setenv("MOSSFORGE_IMAGE_DIRS", str(tmp_path))
    (tmp_path / "fox.png").write_bytes(PNG)
    result = asyncio.run(
        server.mcp.call_tool("caption_image", {"image_path": str(tmp_path / "fox.png"), "detail": "medium"})
    )
    assert seen["path"] == "/v1/moss/caption"
    assert seen["body"] == {"image": base64.b64encode(PNG).decode(), "detail": "medium"}
    text = result.content[0].text
    assert "Long." in text and "Short." in text

    both = asyncio.run(server.mcp.call_tool("caption_image", {"job_id": "a", "image_path": "/x.png"}))
    assert "exactly one" in both.content[0].text


def test_edit_body_needs_exactly_one_source() -> None:
    lora = [{"path": "lora/trained/edit.safetensors"}]
    assert edit_body("make it night", job_id="j1", loras=lora, seed=3) == {
        "prompt": "make it night",
        "job": "j1",
        "loras": lora,
        "seed": 3,
    }
    assert edit_body("x", image_png=b"hi") == {"prompt": "x", "image": "aGk="}
    for kwargs in ({}, {"image_png": b"x", "job_id": "j"}):
        with pytest.raises(ValueError, match="exactly one"):
            edit_body("x", **kwargs)  # type: ignore[arg-type]


def test_edit_tool_chains_on_a_job_and_shows_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeJobServer(running_polls=0, kind="edit")
    monkeypatch.setattr(server, "_client", lambda: mock_client(fake))
    monkeypatch.setenv("MOSSFORGE_OUTPUT_DIR", str(tmp_path))
    lora = {"path": "lora/trained/edit.safetensors"}
    args = {"instruction": "give her a red scarf", "job_id": "g7", "loras": [lora], "seed": 4}
    result = asyncio.run(server.mcp.call_tool("edit_image", args))
    assert fake.submitted_to == "/v1/moss/edit/jobs"
    assert fake.submitted == {
        "prompt": "give her a red scarf",
        "job": "g7",
        "variant": "turbo",
        "seed": 4,
        "loras": [{"path": "lora/trained/edit.safetensors", "strength": 1.0}],
    }
    assert [block.type for block in result.content] == ["image", "text"]
    assert len(list(tmp_path.glob("mossforge-edit-*-j1.png"))) == 1

    none = asyncio.run(server.mcp.call_tool("edit_image", {"instruction": "x", "loras": [lora]}))
    assert "exactly one" in none.content[0].text


def test_edit_layout_is_sent_unless_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert edit_body("x", job_id="j", edit_layout="Moss-Image-Edit")["edit_layout"] == "Moss-Image-Edit"
    assert "edit_layout" not in edit_body("x", job_id="j", edit_layout="auto")

    fake = FakeJobServer(running_polls=0, kind="edit")
    monkeypatch.setattr(server, "_client", lambda: mock_client(fake))
    monkeypatch.setenv("MOSSFORGE_OUTPUT_DIR", str(tmp_path))
    lora = {"path": "lora/akhaliq-edit-sft-armv5.safetensors"}
    args = {"instruction": "make it red", "job_id": "g7", "loras": [lora], "edit_layout": "ai-toolkit"}
    asyncio.run(server.mcp.call_tool("edit_image", args))
    assert fake.submitted["edit_layout"] == "ai-toolkit"
