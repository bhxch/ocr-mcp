import asyncio

import pytest

from ocr_mcp import protocol, server


@pytest.fixture(autouse=True)
def _reset_client():
    # handle_ocr_image caches a module-level BackendClient; reset between tests
    # so each test reconnects through its own monkeypatched ensure_daemon.
    server._client = None
    yield
    server._client = None


class FakeBackend:
    """Stand-in for a connected daemon transport."""

    def __init__(self):
        self.last_sent = None

    async def send(self, data):
        self.last_sent = protocol.decode_message(data)

    async def recv_line(self):
        # echo back a success for whatever id was sent
        return protocol.encode_message(
            protocol.make_success(
                self.last_sent["id"],
                {
                    "image_path": self.last_sent["payload"]["image_path"],
                    "width": 10,
                    "height": 5,
                    "lines": [
                        {"text": "T1", "score": 0.9, "poly": [[0, 0], [1, 0], [1, 1], [0, 1]]}
                    ],
                    "text": "T1",
                    "elapsed_ms": 7,
                },
            )
        )

    async def close(self):
        pass


class StaleOnceBackend(FakeBackend):
    """Simulates a daemon that died after connect: the first recv_line returns
    b"" (EOF on a stale transport), then subsequent calls echo a normal
    success once the client has reconnected."""

    def __init__(self):
        super().__init__()
        self.recv_calls = 0

    async def recv_line(self):
        self.recv_calls += 1
        if self.recv_calls == 1:
            return b""  # stale transport: peer gone, readline() => EOF
        return await super().recv_line()


class FixedSuccessBackend:
    """Concurrency-safe stand-in: ignores the request entirely and returns a
    fixed success on every recv_line (no shared mutable state read in recv)."""

    async def send(self, data):
        pass

    async def recv_line(self):
        return protocol.encode_message(
            protocol.make_success(
                "id",
                {
                    "image_path": "",
                    "width": 1,
                    "height": 1,
                    "lines": [],
                    "text": "T1",
                    "elapsed_ms": 0,
                },
            )
        )

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_call_ocr_image_returns_markdown_table(isolated_env, tmp_path, monkeypatch):
    fake = FakeBackend()

    async def fake_ensure(settings):
        return fake

    monkeypatch.setattr(server.lifecycle, "ensure_daemon", fake_ensure)

    async def fake_ensure(settings):
        return fake

    monkeypatch.setattr(server.lifecycle, "ensure_daemon", fake_ensure)
    out = await server.handle_ocr_image({"image_path": str(tmp_path / "x.png")})
    assert out.isError is False
    md = out.content[0].text
    # FakeBackend returns a line carrying score+poly, so all three columns appear.
    assert md.startswith("| text | score | poly |")
    assert "| --- | --- | --- |" in md
    assert "T1" in md
    assert fake.last_sent["op"] == "ocr"
    assert fake.last_sent["payload"]["image_path"] == str(tmp_path / "x.png")


@pytest.mark.asyncio
async def test_call_ocr_image_missing_path_errors(isolated_env, monkeypatch):
    monkeypatch.setattr(
        server.lifecycle, "ensure_daemon", lambda settings: asyncio.sleep(0, result=FakeBackend())
    )
    out = await server.handle_ocr_image({})
    assert out.isError is True


@pytest.mark.asyncio
async def test_call_ocr_image_propagates_daemon_error(isolated_env, tmp_path, monkeypatch):
    class ErrBackend(FakeBackend):
        async def recv_line(self):
            return protocol.encode_message(
                protocol.make_error(self.last_sent["id"], "IMAGE_NOT_FOUND", "nope")
            )

    fake = ErrBackend()
    monkeypatch.setattr(server.lifecycle, "ensure_daemon", lambda s: asyncio.sleep(0, result=fake))
    out = await server.handle_ocr_image({"image_path": str(tmp_path / "x.png")})
    assert out.isError is True
    assert "IMAGE_NOT_FOUND" in out.content[0].text


def test_create_server_registers_ocr_image_tool():
    """create_server() must return an MCPServer with the ocr_image tool registered.

    Regression guard: the previous 1.x-style create_server used decorators that
    mcp 2.0 removed, so it crashed at runtime; this test exercises that path.
    """
    mcp = server.create_server()
    assert mcp.name == "ocr-mcp"

    tools = asyncio.run(mcp.list_tools())
    names = [t.name for t in tools]
    assert "ocr_image" in names

    tool = next(t for t in tools if t.name == "ocr_image")
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert schema["properties"]["image_path"]["type"] == "string"
    assert "image_path" in schema["required"]
    assert schema["properties"]["return_polys"]["default"] is False
    assert schema["properties"]["return_scores"]["default"] is False


@pytest.mark.asyncio
async def test_reconnects_after_stale_transport(isolated_env, tmp_path, monkeypatch):
    """Spec §10: a stale transport (EOF) must be closed and reconnected once,
    transparently succeeding instead of staying broken."""
    fake = StaleOnceBackend()
    ensure_calls = 0

    async def counting_ensure(settings):
        nonlocal ensure_calls
        ensure_calls += 1
        return fake

    monkeypatch.setattr(server.lifecycle, "ensure_daemon", counting_ensure)

    out = await server.handle_ocr_image({"image_path": str(tmp_path / "x.png")})
    assert out.isError is False
    assert "T1" in out.content[0].text  # markdown table carries the line text
    assert fake.recv_calls == 2  # stale EOF, then success after reconnect
    assert ensure_calls == 2  # initial connect + exactly one reconnect


@pytest.mark.asyncio
async def test_concurrent_first_connect_calls_ensure_once(isolated_env, tmp_path, monkeypatch):
    """Two overlapping first connects must serialize: ensure_daemon runs exactly
    once (lock + double-check), not once per racing caller."""
    ensure_calls = 0

    async def counting_ensure(settings):
        nonlocal ensure_calls
        ensure_calls += 1
        # Force a yield inside ensure_daemon so both callers are mid-connect at
        # once; without the lock this would let both increment ensure_calls.
        await asyncio.sleep(0.05)
        return FixedSuccessBackend()

    monkeypatch.setattr(server.lifecycle, "ensure_daemon", counting_ensure)

    path = str(tmp_path / "x.png")
    results = await asyncio.gather(
        server.handle_ocr_image({"image_path": path}),
        server.handle_ocr_image({"image_path": path}),
    )
    assert ensure_calls == 1
    for out in results:
        assert out.isError is False
        assert out.content[0].text.startswith("| text")  # markdown table
