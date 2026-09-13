import asyncio

import pytest

from ocr_mcp import paths, protocol
from ocr_mcp.backend.worker import DaemonWorker
from ocr_mcp.config import Settings


class FakeEngine:
    def __init__(self):
        self.calls = []
    def predict(self, image_path, return_polys=True, return_scores=True):
        self.calls.append(image_path)
        from ocr_mcp.protocol import OcrLine, OcrResult
        return OcrResult(image_path, 1, 1, [OcrLine("ok", 1.0, None)], "ok", 10)


async def _start_worker(settings, engine):
    w = DaemonWorker(settings, engine)
    task = asyncio.create_task(w.run_async())
    await asyncio.sleep(0.05)  # let server bind
    return w, task


@pytest.mark.asyncio
async def test_ocr_request_succeeds(isolated_env, tmp_path):
    eng = FakeEngine()
    s = Settings.from_env()
    w, task = await _start_worker(s, eng)
    try:
        from ocr_mcp.transport.unix_sock import UnixTransport
        c = UnixTransport(paths.socket_path(s)); await c.connect()
        img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")
        await c.send(protocol.encode_message(
            protocol.make_request("ocr", w.token, {"image_path": str(img)})))
        resp = protocol.decode_message(await c.recv_line())
        assert resp["ok"] is True
        assert resp["result"]["text"] == "ok"
        await c.close()
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass


@pytest.mark.asyncio
async def test_bad_token_rejected(isolated_env, tmp_path):
    eng = FakeEngine()
    s = Settings.from_env()
    _, task = await _start_worker(s, eng)
    try:
        from ocr_mcp.transport.unix_sock import UnixTransport
        c = UnixTransport(paths.socket_path(s)); await c.connect()
        img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")
        await c.send(protocol.encode_message(
            protocol.make_request("ocr", "wrong-token", {"image_path": str(img)})))
        resp = protocol.decode_message(await c.recv_line())
        assert resp["ok"] is False
        assert resp["error"]["code"] == "UNAUTHORIZED"
        await c.close()
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass


@pytest.mark.asyncio
async def test_concurrent_requests_all_processed_serially(isolated_env, tmp_path):
    eng = FakeEngine()
    s = Settings.from_env()
    w, task = await _start_worker(s, eng)
    try:
        from ocr_mcp.transport.unix_sock import UnixTransport
        # Distinct image paths so each request is distinguishable; the old test
        # sent the same path 3x and could not tell requests apart.
        req_paths = []
        for name in ("a.png", "b.png", "c.png"):
            p = tmp_path / name
            p.write_bytes(b"\x89PNG")
            req_paths.append(str(p))
        async def one(path):
            c = UnixTransport(paths.socket_path(s)); await c.connect()
            await c.send(protocol.encode_message(
                protocol.make_request("ocr", w.token, {"image_path": path})))
            r = protocol.decode_message(await c.recv_line()); await c.close()
            return r["result"]["image_path"]
        results = await asyncio.gather(*[one(p) for p in req_paths])
        # FakeEngine.predict is synchronous/instant, so this cannot prove true
        # execution ordering (that would need an injected delay). What it does
        # prove against the single-thread executor: completeness (no request
        # dropped) and correctness (each distinct path served exactly once).
        assert sorted(results) == sorted(req_paths)
        assert sorted(eng.calls) == sorted(req_paths)
        assert len(eng.calls) == 3
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass


@pytest.mark.asyncio
async def test_idle_timeout_stops_daemon(isolated_env, monkeypatch):
    eng = FakeEngine()
    s = Settings.from_env()
    # fast idle timeout via env override of a Settings copy
    from dataclasses import replace
    s = replace(s, idle_timeout=0.05)
    w = DaemonWorker(s, eng)
    await w.run_async()  # should return on its own after idle timeout with no clients
    assert w.stopped


@pytest.mark.asyncio
async def test_idle_timeout_after_client_disconnect(isolated_env):
    # Regression: with the bug, the last client disconnecting left _idle_event
    # set, so the main loop's `if _clients == 0` wait_for returned instantly,
    # never raised TimeoutError, never broke -> 100% CPU busy-loop, daemon hung.
    # This exercises the connect -> disconnect -> idle-timeout path (NOT the
    # no-client-ever path covered above).
    from dataclasses import replace

    from ocr_mcp.transport.unix_sock import UnixTransport
    eng = FakeEngine()

    s = replace(Settings.from_env(), idle_timeout=0.1)
    w = DaemonWorker(s, eng)
    task = asyncio.create_task(w.run_async())
    await asyncio.sleep(0.05)  # let server bind
    # Connect a real client, ping, read response, then close -> 0 clients left.
    c = UnixTransport(paths.socket_path(s))
    await c.connect()
    await c.send(protocol.encode_message(protocol.make_request("ping", w.token, {})))
    resp = protocol.decode_message(await c.recv_line())
    assert resp["ok"] is True
    await c.close()
    # The worker loop MUST exit on its own via idle timeout. Guard with a hard
    # cap so a regression (busy-loop that never breaks) fails fast instead of
    # hanging the whole suite forever.
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        pytest.fail("worker did not exit on idle timeout after client disconnect")
    assert w.stopped is True
