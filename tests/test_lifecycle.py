import asyncio
import json

import pytest

from ocr_mcp import paths
from ocr_mcp.backend import lifecycle
from ocr_mcp.config import Settings


@pytest.mark.asyncio
async def test_read_lock_returns_none_when_abssent(isolated_env):
    s = Settings.from_env()
    assert lifecycle.read_lock(s) is None


@pytest.mark.asyncio
async def test_is_alive_false_for_dead_pid(isolated_env):
    s = Settings.from_env()
    # write a lock pointing at a dead pid and non-listening socket
    paths.lock_file_path(s).write_text(json.dumps(
        {"pid": 2_000_000, "socket": str(paths.socket_path(s)), "token": "t"}))
    assert await lifecycle.is_alive(s, lifecycle.read_lock(s)) is False


@pytest.mark.asyncio
async def test_is_alive_true_for_running_daemon(isolated_env):
    from ocr_mcp.backend.worker import DaemonWorker
    class FakeEngine:
        def predict(self, *a, **k):
            from ocr_mcp.protocol import OcrLine, OcrResult
            return OcrResult("x", 1, 1, [OcrLine("a", 1.0, None)], "a", 1)
    s = Settings.from_env()
    w = DaemonWorker(s, FakeEngine())
    task = asyncio.create_task(w.run_async())
    await asyncio.sleep(0.05)
    try:
        lock = lifecycle.read_lock(s)
        assert lock is not None
        assert await lifecycle.is_alive(s, lock) is True
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass


@pytest.mark.asyncio
async def test_ensure_daemon_reuses_running(isolated_env):
    from ocr_mcp.backend.worker import DaemonWorker
    class FakeEngine:
        def predict(self, *a, **k):
            from ocr_mcp.protocol import OcrLine, OcrResult
            return OcrResult("x", 1, 1, [OcrLine("a", 1.0, None)], "a", 1)
    s = Settings.from_env()
    w = DaemonWorker(s, FakeEngine())
    task = asyncio.create_task(w.run_async())
    await asyncio.sleep(0.05)
    try:
        calls = {"n": 0}
        orig = lifecycle.spawn_daemon
        lifecycle.spawn_daemon = lambda settings: calls.__setitem__("n", calls["n"] + 1)
        try:
            t = await lifecycle.ensure_daemon(s)
            assert calls["n"] == 0  # did NOT spawn, reused
            await t.close()
        finally:
            lifecycle.spawn_daemon = orig
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass


@pytest.mark.asyncio
async def test_ensure_daemon_spawns_when_absent(isolated_env, monkeypatch):
    s = Settings.from_env()
    spawned = {"ok": False}
    # NOTE: real _spawn_and_wait_ready spawns a detached subprocess (not a task
    # in our loop). The fake below starts an in-process DaemonWorker instead;
    # we track it so we can cancel it deterministically, otherwise
    # asyncio.Runner._cancel_all_tasks deadlocks in the worker's
    # `finally: await server.wait_closed()` at loop teardown on py3.11.
    worker_task = {"t": None}

    async def fake_spawn_and_wait(settings):
        # Simulate a daemon coming up by starting a real DaemonWorker
        from ocr_mcp.backend.worker import DaemonWorker
        class FakeEngine:
            def predict(self, *a, **k):
                from ocr_mcp.protocol import OcrLine, OcrResult
                return OcrResult("x", 1, 1, [OcrLine("a", 1.0, None)], "a", 1)
        w = DaemonWorker(settings, FakeEngine())
        worker_task["t"] = asyncio.create_task(w.run_async())
        await asyncio.sleep(0.05)
        spawned["ok"] = True

    monkeypatch.setattr(lifecycle, "_spawn_and_wait_ready", fake_spawn_and_wait)
    t = await lifecycle.ensure_daemon(s)
    try:
        assert spawned["ok"] is True
    finally:
        await t.close()
        task = worker_task["t"]
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
