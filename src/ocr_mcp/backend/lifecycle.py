from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from ocr_mcp import paths, protocol
from ocr_mcp.config import Settings
from ocr_mcp.transport.unix_sock import UnixTransport

log = logging.getLogger("ocr_mcp.lifecycle")

READY_TIMEOUT = 60.0  # seconds to wait for daemon to bind + load model


def read_lock(settings: Settings) -> dict | None:
    lf = paths.lock_file_path(settings)
    if not lf.exists():
        return None
    try:
        return json.loads(lf.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    except OSError:
        return False
    return True


async def is_alive(settings: Settings, lock: dict) -> bool:
    if not lock:
        return False
    if not _pid_alive(int(lock.get("pid", -1))):
        return False
    sock = lock.get("socket")
    if not sock:
        return False
    # Probe by sending a ping.
    try:
        t = UnixTransport(Path(sock))
        await asyncio.wait_for(t.connect(), timeout=2.0)
    except (TimeoutError, OSError):
        return False
    try:
        await t.send(protocol.encode_message(protocol.make_request("ping", lock.get("token", ""), {})))
        line = await asyncio.wait_for(t.recv_line(), timeout=5.0)
        resp = protocol.decode_message(line)
        return bool(resp.get("ok"))
    except Exception:  # noqa: BLE001 - any probe failure means the daemon is unusable
        return False
    finally:
        await t.close()


def spawn_daemon(settings: Settings) -> None:
    """Spawn a detached daemon process. Its stdout/stderr go to a log file,
    never to inherited stdio (which would corrupt the MCP channel on the
    server side)."""
    paths.ensure_data_dir(settings)
    log_path = paths.log_file_path(settings, "daemon")
    log_file = open(log_path, "ab", buffering=0)  # noqa: SIM115 - detached child owns it
    cmd = [sys.executable, "-m", "ocr_mcp", "serve"]
    popen_kwargs = {
        "stdin": subprocess.DEVNULL, "stdout": log_file, "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if sys.platform == "win32":
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        popen_kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    else:
        popen_kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **popen_kwargs)


async def _spawn_and_wait_ready(settings: Settings) -> None:
    spawn_daemon(settings)
    deadline = asyncio.get_event_loop().time() + READY_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        lock = read_lock(settings)
        if lock and await is_alive(settings, lock):
            return
        await asyncio.sleep(0.2)
    raise RuntimeError("daemon failed to become ready in time")


async def _connect(settings: Settings, lock: dict) -> UnixTransport:
    t = UnixTransport(Path(lock["socket"]))
    await t.connect()
    return t


async def ensure_daemon(settings: Settings) -> UnixTransport:
    """Return a connected transport to a running daemon, spawning one if needed.

    Handles stale lock cleanup and concurrent-spawn races via an atomic
    spawn-lock file (O_CREAT|O_EXCL)."""
    paths.ensure_data_dir(settings)
    lock = read_lock(settings)
    if lock and await is_alive(settings, lock):
        return await _connect(settings, lock)

    # Stale: clean up.
    _cleanup_stale(settings, lock)

    # Atomic spawn-lock to avoid two clients racing to spawn.
    spawn_lock = paths.lock_file_path(settings).parent / "spawn.lock"
    try:
        fd = os.open(str(spawn_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        we_spawned = True
    except FileExistsError:
        we_spawned = False

    if we_spawned:
        try:
            # Double-check after acquiring the lock.
            lock = read_lock(settings)
            if not (lock and await is_alive(settings, lock)):
                await _spawn_and_wait_ready(settings)
        finally:
            try:
                spawn_lock.unlink()
            except FileNotFoundError:
                pass
    else:
        # Another client is spawning; wait for it.
        for _ in range(int(READY_TIMEOUT * 5)):
            lock = read_lock(settings)
            if lock and await is_alive(settings, lock):
                break
            await asyncio.sleep(0.2)

    lock = read_lock(settings)
    if not (lock and await is_alive(settings, lock)):
        raise RuntimeError("DAEMON_UNAVAILABLE: could not start or reach daemon")
    return await _connect(settings, lock)


def _cleanup_stale(settings: Settings, lock: dict | None) -> None:
    try:
        paths.lock_file_path(settings).unlink()
    except FileNotFoundError:
        pass
    sock = (lock or {}).get("socket")
    if sock:
        try:
            Path(sock).unlink()
        except (FileNotFoundError, OSError):
            pass
