"""Windows named-pipe transport — intentionally NOT IMPLEMENTED.

Modern Windows (10 1803+, released 2018) supports AF_UNIX natively, so the OCR
daemon and stdio frontend use the AF_UNIX transport (`transport.unix_sock`)
uniformly across Linux / macOS / Windows 10 1803+. No named-pipe path is needed
on supported platforms.

This module exists to give a clear, explicit error on older Windows (< 10 1803)
instead of a confusing failure deep inside asyncio — which has NO public API for
a Windows named-pipe server, and whose `connect_read_pipe`/`connect_write_pipe`
raise NotImplementedError on the Windows ProactorEventLoop.

Supporting pre-1803 Windows would require a full pywin32 named-pipe
implementation (CreateNamedPipe + overlapped I/O + a custom asyncio transport
adapter) — out of scope for this version.
"""
from __future__ import annotations

import asyncio


class PipeTransport:
    """Named-pipe transport stub. All operations raise NotImplementedError."""

    def __init__(self, name: str):
        self.name = name

    async def connect(self) -> None:
        raise NotImplementedError(
            "Windows named-pipe transport is not implemented. "
            "Use Windows 10 1803+ (which supports AF_UNIX), or implement a "
            "pywin32 named-pipe transport."
        )

    async def send(self, data: bytes) -> None:
        raise NotImplementedError("Windows named-pipe transport is not implemented.")

    async def recv_line(self) -> bytes:
        raise NotImplementedError("Windows named-pipe transport is not implemented.")

    async def close(self) -> None:
        raise NotImplementedError("Windows named-pipe transport is not implemented.")


async def serve_pipe(name: str, handler) -> asyncio.base_events.Server:
    raise NotImplementedError(
        "Windows named-pipe server is not implemented (asyncio has no public "
        "API for it). Use Windows 10 1803+ (AF_UNIX), or implement a pywin32 "
        "named-pipe server."
    )
