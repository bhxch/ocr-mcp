from __future__ import annotations

import asyncio
import os
from pathlib import Path


class UnixTransport:
    def __init__(self, path: Path):
        self.path = path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_unix_connection(str(self.path))

    async def send(self, data: bytes) -> None:
        assert self._writer is not None
        self._writer.write(data)
        await self._writer.drain()

    async def recv_line(self) -> bytes:
        assert self._reader is not None
        return await self._reader.readline()

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass


async def serve_unix(path: Path, handler) -> asyncio.base_events.Server:
    # Remove stale socket file before binding.
    try:
        os.unlink(str(path))
    except FileNotFoundError:
        pass
    return await asyncio.start_unix_server(handler, path=str(path))
