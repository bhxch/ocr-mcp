import asyncio

import pytest

from ocr_mcp.transport.unix_sock import UnixTransport, serve_unix


@pytest.mark.asyncio
async def test_loopback_send_recv(tmp_path):
    sock = tmp_path / "s.sock"
    received: list[bytes] = []

    async def handler(reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            received.append(line)
            writer.write(b"PONG\n")
            await writer.drain()
        writer.close()

    server = await serve_unix(sock, handler)
    try:
        client = UnixTransport(sock)
        await client.connect()
        await client.send(b"PING\n")
        line = await client.recv_line()
        assert line == b"PONG\n"
        await client.close()
        await asyncio.sleep(0.05)
        assert received == [b"PING\n"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_stale_socket_file_is_replaced(tmp_path):
    sock = tmp_path / "s.sock"
    sock.write_text("garbage")  # pre-existing stale file

    async def handler(reader, writer):
        writer.close()

    server = await serve_unix(sock, handler)
    try:
        assert sock.exists()  # overwritten, not errored
    finally:
        server.close()
        await server.wait_closed()
