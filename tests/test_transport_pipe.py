import pytest

from ocr_mcp.transport.named_pipe import PipeTransport, serve_pipe


@pytest.mark.asyncio
async def test_pipe_transport_methods_raise_not_implemented():
    t = PipeTransport(r"\\.\pipe\ocr-mcp-test")
    with pytest.raises(NotImplementedError):
        await t.connect()
    with pytest.raises(NotImplementedError):
        await t.send(b"x")
    with pytest.raises(NotImplementedError):
        await t.recv_line()
    with pytest.raises(NotImplementedError):
        await t.close()


@pytest.mark.asyncio
async def test_serve_pipe_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        await serve_pipe(r"\\.\pipe\ocr-mcp-test", lambda r, w: None)
