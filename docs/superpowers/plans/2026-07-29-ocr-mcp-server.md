# OCR MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a uv-managed Python OCR MCP server (stdio) where multiple MCP clients share a single resident PaddleOCR backend daemon via Unix socket / named pipe, loading the model only once to save memory.

**Architecture:** Two-layer process model. A瘦 stdio MCP server frontend (never imports paddle) talks over NDJSON-on-socket to a resident OCR backend daemon (lazy-started, idle-timeout, single-instance serial inference queue). Clients locate the daemon via a lock file; the first client spawns it detached.

**Tech Stack:** Python `>=3.11,<3.13`, uv, hatchling (src layout), `mcp[cli]>=2.0`, `paddleocr`/`paddlepaddle` (PP-OCRv6_medium), `platformdirs`, pytest + pytest-asyncio, ruff.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-07-29-ocr-mcp-server-design.md`:

- `requires-python = ">=3.11,<3.13"` (paddlepaddle constraint).
- Dependencies: `mcp[cli]>=2.0`, `paddleocr>=3.0` (verified 3.7.0), `paddlepaddle>=3.0` (verified 3.3.1), `platformdirs>=4.0`. Build: hatchling, src layout. Dev: `pytest`, `pytest-asyncio`, `ruff`.
- **stdio server process must NEVER `import paddleocr`/`paddlepaddle`.** Paddle imports are deferred inside `backend/engine.py` (daemon-only). `server.py` talks to the daemon only via transport.
- CPU inference MUST use `enable_mkldnn=False` (paddlepaddle 3.3 PIR+onednn crashes otherwise).
- Use `predict()`, not the deprecated `ocr()`. Read results via `res.get("rec_texts")` (dict keys), not `getattr`.
- Cross-platform: Linux / macOS / Windows. Socket = `AF_UNIX` (POSIX + Win10 1803+); named-pipe fallback for older Windows.
- Entry point: `ocr-mcp = "ocr_mcp.__main__:main"`. Subcommands: `stdio` (default) | `serve` | `download`.
- Every task ends with a commit. Commit messages follow Angular convention, Chinese subject, lead with **why**.

## File Structure

```
pyproject.toml                  # build, deps, entry point, tool config
.python-version                 # 3.11
.gitignore                      # Python ignores + .temp
README.md                       # usage, deploy, MCP client config
src/ocr_mcp/
  __init__.py                   # __version__
  __main__.py                   # CLI entry (dispatch subcommands)
  cli.py                        # argparse: stdio | serve | download
  config.py                     # Settings dataclass + env loading
  paths.py                      # cross-platform lock/socket/log paths (platformdirs)
  protocol.py                   # NDJSON envelope + OcrResult/OcrLine + encode/decode
  server.py                     # MCP stdio frontend + ocr_image tool (NO paddle import)
  transport/
    __init__.py
    base.py                     # Transport Protocol (abstract)
    unix_sock.py                # AF_UNIX Transport + serve_unix()
    named_pipe.py               # Windows named-pipe Transport (fallback)
  backend/
    __init__.py
    engine.py                   # PaddleOCR singleton + predict + result normalization (deferred import)
    worker.py                   # daemon: listen + serial queue + refcount + idle timeout + token
    lifecycle.py                # detect/spawn/lock file/stale cleanup/race arbitration
tests/
  conftest.py                   # fixtures (tmp data dir, fake monkeypatched env)
  test_config.py
  test_protocol.py
  test_transport_unix.py
  test_transport_pipe.py        # skipped on non-Windows
  test_engine.py
  test_worker.py
  test_lifecycle.py
  test_server.py
  test_cli.py
  fixtures/                     # sample png images for integration
```

---

### Task 1: Project skeleton, build config, smoke test

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/ocr_mcp/__init__.py`, `src/ocr_mcp/__main__.py`, `src/ocr_mcp/cli.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: `ocr_mcp.__version__`, `ocr_mcp.cli.main(argv) -> int`, entry point `ocr-mcp`.

- [ ] **Step 1: Write `.python-version` and `.gitignore`**

`.python-version`:
```
3.11
```

`.gitignore` (Python + project):
```
__pycache__/
*.py[cod]
.venv/
*.egg-info/
dist/
build/
.ruff_cache/
.pytest_cache/
.temp/
*.log
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "ocr-mcp"
version = "0.1.0"
description = "OCR MCP server sharing a single resident PaddleOCR backend across MCP clients"
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
    "mcp[cli]>=2.0",
    "paddleocr>=3.0,<4.0",
    "paddlepaddle>=3.0,<4.0",
    "platformdirs>=4.0",
]

[project.scripts]
ocr-mcp = "ocr_mcp.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ocr_mcp"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.6",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Write package stubs**

`src/ocr_mcp/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/ocr_mcp/cli.py`:
```python
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--version", "-V"):
        from ocr_mcp import __version__
        print(f"ocr-mcp {__version__}")
        return 0
    # Real subcommand dispatch added in Task 10.
    print("ocr-mcp (stdio/serve/download dispatch pending)", file=sys.stderr)
    return 0
```

`src/ocr_mcp/__main__.py`:
```python
import sys

from ocr_mcp.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

`tests/__init__.py`: empty.
`tests/conftest.py`:
```python
import os
import pytest


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("OCR_MCP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OCR_MCP_LOG_LEVEL", "DEBUG")
    return data_dir
```

- [ ] **Step 4: Write the failing smoke test**

`tests/test_smoke.py`:
```python
import subprocess
import sys


def test_version_flag_prints_version():
    result = subprocess.run(
        [sys.executable, "-m", "ocr_mcp", "--version"],
        capture_output=True, text=True, check=True,
    )
    assert "ocr-mcp" in result.stdout
    assert "0.1.0" in result.stdout
```

- [ ] **Step 5: Install and run test**

Run: `uv sync` (first run downloads paddlepaddle — hundreds of MB, several minutes — be patient).
Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore uv.lock src/ tests/
git commit -m "feat: 初始化项目骨架与构建配置

why: 建立 uv+hatchling src 布局骨架,锁定 python 3.11 与 paddle/mcp 依赖,
为后续分层实现奠定可安装可运行的基础。

what: pyproject/.python-version/.gitignore + 包入口 + smoke 测试。"
```

---

### Task 2: Settings config + cross-platform paths

**Files:**
- Create: `src/ocr_mcp/config.py`, `src/ocr_mcp/paths.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `config.Settings` (frozen dataclass, `.from_env()`), `paths.lock_file_path(s)`, `paths.socket_path(s)`, `paths.log_file_path(s, name)`, `paths.ensure_data_dir(s)`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import os
from pathlib import Path
from ocr_mcp.config import Settings
from ocr_mcp import paths


def test_defaults(isolated_env):
    s = Settings.from_env()
    assert s.model_det == "PP-OCRv6_medium_det"
    assert s.model_rec == "PP-OCRv6_medium_rec"
    assert s.idle_timeout == 600
    assert s.request_timeout == 60
    assert s.data_dir == isolated_env


def test_env_overrides(monkeypatch, tmp_path):
    d = tmp_path / "xd"
    monkeypatch.setenv("OCR_MCP_DATA_DIR", str(d))
    monkeypatch.setenv("OCR_MCP_MODEL_DET", "PP-OCRv6_small_det")
    monkeypatch.setenv("OCR_MCP_IDLE_TIMEOUT", "120")
    s = Settings.from_env()
    assert s.model_det == "PP-OCRv6_small_det"
    assert s.idle_timeout == 120
    assert s.data_dir == d


def test_paths_under_data_dir(isolated_env):
    s = Settings.from_env()
    assert paths.lock_file_path(s) == isolated_env / "daemon.lock"
    assert paths.socket_path(s) == isolated_env / "daemon.sock"
    assert paths.log_file_path(s, "daemon") == isolated_env / "daemon.log"
    # ensure_data_dir creates if missing
    s2 = Settings.from_env()
    s2 = Settings(**{**s2.__dict__, "data_dir": isolated_env / "nested" / "dir"})
    p = paths.ensure_data_dir(s2)
    assert p.exists() and p.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (module `ocr_mcp.config` not found).

- [ ] **Step 3: Write `config.py`**

`src/ocr_mcp/config.py`:
```python
from __future__ import annotations
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

DEFAULT_DET = "PP-OCRv6_medium_det"
DEFAULT_REC = "PP-OCRv6_medium_rec"


@dataclass(frozen=True)
class Settings:
    model_det: str = DEFAULT_DET
    model_rec: str = DEFAULT_REC
    model_dir: str | None = None
    data_dir: Path = None  # type: ignore[assignment]  # filled by from_env
    idle_timeout: float = 600.0
    request_timeout: float = 60.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        import platformdirs
        data_dir = Path(os.environ.get("OCR_MCP_DATA_DIR") or platformdirs.user_data_dir("ocr-mcp"))
        return cls(
            model_det=os.environ.get("OCR_MCP_MODEL_DET", DEFAULT_DET),
            model_rec=os.environ.get("OCR_MCP_MODEL_REC", DEFAULT_REC),
            model_dir=os.environ.get("OCR_MCP_MODEL_DIR") or None,
            data_dir=data_dir,
            idle_timeout=float(os.environ.get("OCR_MCP_IDLE_TIMEOUT", "600")),
            request_timeout=float(os.environ.get("OCR_MCP_REQUEST_TIMEOUT", "60")),
            log_level=os.environ.get("OCR_MCP_LOG_LEVEL", "INFO"),
        )
```

- [ ] **Step 4: Write `paths.py`**

`src/ocr_mcp/paths.py`:
```python
from __future__ import annotations
from pathlib import Path
from ocr_mcp.config import Settings


def lock_file_path(s: Settings) -> Path:
    return s.data_dir / "daemon.lock"


def socket_path(s: Settings) -> Path:
    return s.data_dir / "daemon.sock"


def log_file_path(s: Settings, name: str) -> Path:
    return s.data_dir / f"{name}.log"


def ensure_data_dir(s: Settings) -> Path:
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s.data_dir
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all 3).

- [ ] **Step 6: Commit**

```bash
git add src/ocr_mcp/config.py src/ocr_mcp/paths.py tests/test_config.py
git commit -m "feat: 添加配置项与跨平台路径

why: 后续 daemon/server 需要统一的环境变量配置与跨平台 lock/socket/log 路径。

what: Settings dataclass(from_env) + platformdirs 路径函数。"
```

---

### Task 3: NDJSON communication protocol

**Files:**
- Create: `src/ocr_mcp/protocol.py`, `tests/test_protocol.py`

**Interfaces:**
- Produces: `OcrLine`, `OcrResult` dataclasses; `make_request(op, token, payload, id=None)`, `make_success(id, result)`, `make_error(id, code, message)`, `encode_message(d) -> bytes`, `decode_message(line) -> dict`.

- [ ] **Step 1: Write the failing test**

`tests/test_protocol.py`:
```python
import json
import pytest
from ocr_mcp.protocol import (
    OcrLine, OcrResult, make_request, make_success, make_error,
    encode_message, decode_message,
)


def test_request_envelope_has_required_fields():
    m = make_request("ocr", "tok", {"image_path": "/a.png"})
    assert m["v"] == 1
    assert m["op"] == "ocr"
    assert m["token"] == "tok"
    assert m["payload"] == {"image_path": "/a.png"}
    assert isinstance(m["id"], str) and len(m["id"]) > 0


def test_encode_decode_roundtrip():
    m = make_request("ping", "tok", {})
    data = encode_message(m)
    assert data.endswith(b"\n")
    assert decode_message(data) == m


def test_encode_preserves_unicode():
    m = make_success("1", {"text": "借：销售费用"})
    data = encode_message(m)
    assert "借" in data.decode("utf-8")  # not \u-escaped
    assert decode_message(data) == m


def test_success_and_error_envelopes():
    assert make_success("1", {"a": 1}) == {"v": 1, "id": "1", "ok": True, "result": {"a": 1}}
    assert make_error("1", "IMAGE_NOT_FOUND", "missing") == {
        "v": 1, "id": "1", "ok": False,
        "error": {"code": "IMAGE_NOT_FOUND", "message": "missing"},
    }


def test_decode_rejects_bad_json():
    with pytest.raises(json.JSONDecodeError):
        decode_message(b"not json\n")


def test_ocr_result_to_dict_drops_optional_fields():
    line_full = OcrLine(text="hi", score=0.9, poly=[[0, 0], [1, 1]])
    assert line_full.to_dict() == {"text": "hi", "score": 0.9, "poly": [[0, 0], [1, 1]]}
    line_min = OcrLine(text="hi", score=None, poly=None)
    assert line_min.to_dict() == {"text": "hi"}


def test_ocr_result_dict_has_text_and_lines():
    r = OcrResult(image_path="/a.png", width=10, height=20,
                  lines=[OcrLine("x", 0.1, None)], text="x", elapsed_ms=5)
    d = r.to_dict()
    assert d["text"] == "x"
    assert d["lines"][0]["text"] == "x"
    assert d["elapsed_ms"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `protocol.py`**

`src/ocr_mcp/protocol.py`:
```python
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field

PROTOCOL_VERSION = 1


@dataclass
class OcrLine:
    text: str
    score: float | None = None
    poly: list[list[int]] | None = None

    def to_dict(self) -> dict:
        d: dict = {"text": self.text}
        if self.score is not None:
            d["score"] = self.score
        if self.poly is not None:
            d["poly"] = self.poly
        return d


@dataclass
class OcrResult:
    image_path: str
    width: int
    height: int
    lines: list[OcrLine]
    text: str
    elapsed_ms: int

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "lines": [ln.to_dict() for ln in self.lines],
            "text": self.text,
            "elapsed_ms": self.elapsed_ms,
        }


def make_request(op: str, token: str, payload: dict, id: str | None = None) -> dict:
    return {"v": PROTOCOL_VERSION, "id": id or uuid.uuid4().hex,
            "op": op, "token": token, "payload": payload}


def make_success(id: str, result: dict) -> dict:
    return {"v": PROTOCOL_VERSION, "id": id, "ok": True, "result": result}


def make_error(id: str, code: str, message: str) -> dict:
    return {"v": PROTOCOL_VERSION, "id": id, "ok": False,
            "error": {"code": code, "message": message}}


def encode_message(d: dict) -> bytes:
    return (json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8")


def decode_message(line: bytes) -> dict:
    return json.loads(line.decode("utf-8"))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: PASS (all 7).

- [ ] **Step 5: Commit**

```bash
git add src/ocr_mcp/protocol.py tests/test_protocol.py
git commit -m "feat: 添加 stdio-daemon 间 NDJSON 通信协议

why: 双层进程模型需要一个轻量、可演进、unicode 友好的请求/响应协议。

what: 信封(v/id/op/token/payload + ok/result/error) + OcrResult/OcrLine + 编解码。"
```

---

### Task 4: Transport abstraction + AF_UNIX implementation

**Files:**
- Create: `src/ocr_mcp/transport/__init__.py`, `src/ocr_mcp/transport/base.py`, `src/ocr_mcp/transport/unix_sock.py`, `tests/test_transport_unix.py`

**Interfaces:**
- Produces: `transport.base.Transport` (Protocol), `transport.unix_sock.UnixTransport`, `transport.unix_sock.serve_unix(path, handler)`.

- [ ] **Step 1: Write the failing test**

`tests/test_transport_unix.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transport_unix.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `transport/base.py`**

`src/ocr_mcp/transport/__init__.py`: empty.

`src/ocr_mcp/transport/base.py`:
```python
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    async def connect(self) -> None: ...
    async def send(self, data: bytes) -> None: ...
    async def recv_line(self) -> bytes: ...
    async def close(self) -> None: ...
```

- [ ] **Step 4: Write `transport/unix_sock.py`**

`src/ocr_mcp/transport/unix_sock.py`:
```python
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
    return await asyncio.start_server(handler, path=str(path))
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_transport_unix.py -v`
Expected: PASS (both). Note: on Windows pre-10-1803 this test errors — acceptable (those systems use named pipe, Task 5).

- [ ] **Step 6: Commit**

```bash
git add src/ocr_mcp/transport/__init__.py src/ocr_mcp/transport/base.py src/ocr_mcp/transport/unix_sock.py tests/test_transport_unix.py
git commit -m "feat: 添加 transport 抽象与 AF_UNIX 实现

why: daemon 与 stdio 前端需要一个跨平台、可替换的传输层;AF_UNIX 在 POSIX 与 Win10+ 通用。

what: Transport Protocol + UnixTransport(connect/send/recv_line/close) + serve_unix(含残留 socket 清理)。"
```

---

### Task 5: Windows named-pipe transport (fallback)

**Files:**
- Create: `src/ocr_mcp/transport/named_pipe.py`, `tests/test_transport_pipe.py`
- Modify: `pyproject.toml` (add platform-conditional `pywin32`)

**Interfaces:**
- Produces: `transport.named_pipe.PipeTransport`, `transport.named_pipe.serve_pipe(name, handler)`.

- [ ] **Step 1: Add platform-conditional dependency**

In `pyproject.toml`, add to `dependencies`:
```toml
    "pywin32>=306 ; sys_platform == 'win32'",
```

- [ ] **Step 2: Write the test (skipped off-Windows)**

`tests/test_transport_pipe.py`:
```python
import sys
import pytest
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named pipe is Windows-only")


@pytest.mark.asyncio
async def test_pipe_loopback(tmp_path):
    from ocr_mcp.transport.named_pipe import PipeTransport, serve_pipe
    import asyncio
    name = r"\\.\pipe\ocr-mcp-test"

    async def handler(reader, writer):
        line = await reader.readline()
        writer.write(b"ACK\n")
        await writer.drain()
        writer.close()

    server = await serve_pipe(name, handler)
    try:
        c = PipeTransport(name)
        await c.connect()
        await c.send(b"HI\n")
        assert await c.recv_line() == b"ACK\n"
        await c.close()
    finally:
        server.close()
        await server.wait_closed()
```

- [ ] **Step 3: Run test (skip on Linux/macOS, verify pass on Windows)**

Run: `uv run pytest tests/test_transport_pipe.py -v`
Expected on Linux: 1 skipped. (Real verification happens on a Windows machine.)

- [ ] **Step 4: Write `transport/named_pipe.py`**

`src/ocr_mcp/transport/named_pipe.py`:
```python
from __future__ import annotations
import asyncio
import sys

# pywin32 is Windows-only; import lazily so the module imports on other platforms
# without raising (it is only selected on Windows by the transport factory).


class _PipeStreamProtocol(asyncio.Protocol):
    """Adapts a Windows named pipe into an asyncio stream (reader/writer)."""
    def __init__(self):
        self.transport = None
        self._reader = asyncio.StreamReader()
        self._writer = None

    def connection_made(self, transport):
        self.transport = transport
        self._writer = asyncio.StreamWriter(transport, self, self._reader, asyncio.get_event_loop())

    def data_received(self, data):
        self._reader.feed_data(data)

    def eof_received(self):
        self._reader.feed_eof()

    def connection_lost(self, exc):
        self._reader.feed_eof()


class PipeTransport:
    def __init__(self, name: str):
        self.name = name
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("named pipes are only available on Windows")
        import win32pipe  # type: ignore
        import win32file  # type: ignore
        handle = win32file.CreateFile(
            self.name, win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0, None, win32file.OPEN_EXISTING, 0, None,
        )
        loop = asyncio.get_event_loop()
        proto = _PipeStreamProtocol()
        await loop.connect_pipe(handle)  # wrap into asyncio transport
        self._reader = proto._reader
        self._writer = proto._writer

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


async def serve_pipe(name: str, handler) -> asyncio.base_events.Server:
    if sys.platform != "win32":
        raise RuntimeError("named pipes are only available on Windows")
    import win32pipe  # type: ignore
    # asyncio.start_server on Windows can bind a pipe name via `path` argument;
    # we wrap it so the same handler signature as serve_unix is used.
    return await asyncio.start_server(handler, path=name)
```

> **Verification note:** `asyncio.start_server(path=name)` on Windows treats `path` as a pipe name `\\.\pipe\...`. If the pywin32 manual path above proves fragile on a given Windows build, fall back to `asyncio.start_server(handler, path=name)` for both client and server (using `asyncio.open_unix_connection(path=name)` on the client) — Python 3.11 on Win10+ unifies this path. Confirm during execution on a Windows host and simplify if possible.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ocr_mcp/transport/named_pipe.py tests/test_transport_pipe.py
git commit -m "feat: 添加 Windows 命名管道 transport(老 Windows 回退)

why: 兑现跨平台兼容承诺,为 <Win10 1803(不支持 AF_UNIX)的环境提供回退传输。

what: PipeTransport + serve_pipe + pywin32 平台条件依赖;非 Windows 测试 skip。"
```

---

### Task 6: OCR engine wrapper (deferred paddle import)

**Files:**
- Create: `src/ocr_mcp/backend/__init__.py`, `src/ocr_mcp/backend/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `Settings` (model_det, model_rec, model_dir).
- Produces: `backend.engine.OcrEngine` with `.predict(image_path, return_polys=True, return_scores=True) -> OcrResult`.

- [ ] **Step 1: Write the failing test (paddle mocked)**

`tests/test_engine.py`:
```python
from __future__ import annotations
import sys
import types
import pytest
from ocr_mcp.backend.engine import OcrEngine
from ocr_mcp.config import Settings
from pathlib import Path


def _install_fake_paddle(monkeypatch, rec_texts, rec_scores, dt_polys, width=100, height=50):
    """Inject a fake `paddleocr` module returning a controlled OCRResult-dict."""
    fake = types.ModuleType("paddleocr")

    class FakeResult(dict):
        pass

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def predict(self, path):
            res = FakeResult()
            res["rec_texts"] = list(rec_texts)
            res["rec_scores"] = list(rec_scores)
            res["dt_polys"] = list(dt_polys)
            # PaddleOCR also exposes image shape via 'source_img_obj' etc.; we synthesize:
            res["_fake_w"] = width
            res["_fake_h"] = height
            return [res]

    fake.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake)


def test_predict_normalizes_to_ocr_result(isolated_env, monkeypatch, tmp_path):
    _install_fake_paddle(monkeypatch, ["借：费用", "贷：现金"], [0.97, 0.88],
                         [[[0, 0], [10, 0], [10, 5], [0, 5]]] * 2)
    img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")  # exists, readable
    eng = OcrEngine(Settings.from_env())
    r = eng.predict(str(img), return_polys=True, return_scores=True)
    assert r.text == "借：费用\n贷：现金"
    assert len(r.lines) == 2
    assert r.lines[0].text == "借：费用"
    assert r.lines[0].score == pytest.approx(0.97)
    assert r.lines[0].poly == [[0, 0], [10, 0], [10, 5], [0, 5]]
    assert r.elapsed_ms >= 0


def test_predict_respects_return_flags(isolated_env, monkeypatch, tmp_path):
    _install_fake_paddle(monkeypatch, ["a"], [0.5], [[[0, 0], [1, 0], [1, 1], [0, 1]]])
    img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")
    eng = OcrEngine(Settings.from_env())
    r = eng.predict(str(img), return_polys=False, return_scores=False)
    d = r.to_dict()
    assert d["lines"][0] == {"text": "a"}


def test_singleton_loads_once(isolated_env, monkeypatch, tmp_path):
    _install_fake_paddle(monkeypatch, ["a"], [0.5], [[[0, 0], [1, 0], [1, 1], [0, 1]]])
    img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")
    eng = OcrEngine(Settings.from_env())
    eng.predict(str(img))
    first = eng._ocr
    eng.predict(str(img))
    assert eng._ocr is first  # not reloaded


def test_mkldnn_disabled(isolated_env, monkeypatch, tmp_path):
    _install_fake_paddle(monkeypatch, ["a"], [0.5], [[[0, 0], [1, 0], [1, 1], [0, 1]]])
    img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")
    eng = OcrEngine(Settings.from_env())
    eng.predict(str(img))
    assert eng._ocr.kwargs.get("enable_mkldnn") is False


def test_image_not_found_raises(isolated_env, monkeypatch):
    _install_fake_paddle(monkeypatch, [], [], [])
    eng = OcrEngine(Settings.from_env())
    with pytest.raises(FileNotFoundError):
        eng.predict("/no/such/file.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `backend/engine.py`**

`src/ocr_mcp/backend/__init__.py`: empty.

`src/ocr_mcp/backend/engine.py`:
```python
from __future__ import annotations
import os
import time
from pathlib import Path

from ocr_mcp.config import Settings
from ocr_mcp.protocol import OcrLine, OcrResult


class ImageNotFound(FileNotFoundError):
    pass


class ImageUnreadable(OSError):
    pass


class UnsupportedFormat(ValueError):
    pass


class InferenceFailed(RuntimeError):
    pass


class OcrEngine:
    """Wraps PaddleOCR as a process-singleton. Paddle is imported lazily so the
    stdio frontend process never loads it."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._ocr = None  # type: ignore[assignment]

    def _load(self):
        # Deferred import: keeps paddle out of the stdio server process.
        from paddleocr import PaddleOCR  # type: ignore
        kwargs = dict(
            use_textline_orientation=False,
            lang="ch",
            enable_mkldnn=False,  # MUST be False: paddlepaddle 3.3 PIR+onednn crashes
            text_detection_model_name=self._settings.model_det,
            text_recognition_model_name=self._settings.model_rec,
        )
        if self._settings.model_dir:
            # Custom model cache dir for offline/intranet deploy.
            # Verification point: paddleocr 3.x may read PADDLEX_HOME env instead.
            os.environ.setdefault("PADDLEX_HOME", self._settings.model_dir)
        self._ocr = PaddleOCR(**kwargs)

    def _ensure_loaded(self):
        if self._ocr is None:
            self._load()
        return self._ocr

    def predict(self, image_path: str,
                return_polys: bool = True, return_scores: bool = True) -> OcrResult:
        p = Path(image_path)
        if not p.exists():
            raise ImageNotFound(f"image file not found: {image_path}")
        if not os.access(p, os.R_OK):
            raise ImageUnreadable(f"image file not readable: {image_path}")
        ocr = self._ensure_loaded()
        t0 = time.monotonic()
        try:
            results = ocr.predict(str(p))
        except Exception as e:  # paddle raises various errors
            raise InferenceFailed(str(e)) from e
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        lines: list[OcrLine] = []
        texts_for_join: list[str] = []
        width = height = 0
        for res in results:
            rec_texts = res.get("rec_texts") or []
            rec_scores = res.get("rec_scores") or []
            dt_polys = res.get("dt_polys") or []
            for idx, t in enumerate(rec_texts):
                score = float(rec_scores[idx]) if idx < len(rec_scores) and return_scores else None
                poly = dt_polys[idx] if idx < len(dt_polys) and return_polys else None
                poly = [ [int(x) for x in pt] for pt in poly ] if poly else None
                lines.append(OcrLine(text=str(t), score=score, poly=poly))
                texts_for_join.append(str(t))
            # best-effort image size from result if present
            if "source_img_obj" in res:
                pass  # PaddleOCR-specific; width/height left 0 if unavailable
        return OcrResult(
            image_path=str(p),
            width=width, height=height,
            lines=lines,
            text="\n".join(texts_for_join),
            elapsed_ms=elapsed_ms,
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add src/ocr_mcp/backend/__init__.py src/ocr_mcp/backend/engine.py tests/test_engine.py
git commit -m "feat: 添加 OCR 引擎封装(延迟 import、单例、结果标准化)

why: 将 PaddleOCR 封装为进程内单例,延迟 import 使 stdio 前端不加载 paddle;
标准化 rec_texts/rec_scores/dt_polys 为 OcrResult。

what: OcrEngine.predict + 延迟加载 + enable_mkldnn=False + 错误类型。"
```

---

### Task 7: Daemon worker (listen + serial queue + refcount + idle timeout + token)

**Files:**
- Create: `src/ocr_mcp/backend/worker.py`, `tests/test_worker.py`

**Interfaces:**
- Consumes: `Settings`, `OcrEngine`, `protocol.*`, `transport.unix_sock.serve_unix`.
- Produces: `backend.worker.DaemonWorker` with `.run_async()` (async, serves until idle-timeout or cancelled) and `.token` property.

- [ ] **Step 1: Write the failing test**

`tests/test_worker.py`:
```python
import asyncio
import pytest
from ocr_mcp.backend.worker import DaemonWorker
from ocr_mcp.config import Settings
from ocr_mcp import paths, protocol


class FakeEngine:
    def __init__(self):
        self.calls = []
    def predict(self, image_path, return_polys=True, return_scores=True):
        self.calls.append(image_path)
        from ocr_mcp.protocol import OcrResult, OcrLine
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
    w, task = await _start_worker(s, eng)
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
async def test_serial_queue_preserves_order(isolated_env, tmp_path):
    eng = FakeEngine()
    s = Settings.from_env()
    w, task = await _start_worker(s, eng)
    try:
        from ocr_mcp.transport.unix_sock import UnixTransport
        img = tmp_path / "x.png"; img.write_bytes(b"\x89PNG")
        async def one(path):
            c = UnixTransport(paths.socket_path(s)); await c.connect()
            await c.send(protocol.encode_message(
                protocol.make_request("ocr", w.token, {"image_path": path})))
            r = protocol.decode_message(await c.recv_line()); await c.close()
            return r["result"]["image_path"]
        results = await asyncio.gather(*[one(str(img)) for _ in range(3)])
        assert results == [str(img)] * 3
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_worker.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `backend/worker.py`**

`src/ocr_mcp/backend/worker.py`:
```python
from __future__ import annotations
import asyncio
import concurrent.futures
import logging
import secrets

from ocr_mcp.backend.engine import (ImageNotFound, ImageUnreadable, InferenceFailed)
from ocr_mcp.config import Settings
from ocr_mcp import paths, protocol
from ocr_mcp.transport.unix_sock import serve_unix

log = logging.getLogger("ocr_mcp.daemon")


class DaemonWorker:
    def __init__(self, settings: Settings, engine):
        self.settings = settings
        self.engine = engine
        self.token = secrets.token_hex(16)
        self._clients = 0
        self._idle_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self.stopped = False
        # Single-thread executor => serializes predict calls (paddle not thread-safe).
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def run_async(self) -> None:
        paths.ensure_data_dir(self.settings)
        sock = paths.socket_path(self.settings)
        # Write lock file so clients can find us + authenticate.
        self._write_lock_file(sock)
        server = await serve_unix(sock, self._on_connect)
        log.info("daemon listening on %s", sock)
        try:
            # Wait until idle timeout fires with zero clients.
            while not self._stop_event.is_set():
                if self._clients == 0:
                    try:
                        await asyncio.wait_for(self._idle_event.wait(), timeout=self.settings.idle_timeout)
                    except asyncio.TimeoutError:
                        log.info("idle timeout reached, shutting down")
                        break
                else:
                    # clients connected: wait for them to disconnect
                    self._idle_event.clear()
                    await self._idle_event.wait()
        finally:
            self.stopped = True
            server.close()
            await server.wait_closed()
            self._remove_lock_file()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _write_lock_file(self, sock) -> None:
        import json, os
        lf = paths.lock_file_path(self.settings)
        data = {"pid": os.getpid(), "socket": str(sock), "token": self.token}
        lf.write_text(json.dumps(data))

    def _remove_lock_file(self) -> None:
        try:
            paths.lock_file_path(self.settings).unlink()
        except FileNotFoundError:
            pass

    async def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._clients += 1
        self._idle_event.set()  # wake the run loop: we have a client
        try:
            while not writer.is_closing():
                line = await reader.readline()
                if not line:
                    break
                resp = await self._handle(line)
                writer.write(protocol.encode_message(resp))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._clients -= 1
            try:
                writer.close()
            except Exception:
                pass
            if self._clients == 0:
                self._idle_event.set()  # wake loop to start idle timer

    async def _handle(self, line: bytes) -> dict:
        try:
            msg = protocol.decode_message(line)
        except Exception as e:
            return protocol.make_error("?", "BAD_REQUEST", f"invalid json: {e}")
        mid = msg.get("id", "?")
        if msg.get("token") != self.token:
            return protocol.make_error(mid, "UNAUTHORIZED", "bad token")
        op = msg.get("op")
        if op == "ping":
            import os, time
            return protocol.make_success(mid, {"status": "ready", "pid": os.getpid(),
                                               "uptime_s": 0, "clients": self._clients,
                                               "model": f"{self.settings.model_det} + {self.settings.model_rec}"})
        if op == "ocr":
            return await self._handle_ocr(mid, msg.get("payload", {}))
        return protocol.make_error(mid, "UNKNOWN_OP", f"unknown op: {op}")

    async def _handle_ocr(self, mid: str, payload: dict) -> dict:
        image_path = payload.get("image_path")
        if not image_path:
            return protocol.make_error(mid, "BAD_REQUEST", "image_path required")
        opts = payload.get("options", {}) or {}
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: self.engine.predict(
                        image_path,
                        return_polys=bool(opts.get("return_polys", True)),
                        return_scores=bool(opts.get("return_scores", True)),
                    ),
                ),
                timeout=self.settings.request_timeout,
            )
        except asyncio.TimeoutError:
            return protocol.make_error(mid, "TIMEOUT", "ocr request timed out")
        except ImageNotFound as e:
            return protocol.make_error(mid, "IMAGE_NOT_FOUND", str(e))
        except ImageUnreadable as e:
            return protocol.make_error(mid, "IMAGE_UNREADABLE", str(e))
        except InferenceFailed as e:
            return protocol.make_error(mid, "INFERENCE_FAILED", str(e))
        except Exception as e:
            return protocol.make_error(mid, "INFERENCE_FAILED", f"{type(e).__name__}: {e}")
        return protocol.make_success(mid, result.to_dict())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_worker.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: Commit**

```bash
git add src/ocr_mcp/backend/worker.py tests/test_worker.py
git commit -m "feat: 添加 daemon worker(监听/串行队列/引用计数/空闲超时/token)

why: 常驻后端需正确处理多 client 并发(单实例串行)、生命周期(引用计数+空闲超时)与鉴权。

what: DaemonWorker.run_async + 单线程 executor 串行 predict + token 校验 + lock file 写入/清理。"
```

---

### Task 8: Daemon lifecycle (detect / spawn / lock file / stale cleanup / race)

**Files:**
- Create: `src/ocr_mcp/backend/lifecycle.py`, `tests/test_lifecycle.py`

**Interfaces:**
- Consumes: `Settings`, `paths`, `protocol`, `transport.unix_sock.UnixTransport`.
- Produces: `backend.lifecycle.ensure_daemon(settings) -> UnixTransport`, `backend.lifecycle.read_lock(settings)`, `backend.lifecycle.is_alive(settings, lock)`.

- [ ] **Step 1: Write the failing test**

`tests/test_lifecycle.py`:
```python
import asyncio
import json
import pytest
from ocr_mcp.backend import lifecycle
from ocr_mcp.config import Settings
from ocr_mcp import paths, protocol


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
            from ocr_mcp.protocol import OcrResult, OcrLine
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
            from ocr_mcp.protocol import OcrResult, OcrLine
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

    async def fake_spawn_and_wait(settings):
        # Simulate a daemon coming up by starting a real DaemonWorker
        from ocr_mcp.backend.worker import DaemonWorker
        class FakeEngine:
            def predict(self, *a, **k):
                from ocr_mcp.protocol import OcrResult, OcrLine
                return OcrResult("x", 1, 1, [OcrLine("a", 1.0, None)], "a", 1)
        w = DaemonWorker(settings, FakeEngine())
        asyncio.create_task(w.run_async())
        await asyncio.sleep(0.05)
        spawned["ok"] = True

    monkeypatch.setattr(lifecycle, "_spawn_and_wait_ready", fake_spawn_and_wait)
    t = await lifecycle.ensure_daemon(s)
    assert spawned["ok"] is True
    await t.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `backend/lifecycle.py`**

`src/ocr_mcp/backend/lifecycle.py`:
```python
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import subprocess
from pathlib import Path

from ocr_mcp.config import Settings
from ocr_mcp import paths, protocol
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
    except (OSError, asyncio.TimeoutError):
        return False
    try:
        await t.send(protocol.encode_message(protocol.make_request("ping", lock.get("token", ""), {})))
        line = await asyncio.wait_for(t.recv_line(), timeout=5.0)
        resp = protocol.decode_message(line)
        return bool(resp.get("ok"))
    except Exception:
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
    popen_kwargs = dict(
        stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
        close_fds=True,
    )
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_lifecycle.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add src/ocr_mcp/backend/lifecycle.py tests/test_lifecycle.py
git commit -m "feat: 添加 daemon 生命周期管理(探测/拉起/lock file/陈旧清理/竞争仲裁)

why: stdio 前端需可靠地找到或拉起唯一的常驻后端,并处理陈旧 lock 与多 client 同时拉起的竞争。

what: ensure_daemon + read_lock + is_alive(ping) + detached spawn + 原子 spawn.lock。"
```

---

### Task 9: stdio MCP server frontend (ocr_image tool, no paddle import)

**Files:**
- Create: `src/ocr_mcp/server.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `lifecycle.ensure_daemon`, `protocol`, `transport.UnixTransport`.
- Produces: `server.create_server() -> mcp.server.Server`, `server.BackendClient`.

- [ ] **Step 1: Write the failing test**

`tests/test_server.py`:
```python
import asyncio
import json
import pytest
from ocr_mcp import server, protocol, paths
from ocr_mcp.config import Settings


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
            protocol.make_success(self.last_sent["id"],
                                  {"image_path": self.last_sent["payload"]["image_path"],
                                   "width": 10, "height": 5,
                                   "lines": [{"text": "T1", "score": 0.9, "poly": [[0,0],[1,0],[1,1],[0,1]]}],
                                   "text": "T1", "elapsed_ms": 7}))
    async def close(self):
        pass


@pytest.mark.asyncio
async def test_call_ocr_image_returns_json_text(isolated_env, tmp_path, monkeypatch):
    fake = FakeBackend()
    async def fake_ensure(settings):
        return fake
    monkeypatch.setattr(server.lifecycle, "ensure_daemon", fake_ensure)

    async def fake_ensure(settings):
        return fake
    monkeypatch.setattr(server.lifecycle, "ensure_daemon", fake_ensure)
    out = await server.handle_ocr_image({"image_path": str(tmp_path / "x.png")})
    assert out.isError is False
    payload = json.loads(out.content[0].text)
    assert payload["text"] == "T1"
    assert fake.last_sent["op"] == "ocr"
    assert fake.last_sent["payload"]["image_path"] == str(tmp_path / "x.png")


@pytest.mark.asyncio
async def test_call_ocr_image_missing_path_errors(isolated_env, monkeypatch):
    monkeypatch.setattr(server.lifecycle, "ensure_daemon",
                        lambda settings: asyncio.sleep(0, result=FakeBackend()))
    out = await server.handle_ocr_image({})
    assert out.isError is True


@pytest.mark.asyncio
async def test_call_ocr_image_propagates_daemon_error(isolated_env, tmp_path, monkeypatch):
    class ErrBackend(FakeBackend):
        async def recv_line(self):
            return protocol.encode_message(
                protocol.make_error(self.last_sent["id"], "IMAGE_NOT_FOUND", "nope"))
    fake = ErrBackend()
    monkeypatch.setattr(server.lifecycle, "ensure_daemon", lambda s: asyncio.sleep(0, result=fake))
    out = await server.handle_ocr_image({"image_path": str(tmp_path / "x.png")})
    assert out.isError is True
    assert "IMAGE_NOT_FOUND" in out.content[0].text
```

> **Note:** The test exercises the module-level `handle_ocr_image` helper directly, which wraps the same logic the MCP `call_tool` handler invokes — keeping the test decoupled from mcp SDK internals while still exercising request→daemon→response conversion. An autouse `_reset_client` fixture resets the cached `BackendClient` between tests so each reconnects through its own monkeypatched `ensure_daemon`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write `server.py`**

`src/ocr_mcp/server.py`:
```python
from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from ocr_mcp.backend import lifecycle
from ocr_mcp.config import Settings
from ocr_mcp import protocol

log = logging.getLogger("ocr_mcp.server")

# Re-exported for tests/imports.
__all__ = ["create_server", "handle_ocr_image", "BackendClient", "CallResult"]

OCR_TOOL = Tool(
    name="ocr_image",
    description="Recognize text in a local image file. Returns text + per-line confidence + coordinates.",
    inputSchema={
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "absolute path to a local image file"},
            "return_polys": {"type": "boolean", "default": True, "description": "include per-line polygon coords"},
            "return_scores": {"type": "boolean", "default": True, "description": "include per-line confidence"},
        },
        "required": ["image_path"],
    },
)


@dataclass
class CallResult:
    isError: bool
    content: list[TextContent]


class BackendClient:
    """Holds one persistent connection to the daemon for the server's lifetime.
    Keeping it persistent is what makes the daemon's refcount reflect this client."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.transport = None  # type: ignore[assignment]

    async def connect(self):
        if self.transport is None:
            self.transport = await lifecycle.ensure_daemon(self.settings)

    async def request(self, op: str, payload: dict) -> dict:
        await self.connect()
        lock = lifecycle.read_lock(self.settings) or {}
        token = lock.get("token", "")
        req = protocol.make_request(op, token, payload)
        await self.transport.send(protocol.encode_message(req))
        line = await asyncio.wait_for(self.transport.recv_line(), timeout=self.settings.request_timeout)
        return protocol.decode_message(line)

    async def close(self):
        if self.transport is not None:
            await self.transport.close()
            self.transport = None


# Module-level singleton backend client (one persistent connection per server process).
_client: BackendClient | None = None


async def _get_client() -> BackendClient:
    global _client
    if _client is None:
        _client = BackendClient(Settings.from_env())
    await _client.connect()
    return _client


async def handle_ocr_image(arguments: dict[str, Any]) -> CallResult:
    image_path = arguments.get("image_path")
    if not image_path or not isinstance(image_path, str):
        return CallResult(isError=True, content=[TextContent(
            type="text", text="ERROR: 'image_path' (string) is required.")])
    payload = {"image_path": image_path, "options": {
        "return_polys": bool(arguments.get("return_polys", True)),
        "return_scores": bool(arguments.get("return_scores", True)),
    }}
    try:
        client = await _get_client()
        resp = await client.request("ocr", payload)
    except Exception as e:
        return CallResult(isError=True, content=[TextContent(
            type="text", text=f"ERROR: DAEMON_UNAVAILABLE: {e}")])
    if not resp.get("ok"):
        err = resp.get("error", {})
        return CallResult(isError=True, content=[TextContent(
            type="text", text=f"ERROR: {err.get('code', 'UNKNOWN')}: {err.get('message', '')}")])
    return CallResult(isError=False, content=[TextContent(
        type="text", text=json.dumps(resp["result"], ensure_ascii=False)])


def create_server() -> Server:
    server = Server("ocr-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [OCR_TOOL]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "ocr_image":
            res = await handle_ocr_image(arguments or {})
            return res.content
        return [TextContent(type="text", text=f"ERROR: unknown tool {name}")]

    return server
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (all 3).

- [ ] **Step 5: Commit**

```bash
git add src/ocr_mcp/server.py tests/test_server.py
git commit -m "feat: 添加 stdio MCP server 前端与 ocr_image tool

why: 暴露给 MCP client 的瘦前端,经持久连接转发请求到 daemon,自身不 import paddle。

what: create_server + ocr_image tool + BackendClient(持久连接) + handle_ocr_image 响应转换。"
```

---

### Task 10: CLI dispatch (stdio | serve | download) + entry point

**Files:**
- Modify: `src/ocr_mcp/cli.py`, Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `cli.main(argv)` dispatching `stdio` (default) / `serve` / `download`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import pytest
from ocr_mcp import cli


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "ocr-mcp" in out


def test_unknown_subcommand_errors(capsys):
    assert cli.main(["bogus"]) == 2
    err = capsys.readouterr().err
    assert "bogus" in err or "usage" in err.lower()


@pytest.mark.asyncio
async def test_serve_invokes_worker(monkeypatch, isolated_env):
    called = {"run": False}

    class FakeWorker:
        def __init__(self, *a, **k): pass
        async def run_async(self):
            called["run"] = True

    monkeypatch.setattr(cli, "DaemonWorker", FakeWorker)
    monkeypatch.setattr(cli, "OcrEngine", lambda s: object())
    rc = await cli.serve()
    assert rc == 0
    assert called["run"] is True


@pytest.mark.asyncio
async def test_download_triggers_engine_load(monkeypatch, isolated_env):
    loaded = {"n": 0}
    class FakeEngine:
        def __init__(self, s): self.s = s
        def _load(self): loaded["n"] += 1
        def predict(self, *a, **k): ...
    monkeypatch.setattr(cli, "OcrEngine", FakeEngine)
    rc = await cli.download()
    assert rc == 0
    assert loaded["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (no `serve`/`download` functions).

- [ ] **Step 3: Rewrite `cli.py`**

`src/ocr_mcp/cli.py`:
```python
from __future__ import annotations
import argparse
import asyncio
import logging
import sys

from ocr_mcp import __version__
from ocr_mcp.config import Settings
from ocr_mcp.backend.engine import OcrEngine
from ocr_mcp.backend.worker import DaemonWorker


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ocr-mcp")
    p.add_argument("-V", "--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("stdio", help="run stdio MCP server (default)")
    sub.add_parser("serve", help="run the OCR backend daemon")
    sub.add_parser("download", help="pre-download model files")
    return p


def _run_stdio() -> int:
    import mcp.server.stdio as stdio_server
    from ocr_mcp.server import create_server
    asyncio.run(stdio_server.run(create_server()))  # type: ignore[attr-defined]
    return 0


async def serve() -> int:
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level)
    engine = OcrEngine(settings)
    worker = DaemonWorker(settings, engine)
    await worker.run_async()
    return 0


async def download() -> int:
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level)
    engine = OcrEngine(settings)
    engine._load()  # triggers paddle import + model download
    print("model ready")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"ocr-mcp {__version__}")
        return 0
    command = args.command or "stdio"
    if command == "stdio":
        return _run_stdio()
    if command == "serve":
        return asyncio.run(serve())
    if command == "download":
        return asyncio.run(download())
    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_cli.py tests/test_smoke.py -v`
Expected: PASS (all).

- [ ] **Step 5: Manual smoke check of stdio entry wiring**

Run: `printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' | timeout 10 uv run ocr-mcp stdio 2>/dev/null | head -1`
Expected: a JSON line containing `"serverInfo"` and `"ocr-mcp"`. (This confirms the MCP stdio handshake responds. It will also spawn the daemon on first tool call — not triggered here.)

- [ ] **Step 6: Commit**

```bash
git add src/ocr_mcp/cli.py tests/test_cli.py
git commit -m "feat: 添加 CLI 子命令路由(stdio/serve/download)

why: 统一入口分发三种运行模式;stdio 为默认,方便 MCP client 直接调用。

what: argparse 路由 + serve(DaemonWorker) + download(预下载模型) + stdio(mcp.run)。"
```

---

### Task 11: End-to-end integration test (slow, real model)

**Files:**
- Create: `tests/fixtures/sample.png` (a small image with clear Chinese+digits text), `tests/test_integration.py`

**Interfaces:**
- Exercises the full stack: real daemon (real PaddleOCR) + `lifecycle.ensure_daemon` + `handle_ocr_image`.

- [ ] **Step 1: Prepare a fixture image**

Create `tests/fixtures/sample.png`: a small PNG (e.g. 400×80) containing clear Chinese text + digits (e.g. "订单 20260729 金额 99.00"). Generate it with any image tool and commit it (small binary, acceptable). If generation is hard, capture a screenshot region of the same text.

- [ ] **Step 2: Write the integration test**

`tests/test_integration.py`:
```python
import asyncio
import json
import os
import pytest
from pathlib import Path

pytestmark = pytest.mark.slow

FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"


@pytest.mark.asyncio
@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture image missing")
async def test_end_to_end_ocr(isolated_env):
    from ocr_mcp import server
    out = await server.handle_ocr_image({"image_path": str(FIXTURE)})
    assert out.isError is False, out.content[0].text if out.content else "no content"
    payload = json.loads(out.content[0].text)
    assert isinstance(payload["text"], str)
    assert len(payload["text"]) > 0
    # The model should pick up at least one digit from the fixture.
    assert any(ch.isdigit() for ch in payload["text"])
```

- [ ] **Step 3: Register the `slow` marker**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, add:
```toml
markers = ["slow: end-to-end tests requiring the real PaddleOCR model"]
```

- [ ] **Step 4: Run the integration test (first run downloads/loads the model)**

Run: `uv run pytest tests/test_integration.py -v -m slow`
Expected: PASS (may take 10–60s on first run for model load). If it fails with PIR/onednn error, confirm `enable_mkldnn=False` is set in `engine.py`. If model download fails (no bcebos access), run `uv run ocr-mcp download` first or set `OCR_MCP_MODEL_DIR` to a pre-populated cache.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py tests/fixtures/sample.png pyproject.toml
git commit -m "test: 添加端到端集成测试(真实 PaddleOCR)

why: 单测均 mock 了 paddle,需要一个真实模型测试验证全链路(stdio→daemon→predict→结果)。

what: slow 标记的端到端测试 + fixture 图片。"
```

---

### Task 12: README documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# ocr-mcp

A stdio OCR MCP server that shares a single resident PaddleOCR (PP-OCRv6) backend across all MCP clients on the machine, so the model is loaded only once.

## Install

    uv sync

Pre-download the model (optional; otherwise it auto-downloads on first use):

    uv run ocr-mcp download

## Configure an MCP client

    {
      "mcpServers": {
        "ocr": {
          "command": "uv",
          "args": ["--directory", "/abs/path/ocr-mcp", "run", "ocr-mcp"]
        }
      }
    }

The first `ocr_image` call lazily starts a resident backend daemon (one per user). The daemon unloads itself after `OCR_MCP_IDLE_TIMEOUT` seconds idle (default 600).

## Tool

- `ocr_image(image_path, return_polys=true, return_scores=true)` — recognize a local image file; returns JSON with `text`, per-line `score`, and `poly` coordinates.

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `OCR_MCP_MODEL_DET` | `PP-OCRv6_medium_det` | detection model |
| `OCR_MCP_MODEL_REC` | `PP-OCRv6_medium_rec` | recognition model |
| `OCR_MCP_MODEL_DIR` | `~/.paddlex` | custom model cache (offline deploy) |
| `OCR_MCP_DATA_DIR` | platform user data dir | lock/socket/log location |
| `OCR_MCP_IDLE_TIMEOUT` | `600` | daemon idle exit seconds |
| `OCR_MCP_REQUEST_TIMEOUT` | `60` | per-OCR timeout seconds |
| `OCR_MCP_LOG_LEVEL` | `INFO` | log level |

## Architecture

Two-layer process model: thin stdio MCP servers (one per client, never import paddle) talk NDJSON over Unix socket / named pipe to a single resident daemon that loads the model once and serializes inference. See `docs/superpowers/specs/2026-07-29-ocr-mcp-server-design.md`.

## Development

    uv run pytest -v            # unit tests (paddle mocked)
    uv run pytest -v -m slow    # end-to-end (real model)
    uv run ruff check src tests
```

- [ ] **Step 2: Lint the whole project**

Run: `uv run ruff check src tests`
Expected: no errors (fix any flagged issues inline).

- [ ] **Step 3: Run the full unit suite**

Run: `uv run pytest -v`
Expected: all unit tests PASS (slow test skipped by default unless `-m slow`).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: 添加 README(安装/配置/架构/开发)

why: 让使用者能独立完成安装、MCP client 接入与部署,无需读源码。

what: 安装、client 配置示例、tool 说明、环境变量表、架构概述、开发命令。"
```

---

## Self-Review Notes

**Spec coverage** (spec section → task):

- §3 双层架构 → Tasks 7+8+9
- §4 组件划分 → File Structure (all files mapped to tasks)
- §4 关键约束(server 不 import paddle) → Task 6 (deferred import) + Task 9 (frontend)
- §5 数据流 → Task 9 (`handle_ocr_image`) + Task 8 (`ensure_daemon`) + Task 7 (worker)
- §6 协议 → Task 3
- §6.4 token 鉴权/引用计数/空闲超时/串行 → Task 7
- §7 tool 接口 → Task 9
- §8 CLI 子命令 → Task 10
- §9 配置项 → Task 2
- §10 错误处理 → Task 6 (image/engine) + Task 7 (protocol errors) + Task 8 (DAEMON_UNAVAILABLE/stale) + Task 9 (tool/daemon error mapping)
- §11 测试 → each task's tests + Task 11 integration
- §12 跨平台/依赖 → Task 1 (pyproject) + Task 4 (AF_UNIX) + Task 5 (named pipe) + Task 8 (detached spawn cross-platform)
- §13 已知坑 → Task 6 (enable_mkldnn=False, predict(), res.get())

**Placeholder scan:** The Task 5 "Verification note" (named-pipe asyncio wiring) is an explicit execution-time verification point with a concrete fallback, not a TBD. No other placeholders.

**Type consistency:** `Settings.from_env()` (Task 2) used uniformly. `OcrEngine.predict(image_path, return_polys, return_scores)` (Task 6) matches worker (Task 7) call. `DaemonWorker.run_async()` + `.token` (Task 7) matches lifecycle test (Task 8). `ensure_daemon(settings) -> UnixTransport` (Task 8) matches `BackendClient` (Task 9). `handle_ocr_image(arguments) -> CallResult` (Task 9) matches tests. `cli.serve()/download()` (Task 10) match test signatures.

**Open execution-time verification points (not blockers):**
1. `OCR_MCP_MODEL_DIR` → paddleocr 3.x custom cache mechanism (env `PADDLEX_HOME` vs init kwarg) — Task 6, confirm on a machine with model present.
2. `asyncio.start_server(path=...)` / `open_unix_connection` on Windows for unified AF_UNIX vs the pywin32 manual path — Task 5, confirm on Windows host.
3. `mcp.server.stdio.run` exact import path for SDK 1.9 — Task 10 step 5 confirms via the live handshake; adjust import if API differs (e.g., `stdio_server` context manager).
