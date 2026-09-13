from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from ocr_mcp import protocol
from ocr_mcp.backend import lifecycle
from ocr_mcp.config import Settings

# Re-exported for tests/imports.
__all__ = ["BackendClient", "CallResult", "create_server", "handle_ocr_image"]


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
        # Serializes the initial connect so overlapping first calls don't each
        # spawn/connect their own daemon (refcount/socket leak).
        self._connect_lock = asyncio.Lock()
        # Serializes the full request (send+recv) so overlapping calls don't
        # read the same transport concurrently — asyncio's StreamReader is not
        # safe for parallel readuntil().
        self._request_lock = asyncio.Lock()

    async def connect(self):
        if self.transport is not None:
            return
        async with self._connect_lock:
            if self.transport is not None:  # double-check after acquiring lock
                return
            self.transport = await lifecycle.ensure_daemon(self.settings)

    async def request(self, op: str, payload: dict) -> dict:
        # Serialize send+recv so overlapping requests don't read the same
        # transport concurrently (asyncio StreamReader is not safe for parallel
        # readuntil). Spec §10: on transport failure, close and reconnect once.
        async with self._request_lock:
            last_exc: Exception | None = None
            for _ in range(2):
                await self.connect()
                # Re-read the lock each attempt: a respawned daemon gets a fresh token.
                lockf = lifecycle.read_lock(self.settings) or {}
                token = lockf.get("token", "")
                req = protocol.make_request(op, token, payload)
                try:
                    await self.transport.send(protocol.encode_message(req))
                    line = await asyncio.wait_for(
                        self.transport.recv_line(), timeout=self.settings.request_timeout
                    )
                    return protocol.decode_message(line)
                except (TimeoutError, OSError, ConnectionError, json.JSONDecodeError) as e:
                    # Stale transport (EOF/garbage), timeout, or lost connection:
                    # drop it so the next connect() re-runs ensure_daemon from scratch.
                    last_exc = e
                    await self.close()
                    continue
            raise last_exc  # type: ignore[misc]

    async def close(self):
        if self.transport is not None:
            try:
                await self.transport.close()
            except Exception:
                pass
            self.transport = None


# Module-level singleton backend client (one persistent connection per server process).
_client: BackendClient | None = None


async def _get_client() -> BackendClient:
    global _client
    if _client is None:
        _client = BackendClient(Settings.from_env())
    await _client.connect()
    return _client


def _result_to_markdown(result: dict) -> str:
    """Render the OCR result as a markdown table: header + one row per line.

    Columns: `text` always; `score` / `poly` only when present (i.e. when the
    caller passed return_scores=true / return_polys=true). Other fields
    (image_path, width, height, elapsed_ms, joined text) are omitted.
    """
    lines = result.get("lines") or []
    cols = ["text"]
    if lines and "score" in lines[0]:
        cols.append("score")
    if lines and "poly" in lines[0]:
        cols.append("poly")
    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    if not lines:
        out.append("| (no text recognized)" + " |" * (len(cols) - 1))
        return "\n".join(out)
    for ln in lines:
        cells = []
        for c in cols:
            v = ln.get(c)
            if c == "score":
                cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
            elif c == "poly":
                cells.append(str(v) if v is not None else "")
            else:  # text
                s = str(v) if v is not None else ""
                cells.append(s.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


async def handle_ocr_image(arguments: dict[str, Any]) -> CallResult:
    image_path = arguments.get("image_path")
    if not image_path or not isinstance(image_path, str):
        return CallResult(
            isError=True,
            content=[TextContent(type="text", text="ERROR: 'image_path' (string) is required.")],
        )
    payload = {
        "image_path": image_path,
        "options": {
            "return_polys": bool(arguments.get("return_polys", False)),
            "return_scores": bool(arguments.get("return_scores", False)),
        },
    }
    try:
        client = await _get_client()
        resp = await client.request("ocr", payload)
    except Exception as e:
        return CallResult(
            isError=True, content=[TextContent(type="text", text=f"ERROR: DAEMON_UNAVAILABLE: {e}")]
        )
    if not resp.get("ok"):
        err = resp.get("error", {})
        return CallResult(
            isError=True,
            content=[
                TextContent(
                    type="text",
                    text=f"ERROR: {err.get('code', 'UNKNOWN')}: {err.get('message', '')}",
                )
            ],
        )
    return CallResult(
        isError=False,
        content=[TextContent(type="text", text=_result_to_markdown(resp["result"]))],
    )


def create_server() -> MCPServer:
    """Build the stdio MCP server with the `ocr_image` tool registered.

    mcp 2.0's MCPServer derives the tool's JSON schema from the function
    annotations + docstring, so we no longer hand-build a Tool/inputSchema.
    Errors are signaled the 2.0 way: by returning a CallToolResult with
    is_error=True (convert_result passes a CallToolResult through unchanged).
    """
    mcp = MCPServer("ocr-mcp")

    @mcp.tool()
    async def ocr_image(
        image_path: str, return_polys: bool = False, return_scores: bool = False
    ) -> CallToolResult:
        """Recognize text in a local image file. Returns text + per-line confidence + coordinates."""
        res = await handle_ocr_image(
            {
                "image_path": image_path,
                "return_polys": return_polys,
                "return_scores": return_scores,
            }
        )
        return CallToolResult(content=res.content, is_error=res.isError)

    return mcp
