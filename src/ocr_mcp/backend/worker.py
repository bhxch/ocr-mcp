from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import secrets

from ocr_mcp import paths, protocol
from ocr_mcp.backend.engine import ImageNotFound, ImageUnreadable, InferenceFailed
from ocr_mcp.config import Settings
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
                    # Clear BEFORE waiting: a just-disconnected last client has
                    # set _idle_event, which would make wait() return immediately
                    # and busy-loop forever (no TimeoutError -> no break).
                    # Race-free: asyncio is single-threaded and there is no
                    # `await` between the _clients check above and this clear(),
                    # so a concurrently-arriving connect cannot slip in and have
                    # its set() clobbered (connect runs at the next await below).
                    self._idle_event.clear()
                    try:
                        await asyncio.wait_for(self._idle_event.wait(), timeout=self.settings.idle_timeout)
                    except TimeoutError:
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
        import json
        import os
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
                log.debug("writer.close() failed", exc_info=True)
            if self._clients == 0:
                self._idle_event.set()  # wake loop to start idle timer

    async def _handle(self, line: bytes) -> dict:
        try:
            msg = protocol.decode_message(line)
        except ValueError as e:
            return protocol.make_error("?", "BAD_REQUEST", f"invalid json: {e}")
        mid = msg.get("id", "?")
        if msg.get("token") != self.token:
            return protocol.make_error(mid, "UNAUTHORIZED", "bad token")
        op = msg.get("op")
        if op == "ping":
            import os
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
        loop = asyncio.get_running_loop()
        # NOTE on timeout: wait_for cancels only the asyncio future wrapping the
        # executor call; the underlying engine.predict (paddle inference) is not
        # interruptible and will keep occupying the single-thread executor slot
        # until it returns on its own. A TIMEOUT response is therefore reported
        # to the client while the heavy work may still run to completion. This
        # is an inherent constraint of paddle's non-cancellable C++ inference;
        # document, do not attempt to "fix" by force-killing the thread.
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
        except TimeoutError:
            return protocol.make_error(mid, "TIMEOUT", "ocr request timed out")
        except ImageNotFound as e:
            return protocol.make_error(mid, "IMAGE_NOT_FOUND", str(e))
        except ImageUnreadable as e:
            return protocol.make_error(mid, "IMAGE_UNREADABLE", str(e))
        except InferenceFailed as e:
            return protocol.make_error(mid, "INFERENCE_FAILED", str(e))
        except Exception as e:  # noqa: BLE001 - boundary: engine errors become INFERENCE_FAILED
            return protocol.make_error(mid, "INFERENCE_FAILED", f"{type(e).__name__}: {e}")
        return protocol.make_success(mid, result.to_dict())
