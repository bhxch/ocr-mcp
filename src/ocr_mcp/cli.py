from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from ocr_mcp import __version__
from ocr_mcp.backend.engine import OcrEngine
from ocr_mcp.backend.worker import DaemonWorker
from ocr_mcp.config import Settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ocr-mcp")
    p.add_argument("-V", "--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("stdio", help="run stdio MCP server (default)")
    sub.add_parser("serve", help="run the OCR backend daemon")
    sub.add_parser("download", help="pre-download model files")
    return p


def _run_stdio() -> int:
    # mcp 2.0: MCPServer.run() is synchronous, defaults to the stdio transport,
    # and manages its own event loop via anyio.run(self.run_stdio_async).
    # (The brief's 1.x `stdio_server.run(create_server())` form does not apply here.)
    from ocr_mcp.server import create_server

    create_server().run()
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
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse calls sys.exit(2) on parse errors (e.g. unknown subcommand);
        # convert to a return code so callers get the status without a raise.
        return int(e.code) if isinstance(e.code, int) else 2
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
