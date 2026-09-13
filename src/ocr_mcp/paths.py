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
