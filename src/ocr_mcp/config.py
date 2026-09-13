from __future__ import annotations

import os
from dataclasses import dataclass
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
    def from_env(cls) -> Settings:
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
