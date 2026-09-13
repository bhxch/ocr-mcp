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
        kwargs = {
            "use_textline_orientation": False,
            "lang": "ch",
            "enable_mkldnn": False,  # MUST be False: paddlepaddle 3.3 PIR+onednn crashes
            "text_detection_model_name": self._settings.model_det,
            "text_recognition_model_name": self._settings.model_rec,
        }
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
        assert ocr is not None  # _ensure_loaded() guarantees the model is loaded
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
                poly = [[int(x) for x in pt] for pt in poly] if poly is not None else None
                lines.append(OcrLine(text=str(t), score=score, poly=poly))
                texts_for_join.append(str(t))
        # width/height left 0: PaddleOCR's OCRResult does not expose a stable
        # image-size field; polys already carry absolute pixel coordinates.
        return OcrResult(
            image_path=str(p),
            width=width, height=height,
            lines=lines,
            text="\n".join(texts_for_join),
            elapsed_ms=elapsed_ms,
        )
