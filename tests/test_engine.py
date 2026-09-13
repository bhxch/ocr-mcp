from __future__ import annotations

import sys
import types

import pytest

from ocr_mcp.backend.engine import OcrEngine
from ocr_mcp.config import Settings


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
