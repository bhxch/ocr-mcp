import json

import pytest

from ocr_mcp.protocol import (
    OcrLine,
    OcrResult,
    decode_message,
    encode_message,
    make_error,
    make_request,
    make_success,
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
