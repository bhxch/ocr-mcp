from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

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
