"""End-to-end integration test: the real PaddleOCR stack.

This is the ONLY test that does not mock paddle. It exercises the full chain:
stdio ``handle_ocr_image`` -> ``lifecycle.ensure_daemon`` (spawns a real daemon
holding a real ``OcrEngine``) -> real ``engine.predict`` -> standardized result.

Marked ``slow`` so the default unit-test run skips it; run explicitly with::

    uv run pytest tests/test_integration.py -v -m slow

Skips automatically when the fixture image is absent, so the test stays robust
to source-only checkouts.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture image missing")
@pytest.mark.asyncio
async def test_end_to_end_ocr(isolated_env):
    from ocr_mcp import server

    out = await server.handle_ocr_image({"image_path": str(FIXTURE)})
    assert out.isError is False, out.content[0].text if out.content else "no content"
    payload = json.loads(out.content[0].text)
    assert isinstance(payload["text"], str)
    assert len(payload["text"]) > 0
    # The model should pick up at least one digit from the fixture
    # ("订单 20260729 金额 99.00").
    assert any(ch.isdigit() for ch in payload["text"])
