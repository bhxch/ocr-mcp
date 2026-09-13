"""End-to-end integration test: the real PaddleOCR stack.

This is the ONLY test that does not mock paddle. It exercises the full chain:
stdio ``handle_ocr_image`` -> ``lifecycle.ensure_daemon`` (spawns a real daemon
holding a real ``OcrEngine``) -> real ``engine.predict`` -> standardized result.

Marked ``slow`` so the default unit-test run skips it; run explicitly with::

    uv run pytest tests/test_integration.py -v -m slow

Skips automatically when the fixture image is absent, so the test stays robust
to source-only checkouts.
"""

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
    # ocr_image returns a markdown table: `| text |` header + separator + one row per line.
    text = out.content[0].text
    rows = [ln for ln in text.splitlines()[2:] if ln.strip()]
    assert rows, f"no text rows in output: {text!r}"
    # The model should pick up at least one digit from the fixture
    # ("订单 20260729 金额 99.00").
    assert any(ch.isdigit() for ch in "\n".join(rows))
