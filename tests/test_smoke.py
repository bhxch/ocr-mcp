import subprocess
import sys


def test_version_flag_prints_version():
    result = subprocess.run(
        [sys.executable, "-m", "ocr_mcp", "--version"],
        capture_output=True, text=True, check=True,
    )
    assert "ocr-mcp" in result.stdout
    assert "0.1.0" in result.stdout
