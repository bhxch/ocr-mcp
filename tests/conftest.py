import pytest


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("OCR_MCP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OCR_MCP_LOG_LEVEL", "DEBUG")
    return data_dir
