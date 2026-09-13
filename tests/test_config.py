from ocr_mcp import paths
from ocr_mcp.config import Settings


def test_defaults(isolated_env):
    s = Settings.from_env()
    assert s.model_det == "PP-OCRv6_medium_det"
    assert s.model_rec == "PP-OCRv6_medium_rec"
    assert s.idle_timeout == 600
    assert s.request_timeout == 60
    assert s.data_dir == isolated_env


def test_env_overrides(monkeypatch, tmp_path):
    d = tmp_path / "xd"
    monkeypatch.setenv("OCR_MCP_DATA_DIR", str(d))
    monkeypatch.setenv("OCR_MCP_MODEL_DET", "PP-OCRv6_small_det")
    monkeypatch.setenv("OCR_MCP_IDLE_TIMEOUT", "120")
    s = Settings.from_env()
    assert s.model_det == "PP-OCRv6_small_det"
    assert s.idle_timeout == 120
    assert s.data_dir == d


def test_paths_under_data_dir(isolated_env):
    s = Settings.from_env()
    assert paths.lock_file_path(s) == isolated_env / "daemon.lock"
    assert paths.socket_path(s) == isolated_env / "daemon.sock"
    assert paths.log_file_path(s, "daemon") == isolated_env / "daemon.log"
    # ensure_data_dir creates if missing
    s2 = Settings.from_env()
    s2 = Settings(**{**s2.__dict__, "data_dir": isolated_env / "nested" / "dir"})
    p = paths.ensure_data_dir(s2)
    assert p.exists() and p.is_dir()
