import pytest

from ocr_mcp import cli


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "ocr-mcp" in out


def test_unknown_subcommand_errors(capsys):
    assert cli.main(["bogus"]) == 2
    err = capsys.readouterr().err
    assert "bogus" in err or "usage" in err.lower()


@pytest.mark.asyncio
async def test_serve_invokes_worker(monkeypatch, isolated_env):
    called = {"run": False}

    class FakeWorker:
        def __init__(self, *a, **k):
            pass

        async def run_async(self):
            called["run"] = True

    monkeypatch.setattr(cli, "DaemonWorker", FakeWorker)
    monkeypatch.setattr(cli, "OcrEngine", lambda s: object())
    rc = await cli.serve()
    assert rc == 0
    assert called["run"] is True


@pytest.mark.asyncio
async def test_download_triggers_engine_load(monkeypatch, isolated_env):
    loaded = {"n": 0}

    class FakeEngine:
        def __init__(self, s):
            self.s = s

        def _load(self):
            loaded["n"] += 1

        def predict(self, *a, **k): ...

    monkeypatch.setattr(cli, "OcrEngine", FakeEngine)
    rc = await cli.download()
    assert rc == 0
    assert loaded["n"] == 1
