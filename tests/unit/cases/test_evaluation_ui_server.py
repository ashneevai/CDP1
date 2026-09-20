"""Unit tests for Evaluation UI server."""

from pathlib import Path

from fastapi.testclient import TestClient

import apps.evaluation_ui.main as evaluation_ui_main
from apps.evaluation_ui.main import app


def test_evaluation_ui_health_and_index(tmp_path: Path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<!doctype html><title>IDP Claims</title>", encoding="utf-8"
    )
    monkeypatch.setattr(evaluation_ui_main, "DIST_DIR", dist)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    res = client.get("/")
    assert res.status_code == 200
    assert "IDP" in res.text or "Claims" in res.text
