"""Tests for src/api/main.py — unified API (health, reports, session detail
with ML predictions).

Uses FastAPI's TestClient; routes that touch the filesystem are exercised
against small temp directories that we monkeypatch into the module's
module-level path constants.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def client_with_data(tmp_path: Path, monkeypatch):
    """Boot the API against temp data directories so tests don't depend on
    the actual repo state.  Creates one processed session, one trained
    model, and a Phase 16 metrics file.
    """
    # Temp layout
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    processed.mkdir()
    reports.mkdir()
    models.mkdir()
    figs = reports / "figures"
    figs.mkdir()

    # One minimal session
    s_dir = processed / "test_session_01"
    s_dir.mkdir()
    meta = {
        "session_id": "test_session_01",
        "user_id": "tester",
        "exercise": "wall_slide",
        "display_name": "Wall Slide",
        "filming_angle": "side",
        "fps": 30,
        "reps_detected": 2,
        "safety_note": "stop if pain >3/10",
    }
    (s_dir / "meta.json").write_text(json.dumps(meta))
    reps_df = pd.DataFrame({
        "rep_id": [0, 1],
        "start_frame": [0, 150],
        "end_frame": [149, 299],
        "start_time_s": [0.0, 5.0],
        "end_time_s": [5.0, 10.0],
    })
    reps_df.to_csv(s_dir / "reps.csv", index=False)

    feat_df = pd.DataFrame({
        "session_id": ["test_session_01"] * 2,
        "rep_id": [0, 1],
        "exercise": ["wall_slide"] * 2,
        "user_id": ["tester"] * 2,
        "rom_proxy_max": [0.72, 0.41],
        "rom_proxy_range": [0.45, 0.20],
        "tempo_s": [5.0, 2.2],
        "tempo_deviation": [0.0, 2.8],
        "conf_mean": [0.95, 0.90],
        "conf_min": [0.85, 0.75],
    })
    feat_df.to_csv(s_dir / "features.csv", index=False)
    (s_dir / "flags.json").write_text(json.dumps([
        {
            "rep_id": 0,
            "flagged": False,
            "predicted_label": "good",
            "reasons": [],
            "rom_proxy_max": 0.72,
            "tempo_s": 5.0,
            "confidence_level": "high",
        },
        {
            "rep_id": 1,
            "flagged": True,
            "predicted_label": "bad_rom_partial",
            "reasons": ["rom_below_cutoff"],
            "rom_proxy_max": 0.41,
            "tempo_s": 2.2,
            "confidence_level": "high",
        },
    ]))

    # Fake trained wall_slide model
    model_dir = models / "wall_slide"
    model_dir.mkdir()
    X = np.array([
        [0.75, 0.5, 5.0, 0.0, 0.95, 0.85],
        [0.70, 0.45, 5.2, 0.2, 0.93, 0.83],
        [0.35, 0.15, 2.0, 3.0, 0.9, 0.8],
        [0.40, 0.22, 2.3, 2.7, 0.91, 0.82],
    ])
    y = np.array([0, 0, 1, 1])
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(solver="liblinear").fit(scaler.transform(X), y)
    with open(model_dir / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(model_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    (model_dir / "features.json").write_text(json.dumps([
        "rom_proxy_max", "rom_proxy_range", "tempo_s", "tempo_deviation",
        "conf_mean", "conf_min",
    ]))
    (model_dir / "metrics.json").write_text(json.dumps({
        "exercise": "wall_slide",
        "feature_cols": [
            "rom_proxy_max", "rom_proxy_range", "tempo_s", "tempo_deviation",
            "conf_mean", "conf_min",
        ],
        "threshold": 0.5,
    }))

    # Phase 16 metrics
    (reports / "metrics_wall_slide.json").write_text(json.dumps({
        "exercise": "wall_slide",
        "n_test": 46,
        "n_good": 24,
        "n_bad": 22,
        "n_sessions": 4,
        "threshold": 0.3,
        "auc": 0.93,
        "precision": 0.75,
        "recall": 0.96,
        "f1": 0.84,
        "confusion_matrix": [[18, 6], [1, 21]],
        "bootstrap_auc_ci": {
            "point": 0.93, "lower": 0.78, "upper": 0.99,
            "method": "bca", "cluster": True, "n_groups": 4, "n_valid": 1800,
        },
        "label_detail_breakdown": {"good": 24, "bad_tempo": 10, "bad_rom_partial": 12},
        "per_label_model_recall": {
            "bad_tempo": {"n": 10, "flagged_bad": 9, "recall": 0.9},
            "bad_rom_partial": {"n": 12, "flagged_bad": 12, "recall": 1.0},
        },
    }))
    # Forest plot placeholder
    (figs / "forest_test_metrics.png").write_bytes(b"\x89PNG" + b"\x00" * 600)

    # Patch the module's path constants + clear model cache
    import src.api.main as api_main
    monkeypatch.setattr(api_main, "_PROCESSED_DIR", processed)
    monkeypatch.setattr(api_main, "_REPORTS_DIR", reports)
    monkeypatch.setattr(api_main, "_MODELS_DIR", models)
    api_main._clear_model_cache()

    client = TestClient(api_main.app)
    return client, tmp_path


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client_with_data):
        client, _ = client_with_data
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"  # one session present
        assert body["sessions"] >= 1
        assert "wall_slide" in body["models_available"]
        assert body["reports_dir_present"] is True


# ── Reports ───────────────────────────────────────────────────────────────────

class TestReports:
    def test_list_reports(self, client_with_data):
        client, _ = client_with_data
        r = client.get("/reports/metrics")
        assert r.status_code == 200
        metrics = r.json()["metrics"]
        assert "wall_slide" in metrics
        assert metrics["wall_slide"]["auc"] == 0.93

    def test_specific_metrics(self, client_with_data):
        client, _ = client_with_data
        r = client.get("/reports/metrics/wall_slide")
        assert r.status_code == 200
        assert r.json()["exercise"] == "wall_slide"

    def test_missing_metrics_404(self, client_with_data):
        client, _ = client_with_data
        r = client.get("/reports/metrics/does_not_exist")
        assert r.status_code == 404

    def test_list_figures(self, client_with_data):
        client, _ = client_with_data
        r = client.get("/reports/figures")
        assert r.status_code == 200
        figs = r.json()["figures"]
        assert "forest_test_metrics" in figs
        assert figs["forest_test_metrics"].endswith(".png")


# ── Session detail with ML prediction ─────────────────────────────────────────

class TestSessionDetailWithModel:
    def test_session_detail_includes_model_prediction(self, client_with_data):
        client, _ = client_with_data
        r = client.get("/sessions/test_session_01")
        assert r.status_code == 200
        body = r.json()
        assert len(body["reps"]) == 2

        preds = [rep["model_prediction"] for rep in body["reps"]]
        # Both reps have all required features → both should get predictions
        assert all(p is not None for p in preds)
        for p in preds:
            assert 0.0 <= p["prob_bad"] <= 1.0
            assert isinstance(p["predicted_bad"], bool)
            assert p["threshold"] == pytest.approx(0.5)

        # The low-ROM rep (rep_id=1) should receive a higher prob_bad than
        # the good rep (rep_id=0) given the synthetic training data.
        good = next(r for r in body["reps"] if r["rep_id"] == 0)
        bad = next(r for r in body["reps"] if r["rep_id"] == 1)
        assert bad["model_prediction"]["prob_bad"] > good["model_prediction"]["prob_bad"]

    def test_unknown_session_404(self, client_with_data):
        client, _ = client_with_data
        r = client.get("/sessions/does_not_exist")
        assert r.status_code == 404
