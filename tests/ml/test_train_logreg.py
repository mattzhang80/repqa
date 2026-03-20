"""Tests for src/ml/train_logreg.py — Phase 13."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.train_logreg import (
    FEATURE_COLS,
    load_model,
    save_model,
    select_threshold,
    train_model,
)


def _make_train_df(n_good: int = 30, n_bad: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = [f"s{i:02d}" for i in range(5)]

    rows = []
    for i in range(n_good):
        rows.append({
            "session_id": sessions[i % len(sessions)], "rep_id": i,
            "exercise": "wall_slide", "user_id": "u1", "y_bad": 0,
            "label_detail": "good",
            "rom_proxy_max": float(rng.uniform(0.65, 0.85)),
            "rom_proxy_range": float(rng.uniform(0.4, 0.6)),
            "tempo_s": float(rng.uniform(4.5, 6.0)),
            "tempo_deviation": float(rng.uniform(0.0, 1.0)),
            "conf_mean": float(rng.uniform(0.8, 0.99)),
            "conf_min": float(rng.uniform(0.7, 0.9)),
        })
    for i in range(n_bad):
        rows.append({
            "session_id": sessions[i % len(sessions)], "rep_id": n_good + i,
            "exercise": "wall_slide", "user_id": "u1", "y_bad": 1,
            "label_detail": "bad_tempo",
            "rom_proxy_max": float(rng.uniform(0.3, 0.55)),
            "rom_proxy_range": float(rng.uniform(0.1, 0.3)),
            "tempo_s": float(rng.uniform(1.0, 2.5)),
            "tempo_deviation": float(rng.uniform(2.5, 4.0)),
            "conf_mean": float(rng.uniform(0.75, 0.95)),
            "conf_min": float(rng.uniform(0.6, 0.85)),
        })
    return pd.DataFrame(rows)


class TestTrainModel:
    def test_returns_required_keys(self):
        df = _make_train_df()
        result = train_model(df, "wall_slide", C_values=[0.1, 1.0], cv_folds=3)
        expected = {"model", "scaler", "best_C", "cv_results",
                    "feature_cols", "exercise", "threshold", "train_metrics"}
        assert expected.issubset(result.keys())

    def test_best_C_from_provided_values(self):
        df = _make_train_df()
        C_values = [0.01, 0.1, 1.0]
        result = train_model(df, "wall_slide", C_values=C_values, cv_folds=3)
        assert result["best_C"] in C_values

    def test_cv_results_has_all_C_values(self):
        df = _make_train_df()
        C_values = [0.1, 1.0, 10.0]
        result = train_model(df, "wall_slide", C_values=C_values, cv_folds=3)
        assert set(result["cv_results"].keys()) == set(C_values)

    def test_model_predicts_probabilities(self):
        df = _make_train_df()
        result = train_model(df, "wall_slide", C_values=[1.0], cv_folds=2)
        from sklearn.preprocessing import StandardScaler
        X = df[result["feature_cols"]].to_numpy(dtype=float)
        X_scaled = result["scaler"].transform(X)
        probs = result["model"].predict_proba(X_scaled)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_threshold_in_valid_range(self):
        df = _make_train_df()
        result = train_model(df, "wall_slide", C_values=[1.0], cv_folds=2)
        assert 0.0 < result["threshold"] <= 1.0

    def test_train_metrics_structure(self):
        df = _make_train_df()
        result = train_model(df, "wall_slide", C_values=[1.0], cv_folds=2)
        m = result["train_metrics"]
        assert "precision" in m
        assert "recall" in m
        assert "n_train" in m
        assert m["n_train"] == len(df)

    def test_unknown_exercise_raises(self):
        df = _make_train_df()
        with pytest.raises(ValueError):
            train_model(df, "squat", C_values=[1.0], cv_folds=2)

    def test_no_data_for_exercise_raises(self):
        df = _make_train_df()
        with pytest.raises(ValueError):
            train_model(df, "band_er_side", C_values=[1.0], cv_folds=2)


class TestSelectThreshold:
    def test_returns_float(self):
        rng = np.random.default_rng(0)
        df = _make_train_df()
        result = train_model(df, "wall_slide", C_values=[1.0], cv_folds=2)
        model, scaler = result["model"], result["scaler"]
        X = df[result["feature_cols"]].to_numpy(dtype=float)
        X_scaled = scaler.transform(X)
        y = df["y_bad"].to_numpy(dtype=int)
        thresh = select_threshold(model, X_scaled, y, precision_target=0.8)
        assert isinstance(thresh, float)
        assert 0.0 < thresh <= 1.0


class TestSaveLoadModel:
    def test_round_trip(self, tmp_path):
        df = _make_train_df()
        artifacts = train_model(df, "wall_slide", C_values=[1.0], cv_folds=2)
        save_model(artifacts, tmp_path)

        model_dir = tmp_path / "wall_slide"
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "scaler.pkl").exists()
        assert (model_dir / "features.json").exists()
        assert (model_dir / "metrics.json").exists()

        loaded = load_model(model_dir)
        assert loaded["exercise"] == "wall_slide"
        assert loaded["feature_cols"] == artifacts["feature_cols"]
        assert loaded["threshold"] == pytest.approx(artifacts["threshold"])

    def test_loaded_model_same_predictions(self, tmp_path):
        df = _make_train_df()
        artifacts = train_model(df, "wall_slide", C_values=[1.0], cv_folds=2)
        save_model(artifacts, tmp_path)
        loaded = load_model(tmp_path / "wall_slide")

        X = df[artifacts["feature_cols"]].to_numpy(dtype=float)
        X_orig = artifacts["scaler"].transform(X)
        X_load = loaded["scaler"].transform(X)
        probs_orig = artifacts["model"].predict_proba(X_orig)[:, 1]
        probs_load = loaded["model"].predict_proba(X_load)[:, 1]
        np.testing.assert_allclose(probs_orig, probs_load, rtol=1e-5)
