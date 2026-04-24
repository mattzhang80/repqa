"""Tests for src/ml/eval.py — Phase 16."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.ml.eval import (
    compare_baseline_vs_model,
    evaluate_model,
    generate_plots,
    save_metrics,
)


def _fit_wall_slide_model(train_df: pd.DataFrame) -> tuple:
    cols = [
        "rom_proxy_max", "rom_proxy_range", "tempo_s", "tempo_deviation",
        "conf_mean", "conf_min",
    ]
    X = train_df[cols].to_numpy(dtype=float)
    y = train_df["y_bad"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=1.0, solver="liblinear", class_weight="balanced",
        max_iter=1000, random_state=0,
    ).fit(scaler.transform(X), y)
    return model, scaler, cols


def _make_wall_slide_df(
    n_sessions: int = 6, reps_per_session: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Construct a clearly-separable synthetic wall slide dataset.

    Half sessions are good (full ROM, slow tempo), half are bad (low ROM,
    fast tempo).  Within each session, reps are drawn from a shared mean
    so cluster bootstrap has non-trivial within-group correlation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        is_bad = s % 2
        label = "bad_rom_partial" if is_bad else "good"
        rom_mean = 0.45 if is_bad else 0.75
        tempo_mean = 4.5 if is_bad else 5.0
        for i in range(reps_per_session):
            rows.append({
                "session_id": f"s{s:02d}",
                "rep_id": i,
                "exercise": "wall_slide",
                "user_id": "u1",
                "label_detail": label,
                "y_bad": int(is_bad),
                "rom_proxy_max": float(rng.normal(rom_mean, 0.04)),
                "rom_proxy_range": float(rng.normal(0.45 if not is_bad else 0.25, 0.04)),
                "tempo_s": float(rng.normal(tempo_mean, 0.3)),
                "tempo_deviation": float(abs(rng.normal(tempo_mean - 5.0, 0.3))),
                "conf_mean": float(rng.uniform(0.85, 0.97)),
                "conf_min": float(rng.uniform(0.7, 0.9)),
            })
    return pd.DataFrame(rows)


class TestEvaluateModel:
    def test_returns_required_keys(self):
        df = _make_wall_slide_df()
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=100,
        )
        for k in [
            "exercise", "n_test", "n_good", "n_bad", "n_sessions",
            "threshold", "feature_cols",
            "auc", "precision", "recall", "f1", "confusion_matrix",
            "bootstrap_auc_ci", "bootstrap_precision_ci", "bootstrap_recall_ci",
            "label_detail_breakdown", "per_label_model_recall",
        ]:
            assert k in metrics

    def test_cluster_bootstrap_is_used(self):
        df = _make_wall_slide_df()
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=100, cluster_bootstrap=True,
        )
        assert metrics["bootstrap_auc_ci"]["cluster"] is True
        assert metrics["bootstrap_auc_ci"]["n_groups"] == df["session_id"].nunique()

    def test_point_auc_matches_separable_data(self):
        df = _make_wall_slide_df(n_sessions=8, reps_per_session=20, seed=1)
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=100,
        )
        assert metrics["auc"] > 0.9

    def test_ci_contains_point(self):
        df = _make_wall_slide_df()
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=200,
        )
        ci = metrics["bootstrap_auc_ci"]
        assert ci["lower"] <= ci["point"] <= ci["upper"]

    def test_empty_test_returns_error(self):
        df = _make_wall_slide_df().iloc[0:0]
        model, scaler, cols = _fit_wall_slide_model(_make_wall_slide_df())
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=50,
        )
        assert "error" in metrics

    def test_per_label_recall_computed(self):
        df = _make_wall_slide_df()
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=50,
        )
        # Only 'good' and 'bad_rom_partial' labels in this synthetic data;
        # per_label recall covers non-good labels only.
        assert "bad_rom_partial" in metrics["per_label_model_recall"]
        assert "good" not in metrics["per_label_model_recall"]

    def test_drops_nan_rows(self):
        df = _make_wall_slide_df()
        df.loc[df.index[0], "rom_proxy_max"] = np.nan
        model, scaler, cols = _fit_wall_slide_model(df.dropna())
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=50,
        )
        assert metrics["n_test"] == len(df) - 1


class TestCompareBaselineVsModel:
    def test_returns_expected_keys(self):
        df = _make_wall_slide_df()
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=50,
        )
        comp = compare_baseline_vs_model(df, "wall_slide", metrics)
        for m in ["precision", "recall", "f1"]:
            assert m in comp
            assert "baseline" in comp[m] and "model" in comp[m]

    def test_empty_returns_empty_dict(self):
        empty = _make_wall_slide_df().iloc[0:0]
        comp = compare_baseline_vs_model(empty, "wall_slide", {})
        assert comp == {}


class TestSaveAndPlot:
    def test_save_metrics_strips_arrays(self, tmp_path: Path):
        df = _make_wall_slide_df()
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=50,
        )
        path = save_metrics(metrics, tmp_path)
        assert path.exists()
        import json
        saved = json.loads(path.read_text())
        # Underscore-prefixed entries must NOT be in the file
        for k in saved:
            assert not k.startswith("_")
        assert saved["exercise"] == "wall_slide"

    def test_generate_plots_writes_files(self, tmp_path: Path):
        df = _make_wall_slide_df()
        model, scaler, cols = _fit_wall_slide_model(df)
        metrics = evaluate_model(
            model, scaler, df, cols, threshold=0.5, exercise="wall_slide",
            n_bootstrap=50,
        )
        comp = compare_baseline_vs_model(df, "wall_slide", metrics)
        out = generate_plots(metrics, df, df, comp, tmp_path)
        for key in ["roc", "pr", "confusion_matrix",
                    "label_distribution", "rom_distribution",
                    "baseline_vs_model"]:
            assert key in out
            assert out[key].exists()
            assert out[key].stat().st_size > 0
