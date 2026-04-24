"""Tests for src/ml/bootstrap.py — Phase 15."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.bootstrap import (
    bootstrap_auc,
    bootstrap_metric,
    bootstrap_precision_at_threshold,
    bootstrap_recall_at_threshold,
)


# ── bootstrap_metric core ─────────────────────────────────────────────────────

class TestBootstrapMetric:
    def test_returns_required_keys(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        s = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.6, 0.2])
        out = bootstrap_metric(y, s, _auc_fn, n_bootstrap=200)
        for k in [
            "point_estimate", "ci_lower", "ci_upper", "ci", "method_used",
            "n_bootstrap", "n_valid", "bootstrap_distribution",
            "cluster", "n_groups",
        ]:
            assert k in out

    def test_ci_contains_point_estimate_on_separable_data(self):
        rng = np.random.default_rng(0)
        n = 200
        y = rng.integers(0, 2, size=n)
        s = y.astype(float) + 0.1 * rng.standard_normal(n)  # near-perfect
        out = bootstrap_metric(y, s, _auc_fn, n_bootstrap=500, random_state=0)
        assert out["ci_lower"] <= out["point_estimate"] <= out["ci_upper"]

    def test_larger_n_gives_narrower_ci(self):
        rng = np.random.default_rng(0)
        small_y = rng.integers(0, 2, size=40)
        small_s = small_y + 0.4 * rng.standard_normal(40)
        big_y = rng.integers(0, 2, size=400)
        big_s = big_y + 0.4 * rng.standard_normal(400)

        a = bootstrap_metric(small_y, small_s, _auc_fn, n_bootstrap=400, random_state=1)
        b = bootstrap_metric(big_y, big_s, _auc_fn, n_bootstrap=400, random_state=1)
        assert (b["ci_upper"] - b["ci_lower"]) < (a["ci_upper"] - a["ci_lower"])

    def test_perfect_classifier_ci_near_one(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 5)
        s = y.astype(float)
        out = bootstrap_metric(y, s, _auc_fn, n_bootstrap=500, random_state=0)
        assert out["point_estimate"] == pytest.approx(1.0)
        # Even with BCa fallback, CI should be at the top of the range
        assert out["ci_upper"] == pytest.approx(1.0)
        assert out["ci_lower"] > 0.95

    def test_invalid_method_raises(self):
        y = np.array([0, 1, 0, 1])
        s = np.array([0.2, 0.8, 0.3, 0.7])
        with pytest.raises(ValueError):
            bootstrap_metric(y, s, _auc_fn, method="not-a-method")

    def test_n_valid_reported(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        out = bootstrap_metric(y, s, _auc_fn, n_bootstrap=300, random_state=0)
        assert out["n_valid"] <= out["n_bootstrap"]
        assert out["n_valid"] > 0

    def test_all_one_class_returns_nan_point_estimate(self):
        y = np.zeros(20, dtype=int)
        s = np.linspace(0, 1, 20)
        out = bootstrap_metric(y, s, _auc_fn, n_bootstrap=100)
        assert np.isnan(out["point_estimate"])
        assert np.isnan(out["ci_lower"])
        assert np.isnan(out["ci_upper"])


# ── Cluster bootstrap ─────────────────────────────────────────────────────────

class TestClusterBootstrap:
    def test_cluster_bootstrap_wider_than_rep_level(self):
        # Correlated data: ~10 groups × 15 reps. Group means overlap a
        # lot (small label effect, large within-group homogeneity), so
        # rep-level bootstrap is over-confident and cluster bootstrap
        # widens the CI.
        rng = np.random.default_rng(0)
        groups, y, s = [], [], []
        for g in range(10):
            label = g % 2
            # Large within-group correlation: all reps in group get a
            # shared random offset larger than the label effect.
            group_offset = 0.4 * rng.standard_normal()
            for _ in range(15):
                groups.append(g)
                y.append(label)
                s.append(0.2 * label + group_offset + 0.01 * rng.standard_normal())
        groups = np.asarray(groups)
        y = np.asarray(y, dtype=int)
        s = np.asarray(s)

        rep_boot = bootstrap_metric(
            y, s, _auc_fn, n_bootstrap=400, random_state=0
        )
        grp_boot = bootstrap_metric(
            y, s, _auc_fn, n_bootstrap=400, random_state=0, groups=groups
        )
        rep_width = rep_boot["ci_upper"] - rep_boot["ci_lower"]
        grp_width = grp_boot["ci_upper"] - grp_boot["ci_lower"]
        # Cluster CI should be materially wider (not just a numerical tie).
        assert grp_width > rep_width + 0.02

    def test_cluster_flag_and_group_count(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        out = bootstrap_metric(y, s, _auc_fn, groups=groups, n_bootstrap=100)
        assert out["cluster"] is True
        assert out["n_groups"] == 3

    def test_groups_length_mismatch_raises(self):
        y = np.array([0, 1, 0, 1])
        s = np.array([0.1, 0.9, 0.2, 0.8])
        groups = np.array(["a", "b", "c"])  # wrong length
        with pytest.raises(ValueError):
            bootstrap_metric(y, s, _auc_fn, groups=groups)


# ── Method-specific ───────────────────────────────────────────────────────────

class TestMethodChoice:
    def test_percentile_produces_sensible_ci(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        s = y + 0.3 * rng.standard_normal(100)
        out = bootstrap_metric(
            y, s, _auc_fn, n_bootstrap=300, random_state=0, method="percentile"
        )
        assert out["method_used"] == "percentile"
        assert 0.0 <= out["ci_lower"] <= out["ci_upper"] <= 1.0

    def test_bca_falls_back_when_illdefined(self):
        # Constant scores make the bootstrap distribution degenerate;
        # bca should fall back without crashing.
        y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        s = np.full(8, 0.5)
        out = bootstrap_metric(y, s, _auc_fn, n_bootstrap=100, random_state=0)
        assert out["method_used"] in {"bca", "percentile"}
        assert np.isfinite(out["ci_lower"]) or np.isnan(out["ci_lower"])


# ── Wrappers ──────────────────────────────────────────────────────────────────

class TestAUCWrapper:
    def test_bootstrap_auc_runs(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=100)
        s = y + 0.3 * rng.standard_normal(100)
        out = bootstrap_auc(y, s, n_bootstrap=200, random_state=0)
        assert 0.0 <= out["point_estimate"] <= 1.0


class TestPrecisionWrapper:
    def test_bootstrap_precision_at_threshold(self):
        rng = np.random.default_rng(0)
        n = 200
        y = rng.integers(0, 2, size=n)
        s = y + 0.3 * rng.standard_normal(n)
        out = bootstrap_precision_at_threshold(
            y, s, threshold=0.5, n_bootstrap=200, random_state=0
        )
        assert 0.0 <= out["point_estimate"] <= 1.0

    def test_no_positive_predictions_skipped(self):
        # Threshold so high that no predictions are positive → metric
        # undefined → bootstrap samples all skipped.
        y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        s = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
        out = bootstrap_precision_at_threshold(
            y, s, threshold=0.99, n_bootstrap=50, random_state=0
        )
        assert out["n_valid"] == 0
        assert np.isnan(out["ci_lower"])


class TestRecallWrapper:
    def test_bootstrap_recall_at_threshold(self):
        rng = np.random.default_rng(0)
        n = 100
        y = rng.integers(0, 2, size=n)
        s = y + 0.3 * rng.standard_normal(n)
        out = bootstrap_recall_at_threshold(
            y, s, threshold=0.5, n_bootstrap=200, random_state=0
        )
        assert 0.0 <= out["point_estimate"] <= 1.0


# helper used by several tests
def _auc_fn(yt, ys):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(yt)) < 2:
        raise ValueError("single class")
    return float(roc_auc_score(yt, ys))
