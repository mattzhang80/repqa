"""Smoke tests for src/utils/plotting.py — Phase 16.

Plotting functions are deterministic wrappers over matplotlib; these
tests assert they write non-empty PNGs without raising.  Visual review
is the responsibility of the author.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.plotting import (
    plot_baseline_vs_model,
    plot_confusion_matrix,
    plot_forest,
    plot_label_distribution,
    plot_longitudinal_trend,
    plot_pr_curve,
    plot_rom_distribution,
    plot_roc_curve,
)


def _valid_png(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 500


def _binary_scores(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = y + 0.3 * rng.standard_normal(n)
    return y, s


def test_plot_roc_curve(tmp_path: Path):
    y, s = _binary_scores(100)
    p = tmp_path / "roc.png"
    plot_roc_curve(y, s, "wall_slide", save_path=p)
    assert _valid_png(p)


def test_plot_pr_curve_with_threshold(tmp_path: Path):
    y, s = _binary_scores(100)
    p = tmp_path / "pr.png"
    plot_pr_curve(y, s, "wall_slide", threshold=0.4, save_path=p)
    assert _valid_png(p)


def test_plot_confusion_matrix(tmp_path: Path):
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = tmp_path / "cm.png"
    plot_confusion_matrix(y, y_pred, "wall_slide", save_path=p)
    assert _valid_png(p)


def test_plot_label_distribution(tmp_path: Path):
    df = pd.DataFrame({
        "label_detail": ["good"] * 20 + ["bad_tempo"] * 10 + ["bad_rom_partial"] * 15,
        "exercise": ["wall_slide"] * 45,
    })
    p = tmp_path / "labels.png"
    plot_label_distribution(df, "wall_slide", save_path=p)
    assert _valid_png(p)


def test_plot_rom_distribution(tmp_path: Path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "rom_proxy_max": np.concatenate(
            [rng.normal(0.7, 0.05, 30), rng.normal(0.4, 0.05, 30)]
        ),
        "label_detail": ["good"] * 30 + ["bad_rom_partial"] * 30,
        "exercise": ["wall_slide"] * 60,
    })
    p = tmp_path / "rom.png"
    plot_rom_distribution(df, "wall_slide", save_path=p)
    assert _valid_png(p)


def test_plot_baseline_vs_model(tmp_path: Path):
    comparison = {
        "precision": {"baseline": 0.7, "model": 0.85},
        "recall": {"baseline": 0.6, "model": 0.9},
        "f1": {"baseline": 0.65, "model": 0.87},
    }
    p = tmp_path / "bvm.png"
    plot_baseline_vs_model(comparison, "wall_slide", save_path=p)
    assert _valid_png(p)


def test_plot_forest(tmp_path: Path):
    rows = [
        {"label": "wall_slide: AUC", "point": 0.93, "lower": 0.82, "upper": 0.99,
         "group": "wall_slide"},
        {"label": "wall_slide: Precision", "point": 0.75, "lower": 0.5, "upper": 0.95,
         "group": "wall_slide"},
        {"label": "band_er_side: AUC", "point": 0.54, "lower": 0.14, "upper": 0.71,
         "group": "band_er_side"},
        {"label": "band_er_side: Precision", "point": 0.5, "lower": 0.0, "upper": 1.0,
         "group": "band_er_side"},
    ]
    p = tmp_path / "forest.png"
    plot_forest(rows, save_path=p)
    assert _valid_png(p)


def test_plot_forest_empty_no_crash(tmp_path: Path):
    p = tmp_path / "forest.png"
    plot_forest([], save_path=p)
    # empty rows → no-op
    assert not p.exists() or p.stat().st_size == 0


def test_plot_longitudinal_trend(tmp_path: Path):
    sessions = [
        {"session_id": "s1", "rom_median": 0.7},
        {"session_id": "s2", "rom_median": 0.72},
        {"session_id": "s3", "rom_median": 0.68},
    ]
    p = tmp_path / "trend.png"
    plot_longitudinal_trend(sessions, "wall_slide", save_path=p)
    assert _valid_png(p)
