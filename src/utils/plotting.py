"""Plotting utilities for RepQA model evaluation.

All functions save to file and optionally show the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve


def _save_show(fig: plt.Figure, save_path: Path | None, show: bool) -> None:  # type: ignore[name-defined]
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exercise: str,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """ROC curve with AUC annotation."""
    from sklearn.metrics import roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {exercise}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_show(fig, save_path, show)


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exercise: str,
    threshold: float | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Precision-Recall curve with operating threshold marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="coral", lw=2)
    if threshold is not None:
        # Find closest threshold index
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(recall[idx], precision[idx], s=80, color="crimson",
                   zorder=5, label=f"threshold={threshold:.2f}")
        ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {exercise}")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save_show(fig, save_path, show)


def plot_label_distribution(
    features_df: pd.DataFrame,
    exercise: str,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Bar chart of label counts."""
    col = "label_detail" if "label_detail" in features_df.columns else "label"
    if col not in features_df.columns:
        return
    counts = features_df[col].value_counts()
    colors = {
        "good": "#10b981",
        "bad_tempo": "#f59e0b",
        "bad_rom_partial": "#ef4444",
        "bad_elbow_drift_mild": "#8b5cf6",
        "unknown": "#9ca3af",
    }
    bar_colors = [colors.get(str(k), "#6b7280") for k in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white")
    ax.set_title(f"Label Distribution — {exercise}")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_show(fig, save_path, show)


def plot_rom_distribution(
    features_df: pd.DataFrame,
    exercise: str,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Histogram of ROM proxy values split by label."""
    if "rom_proxy_max" not in features_df.columns:
        return
    col = "label_detail" if "label_detail" in features_df.columns else "label"

    fig, ax = plt.subplots(figsize=(7, 4))
    if col in features_df.columns:
        for label, grp in features_df.groupby(col):
            ax.hist(grp["rom_proxy_max"].dropna(), bins=15, alpha=0.6,
                    label=str(label), density=True)
        ax.legend()
    else:
        ax.hist(features_df["rom_proxy_max"].dropna(), bins=15)
    ax.set_xlabel("ROM Proxy Max")
    ax.set_ylabel("Density")
    ax.set_title(f"ROM Distribution — {exercise}")
    fig.tight_layout()
    _save_show(fig, save_path, show)


def plot_longitudinal_trend(
    sessions_summary: list[dict],
    exercise: str,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Line chart of median ROM proxy across sessions over time.

    Args:
        sessions_summary: List of dicts with keys: session_id, rom_median, date (optional).
    """
    if not sessions_summary:
        return
    labels_x = [s.get("session_id", f"s{i}") for i, s in enumerate(sessions_summary)]
    rom_vals = [s.get("rom_median", float("nan")) for s in sessions_summary]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(labels_x)), rom_vals, "o-", color="steelblue", lw=2)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median ROM Proxy")
    ax.set_title(f"ROM Trend Over Sessions — {exercise}")
    fig.tight_layout()
    _save_show(fig, save_path, show)


def plot_baseline_vs_model(
    comparison_dict: dict,
    exercise: str,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Side-by-side bar chart comparing baseline vs ML model metrics.

    Args:
        comparison_dict: Keys: metric names; values: {baseline: float, model: float}.
    """
    if not comparison_dict:
        return
    metrics = list(comparison_dict.keys())
    baseline_vals = [comparison_dict[m].get("baseline", 0) for m in metrics]
    model_vals = [comparison_dict[m].get("model", 0) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, baseline_vals, w, label="Baseline", color="#9ca3af")
    ax.bar(x + w / 2, model_vals, w, label="Logistic Regression", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Baseline vs Model — {exercise}")
    ax.legend()
    fig.tight_layout()
    _save_show(fig, save_path, show)
