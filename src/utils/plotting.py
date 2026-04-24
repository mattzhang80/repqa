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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exercise: str,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Binary confusion matrix heatmap with cell counts."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["predicted good", "predicted bad"])
    ax.set_yticklabels(["actual good", "actual bad"])
    ax.set_title(f"Confusion Matrix — {exercise}")
    # Annotate counts, with white text on dark cells for legibility
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = "white" if val > vmax * 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_show(fig, save_path, show)


def plot_forest(
    rows: list[dict],
    title: str = "Test-set metrics (bootstrap 95% CIs)",
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Horizontal forest plot of point estimates with 95% CIs.

    The canonical Phase-16 cross-exercise summary figure: one row per
    ``(exercise, metric)`` combo, dot = point estimate, horizontal bar =
    confidence interval.  Communicates magnitude *and* uncertainty in one
    look — the most important thing to show given our small test split.

    Args:
        rows: List of dicts each with keys
              ``label`` (str, y-axis label), ``point`` (float),
              ``lower`` (float), ``upper`` (float), and optional
              ``group`` (for color grouping).
    """
    if not rows:
        return

    group_colors = {
        "wall_slide": "#2563eb",
        "band_er_side": "#dc2626",
    }
    default_color = "#4b5563"

    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.5 * len(rows) + 0.5)))
    y_positions = np.arange(len(rows))[::-1]  # top row first

    for y_pos, row in zip(y_positions, rows):
        color = group_colors.get(row.get("group", ""), default_color)
        point = row["point"]
        lo = row.get("lower", float("nan"))
        hi = row.get("upper", float("nan"))
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y_pos, lo, hi, color=color, lw=2.5, alpha=0.8)
        if np.isfinite(point):
            ax.plot(point, y_pos, "o", color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_xlim(-0.02, 1.02)
    ax.axvline(0.5, color="#9ca3af", lw=0.8, linestyle=":")
    ax.axvline(0.8, color="#d1d5db", lw=0.8, linestyle=":")
    ax.set_xlabel("Metric value")
    ax.set_title(title)

    # Minimal group legend
    seen_groups = [g for g in dict.fromkeys(r.get("group", "") for r in rows) if g]
    if seen_groups:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], marker="o", color=group_colors.get(g, default_color),
                   linestyle="-", lw=2.5, markersize=8, label=g)
            for g in seen_groups
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout()
    _save_show(fig, save_path, show)
