"""Model evaluation suite for RepQA.

Evaluates a trained logistic regression model on the test split and
compares against the baseline threshold flagger.

Usage:
    python src/ml/eval.py \\
        --test   data/features/test.csv \\
        --model  data/models/wall_slide \\
        --out    data/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml.bootstrap import bootstrap_auc, bootstrap_precision_at_threshold
from src.ml.train_logreg import FEATURE_COLS, load_model
from src.utils.plotting import (
    plot_baseline_vs_model,
    plot_label_distribution,
    plot_pr_curve,
    plot_rom_distribution,
    plot_roc_curve,
)


def evaluate_model(
    model: object,
    scaler: object,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    exercise: str,
) -> dict:
    """Full evaluation of a trained model on the test set.

    Args:
        model:        Fitted LogisticRegression.
        scaler:       Fitted StandardScaler.
        test_df:      Test split DataFrame (must have y_bad and feature_cols).
        feature_cols: Feature column names.
        threshold:    Decision probability threshold.
        exercise:     Exercise identifier.

    Returns:
        Evaluation metrics dict.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    ex_df = test_df[test_df["exercise"] == exercise].dropna(subset=feature_cols)  # type: ignore[call-overload]
    if len(ex_df) == 0:
        return {"exercise": exercise, "error": "no test data"}

    X: np.ndarray = ex_df[feature_cols].to_numpy(dtype=float)
    y: np.ndarray = ex_df["y_bad"].to_numpy(dtype=int)

    scaler_: StandardScaler = scaler  # type: ignore[assignment]
    model_: LogisticRegression = model  # type: ignore[assignment]

    X_scaled = scaler_.transform(X)
    y_prob: np.ndarray = model_.predict_proba(X_scaled)[:, 1]
    y_pred: np.ndarray = (y_prob >= threshold).astype(int)

    # Bootstrap CIs
    auc_ci = bootstrap_auc(y, y_prob)
    prec_ci = bootstrap_precision_at_threshold(y, y_prob, threshold)

    # Label breakdown (multi-class distribution in test set)
    label_col = "label_detail" if "label_detail" in ex_df.columns else None
    label_breakdown = (
        ex_df[label_col].value_counts().to_dict() if label_col else {}
    )

    return {
        "exercise": exercise,
        "n_test": int(len(y)),
        "n_bad": int(np.sum(y)),
        "n_good": int(np.sum(y == 0)),
        "threshold": threshold,
        "auc": float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else None,
        "precision": float(precision_score(y, y_pred, zero_division="warn")),  # type: ignore[arg-type]
        "recall": float(recall_score(y, y_pred, zero_division="warn")),  # type: ignore[arg-type]
        "f1": float(f1_score(y, y_pred, zero_division="warn")),  # type: ignore[arg-type]
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "precision_at_threshold": float(precision_score(y, y_pred, zero_division="warn")),  # type: ignore[arg-type]
        "bootstrap_auc_ci": {
            "point": auc_ci["point_estimate"],
            "lower": auc_ci["ci_lower"],
            "upper": auc_ci["ci_upper"],
        },
        "bootstrap_precision_ci": {
            "point": prec_ci["point_estimate"],
            "lower": prec_ci["ci_lower"],
            "upper": prec_ci["ci_upper"],
        },
        "label_detail_breakdown": label_breakdown,
        # Store arrays for plot functions
        "_y": y,
        "_y_prob": y_prob,
    }


def compare_baseline_vs_model(
    test_df: pd.DataFrame,
    exercise: str,
    model_metrics: dict,
) -> dict:
    """Compare baseline threshold flagger vs trained model on test data.

    Uses the test set features directly to apply baseline rules and
    compares precision/recall against the logistic regression.

    Args:
        test_df:       Test split DataFrame.
        exercise:      Exercise identifier.
        model_metrics: Dict from evaluate_model().

    Returns:
        Comparison dict suitable for plot_baseline_vs_model().
    """
    from src.pipeline.baseline import flag_reps_baseline

    ex_df = test_df[test_df["exercise"] == exercise].copy()
    if len(ex_df) == 0 or "y_bad" not in ex_df.columns:
        return {}

    flags = flag_reps_baseline(pd.DataFrame(ex_df), exercise)
    y_true = np.asarray(ex_df["y_bad"], dtype=int)
    y_baseline = np.array([1 if f.flagged else 0 for f in flags])

    baseline_prec = float(precision_score(y_true, y_baseline, zero_division="warn"))  # type: ignore[arg-type]
    baseline_rec = float(recall_score(y_true, y_baseline, zero_division="warn"))  # type: ignore[arg-type]
    baseline_f1 = float(f1_score(y_true, y_baseline, zero_division="warn"))  # type: ignore[arg-type]

    return {
        "precision": {"baseline": baseline_prec, "model": model_metrics.get("precision", 0)},
        "recall":    {"baseline": baseline_rec,  "model": model_metrics.get("recall", 0)},
        "f1":        {"baseline": baseline_f1,   "model": model_metrics.get("f1", 0)},
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split.")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--model", required=True, help="Path to model directory (e.g. data/models/wall_slide)")
    parser.add_argument("--out", default="data/reports", help="Output directory for metrics + figures")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test)
    artifacts = load_model(Path(args.model))
    exercise = artifacts["exercise"]
    out_dir = Path(args.out)
    figs_dir = out_dir / "figures"

    metrics = evaluate_model(
        artifacts["model"], artifacts["scaler"],
        test_df, artifacts["feature_cols"],
        artifacts["threshold"], exercise,
    )

    # Strip non-serialisable arrays before saving
    y = metrics.pop("_y", None)
    y_prob = metrics.pop("_y_prob", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"metrics_{exercise}.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved: {out_dir}/metrics_{exercise}.json")
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("_")}, indent=2))

    if y is not None and y_prob is not None:
        plot_roc_curve(y, y_prob, exercise, figs_dir / f"roc_curve_{exercise}.png")
        plot_pr_curve(y, y_prob, exercise, artifacts["threshold"],
                      figs_dir / f"pr_curve_{exercise}.png")
    plot_label_distribution(test_df, exercise, figs_dir / f"label_distribution_{exercise}.png")
    plot_rom_distribution(test_df, exercise, figs_dir / f"rom_distribution_{exercise}.png")

    comparison = compare_baseline_vs_model(test_df, exercise, metrics)
    if comparison:
        plot_baseline_vs_model(comparison, exercise, figs_dir / f"baseline_vs_model_{exercise}.png")
        print("Baseline vs model:", comparison)

    print(f"Figures saved to: {figs_dir}/")
