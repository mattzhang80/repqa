"""Model evaluation suite for RepQA — Phase 16.

Evaluates a trained per-exercise logistic regression on the held-out
test split and compares it against the hand-tuned baseline flagger
(Phase 5).  All bootstrap confidence intervals are **cluster-bootstrapped
by session_id** because reps within a session are not independent —
rep-level bootstrap would produce over-confident (too narrow) intervals.

Typical usage (via the CLI orchestrator):
    python scripts/eval_all.py

or for a single exercise:
    python src/ml/eval.py \\
        --test  data/features/test.csv \\
        --model data/models/wall_slide \\
        --out   data/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.ml.bootstrap import (
    bootstrap_auc,
    bootstrap_precision_at_threshold,
    bootstrap_recall_at_threshold,
)
from src.ml.train_logreg import load_model
from src.utils.plotting import (
    plot_baseline_vs_model,
    plot_confusion_matrix,
    plot_forest,
    plot_label_distribution,
    plot_pr_curve,
    plot_rom_distribution,
    plot_roc_curve,
)


# ── Core evaluation ───────────────────────────────────────────────────────────

def _ci_dict(result: dict) -> dict:
    """Flatten a bootstrap result into a JSON-serializable CI dict."""
    return {
        "point": result["point_estimate"],
        "lower": result["ci_lower"],
        "upper": result["ci_upper"],
        "method": result["method_used"],
        "cluster": result["cluster"],
        "n_groups": result["n_groups"],
        "n_valid": result["n_valid"],
    }


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    exercise: str,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    cluster_bootstrap: bool = True,
) -> dict:
    """Full test-set evaluation of a trained model.

    Args:
        model:             Fitted LogisticRegression.
        scaler:            Fitted StandardScaler.
        test_df:           Test split containing y_bad, exercise, session_id,
                           and all ``feature_cols``.  Rows with NaN features
                           are silently dropped — expected for columns that
                           do not apply to an exercise (e.g. elbow_drift
                           on wall_slide).
        feature_cols:      Exact feature list the model was trained on.
        threshold:         Decision threshold used at inference.
        exercise:          Exercise identifier.
        n_bootstrap:       Bootstrap iterations for CIs.
        random_state:      RNG seed.
        cluster_bootstrap: When True (default), bootstrap CIs resample
                           sessions — the honest design for clustered data.
                           Set False only for unit tests that need
                           rep-level behavior.

    Returns:
        Dict with: metadata, point metrics, bootstrap CIs (AUC, precision,
        recall), confusion matrix, per-label breakdown + model correctness
        per bad label, and arrays required by downstream plotting (prefixed
        with ``_`` so callers can strip them before JSON-serializing).
    """
    ex_df = test_df[test_df["exercise"] == exercise].copy()
    ex_df = ex_df.dropna(subset=feature_cols)  # type: ignore[call-overload]
    if len(ex_df) == 0:
        return {"exercise": exercise, "error": "no test data"}

    X: np.ndarray = ex_df[feature_cols].to_numpy(dtype=float)
    y: np.ndarray = ex_df["y_bad"].to_numpy(dtype=int)
    sessions: np.ndarray = ex_df["session_id"].to_numpy()
    label_details: np.ndarray | None = (
        ex_df["label_detail"].to_numpy()
        if "label_detail" in ex_df.columns else None
    )

    X_scaled = scaler.transform(X)
    y_prob: np.ndarray = np.asarray(model.predict_proba(X_scaled))[:, 1]
    y_pred: np.ndarray = (y_prob >= threshold).astype(int)

    # ── Bootstrap CIs ────────────────────────────────────────────────────────
    boot_groups = sessions if cluster_bootstrap else None
    boot_kwargs = dict(
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        groups=boot_groups,
    )
    auc_ci = bootstrap_auc(y, y_prob, **boot_kwargs)
    prec_ci = bootstrap_precision_at_threshold(y, y_prob, threshold=threshold, **boot_kwargs)
    rec_ci = bootstrap_recall_at_threshold(y, y_prob, threshold=threshold, **boot_kwargs)

    # ── Point metrics ────────────────────────────────────────────────────────
    both_classes_present = len(np.unique(y)) > 1
    point = {
        "auc": float(roc_auc_score(y, y_prob)) if both_classes_present else None,
        "precision": float(precision_score(y, y_pred, zero_division=0)),  # type: ignore[arg-type]
        "recall": float(recall_score(y, y_pred, zero_division=0)),  # type: ignore[arg-type]
        "f1": float(f1_score(y, y_pred, zero_division=0)),  # type: ignore[arg-type]
    }

    # ── Per-label breakdown + model correctness per bad label ────────────────
    label_breakdown: dict = {}
    per_label_flagged: dict = {}
    if label_details is not None:
        for lab in np.unique(label_details):
            idx = label_details == lab
            label_breakdown[str(lab)] = int(np.sum(idx))
            # For bad labels, report fraction the model flagged as bad
            if lab != "good":
                per_label_flagged[str(lab)] = {
                    "n": int(np.sum(idx)),
                    "flagged_bad": int(np.sum(y_pred[idx] == 1)),
                    "recall": (
                        float(np.mean(y_pred[idx] == 1)) if np.sum(idx) > 0 else None
                    ),
                }

    return {
        "exercise": exercise,
        "n_test": int(len(y)),
        "n_good": int(np.sum(y == 0)),
        "n_bad": int(np.sum(y == 1)),
        "n_sessions": int(len(np.unique(sessions))),
        "threshold": float(threshold),
        "feature_cols": list(feature_cols),
        **point,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "bootstrap_auc_ci": _ci_dict(auc_ci),
        "bootstrap_precision_ci": _ci_dict(prec_ci),
        "bootstrap_recall_ci": _ci_dict(rec_ci),
        "label_detail_breakdown": label_breakdown,
        "per_label_model_recall": per_label_flagged,
        "_y": y,
        "_y_prob": y_prob,
        "_y_pred": y_pred,
        "_sessions": sessions,
    }


# ── Baseline comparison ───────────────────────────────────────────────────────

def compare_baseline_vs_model(
    test_df: pd.DataFrame,
    exercise: str,
    model_metrics: dict,
) -> dict:
    """Apply the Phase 5 hand-tuned threshold flagger to the test rows for
    this exercise and compare to the trained model's point metrics.

    Returns a nested dict suitable for :func:`plot_baseline_vs_model`:
        ``{metric_name: {"baseline": float, "model": float}}``
    Returns an empty dict if the test split is empty.
    """
    from src.pipeline.baseline import flag_reps_baseline

    ex_df = test_df[test_df["exercise"] == exercise].copy()
    if len(ex_df) == 0 or "y_bad" not in ex_df.columns:
        return {}

    flags = flag_reps_baseline(pd.DataFrame(ex_df), exercise)
    y_true = np.asarray(ex_df["y_bad"], dtype=int)
    y_baseline = np.array([1 if f.flagged else 0 for f in flags])

    return {
        "precision": {
            "baseline": float(precision_score(y_true, y_baseline, zero_division=0)),  # type: ignore[arg-type]
            "model": model_metrics.get("precision", 0.0),
        },
        "recall": {
            "baseline": float(recall_score(y_true, y_baseline, zero_division=0)),  # type: ignore[arg-type]
            "model": model_metrics.get("recall", 0.0),
        },
        "f1": {
            "baseline": float(f1_score(y_true, y_baseline, zero_division=0)),  # type: ignore[arg-type]
            "model": model_metrics.get("f1", 0.0),
        },
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, out_dir: Path) -> Path:
    """Write metrics JSON to ``out_dir/metrics_<exercise>.json``.

    Strips the leading-underscore arrays (raw predictions, sessions) so
    the file stays JSON-serializable.  Returns the written path.
    """
    clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{metrics['exercise']}.json"
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2, default=str)
    return path


def generate_plots(
    metrics: dict,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    comparison: dict,
    figs_dir: Path,
) -> dict:
    """Write all Phase 16 figures for a single exercise.

    Writes per-exercise ROC, PR, confusion matrix, label distribution,
    ROM distribution, and baseline-vs-model comparison to ``figs_dir``.
    Returns a dict of output paths keyed by plot name.
    """
    exercise = metrics["exercise"]
    y = metrics.get("_y")
    y_prob = metrics.get("_y_prob")
    y_pred = metrics.get("_y_pred")
    figs_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    if y is not None and y_prob is not None and len(np.unique(y)) > 1:
        p = figs_dir / f"roc_curve_{exercise}.png"
        plot_roc_curve(y, y_prob, exercise, save_path=p)
        out["roc"] = p

        p = figs_dir / f"pr_curve_{exercise}.png"
        plot_pr_curve(y, y_prob, exercise, threshold=metrics["threshold"], save_path=p)
        out["pr"] = p

    if y is not None and y_pred is not None:
        p = figs_dir / f"confusion_matrix_{exercise}.png"
        plot_confusion_matrix(y, y_pred, exercise, save_path=p)
        out["confusion_matrix"] = p

    # Label and ROM distributions over the combined labeled dataset
    # (train + test) so the distribution is informative even when test is
    # tiny.
    combined = test_df if train_df is None else pd.concat(
        [train_df, test_df], ignore_index=True
    )

    p = figs_dir / f"label_distribution_{exercise}.png"
    plot_label_distribution(combined, exercise, save_path=p)
    out["label_distribution"] = p

    p = figs_dir / f"rom_distribution_{exercise}.png"
    plot_rom_distribution(combined, exercise, save_path=p)
    out["rom_distribution"] = p

    if comparison:
        p = figs_dir / f"baseline_vs_model_{exercise}.png"
        plot_baseline_vs_model(comparison, exercise, save_path=p)
        out["baseline_vs_model"] = p

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_test_for(exercise: str, base_test: Path) -> pd.DataFrame:
    """Prefer the personalized test CSV if it exists (produced when
    training with --personalize); fall back to the raw test split."""
    pers = Path("data/models/baselines") / f"test_personalized_{exercise}.csv"
    if pers.exists():
        return pd.read_csv(pers)
    return pd.read_csv(base_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out test split."
    )
    parser.add_argument("--test", default="data/features/test.csv")
    parser.add_argument(
        "--train", default="data/features/train.csv",
        help="Used only for distribution plots (train+test combined)."
    )
    parser.add_argument("--model", required=True,
                        help="Path to model directory (e.g. data/models/wall_slide)")
    parser.add_argument("--out", default="data/reports")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--no-cluster-bootstrap",
        action="store_true",
        help="Disable cluster-by-session bootstrap (default is enabled).",
    )
    args = parser.parse_args()

    artifacts = load_model(Path(args.model))
    exercise = artifacts["exercise"]
    test_df = _load_test_for(exercise, Path(args.test))
    train_df = pd.read_csv(args.train) if Path(args.train).exists() else None

    metrics = evaluate_model(
        artifacts["model"],
        artifacts["scaler"],
        test_df,
        artifacts["feature_cols"],
        artifacts["threshold"],
        exercise,
        n_bootstrap=args.n_bootstrap,
        cluster_bootstrap=not args.no_cluster_bootstrap,
    )

    comparison = compare_baseline_vs_model(test_df, exercise, metrics)
    out_dir = Path(args.out)
    save_metrics(metrics, out_dir)
    generate_plots(metrics, test_df, train_df, comparison, out_dir / "figures")

    print(f"Exercise: {exercise}")
    print(f"  n_test   : {metrics['n_test']} "
          f"(good={metrics['n_good']}, bad={metrics['n_bad']})")
    print(f"  threshold: {metrics['threshold']:.3f}")
    if metrics.get("auc") is not None:
        auc = metrics["bootstrap_auc_ci"]
        print(f"  AUC      : {metrics['auc']:.3f}  "
              f"[{auc['lower']:.3f}, {auc['upper']:.3f}]  "
              f"({auc['method']}, cluster={auc['cluster']})")
    p = metrics["bootstrap_precision_ci"]
    r = metrics["bootstrap_recall_ci"]
    print(f"  P @ t    : {metrics['precision']:.3f}  "
          f"[{p['lower']:.3f}, {p['upper']:.3f}]")
    print(f"  R @ t    : {metrics['recall']:.3f}  "
          f"[{r['lower']:.3f}, {r['upper']:.3f}]")
    print(f"  label breakdown: {metrics['label_detail_breakdown']}")
    print(f"  per-label recall: {metrics['per_label_model_recall']}")
    print(f"  baseline vs model: {comparison}")
