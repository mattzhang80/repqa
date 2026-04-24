"""Phase 16 CLI — evaluate every trained model on the test split.

Loads each model under ``data/models/<exercise>/``, runs the Phase 16
evaluation (cluster bootstrap CIs, per-label breakdown, confusion matrix,
baseline comparison), writes metrics JSON + per-exercise figures, and
produces one cross-exercise forest plot that is the key summary figure
for the paper.

Usage:
    python scripts/eval_all.py
    python scripts/eval_all.py --n-bootstrap 5000 --out data/reports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.eval import (
    _load_test_for,
    compare_baseline_vs_model,
    evaluate_model,
    generate_plots,
    save_metrics,
)
from src.ml.train_logreg import load_model
from src.utils.plotting import plot_forest


def _forest_rows_for(metrics: dict) -> list[dict]:
    """Build forest-plot rows for one exercise.  Metrics with NaN CIs
    (e.g. precision CI when all bootstrap samples had zero positive
    predictions) are omitted so the plot stays readable."""
    ex = metrics["exercise"]
    rows: list[dict] = []

    def _row(label, ci):
        if ci and ci.get("point") is not None:
            return {
                "label": f"{ex}: {label}",
                "point": ci["point"],
                "lower": ci.get("lower", float("nan")),
                "upper": ci.get("upper", float("nan")),
                "group": ex,
            }
        return None

    for lbl, key in [
        ("AUC", "bootstrap_auc_ci"),
        ("Precision", "bootstrap_precision_ci"),
        ("Recall", "bootstrap_recall_ci"),
    ]:
        r = _row(lbl, metrics.get(key))
        if r is not None:
            rows.append(r)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 16: evaluate all trained models and build summary plots."
    )
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--test", default="data/features/test.csv")
    parser.add_argument("--train", default="data/features/train.csv")
    parser.add_argument("--out", default="data/reports")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--no-cluster-bootstrap",
        action="store_true",
        help="Bootstrap at the rep level instead of by session_id. "
             "Not recommended — inflates apparent precision (narrower CI).",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out)
    figs_dir = out_dir / "figures"

    exercises = sorted(
        d.name for d in models_dir.iterdir()
        if d.is_dir() and (d / "model.pkl").exists()
    )
    if not exercises:
        print(f"No trained models found under {models_dir}/")
        return 1

    print(f"Found {len(exercises)} exercise(s): {exercises}")

    train_df = pd.read_csv(args.train) if Path(args.train).exists() else None

    all_forest_rows: list[dict] = []
    summary_rows: list[dict] = []

    for ex in exercises:
        print(f"\n--- {ex} ---")
        artifacts = load_model(models_dir / ex)
        test_df = _load_test_for(ex, Path(args.test))

        metrics = evaluate_model(
            artifacts["model"],
            artifacts["scaler"],
            test_df,
            artifacts["feature_cols"],
            artifacts["threshold"],
            ex,
            n_bootstrap=args.n_bootstrap,
            cluster_bootstrap=not args.no_cluster_bootstrap,
        )

        if "error" in metrics:
            print(f"  SKIP: {metrics['error']}")
            continue

        comparison = compare_baseline_vs_model(test_df, ex, metrics)
        save_metrics(metrics, out_dir)
        generate_plots(metrics, test_df, train_df, comparison, figs_dir)

        # Summary line
        auc_ci = metrics.get("bootstrap_auc_ci", {})
        p_ci = metrics.get("bootstrap_precision_ci", {})
        r_ci = metrics.get("bootstrap_recall_ci", {})
        print(
            f"  n_test={metrics['n_test']} "
            f"(good={metrics['n_good']}, bad={metrics['n_bad']}, "
            f"n_sessions={metrics['n_sessions']})"
        )
        if metrics.get("auc") is not None:
            print(f"  AUC       : {metrics['auc']:.3f}  "
                  f"[{auc_ci['lower']:.3f}, {auc_ci['upper']:.3f}]")
        print(f"  Precision : {metrics['precision']:.3f}  "
              f"[{p_ci['lower']:.3f}, {p_ci['upper']:.3f}]")
        print(f"  Recall    : {metrics['recall']:.3f}  "
              f"[{r_ci['lower']:.3f}, {r_ci['upper']:.3f}]")
        print(f"  Per-label recall: {metrics['per_label_model_recall']}")
        if comparison:
            bp = comparison["precision"]
            br = comparison["recall"]
            print(f"  Baseline P / R: {bp['baseline']:.3f} / {br['baseline']:.3f}  "
                  f"Model P / R: {bp['model']:.3f} / {br['model']:.3f}")

        all_forest_rows.extend(_forest_rows_for(metrics))
        summary_rows.append({
            "exercise": ex,
            "auc": metrics.get("auc"),
            "auc_lo": auc_ci.get("lower"),
            "auc_hi": auc_ci.get("upper"),
            "prec": metrics.get("precision"),
            "prec_lo": p_ci.get("lower"),
            "prec_hi": p_ci.get("upper"),
            "rec": metrics.get("recall"),
            "rec_lo": r_ci.get("lower"),
            "rec_hi": r_ci.get("upper"),
            "n_test": metrics["n_test"],
            "n_sessions": metrics["n_sessions"],
        })

    # Cross-exercise summary figure — the paper's headline plot
    if all_forest_rows:
        forest_path = figs_dir / "forest_test_metrics.png"
        plot_forest(
            all_forest_rows,
            title="Held-out test metrics (95% cluster-bootstrap CI by session)",
            save_path=forest_path,
        )
        print(f"\nForest plot: {forest_path}")

    print(f"\nAll metrics JSON: {out_dir}/metrics_<exercise>.json")
    print(f"All figures     : {figs_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
