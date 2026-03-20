"""Train per-exercise logistic regression models from labeled dataset.

Usage:
    python scripts/train_model.py \\
        --features-dir data/features \\
        --labels       data/labels/labels.csv \\
        --out-dir      data/models

Assembles dataset, splits by session, trains one model per exercise present
in the training data, and saves artifacts + metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.ml.dataset import assemble_dataset, save_splits, split_dataset
from src.ml.train_logreg import FEATURE_COLS, save_model, train_model
from src.utils.config import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RepQA logistic regression models.")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--labels", default="data/labels/labels.csv")
    parser.add_argument("--out-dir", default="data/models")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    cfg = get_config()

    # ── Assemble + split ──────────────────────────────────────────────────────
    print("Assembling dataset...")
    df = assemble_dataset(Path(args.features_dir), Path(args.labels))
    print(f"  {len(df)} labeled reps, {df['session_id'].nunique()} sessions")
    print(f"  y_bad distribution: {df['y_bad'].value_counts().to_dict()}")
    print(f"  Labels: {df['label_detail'].value_counts().to_dict()}")

    train_df, test_df = split_dataset(df, test_size=args.test_size)
    save_splits(train_df, test_df, Path(args.features_dir))
    print(f"  Train: {len(train_df)} reps | Test: {len(test_df)} reps")

    # ── Train one model per exercise ──────────────────────────────────────────
    exercises = [e for e in df["exercise"].unique() if e in FEATURE_COLS]
    for exercise in exercises:
        ex_train = train_df[train_df["exercise"] == exercise]
        if len(ex_train) < 5:
            print(f"\nSkipping {exercise}: too few training reps ({len(ex_train)})")
            continue

        print(f"\nTraining: {exercise} ({len(ex_train)} reps)")
        try:
            artifacts = train_model(
                train_df,
                exercise,
                C_values=cfg["model"]["C_values"],
                cv_folds=cfg["model"]["cv_folds"],
                precision_target=cfg["model"]["precision_target"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        save_model(artifacts, Path(args.out_dir))
        m = artifacts["train_metrics"]
        print(f"  Best C    : {artifacts['best_C']}")
        print(f"  Threshold : {artifacts['threshold']:.2f}")
        print(f"  CV AUC    : {artifacts['cv_results']}")
        print(f"  Train AUC : {m['auc']}")
        print(f"  Train P/R : {m['precision']:.3f} / {m['recall']:.3f}")
        print(f"  Saved to  : {Path(args.out_dir) / exercise}/")


if __name__ == "__main__":
    main()
