"""Train per-exercise logistic regression models from the labeled dataset.

Expects Phase 12 to have produced data/features/train.csv (and test.csv).
Trains one binary classifier per exercise present in the training split and
writes artifacts under data/models/<exercise>/.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --train data/features/train.csv \\
                                  --out-dir data/models \\
                                  --exercises wall_slide band_er_side

This script does NOT re-run Phase 12.  If train.csv is missing or stale,
run ``python src/ml/dataset.py ...`` first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.ml.personalize import personalize_splits
from src.ml.train_logreg import FEATURE_COLS, save_model, train_model
from src.utils.config import get_config


def _print_block(exercise: str, artifacts: dict) -> None:
    cv = artifacts["cv_metrics"]
    tm = artifacts["train_metrics"]
    print(f"\n=== {exercise} ({'personalized' if artifacts.get('personalize') else 'raw'}) ===")
    print(f"  n_train (good/bad) : {tm['n_train']} ({tm['n_good']}/{tm['n_bad']})")
    print(f"  CV folds           : {artifacts['n_cv_folds']}")
    print(f"  best C             : {artifacts['best_C']}")
    print(f"  threshold (OOF)    : {artifacts['threshold']:.3f}")
    print(f"  feature_cols       : {artifacts['feature_cols']}")
    cv_line = "  CV AUC per C       : " + ", ".join(
        f"C={c:<5}: {artifacts['cv_results'][c]:.3f}±{artifacts['cv_results_std'][c]:.3f}"
        for c in artifacts["cv_results"]
    )
    print(cv_line)
    print(f"  OOF AUC            : {cv['oof_auc']:.3f}")
    print(f"  OOF precision      : {cv['oof_precision']:.3f}")
    print(f"  OOF recall         : {cv['oof_recall']:.3f}")
    print(f"  train AUC          : {tm['auc']:.3f}")
    print(f"  train P / R        : {tm['precision']:.3f} / {tm['recall']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train per-exercise logistic regression.")
    parser.add_argument("--train", default="data/features/train.csv")
    parser.add_argument("--test", default="data/features/test.csv",
                        help="Used only when --personalize is set, to apply the"
                             " train-fit baseline onto the test split.")
    parser.add_argument("--out-dir", default="data/models")
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["wall_slide", "band_er_side"],
        choices=["wall_slide", "band_er_side"],
    )
    parser.add_argument(
        "--personalize",
        action="store_true",
        help="Fit per-user baseline from TRAIN good reps and add "
             "personalized features (*_z, *_pct) before training.",
    )
    parser.add_argument(
        "--user-id",
        default="matthew",
        help="User to fit baseline for when --personalize is set.",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(
            f"ERROR: train split not found at {train_path}. "
            "Run `python src/ml/dataset.py ...` first (Phase 12)."
        )
        return 1

    cfg = get_config()
    model_cfg = cfg["model"]
    train_df = pd.read_csv(train_path)
    test_df = (
        pd.read_csv(args.test) if args.personalize and Path(args.test).exists()
        else None
    )

    print(
        f"Loaded {len(train_df)} rows from {train_path} "
        f"({train_df['session_id'].nunique()} sessions)"
    )
    print(f"C_values        : {model_cfg['C_values']}")
    print(f"cv_folds        : {model_cfg['cv_folds']}")
    print(f"precision_target: {model_cfg['precision_target']}")
    print(f"personalize     : {args.personalize}")

    summary_rows = []
    for ex in args.exercises:
        if ex not in FEATURE_COLS:
            print(f"\nSkipping '{ex}': no registered feature list")
            continue
        n_ex_rows = int((train_df["exercise"] == ex).sum())
        if n_ex_rows < 5:
            print(f"\nSkipping {ex}: too few training reps ({n_ex_rows})")
            continue

        # Optionally fit baseline on TRAIN good reps and apply to both splits
        train_use = train_df
        if args.personalize:
            if test_df is None:
                # Still proceed — personalize the train split in place; test
                # features get personalized later (e.g. in eval).
                ex_train, _, baseline = personalize_splits(
                    train_df, train_df.iloc[0:0],  # empty slice as placeholder
                    user_id=args.user_id, exercise=ex, save=True,
                )
                train_use = ex_train
                print(
                    f"\n[{ex}] baseline: {baseline['n_reps_used']} good reps "
                    f"from {baseline['n_sessions_used']} sessions "
                    f"(test split not provided)"
                )
            else:
                ex_train, ex_test, baseline = personalize_splits(
                    train_df, test_df,
                    user_id=args.user_id, exercise=ex, save=True,
                )
                train_use = ex_train
                # Persist personalized test split alongside the baseline for
                # downstream evaluation — Phase 16 picks this up.
                pers_dir = Path(args.out_dir) / "baselines"
                pers_dir.mkdir(parents=True, exist_ok=True)
                ex_test.to_csv(pers_dir / f"test_personalized_{ex}.csv", index=False)
                print(
                    f"\n[{ex}] baseline: {baseline['n_reps_used']} good reps "
                    f"from {baseline['n_sessions_used']} sessions; "
                    f"personalized test saved to "
                    f"{pers_dir}/test_personalized_{ex}.csv"
                )

        artifacts = train_model(
            train_use,
            exercise=ex,
            C_values=model_cfg["C_values"],
            cv_folds=model_cfg["cv_folds"],
            precision_target=model_cfg["precision_target"],
            random_state=model_cfg.get("random_state", 42),
            personalize=args.personalize,
        )
        save_model(artifacts, Path(args.out_dir))
        _print_block(ex, artifacts)
        cv = artifacts["cv_metrics"]
        summary_rows.append(
            {
                "exercise": ex,
                "best_C": artifacts["best_C"],
                "threshold": artifacts["threshold"],
                "oof_auc": cv["oof_auc"],
                "oof_precision": cv["oof_precision"],
                "oof_recall": cv["oof_recall"],
            }
        )

    if not summary_rows:
        print("\nNo models trained.")
        return 1

    print("\n=== summary ===")
    header = f'{"exercise":<14} {"C":>5} {"thr":>6} {"OOF AUC":>8} {"OOF P":>7} {"OOF R":>7}'
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f'{r["exercise"]:<14} {r["best_C"]:>5} {r["threshold"]:>6.3f} '
            f'{r["oof_auc"]:>8.3f} {r["oof_precision"]:>7.3f} {r["oof_recall"]:>7.3f}'
        )
    print(f"\nModels written to {args.out_dir}/<exercise>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
