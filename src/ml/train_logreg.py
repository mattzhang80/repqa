"""Logistic regression training for RepQA rep quality classification.

Trains one binary classifier per exercise (good vs any bad label) with:
  - StandardScaler fitted on train only
  - GroupKFold cross-validation by session_id to tune C
  - Threshold selection to meet a precision target
  - Saved artifacts: model pickle, scaler pickle, feature list JSON, metrics JSON

Usage:
    python src/ml/train_logreg.py \\
        --train data/features/train.csv \\
        --exercise wall_slide \\
        --out-dir data/models
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Feature columns per exercise
FEATURE_COLS_WALL_SLIDE = [
    "rom_proxy_max", "rom_proxy_range",
    "tempo_s", "tempo_deviation",
    "conf_mean", "conf_min",
]
FEATURE_COLS_BAND_ER_SIDE = [
    "rom_proxy_max", "rom_proxy_range",
    "tempo_s", "tempo_deviation",
    "conf_mean", "conf_min",
    "elbow_drift_max", "elbow_drift_mean",
]

FEATURE_COLS: dict[str, list[str]] = {
    "wall_slide": FEATURE_COLS_WALL_SLIDE,
    "band_er_side": FEATURE_COLS_BAND_ER_SIDE,
}


def _feature_cols(exercise: str) -> list[str]:
    if exercise not in FEATURE_COLS:
        raise ValueError(f"Unknown exercise: '{exercise}'")
    return FEATURE_COLS[exercise]


def train_model(
    train_df: pd.DataFrame,
    exercise: str,
    C_values: list[float],
    cv_folds: int = 5,
    precision_target: float = 0.8,
) -> dict:
    """Train logistic regression with GroupKFold CV to select C.

    Args:
        train_df:         Training split from split_dataset().
        exercise:         Exercise identifier.
        C_values:         Regularisation strengths to try.
        cv_folds:         Number of GroupKFold folds.
        precision_target: Target precision for threshold selection.

    Returns:
        Dict with keys: model, scaler, best_C, cv_results, feature_cols, exercise,
                        threshold, train_metrics.
    """
    feat_cols = _feature_cols(exercise)
    ex_df = train_df[train_df["exercise"] == exercise].copy()

    if len(ex_df) == 0:
        raise ValueError(f"No training data for exercise '{exercise}'")

    # Drop rows with NaN features
    ex_df = ex_df.dropna(subset=feat_cols)  # type: ignore[call-overload]
    X: np.ndarray = ex_df[feat_cols].to_numpy(dtype=float)
    y: np.ndarray = ex_df["y_bad"].to_numpy(dtype=int)
    groups: np.ndarray = ex_df["session_id"].to_numpy()

    n_splits = min(cv_folds, int(ex_df["session_id"].nunique()))
    gkf = GroupKFold(n_splits=n_splits)

    # ── C selection via CV AUC ────────────────────────────────────────────────
    cv_results: dict[float, float] = {}
    for C in C_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ])
        scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="roc_auc")
        cv_results[C] = float(np.mean(scores))

    best_C = max(cv_results, key=lambda c: cv_results[c])

    # ── Final model on full training set ──────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # ── Threshold selection ────────────────────────────────────────────────────
    threshold = select_threshold(model, X_scaled, y, precision_target)

    # ── Train metrics ──────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    train_metrics = {
        "auc": float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else None,
        "precision": float(precision_score(y, y_pred, zero_division="warn")),  # type: ignore[arg-type]
        "recall": float(recall_score(y, y_pred, zero_division="warn")),  # type: ignore[arg-type]
        "n_train": int(len(y)),
        "n_bad": int(np.sum(y)),
        "n_good": int(np.sum(y == 0)),
    }

    return {
        "model": model,
        "scaler": scaler,
        "best_C": best_C,
        "cv_results": cv_results,
        "feature_cols": feat_cols,
        "exercise": exercise,
        "threshold": threshold,
        "train_metrics": train_metrics,
    }


def select_threshold(
    model: LogisticRegression,
    X_scaled: np.ndarray,
    y: np.ndarray,
    precision_target: float,
) -> float:
    """Find the lowest probability threshold where precision >= precision_target.

    Falls back to 0.5 if target cannot be met.

    Args:
        model:            Fitted LogisticRegression.
        X_scaled:         Scaled feature matrix.
        y:                True binary labels.
        precision_target: Desired minimum precision.

    Returns:
        Probability threshold in (0, 1).
    """
    y_prob = model.predict_proba(X_scaled)[:, 1]
    best_threshold = 0.5
    for thresh in np.arange(0.3, 0.95, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        if np.sum(y_pred) == 0:
            continue
        prec = precision_score(y, y_pred, zero_division="warn")  # type: ignore[arg-type]
        if prec >= precision_target:
            best_threshold = float(thresh)
            break
    return best_threshold


def save_model(artifacts: dict, out_dir: Path) -> None:
    """Save model artifacts to out_dir/<exercise>/.

    Writes:
        model.pkl      — fitted LogisticRegression
        scaler.pkl     — fitted StandardScaler
        features.json  — list of feature column names
        metrics.json   — CV results + train metrics + threshold

    Args:
        artifacts: Dict returned by train_model().
        out_dir:   Parent directory; exercise sub-directory created automatically.
    """
    exercise = artifacts["exercise"]
    model_dir = Path(out_dir) / exercise
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "model.pkl", "wb") as fh:
        pickle.dump(artifacts["model"], fh)
    with open(model_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(artifacts["scaler"], fh)

    meta = {
        "exercise": exercise,
        "feature_cols": artifacts["feature_cols"],
        "best_C": artifacts["best_C"],
        "threshold": artifacts["threshold"],
        "cv_results": {str(k): v for k, v in artifacts["cv_results"].items()},
        "train_metrics": artifacts["train_metrics"],
    }
    with open(model_dir / "metrics.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(model_dir / "features.json", "w") as fh:
        json.dump(artifacts["feature_cols"], fh, indent=2)


def load_model(model_dir: Path) -> dict:
    """Load saved model artifacts from a model directory.

    Args:
        model_dir: Directory produced by save_model() (contains model.pkl etc.)

    Returns:
        Dict with model, scaler, feature_cols, threshold, exercise.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    with open(model_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    with open(model_dir / "features.json") as fh:
        feature_cols = json.load(fh)
    with open(model_dir / "metrics.json") as fh:
        metrics = json.load(fh)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "threshold": metrics["threshold"],
        "exercise": metrics["exercise"],
        "metrics": metrics,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.config import get_config

    parser = argparse.ArgumentParser(description="Train logistic regression for one exercise.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--exercise", required=True, choices=["wall_slide", "band_er_side"])
    parser.add_argument("--out-dir", default="data/models")
    args = parser.parse_args()

    cfg = get_config()
    train_df = pd.read_csv(args.train)
    artifacts = train_model(
        train_df,
        args.exercise,
        C_values=cfg["model"]["C_values"],
        cv_folds=cfg["model"]["cv_folds"],
        precision_target=cfg["model"]["precision_target"],
    )
    save_model(artifacts, Path(args.out_dir))

    print(f"Exercise  : {args.exercise}")
    print(f"Best C    : {artifacts['best_C']}")
    print(f"Threshold : {artifacts['threshold']:.2f}")
    print(f"CV AUC    : {artifacts['cv_results']}")
    print(f"Train AUC : {artifacts['train_metrics']['auc']}")
    print(f"Train P/R : {artifacts['train_metrics']['precision']:.3f} / {artifacts['train_metrics']['recall']:.3f}")
    print(f"Saved to  : {Path(args.out_dir) / args.exercise}/")
