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
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
    cross_val_score,
)
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

# Added when personalization is enabled — originals are kept so the model
# can weight absolute vs relative signal via regularisation.
_PERSONALIZED_EXTRA_COLS = [
    "rom_proxy_max_z", "rom_proxy_max_pct",
    "tempo_s_z", "tempo_s_pct",
]


def get_feature_cols(exercise: str, personalize: bool = False) -> list[str]:
    """Return the feature columns used for a given exercise.

    Args:
        exercise:    Exercise identifier.
        personalize: When True, append the personalized feature columns
                     (``*_z`` and ``*_pct`` for ROM and tempo).  The caller
                     is responsible for ensuring those columns exist in the
                     input DataFrame.
    """
    if exercise not in FEATURE_COLS:
        raise ValueError(f"Unknown exercise: '{exercise}'")
    cols = list(FEATURE_COLS[exercise])
    if personalize:
        cols = cols + list(_PERSONALIZED_EXTRA_COLS)
    return cols


def _feature_cols(exercise: str) -> list[str]:
    # Kept for backwards compatibility with older callers/tests.
    return get_feature_cols(exercise, personalize=False)


def _build_pipeline(C: float, random_state: int) -> Pipeline:
    """Build the scaler + logistic regression pipeline used for both CV and
    the final fit.  ``class_weight='balanced'`` corrects the good/bad class
    imbalance (wall_slide train is ~1:2.2) without changing the decision
    threshold; the threshold is tuned separately from out-of-fold scores.
    ``solver='liblinear'`` is deterministic and well-suited to small datasets.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=C,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
        ),
    ])


def train_model(
    train_df: pd.DataFrame,
    exercise: str,
    C_values: list[float],
    cv_folds: int = 5,
    precision_target: float = 0.8,
    random_state: int = 42,
    personalize: bool = False,
) -> dict:
    """Train logistic regression with GroupKFold CV to select C.

    Training protocol:
      1. Filter to this exercise's rows; drop any with NaN feature values.
      2. For each candidate C, compute GroupKFold(session_id) CV ROC-AUC to
         score model quality independently of the decision threshold.
      3. Pick the C with the highest mean CV AUC (ties: smallest C — prefer
         more regularisation / simpler model).
      4. Generate out-of-fold probabilities at the best C via
         cross_val_predict.  Select the threshold that achieves
         ``precision_target`` on these OOF scores — this yields an honest
         unbiased threshold rather than one optimised on train predictions.
      5. Refit the pipeline on the full training set for deployment.

    Args:
        train_df:         Training split from split_dataset().
        exercise:         Exercise identifier.
        C_values:         Regularisation strengths to try (higher = weaker L2).
        cv_folds:         Requested number of GroupKFold folds; clamped to
                          the number of distinct sessions for this exercise.
        precision_target: Target minimum precision for threshold selection
                          (picked on out-of-fold predictions).
        random_state:     RNG seed for LogisticRegression.

    Returns:
        Dict with keys: model, scaler, best_C, cv_results, feature_cols,
                        exercise, threshold, train_metrics, cv_metrics,
                        n_cv_folds.
    """
    feat_cols = get_feature_cols(exercise, personalize=personalize)
    ex_df = train_df[train_df["exercise"] == exercise].copy()
    missing = [c for c in feat_cols if c not in ex_df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns for exercise='{exercise}' "
            f"(personalize={personalize}): {missing}"
        )

    if len(ex_df) == 0:
        raise ValueError(f"No training data for exercise '{exercise}'")

    # Drop rows with NaN features
    ex_df = ex_df.dropna(subset=feat_cols)  # type: ignore[call-overload]
    X: np.ndarray = ex_df[feat_cols].to_numpy(dtype=float)
    y: np.ndarray = ex_df["y_bad"].to_numpy(dtype=int)
    groups: np.ndarray = ex_df["session_id"].to_numpy()

    n_sessions = int(ex_df["session_id"].nunique())
    n_splits = max(2, min(cv_folds, n_sessions))
    # StratifiedGroupKFold keeps sessions whole (no rep in two folds) AND
    # tries to balance the good/bad ratio across folds so every fold has
    # both classes in its validation slice — without this, a small-session
    # exercise like band_er_side can produce folds with only one class and
    # roc_auc returns nan.
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ── C selection via CV AUC ────────────────────────────────────────────────
    cv_results: dict[float, float] = {}
    cv_results_std: dict[float, float] = {}
    for C in C_values:
        scores = cross_val_score(
            _build_pipeline(C, random_state),
            X, y, cv=gkf, groups=groups, scoring="roc_auc",
        )
        cv_results[C] = float(np.mean(scores))
        cv_results_std[C] = float(np.std(scores))

    # Tie-break: higher AUC wins; among ties, smaller C (stronger regularisation)
    best_C = max(cv_results, key=lambda c: (cv_results[c], -c))

    # ── Out-of-fold predictions for honest threshold selection ────────────────
    oof_proba = cross_val_predict(
        _build_pipeline(best_C, random_state),
        X, y, cv=gkf, groups=groups, method="predict_proba",
    )
    oof_prob: np.ndarray = np.asarray(oof_proba)[:, 1]

    threshold = _select_threshold_from_scores(y, oof_prob, precision_target)

    oof_pred = (oof_prob >= threshold).astype(int)
    cv_metrics = {
        "mean_auc": cv_results[best_C],
        "std_auc": cv_results_std[best_C],
        "oof_auc": (
            float(roc_auc_score(y, oof_prob)) if len(np.unique(y)) > 1 else None
        ),
        "oof_precision": float(
            precision_score(y, oof_pred, zero_division=0)  # type: ignore[arg-type]
        ),
        "oof_recall": float(
            recall_score(y, oof_pred, zero_division=0)  # type: ignore[arg-type]
        ),
    }

    # ── Final model on full training set ──────────────────────────────────────
    final_pipe = _build_pipeline(best_C, random_state).fit(X, y)
    scaler: StandardScaler = final_pipe.named_steps["scaler"]
    model: LogisticRegression = final_pipe.named_steps["clf"]
    X_scaled = scaler.transform(X)

    # ── Train metrics (for sanity check — do not use to select threshold) ─────
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    train_metrics = {
        "auc": (
            float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else None
        ),
        "precision": float(
            precision_score(y, y_pred, zero_division=0)  # type: ignore[arg-type]
        ),
        "recall": float(
            recall_score(y, y_pred, zero_division=0)  # type: ignore[arg-type]
        ),
        "n_train": int(len(y)),
        "n_bad": int(np.sum(y)),
        "n_good": int(np.sum(y == 0)),
    }

    return {
        "model": model,
        "scaler": scaler,
        "best_C": best_C,
        "cv_results": cv_results,
        "cv_results_std": cv_results_std,
        "feature_cols": feat_cols,
        "exercise": exercise,
        "threshold": threshold,
        "train_metrics": train_metrics,
        "cv_metrics": cv_metrics,
        "n_cv_folds": n_splits,
        "personalize": personalize,
    }


def _select_threshold_from_scores(
    y: np.ndarray,
    scores: np.ndarray,
    precision_target: float,
) -> float:
    """Pick the smallest threshold that achieves ``precision_target`` on the
    given scores.  Smallest qualifying threshold → highest recall among
    points that meet the precision target.

    Falls back to 0.5 if no threshold hits the target (the model will then
    have train-precision below the target, which gets surfaced in metrics).
    """
    best_threshold = 0.5
    for thresh in np.arange(0.30, 0.95 + 1e-9, 0.01):
        y_pred = (scores >= thresh).astype(int)
        if int(np.sum(y_pred)) == 0:
            continue
        prec = precision_score(y, y_pred, zero_division=0)  # type: ignore[arg-type]
        if float(prec) >= precision_target:
            best_threshold = float(thresh)
            break
    return best_threshold


def select_threshold(
    model: LogisticRegression,
    X_scaled: np.ndarray,
    y: np.ndarray,
    precision_target: float,
) -> float:
    """Pick a threshold from in-sample model probabilities.

    Kept for backwards compatibility with test fixtures that select a
    threshold directly from a fitted model.  Production threshold tuning
    goes through out-of-fold scores inside :func:`train_model`.

    Args:
        model:            Fitted LogisticRegression.
        X_scaled:         Scaled feature matrix.
        y:                True binary labels.
        precision_target: Desired minimum precision.

    Returns:
        Probability threshold in (0, 1].
    """
    y_prob = model.predict_proba(X_scaled)[:, 1]
    return _select_threshold_from_scores(y, y_prob, precision_target)


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
        "n_cv_folds": artifacts.get("n_cv_folds"),
        "personalize": artifacts.get("personalize", False),
        "cv_results": {str(k): v for k, v in artifacts["cv_results"].items()},
        "cv_results_std": {
            str(k): v for k, v in artifacts.get("cv_results_std", {}).items()
        },
        "cv_metrics": artifacts.get("cv_metrics"),
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
        random_state=cfg["model"].get("random_state", 42),
    )
    save_model(artifacts, Path(args.out_dir))

    cv = artifacts["cv_metrics"]
    tm = artifacts["train_metrics"]
    print(f"Exercise        : {args.exercise}")
    print(f"CV folds used   : {artifacts['n_cv_folds']}")
    print(f"Best C          : {artifacts['best_C']}")
    print(f"Threshold (OOF) : {artifacts['threshold']:.3f}")
    print(f"CV AUC per C    : " + ", ".join(
        f"C={c:<5}: {artifacts['cv_results'][c]:.3f}±{artifacts['cv_results_std'][c]:.3f}"
        for c in artifacts['cv_results']
    ))
    print(f"OOF AUC         : {cv['oof_auc']:.3f}")
    print(f"OOF P / R       : {cv['oof_precision']:.3f} / {cv['oof_recall']:.3f}")
    print(f"Train AUC       : {tm['auc']:.3f}")
    print(f"Train P / R     : {tm['precision']:.3f} / {tm['recall']:.3f}")
    print(f"n_good / n_bad  : {tm['n_good']} / {tm['n_bad']}")
    print(f"Saved to        : {Path(args.out_dir) / args.exercise}/")
