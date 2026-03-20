"""Per-user baseline computation and personalized feature transforms.

Fits a baseline (median + IQR) from a user's first N sessions, then
adds z-score and percentile columns to new session features.

Usage:
    python src/ml/personalize.py \\
        --features-dir data/features \\
        --user-id matthew \\
        --exercise wall_slide
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_BASELINES_DIR = Path("data/models/baselines")
_BASELINE_FEATURES = ["rom_proxy_max", "tempo_s"]


def fit_user_baseline(
    features_df: pd.DataFrame,
    user_id: str,
    exercise: str,
    n_sessions: int = 5,
) -> dict:
    """Compute per-user baseline statistics from the first N sessions.

    Args:
        features_df: All available feature rows for this user/exercise.
        user_id:     User identifier.
        exercise:    Exercise identifier.
        n_sessions:  Number of earliest sessions to use as baseline.

    Returns:
        Baseline dict with median, IQR, and std for each baseline feature.
        Saved to data/models/baselines/<user_id>_<exercise>.json.
    """
    user_df = features_df[
        (features_df["user_id"] == user_id) &
        (features_df["exercise"] == exercise)
    ].copy()

    if len(user_df) == 0:
        raise ValueError(f"No features found for user='{user_id}', exercise='{exercise}'")

    # Take first n_sessions (by session_id alphabetical order as a proxy for time)
    sessions = sorted(user_df["session_id"].unique().tolist())[:n_sessions]
    baseline_df = user_df[user_df["session_id"].isin(sessions)]

    stats: dict = {
        "user_id": user_id,
        "exercise": exercise,
        "n_sessions_used": len(sessions),
        "sessions": sessions,
    }
    for feat in _BASELINE_FEATURES:
        series: pd.Series = baseline_df[feat]  # type: ignore[assignment]
        vals: np.ndarray = np.asarray(series.values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            stats[f"{feat}_median"] = None
            stats[f"{feat}_iqr"] = None
            stats[f"{feat}_std"] = None
            continue
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        stats[f"{feat}_median"] = float(np.median(vals))
        stats[f"{feat}_iqr"] = float(q75 - q25)
        stats[f"{feat}_std"] = float(np.std(vals))

    # Save baseline
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _BASELINES_DIR / f"{user_id}_{exercise}.json"
    with open(out_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    return stats


def load_user_baseline(user_id: str, exercise: str) -> dict | None:
    """Load a saved user baseline, or None if not found."""
    path = _BASELINES_DIR / f"{user_id}_{exercise}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def apply_personalization(
    features_df: pd.DataFrame,
    baseline: dict,
) -> pd.DataFrame:
    """Add personalized z-score columns to a features DataFrame.

    Adds for each baseline feature:
        <feat>_z:   (value - median) / IQR  (robust z-score)

    Original columns are preserved.

    Args:
        features_df: Feature DataFrame for new sessions.
        baseline:    Dict from fit_user_baseline() or load_user_baseline().

    Returns:
        Copy of features_df with additional *_z columns.
    """
    df = features_df.copy()
    for feat in _BASELINE_FEATURES:
        median = baseline.get(f"{feat}_median")
        iqr = baseline.get(f"{feat}_iqr")
        if median is None or iqr is None or iqr == 0:
            df[f"{feat}_z"] = float("nan")
        else:
            df[f"{feat}_z"] = (df[feat] - median) / iqr
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit user baseline and apply personalization.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--exercise", required=True, choices=["wall_slide", "band_er_side"])
    parser.add_argument("--n-sessions", type=int, default=5)
    args = parser.parse_args()

    feat_frames = []
    for p in Path(args.features_dir).rglob("features.csv"):
        feat_frames.append(pd.read_csv(p))
    if not feat_frames:
        raise SystemExit(f"No features.csv found under {args.features_dir}")

    features_df = pd.concat(feat_frames, ignore_index=True)
    baseline = fit_user_baseline(
        features_df, args.user_id, args.exercise, args.n_sessions
    )
    print(f"Baseline for {args.user_id} / {args.exercise}:")
    for k, v in baseline.items():
        print(f"  {k}: {v}")
