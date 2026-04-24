"""Per-user baseline computation and personalized feature transforms.

Fits a per-user, per-exercise baseline from the user's GOOD reps only, then
adds robust z-score and percentile columns to new session features.

The baseline captures what *this user's normal good form* looks like:
their typical ROM and tempo.  Personalized features express each new rep
relative to that baseline — a rep that looks "too short" in absolute
terms may be normal for a particular user, and vice versa.

Why good reps only:
    A baseline meant to encode "normal form" must not be contaminated by
    bad reps.  Otherwise a user with many rushed bad_tempo reps would end
    up with a baseline median that already looks fast, and the
    personalization would undo the very signal the model needs.

Why explicit baseline sessions:
    Session IDs in this project do not carry timestamps, so sorting by
    session_id is meaningless as a proxy for "earliest sessions".  Callers
    must either pass the list of baseline sessions explicitly (preferred)
    or rely on the label filter to restrict to good reps.

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

# Features personalized per user/exercise.  ROM proxy and tempo are the two
# interpretable dimensions that vary most across users (mobility, cadence);
# confidence features are pose-quality indicators and have no user-specific
# meaning.
_BASELINE_FEATURES = ["rom_proxy_max", "tempo_s"]


def fit_user_baseline(
    features_df: pd.DataFrame,
    user_id: str,
    exercise: str,
    n_sessions: int | None = None,
    baseline_session_ids: list[str] | None = None,
    label_col: str = "label_detail",
    good_label: str = "good",
    save: bool = True,
) -> dict:
    """Compute per-user baseline statistics from the user's good reps.

    Selection logic (in order of precedence):
      1. If ``baseline_session_ids`` is given, use exactly those sessions.
      2. Else if ``label_col`` is present, filter to rows where the label
         equals ``good_label``.
      3. Else use all rows for this user/exercise (back-compat; emits a
         warning via dict entry).
      4. Then cap at ``n_sessions`` if provided (chronological order
         assumed preserved by the caller; we sort by session_id as a
         stable tie-break but do not rely on it for correctness).

    Args:
        features_df:          DataFrame with at least ``user_id``,
                              ``exercise``, ``session_id``, and the baseline
                              feature columns.  ``label_col`` is required
                              unless ``baseline_session_ids`` is passed.
        user_id:              User identifier.
        exercise:             Exercise identifier.
        n_sessions:           Optional cap on the number of baseline
                              sessions after filtering.
        baseline_session_ids: Explicit session list; overrides label-based
                              filtering.
        label_col:            Name of the label column used for the good-rep
                              filter.  Ignored if ``baseline_session_ids``
                              is set.
        good_label:           Value of ``label_col`` identifying good reps.
        save:                 Whether to persist the baseline JSON to
                              ``data/models/baselines/<user>_<exercise>.json``.

    Returns:
        Baseline dict with: user_id, exercise, sessions, n_sessions_used,
        contamination_warning, and for each feature ``f`` in
        :data:`_BASELINE_FEATURES`:
            ``f_median``, ``f_iqr``, ``f_std``, ``f_values`` (list of raw
            values used to compute percentiles later).
    """
    user_df = features_df[
        (features_df["user_id"] == user_id)
        & (features_df["exercise"] == exercise)
    ].copy()

    if len(user_df) == 0:
        raise ValueError(
            f"No features found for user='{user_id}', exercise='{exercise}'"
        )

    contamination_warning: str | None = None
    if baseline_session_ids is not None:
        missing = [s for s in baseline_session_ids if s not in set(user_df["session_id"])]
        if missing:
            raise ValueError(
                f"baseline_session_ids not found in features: {missing}"
            )
        baseline_df = user_df[user_df["session_id"].isin(baseline_session_ids)]
    elif label_col in user_df.columns:
        baseline_df = user_df[user_df[label_col] == good_label]
        if len(baseline_df) == 0:
            raise ValueError(
                f"No '{good_label}' rows found for user='{user_id}', "
                f"exercise='{exercise}'"
            )
    else:
        baseline_df = user_df
        contamination_warning = (
            f"No '{label_col}' column and no explicit baseline_session_ids; "
            f"baseline is computed from ALL reps including bad ones. "
            f"This likely harms personalization quality."
        )

    sessions = sorted(baseline_df["session_id"].unique().tolist())
    if n_sessions is not None:
        sessions = sessions[:n_sessions]
        baseline_df = baseline_df[baseline_df["session_id"].isin(sessions)]

    stats: dict = {
        "user_id": user_id,
        "exercise": exercise,
        "n_sessions_used": len(sessions),
        "n_reps_used": int(len(baseline_df)),
        "sessions": sessions,
        "label_col": label_col,
        "good_label": good_label,
        "contamination_warning": contamination_warning,
    }
    for feat in _BASELINE_FEATURES:
        if feat not in baseline_df.columns:
            stats[f"{feat}_median"] = None
            stats[f"{feat}_iqr"] = None
            stats[f"{feat}_std"] = None
            stats[f"{feat}_values"] = []
            continue
        series: pd.Series = baseline_df[feat]  # type: ignore[assignment]
        vals: np.ndarray = np.asarray(series.values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            stats[f"{feat}_median"] = None
            stats[f"{feat}_iqr"] = None
            stats[f"{feat}_std"] = None
            stats[f"{feat}_values"] = []
            continue
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        stats[f"{feat}_median"] = float(np.median(vals))
        stats[f"{feat}_iqr"] = float(q75 - q25)
        stats[f"{feat}_std"] = float(np.std(vals))
        stats[f"{feat}_values"] = [float(v) for v in vals]

    if save:
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


def _percentile_of(values: list[float], x: float) -> float:
    """Return the percentile rank of ``x`` within ``values`` (0–100).

    Uses the "mean" convention: average of "strictly less than" and
    "less than or equal to" to handle ties symmetrically — so a baseline
    median value lands at ~50 rather than drifting above or below.
    """
    if not values or not np.isfinite(x):
        return float("nan")
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    lt = float(np.sum(arr < x))
    le = float(np.sum(arr <= x))
    return 100.0 * ((lt + le) / 2.0) / n


def apply_personalization(
    features_df: pd.DataFrame,
    baseline: dict,
) -> pd.DataFrame:
    """Add personalized z-score and percentile columns.

    Adds for each feature ``f`` in :data:`_BASELINE_FEATURES`:
      - ``f_z``:   robust z-score, ``(value − median) / IQR``.  Uses IQR
                   (not std) so the statistic is insensitive to tails and
                   stays in a comparable scale across features.
      - ``f_pct``: percentile rank in the baseline distribution (0–100),
                   using the "mean" convention so the baseline median
                   lands near 50.

    Original columns are preserved.  When a baseline is missing (None
    median/IQR, or zero IQR), the corresponding *_z / *_pct columns are
    filled with NaN; the caller (e.g. training) must then either drop
    those rows or impute.

    Args:
        features_df: Feature DataFrame to extend.
        baseline:    Dict from :func:`fit_user_baseline` /
                     :func:`load_user_baseline`.

    Returns:
        Copy of ``features_df`` with new ``*_z`` and ``*_pct`` columns.
    """
    df = features_df.copy()
    for feat in _BASELINE_FEATURES:
        median = baseline.get(f"{feat}_median")
        iqr = baseline.get(f"{feat}_iqr")
        values = baseline.get(f"{feat}_values") or []
        z_col = f"{feat}_z"
        pct_col = f"{feat}_pct"

        if median is None or iqr is None or iqr == 0 or feat not in df.columns:
            df[z_col] = float("nan")
        else:
            df[z_col] = (df[feat].astype(float) - float(median)) / float(iqr)

        if not values or feat not in df.columns:
            df[pct_col] = float("nan")
        else:
            df[pct_col] = df[feat].astype(float).apply(
                lambda x, v=values: _percentile_of(v, x)
            )
    return df


def personalize_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_id: str,
    exercise: str,
    save: bool = True,
    **fit_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit a baseline on TRAIN good reps and apply personalization to both splits.

    Critical: the baseline is fit on ``train_df`` only; applying it to
    ``test_df`` does not leak any test-set information into training.

    Args:
        train_df:    Training split with a ``label_detail`` column.
        test_df:     Test split.
        user_id:     User identifier.
        exercise:    Exercise identifier.
        save:        Whether to persist the baseline JSON.
        **fit_kwargs: Forwarded to :func:`fit_user_baseline`
                     (e.g. ``baseline_session_ids``, ``n_sessions``).

    Returns:
        Tuple ``(train_pers, test_pers, baseline)``.
    """
    baseline = fit_user_baseline(
        train_df, user_id=user_id, exercise=exercise, save=save, **fit_kwargs
    )
    train_pers = apply_personalization(train_df, baseline)
    test_pers = apply_personalization(test_df, baseline)
    return train_pers, test_pers, baseline


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit user baseline from good reps and apply personalization."
    )
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--labels", default="data/labels/labels.csv")
    parser.add_argument("--user-id", required=True)
    parser.add_argument(
        "--exercise", required=True, choices=["wall_slide", "band_er_side"]
    )
    parser.add_argument("--n-sessions", type=int, default=None)
    args = parser.parse_args()

    feat_frames = []
    for p in Path(args.features_dir).rglob("features.csv"):
        feat_frames.append(pd.read_csv(p))
    if not feat_frames:
        raise SystemExit(f"No features.csv found under {args.features_dir}")

    features_df = pd.concat(feat_frames, ignore_index=True)

    # Merge labels so we can filter baseline to good reps
    labels_df = pd.read_csv(args.labels)[
        ["session_id", "rep_id", "label"]
    ].rename(columns={"label": "label_detail"})
    features_df["rep_id"] = features_df["rep_id"].astype(int)
    labels_df["rep_id"] = labels_df["rep_id"].astype(int)
    features_df = features_df.merge(
        labels_df, on=["session_id", "rep_id"], how="left"
    )

    baseline = fit_user_baseline(
        features_df,
        user_id=args.user_id,
        exercise=args.exercise,
        n_sessions=args.n_sessions,
    )
    print(f"Baseline for {args.user_id} / {args.exercise}:")
    for k, v in baseline.items():
        if k.endswith("_values"):
            print(f"  {k}: {len(v)} values")
        else:
            print(f"  {k}: {v}")
