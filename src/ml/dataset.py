"""Dataset assembly and train/test split for RepQA ML pipeline.

Merges per-session features with human labels, creates a binary target,
and splits by session_id (group split) to prevent data leakage.

Usage:
    python src/ml/dataset.py \\
        --features-dir data/features \\
        --labels       data/labels/labels.csv \\
        --out-dir      data/features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def assemble_dataset(
    features_dir: Path,
    labels_path: Path,
) -> pd.DataFrame:
    """Merge all session feature CSVs with human labels.

    Walks features_dir for features.csv files, concatenates them, then
    inner-joins with labels on (session_id, rep_id).  Adds:
      - y_bad (int):     1 if label != 'good', else 0
      - label_detail (str): the original label string

    Args:
        features_dir: Directory containing <session>/features.csv files.
        labels_path:  Path to data/labels/labels.csv.

    Returns:
        DataFrame with all feature columns plus y_bad and label_detail.

    Raises:
        FileNotFoundError: If labels_path does not exist.
        ValueError:        If no features CSVs found.
    """
    features_dir = Path(features_dir)
    labels_path = Path(labels_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Collect all features CSVs
    feat_frames = []
    for csv_path in sorted(features_dir.rglob("features.csv")):
        feat_frames.append(pd.read_csv(csv_path))

    if not feat_frames:
        raise ValueError(f"No features.csv files found under {features_dir}")

    features_df = pd.concat(feat_frames, ignore_index=True)

    # Load labels
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[["session_id", "rep_id", "label"]].copy()
    labels_df["rep_id"] = labels_df["rep_id"].astype(int)
    features_df["rep_id"] = features_df["rep_id"].astype(int)

    # Deduplicate labels: keep last (most recent label wins)
    labels_df = labels_df.drop_duplicates(subset=["session_id", "rep_id"], keep="last")  # type: ignore[call-overload]

    merged = features_df.merge(labels_df, on=["session_id", "rep_id"], how="inner")
    merged["y_bad"] = (merged["label"] != "good").astype(int)
    merged = merged.rename(columns={"label": "label_detail"})

    return merged.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by session_id to prevent leakage.

    No session appears in both train and test.  When stratify=True (default),
    sessions are bucketed by (exercise, label_detail) and drawn proportionally
    from each bucket so every class in every exercise is represented in both
    splits — critical with a small dataset (38 sessions) where a naive
    GroupShuffleSplit can drop an entire class from test by chance.

    Stratification is done at the session level (one stratum key per session),
    not the rep level, to keep each session whole.  This requires each session
    to have a single dominant label_detail, which holds for RepQA since every
    video is recorded with one intended form.

    Args:
        df:           Output of assemble_dataset().
        test_size:    Fraction of sessions to hold out for test.
        random_state: RNG seed for reproducibility.
        stratify:     If True, stratify by (exercise, label_detail).  If False,
                      fall back to unstratified GroupShuffleSplit.

    Returns:
        (train_df, test_df) — non-overlapping by session_id.
    """
    if not stratify or "label_detail" not in df.columns:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        groups = df["session_id"].values
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    # Session-level stratification: one row per session with its stratum key
    per_session = (
        df.groupby("session_id")
        .agg(exercise=("exercise", "first"), label_detail=("label_detail", "first"))
        .reset_index()
    )
    per_session["stratum"] = (
        per_session["exercise"].astype(str) + "__" + per_session["label_detail"].astype(str)
    )

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_s_idx, test_s_idx = next(
        sss.split(per_session, per_session["stratum"])
    )
    train_sessions = list(per_session.iloc[train_s_idx]["session_id"])
    test_sessions = list(per_session.iloc[test_s_idx]["session_id"])

    train_df = df[df["session_id"].isin(train_sessions)].reset_index(drop=True)
    test_df = df[df["session_id"].isin(test_sessions)].reset_index(drop=True)
    return train_df, test_df  # type: ignore[return-value]


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save train.csv and test.csv to out_dir.

    Args:
        train_df: Training split from split_dataset().
        test_df:  Test split from split_dataset().
        out_dir:  Destination directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble and split ML dataset.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = assemble_dataset(Path(args.features_dir), Path(args.labels))
    print(f"Assembled: {len(df)} labeled reps from {df['session_id'].nunique()} sessions")
    print(f"  y_bad=1: {df['y_bad'].sum()}  y_bad=0: {(df['y_bad']==0).sum()}")
    print(f"  Labels: {df['label_detail'].value_counts().to_dict()}")

    train_df, test_df = split_dataset(df, test_size=args.test_size)
    print(f"Train: {len(train_df)} reps ({train_df['session_id'].nunique()} sessions)")
    print(f"Test : {len(test_df)} reps ({test_df['session_id'].nunique()} sessions)")

    save_splits(train_df, test_df, Path(args.out_dir))
    print(f"Saved to {args.out_dir}/train.csv and test.csv")
