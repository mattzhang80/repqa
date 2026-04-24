"""Per-rep feature extraction for RepQA.

Computes interpretable, physics-inspired features for each detected rep:
  - ROM proxy: how much range of motion was achieved (normalised, not clinical)
  - Tempo: rep duration in seconds
  - Tempo deviation: how far from the prescribed tempo centre
  - Confidence: pose-estimation quality within the rep

Wall Slide (side view):
    ROM proxy = max vertical wrist travel normalised by torso height.

Band ER Side (front view):
    ROM proxy = max lateral wrist displacement normalised by shoulder width.
    (Added in Phase 7.)

Usage:
    python src/pipeline/features.py \\
        --poses data/poses/<session>/poses.parquet \\
        --reps  data/processed/<session>/reps.csv \\
        --exercise wall_slide \\
        --session-id <session_id> \\
        --user-id   <user_id> \\
        --fps 30 \\
        --output data/features/<session>/features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.rep_segment import Rep, select_signal_arm, select_working_arm_band_er

# ── Key joints used for confidence scoring ────────────────────────────────────

_KEY_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
]

# ── Nominal tempo centres (seconds) per exercise ──────────────────────────────
# 2s up + 1s hold + 2s down = 5s for both exercises.

_TEMPO_NOMINAL_S = {
    "wall_slide": 5.0,
    "band_er_side": 5.0,  # 2s out + 1s hold + 2s back = 5s nominal
}


# ── ROM proxy ─────────────────────────────────────────────────────────────────

def compute_rom_proxy_wall_slide(
    pose_df: pd.DataFrame,
    rep: Rep,
) -> dict[str, float]:
    """ROM proxy for the Wall Slide (side view).

    Measures the vertical range of wrist travel within the rep, normalised by
    torso height.  Uses the same arm selection logic as the segmentation signal.

    The proxy is intentionally labelled 'proxy' — it is not a clinical angle.

    Args:
        pose_df: Wide-format pose DataFrame (all frames in the session).
        rep:     Rep dataclass with frame boundaries.

    Returns:
        Dict with keys:
            "rom_proxy_max":   max normalised height reached (signal peak value)
            "rom_proxy_range": peak-to-trough range within the rep
    """
    arm = select_signal_arm(pose_df)
    rep_df = pose_df.iloc[rep.start_frame : rep.end_frame + 1]

    wrist_y = rep_df[f"{arm}_wrist_y"].values.astype(float)
    shoulder_y = rep_df[f"{arm}_shoulder_y"].values.astype(float)
    hip_y = rep_df[f"{arm}_hip_y"].values.astype(float)

    torso = hip_y - shoulder_y
    med_torso = float(np.nanmedian(torso))
    if not (med_torso > 0):
        med_torso = 0.2
    torso = np.where(~np.isfinite(torso) | (np.abs(torso) < 1e-3), med_torso, torso)

    signal = (shoulder_y - wrist_y) / torso   # positive = wrist above shoulder

    finite = signal[np.isfinite(signal)]
    if len(finite) == 0:
        return {"rom_proxy_max": float("nan"), "rom_proxy_range": float("nan")}

    return {
        "rom_proxy_max": float(np.nanmax(signal)),
        "rom_proxy_range": float(np.nanmax(signal) - np.nanmin(signal)),
    }


def compute_rom_proxy_band_er_side(
    pose_df: pd.DataFrame,
    rep: Rep,
) -> dict[str, float]:
    """ROM proxy for Band ER Side (front view).

    Measures the max lateral displacement of the wrist from the elbow,
    normalised by shoulder width.  Captures how far the forearm rotated
    outward during the rep.

    Args:
        pose_df: Wide-format pose DataFrame.
        rep:     Rep dataclass with frame boundaries.

    Returns:
        Dict with "rom_proxy_max" and "rom_proxy_range".
    """
    arm = select_working_arm_band_er(pose_df)
    rep_df = pose_df.iloc[rep.start_frame : rep.end_frame + 1]

    wrist_x = rep_df[f"{arm}_wrist_x"].values.astype(float)
    elbow_x = rep_df[f"{arm}_elbow_x"].values.astype(float)

    left_sh_x = rep_df["left_shoulder_x"].values.astype(float)
    right_sh_x = rep_df["right_shoulder_x"].values.astype(float)
    shoulder_width = np.abs(left_sh_x - right_sh_x)
    med_width = float(np.nanmedian(shoulder_width))
    if not (med_width > 0):
        med_width = 0.2

    lateral = np.abs(wrist_x - elbow_x) / np.where(
        ~np.isfinite(shoulder_width) | (shoulder_width < 1e-3),
        med_width,
        shoulder_width,
    )

    finite = lateral[np.isfinite(lateral)]
    if len(finite) == 0:
        return {"rom_proxy_max": float("nan"), "rom_proxy_range": float("nan")}

    return {
        "rom_proxy_max": float(np.nanmax(lateral)),
        "rom_proxy_range": float(np.nanmax(lateral) - np.nanmin(lateral)),
    }


def compute_elbow_drift(
    pose_df: pd.DataFrame,
    rep: Rep,
) -> dict[str, float]:
    """Track elbow displacement from its starting position during a rep.

    For Band ER Side: a stable elbow (pinned to the side) should barely move.
    If the elbow lifts away from the ribs during rotation, drift increases.

    Displacement is Euclidean (x + y) and normalised by shoulder width.

    Args:
        pose_df: Wide-format pose DataFrame.
        rep:     Rep dataclass with frame boundaries.

    Returns:
        Dict with "elbow_drift_max" and "elbow_drift_mean".
    """
    arm = select_working_arm_band_er(pose_df)
    rep_df = pose_df.iloc[rep.start_frame : rep.end_frame + 1]

    elbow_x = rep_df[f"{arm}_elbow_x"].values.astype(float)
    elbow_y = rep_df[f"{arm}_elbow_y"].values.astype(float)

    # Starting position: mean of first 5 frames
    n_start = min(5, len(elbow_x))
    start_x = float(np.nanmean(elbow_x[:n_start]))
    start_y = float(np.nanmean(elbow_y[:n_start]))

    drift = np.sqrt((elbow_x - start_x) ** 2 + (elbow_y - start_y) ** 2)

    # Normalise by shoulder width
    left_sh_x = rep_df["left_shoulder_x"].values.astype(float)
    right_sh_x = rep_df["right_shoulder_x"].values.astype(float)
    med_width = float(np.nanmedian(np.abs(left_sh_x - right_sh_x)))
    if not (med_width > 0):
        med_width = 0.2

    drift_norm = drift / med_width

    finite = drift_norm[np.isfinite(drift_norm)]
    if len(finite) == 0:
        return {"elbow_drift_max": float("nan"), "elbow_drift_mean": float("nan")}

    return {
        "elbow_drift_max": float(np.nanmax(drift_norm)),
        "elbow_drift_mean": float(np.nanmean(drift_norm)),
    }


# ── Tempo ─────────────────────────────────────────────────────────────────────

def compute_tempo(rep: Rep, fps: int) -> float:
    """Rep duration in seconds.

    Args:
        rep: Rep dataclass.
        fps: Video frame rate (unused if end_time_s / start_time_s already set,
             but kept for API consistency).

    Returns:
        Duration in seconds.
    """
    return rep.end_time_s - rep.start_time_s


def compute_tempo_deviation(tempo_s: float, exercise: str) -> float:
    """Absolute deviation from the prescribed tempo centre.

    Both exercises have a nominal 5-second rep (2s out + 1s hold + 2s back).

    Args:
        tempo_s:  Actual rep duration in seconds.
        exercise: Exercise identifier.

    Returns:
        |tempo_s - nominal_s| in seconds.  0 = perfect, >0 = deviation.
    """
    nominal = _TEMPO_NOMINAL_S.get(exercise, 5.0)
    return abs(tempo_s - nominal)


# ── Confidence ────────────────────────────────────────────────────────────────

def compute_confidence_features(
    pose_df: pd.DataFrame,
    rep: Rep,
) -> dict[str, float]:
    """Mean and min visibility of key joints within a rep.

    Args:
        pose_df: Wide-format pose DataFrame.
        rep:     Rep dataclass with frame boundaries.

    Returns:
        Dict with keys:
            "conf_mean": mean visibility across key joints and frames
            "conf_min":  minimum visibility across key joints and frames
    """
    rep_df = pose_df.iloc[rep.start_frame : rep.end_frame + 1]

    vis_cols = [f"{j}_vis" for j in _KEY_JOINTS if f"{j}_vis" in rep_df.columns]
    if not vis_cols:
        return {"conf_mean": float("nan"), "conf_min": float("nan")}

    vis_values = rep_df[vis_cols].values.astype(float).ravel()
    finite = vis_values[np.isfinite(vis_values)]
    if len(finite) == 0:
        return {"conf_mean": float("nan"), "conf_min": float("nan")}

    return {
        "conf_mean": float(np.mean(finite)),
        "conf_min": float(np.min(finite)),
    }


# ── Top-level extractor ───────────────────────────────────────────────────────

def extract_rep_features(
    pose_df: pd.DataFrame,
    reps: list[Rep],
    exercise: str,
    fps: int,
    session_meta: dict,
) -> pd.DataFrame:
    """Extract features for all reps in a session.

    Produces one row per rep.  The ROM proxy column names are the same for both
    exercises ('rom_proxy_max', 'rom_proxy_range') so downstream ML code can
    treat them uniformly; the *meaning* depends on exercise type and is
    documented in config.yaml under features.rom_proxy.

    Args:
        pose_df:      Wide-format pose DataFrame.
        reps:         List of Rep dataclasses from segment_reps().
        exercise:     Exercise identifier ('wall_slide' or 'band_er_side').
        fps:          Video frame rate.
        session_meta: Dict with at least 'session_id' and 'user_id' keys.

    Returns:
        DataFrame with columns:
            session_id, rep_id, exercise, user_id,
            rom_proxy_max, rom_proxy_range,
            tempo_s, tempo_deviation,
            conf_mean, conf_min

    Raises:
        NotImplementedError: If exercise == 'band_er_side' (Phase 7).
        ValueError:          If exercise is unrecognised.
    """
    if exercise not in ("wall_slide", "band_er_side"):
        raise ValueError(
            f"Unknown exercise: '{exercise}'. Supported: wall_slide, band_er_side"
        )

    session_id = session_meta.get("session_id", "unknown")
    user_id = session_meta.get("user_id", "unknown")

    rows = []
    for rep in reps:
        if exercise == "wall_slide":
            rom = compute_rom_proxy_wall_slide(pose_df, rep)
        else:
            rom = compute_rom_proxy_band_er_side(pose_df, rep)

        tempo_s = compute_tempo(rep, fps)
        tempo_dev = compute_tempo_deviation(tempo_s, exercise)
        conf = compute_confidence_features(pose_df, rep)

        row_dict: dict[str, object] = {
            "session_id": session_id,
            "rep_id": rep.rep_id,
            "exercise": exercise,
            "user_id": user_id,
            "rom_proxy_max": rom["rom_proxy_max"],
            "rom_proxy_range": rom["rom_proxy_range"],
            "tempo_s": tempo_s,
            "tempo_deviation": tempo_dev,
            "conf_mean": conf["conf_mean"],
            "conf_min": conf["conf_min"],
        }

        # Band ER Side gets elbow drift features
        if exercise == "band_er_side":
            drift = compute_elbow_drift(pose_df, rep)
            row_dict["elbow_drift_max"] = drift["elbow_drift_max"]
            row_dict["elbow_drift_mean"] = drift["elbow_drift_mean"]

        rows.append(row_dict)

    cols = [
        "session_id", "rep_id", "exercise", "user_id",
        "rom_proxy_max", "rom_proxy_range",
        "tempo_s", "tempo_deviation",
        "conf_mean", "conf_min",
    ]
    if exercise == "band_er_side":
        cols.extend(["elbow_drift_max", "elbow_drift_mean"])

    return pd.DataFrame(rows, columns=cols)  # type: ignore[arg-type]


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_features(features_df: pd.DataFrame, out_path: Path) -> None:
    """Save features DataFrame to CSV.

    Args:
        features_df: DataFrame from extract_rep_features().
        out_path:    Destination path; parent dirs created if absent.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)


def load_features(path: Path) -> pd.DataFrame:
    """Load a features CSV produced by save_features().

    Args:
        path: Path to features.csv.

    Returns:
        DataFrame with correct dtypes.
    """
    return pd.read_csv(path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-rep features from poses + reps."
    )
    parser.add_argument("--poses", required=True, help="Path to poses.parquet")
    parser.add_argument("--reps", required=True, help="Path to reps.csv")
    parser.add_argument(
        "--exercise", required=True, choices=["wall_slide", "band_er_side"]
    )
    parser.add_argument("--session-id", required=True, help="Session identifier")
    parser.add_argument("--user-id", required=True, help="User identifier")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default 30)")
    parser.add_argument("--output", required=True, help="Path for output features.csv")
    args = parser.parse_args()

    from src.pipeline.pose_extract import load_poses
    from src.pipeline.rep_segment import Rep

    pose_df = load_poses(Path(args.poses))

    reps_df = pd.read_csv(args.reps)
    reps = [
        Rep(
            rep_id=int(row["rep_id"]),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            start_time_s=float(row["start_time_s"]),
            end_time_s=float(row["end_time_s"]),
        )
        for _, row in reps_df.iterrows()
    ]

    session_meta = {"session_id": args.session_id, "user_id": args.user_id}
    features_df = extract_rep_features(
        pose_df, reps, args.exercise, args.fps, session_meta
    )
    save_features(features_df, Path(args.output))

    print(f"Features extracted: {len(features_df)} reps")
    print(features_df.to_string(index=False))
    print(f"Saved to: {args.output}")
