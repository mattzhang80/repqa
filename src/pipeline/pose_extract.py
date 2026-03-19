"""Pose extraction using MediaPipe PoseLandmarker (Tasks API).

Produces a wide-format DataFrame: one row per frame, one column set per
landmark (x, y, z, vis). Coordinates are image-normalised [0, 1].

Usage:
    python src/pipeline/pose_extract.py \\
        --video data/processed/<session>/video.mp4 \\
        --output data/poses/<session>/poses.parquet
"""

import argparse
import os
import urllib.request
from pathlib import Path

# Suppress MediaPipe / TensorFlow info/warning logs before importing them
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmark,
    RunningMode,
)

from src.utils.config import get_section

# ── Landmark name list (index-aligned with MediaPipe PoseLandmark enum) ──────
LANDMARK_NAMES: list[str] = [lm.name.lower() for lm in PoseLandmark]

# ── Key joints used for quality checks ───────────────────────────────────────
KEY_JOINTS: list[str] = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

# Bilateral joint pairs: for side-view exercises the occluded side will have
# low visibility — use the better of the two when assessing quality.
_BILATERAL_PAIRS: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
]


# ── Model path helpers ────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_model_path() -> Path:
    """Return the pose landmarker model path, downloading it if absent."""
    cfg = get_section("pose")
    model_path = _project_root() / cfg["model_path"]
    if not model_path.exists():
        url = cfg["model_download_url"]
        print(f"Pose model not found. Downloading from:\n  {url}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        print(f"Model saved to {model_path}")
    return model_path


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_poses(
    video_path: Path,
    fps: int = 30,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Run MediaPipe PoseLandmarker on a preprocessed video.

    Uses VIDEO running mode for temporal smoothing across frames.
    Coordinates are image-normalised: x, y in [0, 1] (top-left origin).

    Args:
        video_path:     Path to a preprocessed .mp4 video.
        fps:            Expected frame rate of the video (used for timestamps).
        show_progress:  Print a progress line every 150 frames.

    Returns:
        Wide-format DataFrame with columns:
            frame_idx, timestamp_s,
            <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_vis  (×33)
        When no pose is detected in a frame, x/y/z are NaN and vis = 0.0.

    Raises:
        FileNotFoundError: if video_path does not exist.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_path = get_model_path()
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    rows: list[dict] = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row: dict = {
                "frame_idx": frame_idx,
                "timestamp_s": round(frame_idx / fps, 6),
            }

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first (only) person
                for i, name in enumerate(LANDMARK_NAMES):
                    lm = landmarks[i]
                    row[f"{name}_x"] = float(lm.x)
                    row[f"{name}_y"] = float(lm.y)
                    row[f"{name}_z"] = float(lm.z)
                    row[f"{name}_vis"] = float(lm.visibility if lm.visibility is not None else 0.0)
            else:
                for name in LANDMARK_NAMES:
                    row[f"{name}_x"] = float("nan")
                    row[f"{name}_y"] = float("nan")
                    row[f"{name}_z"] = float("nan")
                    row[f"{name}_vis"] = 0.0

            rows.append(row)
            frame_idx += 1

            if show_progress and frame_idx % 150 == 0:
                print(f"  processed {frame_idx} frames…")

    cap.release()

    if not rows:
        raise ValueError(f"No frames read from {video_path}")

    return pd.DataFrame(rows)


# ── Quality assessment ────────────────────────────────────────────────────────

def check_pose_quality(
    pose_df: pd.DataFrame,
    confidence_threshold: float,
    max_low_pct: float,
) -> dict:
    """Assess whether pose tracking is reliable enough for analysis.

    For bilateral joints (left/right shoulder, elbow, wrist, hip) the BETTER
    of the two sides is used, because side-view exercises will always have one
    occluded side — that should not cause a false quality failure.

    Args:
        pose_df:              Wide-format DataFrame from extract_poses().
        confidence_threshold: Minimum acceptable visibility per joint per frame.
        max_low_pct:          Maximum fraction of frames allowed to be below
                              the threshold before a joint is flagged.

    Returns:
        dict with keys:
            pass (bool)           — True if quality is acceptable
            low_confidence_pct    — worst per-joint low-confidence fraction
            worst_joint           — name of the least-visible joint
            per_joint (dict)      — {joint_name: low_confidence_fraction}
    """
    per_joint: dict[str, float] = {}

    # Evaluate bilateral pairs using the better side
    evaluated_pairs: set[str] = set()
    for left, right in _BILATERAL_PAIRS:
        left_vis = pose_df.get(f"{left}_vis", pd.Series(dtype=float))
        right_vis = pose_df.get(f"{right}_vis", pd.Series(dtype=float))
        best_vis = pd.concat([left_vis, right_vis], axis=1).max(axis=1)
        low_pct = (best_vis < confidence_threshold).mean()
        # Report under the pair name
        pair_name = left.replace("left_", "")
        per_joint[pair_name] = float(low_pct)
        evaluated_pairs.update([left, right])

    # Any remaining key joints not covered by bilateral pairs
    for joint in KEY_JOINTS:
        if joint not in evaluated_pairs:
            col = f"{joint}_vis"
            if col in pose_df.columns:
                low_pct = (pose_df[col] < confidence_threshold).mean()
                per_joint[joint] = float(low_pct)

    worst_joint = max(per_joint, key=per_joint.get) if per_joint else "unknown"
    worst_pct = per_joint.get(worst_joint, 0.0)

    return {
        "pass": worst_pct <= max_low_pct,
        "low_confidence_pct": round(worst_pct, 4),
        "worst_joint": worst_joint,
        "per_joint": {k: round(v, 4) for k, v in per_joint.items()},
    }


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_poses(pose_df: pd.DataFrame, out_path: Path) -> None:
    """Save pose DataFrame to Parquet (snappy compression)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pose_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)


def load_poses(path: Path) -> pd.DataFrame:
    """Load a pose Parquet file produced by save_poses()."""
    return pd.read_parquet(Path(path), engine="pyarrow")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks from a preprocessed video."
    )
    parser.add_argument("--video", required=True, help="Path to preprocessed .mp4")
    parser.add_argument("--output", required=True, help="Path for output .parquet")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default 30)")
    args = parser.parse_args()

    cfg = get_section("pose")

    print(f"Extracting poses from: {args.video}")
    pose_df = extract_poses(
        video_path=Path(args.video),
        fps=args.fps,
        show_progress=True,
    )

    quality = check_pose_quality(
        pose_df,
        confidence_threshold=cfg["confidence_threshold"],
        max_low_pct=cfg["low_confidence_frame_pct"],
    )

    save_poses(pose_df, Path(args.output))

    print(f"\nFrames processed : {len(pose_df)}")
    print(f"Poses detected   : {pose_df['nose_x'].notna().sum()} "
          f"({100 * pose_df['nose_x'].notna().mean():.1f}%)")
    print(f"Quality check    : {'PASS' if quality['pass'] else 'FAIL'}")
    print(f"Worst joint      : {quality['worst_joint']} "
          f"({100 * quality['low_confidence_pct']:.1f}% low-confidence frames)")
    print(f"Output saved to  : {args.output}")
