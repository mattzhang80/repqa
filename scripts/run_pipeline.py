"""End-to-end pipeline: video → preprocessed → poses → reps → features → flags.

Run a single Wall Slide (or Band ER Side) video through all pipeline stages and
produce a structured session directory with every intermediate artifact.

Usage:
    python scripts/run_pipeline.py \\
        --video data/raw/matthew/wall_slide_good_01.mov \\
        --exercise wall_slide \\
        --user-id matthew

    python scripts/run_pipeline.py \\
        --video data/raw/matthew/wall_slide_good_01.mov \\
        --exercise wall_slide \\
        --user-id matthew \\
        --session-id 2026-03-05_wall_slide_good_01

Output layout:
    data/processed/<session_id>/
        meta.json
        video.mp4           (preprocessed)
        poses.parquet       (saved to data/poses/<session_id>/ too)
        reps.csv
        segmentation_plot.png
        features.csv
        flags.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ── make repo root importable when run directly ───────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.baseline import flag_reps_baseline, save_flags, summarize_flags
from src.pipeline.features import extract_rep_features, save_features
from src.pipeline.pose_extract import check_pose_quality, extract_poses, save_poses
from src.pipeline.preprocess import preprocess_video
from src.pipeline.rep_segment import (
    build_signal_wall_slide,
    plot_segmentation,
    save_reps_csv,
    segment_reps,
    smooth_signal,
)
from src.utils.config import get_config, get_exercise_config, get_section


def run_pipeline(
    video_path: Path,
    exercise: str,
    user_id: str,
    session_id: str | None = None,
) -> dict:
    """Run the full RepQA pipeline for one video.

    Steps:
        1. Validate exercise is in registry.
        2. Create session directory.
        3. Preprocess video (standardise fps / resolution).
        4. Extract poses → save Parquet.
        5. Segment reps → save reps.csv + debug plot.
        6. Extract features → save features.csv.
        7. Run baseline flagger → save flags.json.
        8. Write meta.json.

    Args:
        video_path: Path to raw input video (.mov / .mp4 / etc.).
        exercise:   Exercise identifier ('wall_slide' or 'band_er_side').
        user_id:    Identifier for the recording user.
        session_id: Optional session name.  Auto-generated from date + exercise
                    if not provided.

    Returns:
        Dict with keys:
            session_id, session_dir, all output paths, and summary stats.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ── Step 1: Validate exercise ──────────────────────────────────────────────
    ex_config = get_exercise_config(exercise)  # raises KeyError for unknown exercise
    cfg = get_config()

    # ── Step 2: Build session ID + directory ──────────────────────────────────
    if session_id is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        session_id = f"{date_str}_{exercise}_{video_path.stem}"

    session_dir = Path("data/processed") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    poses_dir = Path("data/poses") / session_id
    poses_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path("data/features") / session_id
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Session: {session_id}")
    print(f"      Dir   : {session_dir}")

    # ── Step 3: Preprocess video ───────────────────────────────────────────────
    video_out = session_dir / "video.mp4"
    fps = cfg["video"]["fps"]
    width = cfg["video"]["width"]
    print(f"[2/7] Preprocessing video → {video_out}")
    video_meta = preprocess_video(video_path, video_out, fps=fps, width=width)
    print(f"      Duration : {video_meta['duration_s']:.1f}s  "
          f"Frames: {video_meta['frame_count']}")

    # ── Step 4: Pose extraction ────────────────────────────────────────────────
    poses_out = poses_dir / "poses.parquet"
    print(f"[3/7] Extracting poses → {poses_out}")
    pose_df = extract_poses(video_out, fps=fps)

    # Quality check
    pose_cfg = cfg["pose"]
    quality = check_pose_quality(
        pose_df,
        confidence_threshold=pose_cfg["confidence_threshold"],
        max_low_pct=pose_cfg["low_confidence_frame_pct"],
    )
    if not quality["pass"]:
        print(f"      WARNING: Pose quality check FAILED — "
              f"{quality['low_confidence_pct']:.1%} low-confidence frames "
              f"(worst joint: {quality['worst_joint']})")
    else:
        print(f"      Pose quality: OK  "
              f"(low-conf frames: {quality['low_confidence_pct']:.1%})")
    save_poses(pose_df, poses_out)
    # Also copy parquet into session dir for convenience
    import shutil
    shutil.copy(poses_out, session_dir / "poses.parquet")

    # ── Step 5: Rep segmentation ───────────────────────────────────────────────
    reps_out = session_dir / "reps.csv"
    plot_out = session_dir / "segmentation_plot.png"
    print(f"[4/7] Segmenting reps → {reps_out}")
    reps = segment_reps(pose_df, exercise, fps=fps)
    save_reps_csv(reps, reps_out)
    print(f"      Detected {len(reps)} rep(s)")

    # Debug plot
    seg_cfg = get_section("segmentation")[exercise]
    if exercise == "wall_slide":
        raw_signal = build_signal_wall_slide(pose_df)
    else:
        raise NotImplementedError("band_er_side signal not yet implemented (Phase 7)")
    smoothed = smooth_signal(
        raw_signal, seg_cfg["smoothing_window"], seg_cfg["smoothing_polyorder"]
    )
    plot_segmentation(smoothed, reps, fps=fps,
                      title=f"{exercise} — {session_id}",
                      save_path=plot_out)

    # ── Step 6: Feature extraction ────────────────────────────────────────────
    features_out = features_dir / "features.csv"
    session_features_out = session_dir / "features.csv"
    print(f"[5/7] Extracting features → {features_out}")
    session_meta = {"session_id": session_id, "user_id": user_id}
    features_df = extract_rep_features(pose_df, reps, exercise, fps, session_meta)
    save_features(features_df, features_out)
    save_features(features_df, session_features_out)

    # ── Step 7: Baseline flagging ─────────────────────────────────────────────
    flags_out = session_dir / "flags.json"
    print(f"[6/7] Flagging reps → {flags_out}")
    flags = flag_reps_baseline(features_df, exercise)
    save_flags(flags, flags_out)
    summary = summarize_flags(flags)
    print(f"      Flagged: {summary['flagged_count']}/{summary['total_reps']}  "
          f"labels={summary['label_distribution']}")

    # ── Step 8: Write meta.json ────────────────────────────────────────────────
    meta = {
        "session_id": session_id,
        "user_id": user_id,
        "exercise": exercise,
        "display_name": ex_config["display_name"],
        "filming_angle": ex_config["filming_angle"],
        "fps": fps,
        "video_path": str(video_out),
        "poses_path": str(poses_out),
        "duration_s": video_meta["duration_s"],
        "frame_count": video_meta["frame_count"],
        "reps_detected": len(reps),
        "pose_quality": quality,
        "safety_note": cfg["report"]["safety_note"],
    }
    meta_out = session_dir / "meta.json"
    with open(meta_out, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[7/7] Metadata → {meta_out}")

    result = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "video": str(video_out),
        "poses": str(poses_out),
        "reps_csv": str(reps_out),
        "segmentation_plot": str(plot_out),
        "features_csv": str(features_out),
        "flags_json": str(flags_out),
        "meta_json": str(meta_out),
        "summary": summary,
        "pose_quality": quality,
    }
    print("\nDone.")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RepQA end-to-end pipeline: video → features + flags."
    )
    parser.add_argument("--video", required=True, help="Path to raw input video")
    parser.add_argument(
        "--exercise", required=True, choices=["wall_slide", "band_er_side"],
        help="Exercise type"
    )
    parser.add_argument("--user-id", required=True, help="User identifier")
    parser.add_argument(
        "--session-id", default=None,
        help="Session identifier (auto-generated if omitted)"
    )
    args = parser.parse_args()

    result = run_pipeline(
        video_path=Path(args.video),
        exercise=args.exercise,
        user_id=args.user_id,
        session_id=args.session_id,
    )

    print("\n=== Output files ===")
    for key, val in result.items():
        if key not in ("summary", "pose_quality"):
            print(f"  {key:25s}: {val}")
    print(f"\n=== Summary ===")
    for key, val in result["summary"].items():
        print(f"  {key}: {val}")
