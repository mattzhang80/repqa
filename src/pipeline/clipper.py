"""Clip extraction and thumbnail generation for flagged reps.

Extracts short video clips and a thumbnail frame for each flagged rep using
FFmpeg.  Clips are padded slightly before and after the rep boundaries so the
reviewer sees context.

Usage:
    python src/pipeline/clipper.py \\
        --video   data/processed/<session>/video.mp4 \\
        --reps    data/processed/<session>/reps.csv \\
        --flags   data/processed/<session>/flags.json \\
        --session-dir data/processed/<session>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ffmpeg

from src.pipeline.baseline import load_flags
from src.pipeline.rep_segment import Rep


def extract_clip(
    video_path: Path,
    start_s: float,
    end_s: float,
    padding_s: float,
    out_path: Path,
) -> Path:
    """Extract a padded clip from a video file.

    Args:
        video_path: Source preprocessed video.
        start_s:    Rep start time in seconds.
        end_s:      Rep end time in seconds.
        padding_s:  Seconds to add before and after the rep.
        out_path:   Destination .mp4 path.

    Returns:
        Path to the written clip.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = max(0.0, start_s - padding_s)
    duration = (end_s + padding_s) - t_start

    (
        ffmpeg
        .input(str(video_path), ss=t_start, t=duration)
        .output(
            str(out_path),
            vcodec="libx264",
            crf=23,
            preset="fast",
            an=None,
            y=None,
        )
        .run(quiet=True, overwrite_output=True)
    )
    return out_path


def extract_thumbnail(
    video_path: Path,
    timestamp_s: float,
    out_path: Path,
) -> Path:
    """Extract a single frame as a JPEG thumbnail.

    Args:
        video_path:   Source video.
        timestamp_s:  Seek to this timestamp (seconds).
        out_path:     Destination .jpg path.

    Returns:
        Path to the written thumbnail.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg
        .input(str(video_path), ss=timestamp_s)
        .output(
            str(out_path),
            vframes=1,
            format="image2",
            vcodec="mjpeg",
            y=None,
        )
        .run(quiet=True, overwrite_output=True)
    )
    return out_path


def clip_flagged_reps(
    video_path: Path,
    reps: list[Rep],
    flags: list,
    session_dir: Path,
    padding_s: float = 0.3,
) -> list[dict]:
    """Extract clips and thumbnails for all flagged reps.

    Only reps where RepFlag.flagged == True are processed.  The clip and
    thumbnail for each flagged rep are saved under session_dir/clips/.

    Args:
        video_path:  Preprocessed session video.
        reps:        Rep dataclasses from segment_reps().
        flags:       RepFlag list from flag_reps_baseline().
        session_dir: Session directory; clips saved to session_dir/clips/.
        padding_s:   Padding added before/after each clip (seconds).

    Returns:
        List of dicts, one per flagged rep:
            {rep_id, clip_path, thumbnail_path, predicted_label, reasons,
             start_s, end_s, duration_s}
    """
    video_path = Path(video_path)
    clips_dir = Path(session_dir) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rep_by_id = {r.rep_id: r for r in reps}
    results = []

    for flag in flags:
        if not flag.flagged:
            continue

        rep = rep_by_id.get(flag.rep_id)
        if rep is None:
            continue

        clip_path = clips_dir / f"rep_{flag.rep_id:02d}.mp4"
        thumb_path = clips_dir / f"rep_{flag.rep_id:02d}_thumb.jpg"
        mid_s = (rep.start_time_s + rep.end_time_s) / 2.0

        extract_clip(video_path, rep.start_time_s, rep.end_time_s, padding_s, clip_path)
        extract_thumbnail(video_path, mid_s, thumb_path)

        results.append(
            {
                "rep_id": flag.rep_id,
                "clip_path": str(clip_path),
                "thumbnail_path": str(thumb_path),
                "predicted_label": flag.predicted_label,
                "reasons": flag.reasons,
                "start_s": rep.start_time_s,
                "end_s": rep.end_time_s,
                "duration_s": rep.end_time_s - rep.start_time_s,
            }
        )

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clips + thumbnails for flagged reps."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--reps", required=True, help="Path to reps.csv")
    parser.add_argument("--flags", required=True, help="Path to flags.json")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--padding", type=float, default=0.3)
    args = parser.parse_args()

    import pandas as pd
    from src.pipeline.rep_segment import Rep

    reps_df = pd.read_csv(args.reps)
    reps = [
        Rep(int(r["rep_id"]), int(r["start_frame"]), int(r["end_frame"]),
            float(r["start_time_s"]), float(r["end_time_s"]))
        for _, r in reps_df.iterrows()
    ]
    flags = load_flags(Path(args.flags))
    clips = clip_flagged_reps(
        Path(args.video), reps, flags,
        Path(args.session_dir), padding_s=args.padding,
    )
    print(f"Extracted {len(clips)} clip(s):")
    for c in clips:
        print(f"  rep {c['rep_id']:2d}  {c['predicted_label']:20s}  {c['clip_path']}")
