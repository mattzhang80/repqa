"""Video preprocessing: re-encode to standard fps, resolution, and codec.

FFmpeg 4+ automatically applies rotation metadata during encoding, so
the output is always in the correct display orientation regardless of
how the phone stored the rotation tag.
"""

import argparse
from pathlib import Path

import ffmpeg

from src.utils.video import find_binary, get_video_metadata


def preprocess_video(
    in_path: Path,
    out_path: Path,
    fps: int = 30,
    width: int = 720,
) -> dict:
    """Re-encode a video to standard format for the RepQA pipeline.

    Applies rotation metadata (phones often store portrait video with a rotation
    tag), scales to target width while preserving aspect ratio, forces target fps,
    strips audio, and outputs H.264 mp4.

    Args:
        in_path:  Path to raw input video (.mov, .mp4, etc.)
        out_path: Path for preprocessed output (.mp4)
        fps:      Target frames per second (default 30)
        width:    Target width in pixels (default 720); height scales proportionally
                  and is rounded down to the nearest even number (-2 filter).

    Returns:
        dict with keys:
            in_path, out_path
            original_fps, original_width, original_height, original_rotation
            target_fps, target_width
            actual_width, actual_height    (output dimensions after scale)
            duration_s, frame_count        (based on input duration × target fps)

    Raises:
        FileNotFoundError: if in_path does not exist.
        ffmpeg.Error:      if encoding fails (stderr included in exception).
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = get_video_metadata(in_path)

    ffmpeg_bin = find_binary("ffmpeg")

    # FFmpeg 4+ auto-applies rotation during encode when a custom vf is used,
    # so scale=width:-2 operates on the correctly-oriented frame.
    (
        ffmpeg
        .input(str(in_path))
        .output(
            str(out_path),
            vf=f"scale={width}:-2",
            r=fps,
            vcodec="libx264",
            crf=23,
            preset="fast",
            an=None,           # strip audio
        )
        .overwrite_output()
        .run(cmd=ffmpeg_bin, quiet=True)
    )

    # Probe output to get the actual encoded dimensions
    out_meta = get_video_metadata(out_path)

    return {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "original_fps": meta["fps"],
        "original_width": meta["width"],
        "original_height": meta["height"],
        "original_rotation": meta["rotation"],
        "target_fps": fps,
        "target_width": width,
        "actual_width": out_meta["width"],
        "actual_height": out_meta["height"],
        "duration_s": meta["duration_s"],
        "frame_count": round(meta["duration_s"] * fps),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a video to standard format for RepQA."
    )
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path for output .mp4")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default 30)")
    parser.add_argument("--width", type=int, default=720, help="Target width (default 720)")
    args = parser.parse_args()

    result = preprocess_video(
        in_path=Path(args.input),
        out_path=Path(args.output),
        fps=args.fps,
        width=args.width,
    )

    print(f"Input:    {result['in_path']}")
    print(f"Output:   {result['out_path']}")
    print(f"Original: {result['original_width']}x{result['original_height']} "
          f"@ {result['original_fps']:.2f}fps  rotation={result['original_rotation']}°")
    print(f"Output:   {result['actual_width']}x{result['actual_height']} "
          f"@ {result['target_fps']}fps")
    print(f"Duration: {result['duration_s']:.2f}s  (~{result['frame_count']} frames)")
