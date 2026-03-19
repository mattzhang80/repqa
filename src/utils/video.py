"""Video utility functions: probing metadata and locating FFmpeg binaries."""

import shutil
from pathlib import Path

import ffmpeg


def find_binary(name: str) -> str:
    """Find an FFmpeg-suite binary (ffmpeg or ffprobe), checking PATH and common macOS locations."""
    path = shutil.which(name)
    if path:
        return path
    for candidate in [f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"]:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        f"'{name}' binary not found. Install with: brew install ffmpeg"
    )


def _parse_fps(fps_str: str) -> float:
    """Parse a fractional FPS string like '30/1' or '24000/1001' to float."""
    if not fps_str or fps_str == "0/0":
        return 0.0
    parts = fps_str.split("/")
    if len(parts) == 2:
        num, den = int(parts[0]), int(parts[1])
        return num / den if den != 0 else 0.0
    return float(fps_str)


def _get_rotation(video_stream: dict) -> int:
    """Extract rotation angle (0, 90, -90, 180, 270) from stream side_data_list."""
    for sd in video_stream.get("side_data_list", []):
        if "rotation" in sd:
            return int(sd["rotation"])
    # Fallback: older ffmpeg stores rotation in tags
    tags = video_stream.get("tags", {})
    if "rotate" in tags:
        return int(tags["rotate"])
    return 0


def get_video_metadata(path: Path) -> dict:
    """Probe a video file and return a metadata dict.

    Returns keys:
        width, height          — displayed dimensions (rotation applied)
        stored_width/height    — raw stream dimensions (before rotation)
        fps                    — frames per second (float)
        duration_s             — duration in seconds
        rotation               — rotation degrees from metadata (0 if none)
        codec                  — video codec name
        nb_frames              — estimated frame count (duration * fps)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    probe = ffmpeg.probe(str(path), cmd=find_binary("ffprobe"))
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in: {path}")

    stored_w = video_stream["width"]
    stored_h = video_stream["height"]
    rotation = _get_rotation(video_stream)

    # Displayed dimensions swap when rotation is ±90°
    if abs(rotation) in (90, 270):
        display_w, display_h = stored_h, stored_w
    else:
        display_w, display_h = stored_w, stored_h

    # Prefer avg_frame_rate for variable-rate footage; fall back to r_frame_rate
    fps = _parse_fps(video_stream.get("avg_frame_rate", "0/0"))
    if fps < 1.0:
        fps = _parse_fps(video_stream.get("r_frame_rate", "30/1"))

    duration_s = float(probe["format"].get("duration", 0))
    nb_frames = round(duration_s * fps) if fps > 0 else 0

    return {
        "width": display_w,
        "height": display_h,
        "stored_width": stored_w,
        "stored_height": stored_h,
        "fps": fps,
        "duration_s": duration_s,
        "rotation": rotation,
        "codec": video_stream.get("codec_name", "unknown"),
        "nb_frames": nb_frames,
    }
