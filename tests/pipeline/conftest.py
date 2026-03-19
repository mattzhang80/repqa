"""Shared fixtures for pipeline tests."""

import subprocess
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
REAL_VIDEO_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "matthew"


@pytest.fixture(scope="session")
def test_video_path() -> Path:
    """Generate a synthetic 1280x720@24fps test video if it doesn't exist.

    Uses FFmpeg's testsrc to create a deterministic 3-second test clip.
    Requires FFmpeg to be installed.
    """
    path = FIXTURES_DIR / "test_input.mp4"
    if not path.exists():
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        from src.utils.video import find_binary
        subprocess.run(
            [
                find_binary("ffmpeg"),
                "-f", "lavfi",
                "-i", "testsrc=duration=3:size=1280x720:rate=24",
                "-c:v", "libx264",
                "-y", str(path),
            ],
            check=True,
            capture_output=True,
        )
    return path


@pytest.fixture(scope="session")
def real_video_path() -> Path | None:
    """Return path to a real good wall-slide video if it exists, else None."""
    candidate = REAL_VIDEO_DIR / "wall_slide_good_01.mov"
    return candidate if candidate.exists() else None


@pytest.fixture(scope="session")
def short_preprocessed_video_path() -> Path | None:
    """Return a ~10s preprocessed clip for pose tests.

    Trims the first 10 seconds of wall_slide_good_01.mov and preprocesses it
    (rotation correction, scale to 720px wide, 30fps). Cached to fixtures dir.
    Returns None if the raw video is not available.
    """
    raw = REAL_VIDEO_DIR / "wall_slide_good_01.mov"
    if not raw.exists():
        return None

    out = FIXTURES_DIR / "wall_slide_short_preprocessed.mp4"
    if out.exists():
        return out

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    import ffmpeg as ffmpeg_lib  # noqa: F401 (imported for side-effects)
    from src.utils.video import find_binary

    # FFmpeg auto-applies rotation; scale + trim to 10s
    (
        ffmpeg_lib
        .input(str(raw), t=10)
        .output(
            str(out),
            vf="scale=720:-2",
            r=30,
            vcodec="libx264",
            crf=23,
            preset="fast",
            an=None,
        )
        .overwrite_output()
        .run(cmd=find_binary("ffmpeg"), quiet=True)
    )

    return out


@pytest.fixture(scope="session")
def short_pose_df(short_preprocessed_video_path):
    """Extract and cache poses from the short preprocessed video.

    Shared by both pose and segmentation tests to avoid re-running the
    ~80-second MediaPipe extraction more than once per test session.
    Returns None if the short preprocessed video is not available.
    """
    if short_preprocessed_video_path is None:
        return None
    from src.pipeline.pose_extract import extract_poses
    return extract_poses(short_preprocessed_video_path, fps=30)
