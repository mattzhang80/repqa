"""Tests for src/pipeline/preprocess.py and src/utils/video.py"""

import pytest

from src.pipeline.preprocess import preprocess_video
from src.utils.video import get_video_metadata


# ---------------------------------------------------------------------------
# get_video_metadata tests
# ---------------------------------------------------------------------------

class TestGetVideoMetadata:
    def test_returns_expected_keys(self, test_video_path):
        meta = get_video_metadata(test_video_path)
        expected = {
            "width", "height", "stored_width", "stored_height",
            "fps", "duration_s", "rotation", "codec", "nb_frames",
        }
        assert expected.issubset(meta.keys())

    def test_synthetic_video_dimensions(self, test_video_path):
        meta = get_video_metadata(test_video_path)
        assert meta["width"] == 1280
        assert meta["height"] == 720

    def test_synthetic_video_fps(self, test_video_path):
        meta = get_video_metadata(test_video_path)
        assert abs(meta["fps"] - 24.0) < 1.0  # within 1 fps

    def test_synthetic_video_duration(self, test_video_path):
        meta = get_video_metadata(test_video_path)
        assert abs(meta["duration_s"] - 3.0) < 0.5  # within 0.5s

    def test_no_rotation_on_synthetic(self, test_video_path):
        meta = get_video_metadata(test_video_path)
        assert meta["rotation"] == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_video_metadata(tmp_path / "nonexistent.mp4")

    def test_real_video_dimensions(self, real_video_path):
        """Phone-recorded video should have correct display dimensions."""
        if real_video_path is None:
            pytest.skip("Real video not available")
        meta = get_video_metadata(real_video_path)
        # Displayed dimensions: rotation is applied, so stored 1920x1080 with
        # -90 rotation → displayed 1080x1920 (portrait)
        assert meta["width"] > 0
        assert meta["height"] > 0
        assert meta["fps"] > 0
        assert meta["duration_s"] > 5.0  # real session is >5 seconds

    def test_real_video_rotation_detected(self, real_video_path):
        """Phone video should have a non-zero rotation tag."""
        if real_video_path is None:
            pytest.skip("Real video not available")
        meta = get_video_metadata(real_video_path)
        # iPhone portrait recording typically has ±90° rotation
        assert meta["rotation"] != 0


# ---------------------------------------------------------------------------
# preprocess_video tests
# ---------------------------------------------------------------------------

class TestPreprocessVideo:
    def test_output_file_created(self, test_video_path, tmp_path):
        out = tmp_path / "out.mp4"
        preprocess_video(test_video_path, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_is_valid_mp4(self, test_video_path, tmp_path):
        out = tmp_path / "out.mp4"
        preprocess_video(test_video_path, out)
        # ffprobe can read it without error
        meta = get_video_metadata(out)
        assert meta["codec"] == "h264"

    def test_output_width(self, test_video_path, tmp_path):
        out = tmp_path / "out.mp4"
        result = preprocess_video(test_video_path, out, width=720)
        assert result["actual_width"] == 720

    def test_output_height_even(self, test_video_path, tmp_path):
        """Height must be even (H.264 requirement)."""
        out = tmp_path / "out.mp4"
        result = preprocess_video(test_video_path, out, width=720)
        assert result["actual_height"] % 2 == 0

    def test_output_fps(self, test_video_path, tmp_path):
        out = tmp_path / "out.mp4"
        result = preprocess_video(test_video_path, out, fps=30)
        out_meta = get_video_metadata(out)
        assert abs(out_meta["fps"] - 30.0) < 1.0

    def test_returns_metadata_dict(self, test_video_path, tmp_path):
        out = tmp_path / "out.mp4"
        result = preprocess_video(test_video_path, out)
        required_keys = {
            "in_path", "out_path",
            "original_fps", "original_width", "original_height", "original_rotation",
            "target_fps", "target_width",
            "actual_width", "actual_height",
            "duration_s", "frame_count",
        }
        assert required_keys.issubset(result.keys())

    def test_synthetic_1280x720_to_720_scaled(self, test_video_path, tmp_path):
        """1280x720@24fps → width 720, height ≈405 rounded to nearest even (404 or 406)."""
        out = tmp_path / "out.mp4"
        result = preprocess_video(test_video_path, out, fps=30, width=720)
        assert result["actual_width"] == 720
        # scale=720:-2: exact height is 720*(720/1280)=405, rounded to nearest even
        assert abs(result["actual_height"] - 405) <= 2
        assert result["actual_height"] % 2 == 0

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess_video(tmp_path / "ghost.mov", tmp_path / "out.mp4")

    def test_creates_output_directory(self, test_video_path, tmp_path):
        """Output directory should be created if it doesn't exist."""
        out = tmp_path / "nested" / "deep" / "out.mp4"
        preprocess_video(test_video_path, out)
        assert out.exists()

    def test_overwrite_existing(self, test_video_path, tmp_path):
        """Calling twice on the same output path should not raise."""
        out = tmp_path / "out.mp4"
        preprocess_video(test_video_path, out)
        preprocess_video(test_video_path, out)  # overwrite — no error

    def test_real_video_preprocesses(self, real_video_path, tmp_path):
        """Real phone-recorded video should preprocess cleanly."""
        if real_video_path is None:
            pytest.skip("Real video not available")
        out = tmp_path / "real_out.mp4"
        result = preprocess_video(real_video_path, out, fps=30, width=720)
        assert out.exists()
        assert result["actual_width"] == 720
        assert result["actual_height"] % 2 == 0
        assert result["frame_count"] > 0
        # Display rotation should have been applied; rotation tag absent from output
        out_meta = get_video_metadata(out)
        assert out_meta["rotation"] == 0  # rotation baked in, no tag on output

    def test_real_video_correct_orientation(self, real_video_path, tmp_path):
        """Portrait phone video should produce taller-than-wide output."""
        if real_video_path is None:
            pytest.skip("Real video not available")
        out = tmp_path / "real_out.mp4"
        result = preprocess_video(real_video_path, out, fps=30, width=720)
        # Portrait orientation: height > width
        assert result["actual_height"] > result["actual_width"]
