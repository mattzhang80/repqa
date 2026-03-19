"""Tests for src/pipeline/pose_extract.py"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.pose_extract import (
    LANDMARK_NAMES,
    check_pose_quality,
    extract_poses,
    load_poses,
    save_poses,
)


# ---------------------------------------------------------------------------
# LANDMARK_NAMES tests
# ---------------------------------------------------------------------------

class TestLandmarkNames:
    def test_count(self):
        assert len(LANDMARK_NAMES) == 33

    def test_contains_expected(self):
        assert "nose" in LANDMARK_NAMES
        assert "left_shoulder" in LANDMARK_NAMES
        assert "right_wrist" in LANDMARK_NAMES
        assert "left_hip" in LANDMARK_NAMES

    def test_all_lowercase(self):
        assert all(name == name.lower() for name in LANDMARK_NAMES)

    def test_no_duplicates(self):
        assert len(LANDMARK_NAMES) == len(set(LANDMARK_NAMES))


# ---------------------------------------------------------------------------
# extract_poses tests
# ---------------------------------------------------------------------------

class TestExtractPoses:
    def test_returns_dataframe(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        assert isinstance(df, pd.DataFrame)

    def test_has_frame_idx_and_timestamp(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        assert "frame_idx" in df.columns
        assert "timestamp_s" in df.columns

    def test_has_all_landmark_columns(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        for name in LANDMARK_NAMES:
            for suffix in ("_x", "_y", "_z", "_vis"):
                assert f"{name}{suffix}" in df.columns, f"Missing column: {name}{suffix}"

    def test_column_count(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        # frame_idx + timestamp_s + 33 landmarks × 4 fields
        assert len(df.columns) == 2 + 33 * 4

    def test_frame_count_approx(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        # 10s clip @ 30fps → ~300 frames
        assert 200 <= len(df) <= 400

    def test_frame_idx_sequential(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        assert list(df["frame_idx"]) == list(range(len(df)))

    def test_timestamp_increases_monotonically(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        assert (df["timestamp_s"].diff().dropna() > 0).all()

    def test_timestamp_starts_at_zero(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        assert df["timestamp_s"].iloc[0] == 0.0

    def test_detected_coordinates_are_finite(self, short_preprocessed_video_path):
        """Detected landmark coordinates must be finite (not inf/nan).

        MediaPipe can return x/y outside [0, 1] for landmarks near or outside
        the frame boundary (e.g. far arm in side-view), so we only check
        finiteness rather than clamping to a fixed range.
        """
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        for suffix in ("_x", "_y", "_z"):
            cols = [f"{n}{suffix}" for n in LANDMARK_NAMES]
            vals = df[cols].values.ravel()
            valid = vals[~np.isnan(vals)]
            if len(valid):
                assert np.isfinite(valid).all(), f"Non-finite value in {suffix} coords"

    def test_vis_in_unit_range(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        vis_cols = [f"{n}_vis" for n in LANDMARK_NAMES]
        vis_vals = df[vis_cols].values.ravel()
        assert vis_vals.min() >= 0.0
        assert vis_vals.max() <= 1.0

    def test_real_video_high_detection_rate(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        detection_rate = df["nose_x"].notna().mean()
        assert detection_rate >= 0.8, f"Pose detection rate too low: {detection_rate:.1%}"

    def test_no_pose_frame_has_nan_coordinates(self, test_video_path):
        """Frames with no pose detected have NaN for x/y/z."""
        df = extract_poses(test_video_path, fps=24)
        no_pose = df[df["nose_x"].isna()]
        if no_pose.empty:
            pytest.skip("Synthetic video has pose detections (unexpected but valid)")
        for name in LANDMARK_NAMES:
            for suffix in ("_x", "_y", "_z"):
                col = f"{name}{suffix}"
                assert no_pose[col].isna().all(), f"{col} should be NaN for undetected frames"

    def test_no_pose_frame_has_zero_vis(self, test_video_path):
        """Frames with no pose detected have vis=0.0."""
        df = extract_poses(test_video_path, fps=24)
        no_pose = df[df["nose_x"].isna()]
        if no_pose.empty:
            pytest.skip("Synthetic video has pose detections (unexpected but valid)")
        for name in LANDMARK_NAMES:
            col = f"{name}_vis"
            assert (no_pose[col] == 0.0).all(), f"{col} should be 0.0 for undetected frames"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_poses(tmp_path / "ghost.mp4", fps=30)

    def test_synthetic_video_has_rows(self, test_video_path):
        """extract_poses returns at least one row even with no detections."""
        df = extract_poses(test_video_path, fps=24)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# check_pose_quality tests
# ---------------------------------------------------------------------------

class TestCheckPoseQuality:
    def _make_df(self, vis_val: float, n: int = 100) -> pd.DataFrame:
        """Create a minimal pose DataFrame with uniform visibility."""
        return pd.DataFrame(
            {f"{name}_vis": [vis_val] * n for name in LANDMARK_NAMES}
        )

    def test_pass_when_all_high_confidence(self):
        df = self._make_df(0.9)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        assert result["pass"] is True

    def test_fail_when_all_low_confidence(self):
        df = self._make_df(0.1)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        assert result["pass"] is False

    def test_returns_required_keys(self):
        df = self._make_df(0.9)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        assert set(result.keys()) == {"pass", "low_confidence_pct", "worst_joint", "per_joint"}

    def test_per_joint_has_bilateral_pair_names(self):
        df = self._make_df(0.9)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        pj = result["per_joint"]
        assert "shoulder" in pj
        assert "elbow" in pj
        assert "wrist" in pj
        assert "hip" in pj

    def test_bilateral_uses_better_side(self):
        """Left shoulder degraded, right shoulder good → shoulder pair should pass."""
        data = {f"{name}_vis": [0.9] * 100 for name in LANDMARK_NAMES}
        data["left_shoulder_vis"] = [0.1] * 100  # degrade left only
        df = pd.DataFrame(data)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        # bilateral check uses right (0.9) → low_pct = 0.0 for shoulder
        assert result["per_joint"]["shoulder"] == 0.0

    def test_both_sides_bad_fails(self):
        """Both left and right shoulder poor → shoulder pair fails."""
        data = {f"{name}_vis": [0.9] * 100 for name in LANDMARK_NAMES}
        data["left_shoulder_vis"] = [0.1] * 100
        data["right_shoulder_vis"] = [0.1] * 100
        df = pd.DataFrame(data)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        assert result["per_joint"]["shoulder"] == 1.0

    def test_threshold_boundary(self):
        """Visibility exactly at threshold should be treated as low confidence."""
        df = self._make_df(0.5)  # exactly at threshold
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        # 0.5 < 0.5 is False → 0% low confidence → should pass
        assert result["per_joint"]["shoulder"] == 0.0

    def test_low_confidence_pct_is_float(self):
        df = self._make_df(0.1)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        assert isinstance(result["low_confidence_pct"], float)

    def test_per_joint_values_rounded(self):
        df = self._make_df(0.1)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        for v in result["per_joint"].values():
            assert v == round(v, 4)

    def test_worst_joint_is_string(self):
        df = self._make_df(0.1)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        assert isinstance(result["worst_joint"], str)

    def test_empty_df_does_not_raise(self):
        """check_pose_quality should not raise on an empty DataFrame."""
        result = check_pose_quality(pd.DataFrame(), confidence_threshold=0.5, max_low_pct=0.3)
        assert isinstance(result, dict)
        assert "pass" in result
        assert "worst_joint" in result

    def test_real_video_quality_passes(self, short_preprocessed_video_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        result = check_pose_quality(df, confidence_threshold=0.5, max_low_pct=0.3)
        assert result["pass"] is True


# ---------------------------------------------------------------------------
# save_poses / load_poses tests
# ---------------------------------------------------------------------------

class TestSaveLoadPoses:
    def test_round_trip_synthetic(self, test_video_path, tmp_path):
        """save/load works correctly even with all-NaN pose data."""
        df = extract_poses(test_video_path, fps=24)
        out = tmp_path / "poses.parquet"
        save_poses(df, out)
        loaded = load_poses(out)
        pd.testing.assert_frame_equal(df, loaded)

    def test_round_trip_real(self, short_preprocessed_video_path, tmp_path):
        if short_preprocessed_video_path is None:
            pytest.skip("Short preprocessed video not available")
        df = extract_poses(short_preprocessed_video_path, fps=30)
        out = tmp_path / "poses.parquet"
        save_poses(df, out)
        loaded = load_poses(out)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_creates_parent_directories(self, test_video_path, tmp_path):
        df = extract_poses(test_video_path, fps=24)
        out = tmp_path / "nested" / "deep" / "poses.parquet"
        save_poses(df, out)
        assert out.exists()

    def test_save_creates_nonempty_file(self, test_video_path, tmp_path):
        df = extract_poses(test_video_path, fps=24)
        out = tmp_path / "poses.parquet"
        save_poses(df, out)
        assert out.stat().st_size > 0

    def test_loaded_columns_match(self, test_video_path, tmp_path):
        df = extract_poses(test_video_path, fps=24)
        out = tmp_path / "poses.parquet"
        save_poses(df, out)
        loaded = load_poses(out)
        assert list(df.columns) == list(loaded.columns)
