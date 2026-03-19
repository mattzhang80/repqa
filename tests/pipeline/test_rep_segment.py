"""Tests for src/pipeline/rep_segment.py"""

import matplotlib
matplotlib.use("Agg")  # headless backend — must precede any pyplot import

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.rep_segment import (
    Rep,
    build_signal_wall_slide,
    find_rep_boundaries,
    plot_segmentation,
    save_reps_csv,
    segment_reps,
    select_signal_arm,
    smooth_signal,
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _make_wall_slide_df(
    n_frames: int,
    wrist_y: np.ndarray | float,
    shoulder_y: float = 0.5,
    hip_y: float = 0.7,
    left_vis: float = 0.9,
    right_vis: float = 0.1,
) -> pd.DataFrame:
    """Minimal pose DataFrame for wall-slide signal tests."""
    if np.isscalar(wrist_y):
        wrist_y_arr = np.full(n_frames, float(wrist_y))
    else:
        wrist_y_arr = np.asarray(wrist_y, dtype=float)
    return pd.DataFrame({
        "left_wrist_y":    wrist_y_arr,
        "left_wrist_vis":  np.full(n_frames, left_vis),
        "right_wrist_y":   wrist_y_arr,
        "right_wrist_vis": np.full(n_frames, right_vis),
        "left_shoulder_y": np.full(n_frames, shoulder_y),
        "left_hip_y":      np.full(n_frames, hip_y),
        "right_shoulder_y": np.full(n_frames, shoulder_y),
        "right_hip_y":     np.full(n_frames, hip_y),
    })


def _sinusoidal_signal(n_reps: int = 5, rep_duration_s: float = 5.0, fps: int = 30) -> np.ndarray:
    """Return a clean sinusoidal signal with exactly n_reps peaks.

    Uses (1 - cos(t))/2 with endpoint=False so each period [0, 2π)
    produces exactly one trough→peak→trough cycle.
    Amplitude: 0 at troughs, 1 at peaks.
    """
    n_frames = int(n_reps * rep_duration_s * fps)
    t = np.linspace(0, 2 * np.pi * n_reps, n_frames, endpoint=False)
    return (1 - np.cos(t)) / 2


def _sinusoidal_pose_df(
    n_reps: int = 5,
    rep_duration_s: float = 5.0,
    fps: int = 30,
) -> pd.DataFrame:
    """Wall-slide pose DataFrame whose wrist_y traces a sinusoidal rep pattern.

    wrist_y oscillates so that the *signal* (shoulder_y - wrist_y) / torso_height
    oscillates between 0 and 1 with n_reps complete cycles.
    torso_height = hip_y - shoulder_y = 0.7 - 0.5 = 0.2.
    wrist_y = shoulder_y - signal * torso_height = 0.5 - signal * 0.2.
    """
    signal = _sinusoidal_signal(n_reps, rep_duration_s, fps)
    shoulder_y, hip_y = 0.5, 0.7
    torso_height = hip_y - shoulder_y                    # 0.2
    wrist_y = shoulder_y - signal * torso_height         # oscillates 0.3 ↔ 0.5
    n_frames = len(wrist_y)
    return _make_wall_slide_df(n_frames, wrist_y, shoulder_y=shoulder_y, hip_y=hip_y)


# ---------------------------------------------------------------------------
# TestSelectSignalArm
# ---------------------------------------------------------------------------

class TestSelectSignalArm:
    def test_picks_left_when_higher_visibility(self):
        df = _make_wall_slide_df(50, 0.4, left_vis=0.9, right_vis=0.1)
        assert select_signal_arm(df) == "left"

    def test_picks_right_when_higher_visibility(self):
        df = _make_wall_slide_df(50, 0.4, left_vis=0.1, right_vis=0.9)
        assert select_signal_arm(df) == "right"

    def test_ties_default_to_left(self):
        df = _make_wall_slide_df(50, 0.4, left_vis=0.8, right_vis=0.8)
        assert select_signal_arm(df) == "left"

    def test_missing_right_returns_left(self):
        df = pd.DataFrame({"left_wrist_vis": [0.9] * 10})
        assert select_signal_arm(df) == "left"


# ---------------------------------------------------------------------------
# TestBuildSignalWallSlide
# ---------------------------------------------------------------------------

class TestBuildSignalWallSlide:
    def test_arms_raised_gives_positive_signal(self):
        """Wrist above shoulder (wrist_y < shoulder_y) → positive signal."""
        df = _make_wall_slide_df(50, wrist_y=0.3, shoulder_y=0.5, hip_y=0.7)
        sig = build_signal_wall_slide(df)
        assert sig.mean() > 0

    def test_arms_at_rest_gives_near_zero_signal(self):
        """Wrist at shoulder level → signal ≈ 0."""
        df = _make_wall_slide_df(50, wrist_y=0.5, shoulder_y=0.5, hip_y=0.7)
        sig = build_signal_wall_slide(df)
        assert np.abs(sig.mean()) < 0.05

    def test_returns_array_of_correct_length(self):
        df = _make_wall_slide_df(100, 0.4)
        sig = build_signal_wall_slide(df)
        assert len(sig) == 100

    def test_nan_frames_are_interpolated(self):
        """NaN wrist values should be interpolated, not left as NaN."""
        wrist_y = np.full(50, 0.4)
        wrist_y[10:15] = np.nan                 # inject 5 NaN frames
        df = _make_wall_slide_df(50, wrist_y)
        sig = build_signal_wall_slide(df)
        assert np.isfinite(sig).all()

    def test_signal_tracks_wrist_motion(self):
        """Signal amplitude matches expected normalised wrist travel."""
        pose_df = _sinusoidal_pose_df(n_reps=3, rep_duration_s=5.0, fps=30)
        sig = build_signal_wall_slide(pose_df)
        # Signal should oscillate between ~0 and ~1
        assert sig.max() > 0.8
        assert sig.min() < 0.2


# ---------------------------------------------------------------------------
# TestSmoothSignal
# ---------------------------------------------------------------------------

class TestSmoothSignal:
    def test_reduces_variance(self):
        rng = np.random.default_rng(42)
        noisy = np.sin(np.linspace(0, 4 * np.pi, 300)) + rng.normal(0, 0.3, 300)
        smoothed = smooth_signal(noisy, window=11, polyorder=2)
        assert smoothed.var() < noisy.var()

    def test_length_preserved(self):
        sig = np.random.default_rng(0).random(200)
        assert len(smooth_signal(sig)) == 200

    def test_short_signal_no_crash(self):
        """Signal shorter than window should not raise."""
        sig = np.array([0.1, 0.5, 0.3])
        result = smooth_signal(sig, window=7)
        assert len(result) == 3

    def test_constant_signal_unchanged(self):
        sig = np.ones(100)
        smoothed = smooth_signal(sig)
        np.testing.assert_allclose(smoothed, sig, atol=1e-10)


# ---------------------------------------------------------------------------
# TestFindRepBoundaries
# ---------------------------------------------------------------------------

class TestFindRepBoundaries:
    _FPS = 30
    _BOUNDS = (3.0, 8.0)

    def test_detects_correct_rep_count(self):
        signal = _sinusoidal_signal(n_reps=5, rep_duration_s=5.0, fps=self._FPS)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        assert len(reps) == 5

    def test_rep_durations_within_bounds(self):
        signal = _sinusoidal_signal(n_reps=5, rep_duration_s=5.0, fps=self._FPS)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        for rep in reps:
            dur = rep.end_time_s - rep.start_time_s
            assert self._BOUNDS[0] <= dur <= self._BOUNDS[1], (
                f"Rep {rep.rep_id} duration {dur:.2f}s outside {self._BOUNDS}"
            )

    def test_rep_ids_are_sequential_from_zero(self):
        signal = _sinusoidal_signal(n_reps=3, rep_duration_s=5.0, fps=self._FPS)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        assert [r.rep_id for r in reps] == list(range(len(reps)))

    def test_reps_do_not_overlap(self):
        signal = _sinusoidal_signal(n_reps=5, rep_duration_s=5.0, fps=self._FPS)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        for i in range(len(reps) - 1):
            assert reps[i].end_frame <= reps[i + 1].start_frame

    def test_flat_signal_returns_no_reps(self):
        signal = np.zeros(900)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        assert reps == []

    def test_too_slow_reps_filtered_out(self):
        """Reps slower than max duration should be dropped."""
        # 5 reps × 9s each = 45s — above the 8.0s maximum
        signal = _sinusoidal_signal(n_reps=5, rep_duration_s=9.0, fps=self._FPS)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        assert len(reps) == 0

    def test_start_end_frames_are_integers(self):
        signal = _sinusoidal_signal(n_reps=3, rep_duration_s=5.0, fps=self._FPS)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        for rep in reps:
            assert isinstance(rep.start_frame, int)
            assert isinstance(rep.end_frame, int)

    def test_timestamps_match_frames(self):
        signal = _sinusoidal_signal(n_reps=3, rep_duration_s=5.0, fps=self._FPS)
        reps = find_rep_boundaries(signal, self._FPS, 3.0, self._BOUNDS)
        for rep in reps:
            assert abs(rep.start_time_s - rep.start_frame / self._FPS) < 0.01
            assert abs(rep.end_time_s - rep.end_frame / self._FPS) < 0.01


# ---------------------------------------------------------------------------
# TestSegmentReps
# ---------------------------------------------------------------------------

class TestSegmentReps:
    def test_wall_slide_returns_list(self):
        pose_df = _sinusoidal_pose_df(n_reps=5)
        reps = segment_reps(pose_df, "wall_slide", fps=30)
        assert isinstance(reps, list)

    def test_wall_slide_detects_reps_on_synthetic(self):
        pose_df = _sinusoidal_pose_df(n_reps=5)
        reps = segment_reps(pose_df, "wall_slide", fps=30)
        assert len(reps) == 5

    def test_unknown_exercise_raises_value_error(self):
        pose_df = _sinusoidal_pose_df(n_reps=2)
        with pytest.raises(ValueError, match="Unknown exercise"):
            segment_reps(pose_df, "squat", fps=30)

    def test_band_er_side_raises_not_implemented(self):
        pose_df = _sinusoidal_pose_df(n_reps=2)
        with pytest.raises(NotImplementedError):
            segment_reps(pose_df, "band_er_side", fps=30)


# ---------------------------------------------------------------------------
# TestPlotSegmentation
# ---------------------------------------------------------------------------

class TestPlotSegmentation:
    def test_saves_png_file(self, tmp_path):
        signal = _sinusoidal_signal(n_reps=3)
        reps = [
            Rep(0, 0, 150, 0.0, 5.0),
            Rep(1, 150, 300, 5.0, 10.0),
            Rep(2, 300, 450, 10.0, 15.0),
        ]
        out = tmp_path / "seg.png"
        plot_segmentation(signal, reps, fps=30, title="Test", save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_reps_no_crash(self, tmp_path):
        signal = np.zeros(300)
        out = tmp_path / "empty.png"
        plot_segmentation(signal, [], fps=30, title="No reps", save_path=out)
        assert out.exists()

    def test_creates_parent_directories(self, tmp_path):
        signal = _sinusoidal_signal(n_reps=2)
        out = tmp_path / "nested" / "dir" / "seg.png"
        plot_segmentation(signal, [], fps=30, title="X", save_path=out)
        assert out.exists()

    def test_no_save_path_no_crash(self):
        signal = _sinusoidal_signal(n_reps=2)
        # Should not raise even with no save_path and show=False
        plot_segmentation(signal, [], fps=30, title="X", save_path=None, show=False)


# ---------------------------------------------------------------------------
# TestSaveRepsCsv
# ---------------------------------------------------------------------------

class TestSaveRepsCsv:
    def _make_reps(self) -> list[Rep]:
        return [
            Rep(0, 0, 150, 0.0, 5.0),
            Rep(1, 150, 300, 5.0, 10.0),
        ]

    def test_creates_csv_file(self, tmp_path):
        out = tmp_path / "reps.csv"
        save_reps_csv(self._make_reps(), out)
        assert out.exists()

    def test_csv_schema(self, tmp_path):
        out = tmp_path / "reps.csv"
        save_reps_csv(self._make_reps(), out)
        df = pd.read_csv(out)
        assert set(df.columns) == {"rep_id", "start_frame", "end_frame",
                                   "start_time_s", "end_time_s"}

    def test_csv_row_count_matches(self, tmp_path):
        reps = self._make_reps()
        out = tmp_path / "reps.csv"
        save_reps_csv(reps, out)
        df = pd.read_csv(out)
        assert len(df) == len(reps)

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "sub" / "reps.csv"
        save_reps_csv(self._make_reps(), out)
        assert out.exists()

    def test_empty_reps_creates_empty_csv(self, tmp_path):
        out = tmp_path / "reps.csv"
        save_reps_csv([], out)
        df = pd.read_csv(out)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Integration tests (require real pose data)
# ---------------------------------------------------------------------------

class TestIntegrationWallSlide:
    def test_segment_reps_returns_list_of_reps(self, short_pose_df):
        if short_pose_df is None:
            pytest.skip("Short pose data not available")
        reps = segment_reps(short_pose_df, "wall_slide", fps=30)
        assert isinstance(reps, list)
        assert all(isinstance(r, Rep) for r in reps)

    def test_detected_reps_have_valid_frames(self, short_pose_df):
        if short_pose_df is None:
            pytest.skip("Short pose data not available")
        reps = segment_reps(short_pose_df, "wall_slide", fps=30)
        n_frames = len(short_pose_df)
        for rep in reps:
            assert 0 <= rep.start_frame < rep.end_frame <= n_frames
            assert rep.start_time_s < rep.end_time_s

    def test_plot_saves_from_real_poses(self, short_pose_df, tmp_path):
        if short_pose_df is None:
            pytest.skip("Short pose data not available")
        from src.utils.config import get_section
        cfg = get_section("segmentation")["wall_slide"]
        raw = build_signal_wall_slide(short_pose_df)
        smoothed = smooth_signal(raw, cfg["smoothing_window"], cfg["smoothing_polyorder"])
        reps = segment_reps(short_pose_df, "wall_slide", fps=30)
        out = tmp_path / "seg.png"
        plot_segmentation(smoothed, reps, fps=30,
                          title="wall_slide integration", save_path=out)
        assert out.exists()
