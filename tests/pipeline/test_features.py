"""Tests for src/pipeline/features.py — Phase 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import (
    compute_confidence_features,
    compute_rom_proxy_wall_slide,
    compute_tempo,
    compute_tempo_deviation,
    extract_rep_features,
    load_features,
    save_features,
)
from src.pipeline.rep_segment import Rep


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rep(
    rep_id: int = 0,
    start_frame: int = 0,
    end_frame: int = 149,
    fps: int = 30,
) -> Rep:
    return Rep(
        rep_id=rep_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time_s=round(start_frame / fps, 3),
        end_time_s=round(end_frame / fps, 3),
    )


def _make_pose_df(
    n_frames: int = 150,
    wrist_y_start: float = 0.8,
    wrist_y_end: float = 0.3,
    shoulder_y: float = 0.5,
    hip_y: float = 0.7,
    vis: float = 0.9,
) -> pd.DataFrame:
    """Synthetic pose DataFrame.

    Wrist moves linearly from wrist_y_start (low = resting) to wrist_y_end
    (high = top-of-slide) and back.  shoulder_y and hip_y are constant.
    torso_height = hip_y - shoulder_y = 0.2 by default.
    """
    frames = np.arange(n_frames)
    # linear ramp: go from start to end
    wrist_y = np.linspace(wrist_y_start, wrist_y_end, n_frames)

    rows = {
        "frame_idx": frames,
        "timestamp_s": frames / 30.0,
        "left_wrist_y": wrist_y,
        "right_wrist_y": wrist_y + 0.05,           # slightly worse
        "left_shoulder_y": np.full(n_frames, shoulder_y),
        "right_shoulder_y": np.full(n_frames, shoulder_y),
        "left_hip_y": np.full(n_frames, hip_y),
        "right_hip_y": np.full(n_frames, hip_y),
        "left_elbow_y": np.full(n_frames, 0.6),
        "right_elbow_y": np.full(n_frames, 0.6),
        "left_wrist_vis": np.full(n_frames, vis),
        "right_wrist_vis": np.full(n_frames, vis - 0.1),
        "left_shoulder_vis": np.full(n_frames, vis),
        "right_shoulder_vis": np.full(n_frames, vis),
        "left_elbow_vis": np.full(n_frames, vis),
        "right_elbow_vis": np.full(n_frames, vis),
        "left_hip_vis": np.full(n_frames, vis),
        "right_hip_vis": np.full(n_frames, vis),
    }
    return pd.DataFrame(rows)


# ── compute_rom_proxy_wall_slide ──────────────────────────────────────────────

class TestComputeRomProxyWallSlide:
    def test_basic_range_is_correct(self):
        """Wrist goes from y=0.8 (low) to y=0.3 (high-up).
        signal = (shoulder_y - wrist_y) / torso  =  (0.5 - wrist_y) / 0.2
        At rest:  (0.5 - 0.8) / 0.2 = -1.5
        At top:   (0.5 - 0.3) / 0.2 = +1.0
        range = 1.0 - (-1.5) = 2.5
        """
        pose_df = _make_pose_df(wrist_y_start=0.8, wrist_y_end=0.3)
        rep = _make_rep(start_frame=0, end_frame=149)
        result = compute_rom_proxy_wall_slide(pose_df, rep)

        assert "rom_proxy_max" in result
        assert "rom_proxy_range" in result
        assert abs(result["rom_proxy_max"] - 1.0) < 0.05
        assert abs(result["rom_proxy_range"] - 2.5) < 0.05

    def test_small_travel_gives_small_range(self):
        """Wrist barely moves → small ROM proxy range."""
        pose_df = _make_pose_df(wrist_y_start=0.6, wrist_y_end=0.55)
        rep = _make_rep(start_frame=0, end_frame=149)
        result = compute_rom_proxy_wall_slide(pose_df, rep)
        assert result["rom_proxy_range"] < 0.5

    def test_larger_travel_gives_larger_range_than_smaller(self):
        """Larger wrist travel → larger ROM proxy range."""
        pose_df_large = _make_pose_df(wrist_y_start=0.8, wrist_y_end=0.2)
        pose_df_small = _make_pose_df(wrist_y_start=0.7, wrist_y_end=0.5)
        rep = _make_rep()
        large = compute_rom_proxy_wall_slide(pose_df_large, rep)["rom_proxy_range"]
        small = compute_rom_proxy_wall_slide(pose_df_small, rep)["rom_proxy_range"]
        assert large > small

    def test_returns_nan_when_all_nan(self):
        """If all wrist coordinates are NaN, result should be NaN."""
        pose_df = _make_pose_df()
        pose_df["left_wrist_y"] = float("nan")
        pose_df["right_wrist_y"] = float("nan")
        rep = _make_rep()
        result = compute_rom_proxy_wall_slide(pose_df, rep)
        # If all NaN, signal is all NaN → nan output
        assert np.isnan(result["rom_proxy_max"]) or np.isnan(result["rom_proxy_range"])

    def test_uses_sub_slice_of_full_df(self):
        """Only frames within rep boundaries are used."""
        # First half: large travel; second half: no travel
        pose_df = _make_pose_df(n_frames=300)
        # Overwrite second half to have no wrist movement
        pose_df.loc[150:, "left_wrist_y"] = 0.5

        rep_first = _make_rep(start_frame=0, end_frame=149)
        rep_second = _make_rep(rep_id=1, start_frame=150, end_frame=299)

        rom_first = compute_rom_proxy_wall_slide(pose_df, rep_first)["rom_proxy_range"]
        rom_second = compute_rom_proxy_wall_slide(pose_df, rep_second)["rom_proxy_range"]
        assert rom_first > rom_second


# ── compute_tempo ─────────────────────────────────────────────────────────────

class TestComputeTempo:
    def test_correct_duration(self):
        rep = Rep(
            rep_id=0,
            start_frame=0,
            end_frame=150,
            start_time_s=0.0,
            end_time_s=5.0,
        )
        assert compute_tempo(rep, fps=30) == pytest.approx(5.0)

    def test_short_rep(self):
        rep = Rep(
            rep_id=0,
            start_frame=0,
            end_frame=45,
            start_time_s=0.0,
            end_time_s=1.5,
        )
        assert compute_tempo(rep, fps=30) == pytest.approx(1.5)

    def test_uses_time_not_frames(self):
        """compute_tempo reads from .start_time_s / .end_time_s fields."""
        rep = Rep(
            rep_id=0,
            start_frame=100,
            end_frame=250,
            start_time_s=10.0,
            end_time_s=15.0,
        )
        assert compute_tempo(rep, fps=30) == pytest.approx(5.0)


# ── compute_tempo_deviation ───────────────────────────────────────────────────

class TestComputeTempoDeviation:
    def test_zero_for_perfect_tempo(self):
        assert compute_tempo_deviation(5.0, "wall_slide") == pytest.approx(0.0)
        assert compute_tempo_deviation(5.0, "band_er_side") == pytest.approx(0.0)

    def test_positive_for_fast_rep(self):
        dev = compute_tempo_deviation(1.5, "wall_slide")
        assert dev == pytest.approx(3.5)

    def test_positive_for_slow_rep(self):
        dev = compute_tempo_deviation(9.0, "wall_slide")
        assert dev == pytest.approx(4.0)

    def test_symmetric(self):
        """Same deviation above and below nominal."""
        dev_fast = compute_tempo_deviation(3.0, "wall_slide")
        dev_slow = compute_tempo_deviation(7.0, "wall_slide")
        assert dev_fast == pytest.approx(dev_slow)

    def test_unknown_exercise_uses_default_5s(self):
        """Falls back to 5.0s nominal for unknown exercise."""
        dev = compute_tempo_deviation(5.0, "unknown_exercise")
        assert dev == pytest.approx(0.0)


# ── compute_confidence_features ───────────────────────────────────────────────

class TestComputeConfidenceFeatures:
    def test_returns_correct_keys(self):
        pose_df = _make_pose_df()
        rep = _make_rep()
        result = compute_confidence_features(pose_df, rep)
        assert "conf_mean" in result
        assert "conf_min" in result

    def test_values_in_0_1_range(self):
        pose_df = _make_pose_df(vis=0.85)
        rep = _make_rep()
        result = compute_confidence_features(pose_df, rep)
        assert 0.0 <= result["conf_mean"] <= 1.0
        assert 0.0 <= result["conf_min"] <= 1.0

    def test_mean_approx_vis(self):
        pose_df = _make_pose_df(vis=0.9)
        rep = _make_rep()
        result = compute_confidence_features(pose_df, rep)
        # left joints are 0.9, right joints are 0.9 as well (except wrist: 0.8)
        assert result["conf_mean"] == pytest.approx(0.9 - 0.1 / 8, abs=0.05)

    def test_min_reflects_lowest_visibility(self):
        pose_df = _make_pose_df(vis=0.9)
        # Force one joint to very low visibility
        pose_df["left_hip_vis"] = 0.1
        rep = _make_rep()
        result = compute_confidence_features(pose_df, rep)
        assert result["conf_min"] == pytest.approx(0.1, abs=0.05)

    def test_nan_returned_when_no_vis_columns(self):
        pose_df = _make_pose_df()
        vis_cols = [c for c in pose_df.columns if c.endswith("_vis")]
        pose_df = pose_df.drop(columns=vis_cols)
        rep = _make_rep()
        result = compute_confidence_features(pose_df, rep)
        assert np.isnan(result["conf_mean"])
        assert np.isnan(result["conf_min"])


# ── extract_rep_features ──────────────────────────────────────────────────────

class TestExtractRepFeatures:
    def _build_reps(self, n: int = 3, frames_per_rep: int = 150) -> list[Rep]:
        reps = []
        for i in range(n):
            s = i * frames_per_rep
            e = s + frames_per_rep - 1
            reps.append(
                Rep(
                    rep_id=i,
                    start_frame=s,
                    end_frame=e,
                    start_time_s=round(s / 30, 3),
                    end_time_s=round(e / 30, 3),
                )
            )
        return reps

    def test_returns_one_row_per_rep(self):
        n = 3
        pose_df = _make_pose_df(n_frames=n * 150)
        reps = self._build_reps(n)
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, reps, "wall_slide", 30, meta)
        assert len(df) == n

    def test_correct_columns(self):
        pose_df = _make_pose_df(n_frames=150)
        reps = self._build_reps(1)
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, reps, "wall_slide", 30, meta)
        expected = {
            "session_id", "rep_id", "exercise", "user_id",
            "rom_proxy_max", "rom_proxy_range",
            "tempo_s", "tempo_deviation",
            "conf_mean", "conf_min",
        }
        assert set(df.columns) == expected

    def test_session_and_user_propagated(self):
        pose_df = _make_pose_df(n_frames=150)
        reps = self._build_reps(1)
        meta = {"session_id": "test_session", "user_id": "alice"}
        df = extract_rep_features(pose_df, reps, "wall_slide", 30, meta)
        assert df["session_id"].iloc[0] == "test_session"
        assert df["user_id"].iloc[0] == "alice"

    def test_exercise_column_set(self):
        pose_df = _make_pose_df(n_frames=150)
        reps = self._build_reps(1)
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, reps, "wall_slide", 30, meta)
        assert (df["exercise"] == "wall_slide").all()

    def test_no_nans_for_clean_input(self):
        pose_df = _make_pose_df(n_frames=3 * 150)
        reps = self._build_reps(3)
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, reps, "wall_slide", 30, meta)
        num_cols = ["rom_proxy_max", "rom_proxy_range", "tempo_s",
                    "tempo_deviation", "conf_mean", "conf_min"]
        for col in num_cols:
            assert df[col].notna().all(), f"NaN found in column {col}"

    def test_tempo_s_matches_rep_duration(self):
        pose_df = _make_pose_df(n_frames=150)
        rep = Rep(
            rep_id=0, start_frame=0, end_frame=149,
            start_time_s=0.0, end_time_s=5.0,
        )
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, [rep], "wall_slide", 30, meta)
        assert df["tempo_s"].iloc[0] == pytest.approx(5.0)

    def test_fast_rep_has_high_tempo_deviation(self):
        pose_df = _make_pose_df(n_frames=60)
        rep = Rep(
            rep_id=0, start_frame=0, end_frame=59,
            start_time_s=0.0, end_time_s=2.0,
        )
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, [rep], "wall_slide", 30, meta)
        assert df["tempo_deviation"].iloc[0] > 2.0

    def test_good_tempo_has_low_deviation(self):
        pose_df = _make_pose_df(n_frames=150)
        rep = Rep(
            rep_id=0, start_frame=0, end_frame=149,
            start_time_s=0.0, end_time_s=5.0,
        )
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, [rep], "wall_slide", 30, meta)
        assert df["tempo_deviation"].iloc[0] == pytest.approx(0.0)

    def test_empty_reps_returns_empty_df_with_columns(self):
        pose_df = _make_pose_df(n_frames=150)
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, [], "wall_slide", 30, meta)
        assert len(df) == 0
        assert "rom_proxy_max" in df.columns

    def test_band_er_side_raises_not_implemented(self):
        pose_df = _make_pose_df(n_frames=150)
        reps = self._build_reps(1)
        meta = {"session_id": "s01", "user_id": "u01"}
        with pytest.raises(NotImplementedError):
            extract_rep_features(pose_df, reps, "band_er_side", 30, meta)

    def test_unknown_exercise_raises_value_error(self):
        pose_df = _make_pose_df(n_frames=150)
        reps = self._build_reps(1)
        meta = {"session_id": "s01", "user_id": "u01"}
        with pytest.raises(ValueError):
            extract_rep_features(pose_df, reps, "squat", 30, meta)


# ── save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoadFeatures:
    def test_round_trip(self, tmp_path):
        pose_df = _make_pose_df(n_frames=150)
        rep = _make_rep()
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, [rep], "wall_slide", 30, meta)

        out = tmp_path / "features" / "features.csv"
        save_features(df, out)
        assert out.exists()

        loaded = load_features(out)
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)
        pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                       loaded.reset_index(drop=True))

    def test_save_creates_parent_dirs(self, tmp_path):
        pose_df = _make_pose_df(n_frames=150)
        rep = _make_rep()
        meta = {"session_id": "s01", "user_id": "u01"}
        df = extract_rep_features(pose_df, [rep], "wall_slide", 30, meta)

        out = tmp_path / "a" / "b" / "c" / "features.csv"
        save_features(df, out)
        assert out.exists()

    def test_save_empty_df(self, tmp_path):
        out = tmp_path / "features.csv"
        meta = {"session_id": "s01", "user_id": "u01"}
        pose_df = _make_pose_df()
        df = extract_rep_features(pose_df, [], "wall_slide", 30, meta)
        save_features(df, out)
        loaded = load_features(out)
        assert len(loaded) == 0


# ── Integration: run on a real session ───────────────────────────────────────

class TestIntegrationRealSession:
    """Run feature extraction on a real processed session if artifacts exist."""

    def test_real_session_good_01(self, tmp_path):
        poses_path = Path("data/poses/wall_slide_good_01/poses.parquet")
        reps_path = Path("data/processed/wall_slide_good_01/reps.csv")
        if not poses_path.exists() or not reps_path.exists():
            pytest.skip("Real session artifacts not available.")

        pose_df = pd.read_parquet(poses_path)
        reps_df = pd.read_csv(reps_path)
        reps = [
            Rep(
                rep_id=int(r["rep_id"]),
                start_frame=int(r["start_frame"]),
                end_frame=int(r["end_frame"]),
                start_time_s=float(r["start_time_s"]),
                end_time_s=float(r["end_time_s"]),
            )
            for _, r in reps_df.iterrows()
        ]

        meta = {"session_id": "wall_slide_good_01", "user_id": "matthew"}
        df = extract_rep_features(pose_df, reps, "wall_slide", 30, meta)

        assert len(df) == len(reps)
        num_cols = ["rom_proxy_max", "rom_proxy_range", "tempo_s",
                    "tempo_deviation", "conf_mean", "conf_min"]
        for col in num_cols:
            assert df[col].notna().all(), f"NaN in {col}"
        # Good reps should have reasonable ROM proxy
        assert df["rom_proxy_max"].median() > 0.0
        # Tempo ~4-8s for good reps
        assert (df["tempo_s"].between(3.0, 10.0)).all()
