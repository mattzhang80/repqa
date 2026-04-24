"""Tests for src/pipeline/baseline.py — Phase 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.baseline import (
    RepFlag,
    flag_reps_baseline,
    load_flags,
    save_flags,
    summarize_flags,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_features_row(
    rep_id: int = 0,
    rom_proxy_max: float = 0.7,
    rom_proxy_range: float = 0.5,
    tempo_s: float = 5.0,
    tempo_deviation: float = 0.0,
    conf_mean: float = 0.9,
    conf_min: float = 0.8,
    session_id: str = "test_session",
    user_id: str = "test_user",
    exercise: str = "wall_slide",
) -> dict:
    return {
        "session_id": session_id,
        "rep_id": rep_id,
        "exercise": exercise,
        "user_id": user_id,
        "rom_proxy_max": rom_proxy_max,
        "rom_proxy_range": rom_proxy_range,
        "tempo_s": tempo_s,
        "tempo_deviation": tempo_deviation,
        "conf_mean": conf_mean,
        "conf_min": conf_min,
    }


def _make_features_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# Minimal config that mirrors the relevant sections of config.yaml
_MOCK_CONFIG = {
    "pose": {
        "confidence_threshold": 0.5,
    },
    "baseline": {
        "rom_cutoffs": {
            "wall_slide": 0.3,
            "band_er_side": 0.2,
        },
        "tempo_bounds_s": {
            "wall_slide": [3.0, 8.0],    # half-width = 2.5s threshold
            "band_er_side": [3.0, 8.0],
        },
        "flag_unknown_on_low_confidence": True,
    },
}


# ── flag_reps_baseline ────────────────────────────────────────────────────────

class TestFlagRepsBaseline:
    def test_good_rep_not_flagged(self):
        """Good ROM, good tempo, high confidence → predicted_label='good'."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.7, tempo_deviation=0.0, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert len(flags) == 1
        f = flags[0]
        assert not f.flagged
        assert f.predicted_label == "good"
        assert f.reasons == []

    def test_bad_rom_flagged(self):
        """ROM below cutoff (0.3) with good confidence → bad_rom_partial."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.1, tempo_deviation=0.0, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        f = flags[0]
        assert f.flagged
        assert f.predicted_label == "bad_rom_partial"
        assert "rom_below_cutoff" in f.reasons

    def test_bad_tempo_flagged(self):
        """tempo_s outside bounds [3.0, 8.0] → bad_tempo."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.7, tempo_s=1.5, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        f = flags[0]
        assert f.flagged
        assert f.predicted_label == "bad_tempo"
        assert "tempo_out_of_range" in f.reasons

    def test_low_confidence_flagged_as_unknown(self):
        """conf_mean below confidence_threshold → predicted_label='unknown'."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.7, tempo_deviation=0.0, conf_mean=0.3
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        f = flags[0]
        assert f.flagged
        assert f.predicted_label == "unknown"
        assert "pose_low_confidence" in f.reasons

    def test_tempo_takes_priority_over_rom(self):
        """Both tempo and ROM issues → predicted_label='bad_tempo' (tempo checked first)."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.1, tempo_s=1.5, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        f = flags[0]
        assert f.predicted_label == "bad_tempo"
        assert "tempo_out_of_range" in f.reasons
        assert "rom_below_cutoff" in f.reasons  # still collected

    def test_low_conf_suppresses_rom_flag(self):
        """Low confidence should not produce a rom_below_cutoff reason."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.1, tempo_deviation=0.0, conf_mean=0.2
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        f = flags[0]
        assert f.predicted_label == "unknown"
        assert "rom_below_cutoff" not in f.reasons

    def test_rom_just_above_cutoff_not_flagged(self):
        """ROM at cutoff + ε → not flagged."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.31, tempo_deviation=0.0, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].predicted_label == "good"

    def test_tempo_at_boundary_not_flagged(self):
        """tempo_s exactly at lower bound → not flagged."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.7, tempo_s=3.0, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].predicted_label == "good"

    def test_tempo_just_below_lower_bound_flagged(self):
        """tempo_s below lower bound → flagged."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.7, tempo_s=2.9, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].predicted_label == "bad_tempo"

    def test_tempo_above_upper_bound_flagged(self):
        """tempo_s above upper bound → flagged."""
        df = _make_features_df(_make_features_row(
            rom_proxy_max=0.7, tempo_s=8.1, conf_mean=0.9
        ))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].predicted_label == "bad_tempo"

    def test_multiple_reps_returned(self):
        """Returns one flag per row."""
        rows = [
            _make_features_row(rep_id=0, rom_proxy_max=0.7, tempo_deviation=0.0),
            _make_features_row(rep_id=1, rom_proxy_max=0.1, tempo_deviation=0.0),
            _make_features_row(rep_id=2, rom_proxy_max=0.7, tempo_s=1.5),
        ]
        df = _make_features_df(*rows)
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert len(flags) == 3
        assert flags[0].predicted_label == "good"
        assert flags[1].predicted_label == "bad_rom_partial"
        assert flags[2].predicted_label == "bad_tempo"

    def test_rep_ids_preserved(self):
        """rep_id in each RepFlag matches the input row."""
        rows = [_make_features_row(rep_id=i) for i in range(5)]
        df = _make_features_df(*rows)
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        for i, f in enumerate(flags):
            assert f.rep_id == i

    def test_empty_df_returns_empty_list(self):
        """Empty features DataFrame → empty flag list."""
        df = pd.DataFrame(columns=list(_make_features_row().keys()))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags == []

    def test_confidence_level_high_for_good_confidence(self):
        df = _make_features_df(_make_features_row(conf_mean=0.9))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].confidence_level == "high"

    def test_confidence_level_low_for_bad_confidence(self):
        df = _make_features_df(_make_features_row(conf_mean=0.2))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].confidence_level == "low"

    def test_config_loaded_from_yaml_if_none(self):
        """When config=None, loads from config.yaml without error."""
        df = _make_features_df(_make_features_row())
        flags = flag_reps_baseline(df, "wall_slide", config=None)
        assert len(flags) == 1

    def test_rom_proxy_max_stored_on_flag(self):
        df = _make_features_df(_make_features_row(rom_proxy_max=0.42))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].rom_proxy_max == pytest.approx(0.42)

    def test_tempo_s_stored_on_flag(self):
        df = _make_features_df(_make_features_row(tempo_s=3.14))
        flags = flag_reps_baseline(df, "wall_slide", _MOCK_CONFIG)
        assert flags[0].tempo_s == pytest.approx(3.14)


# ── summarize_flags ───────────────────────────────────────────────────────────

class TestSummarizeFlags:
    def _build_flags(self) -> list[RepFlag]:
        return [
            RepFlag(0, False, "good",           [],                          0.7, 5.0, "high"),
            RepFlag(1, True,  "bad_rom_partial", ["rom_below_cutoff"],       0.1, 5.0, "high"),
            RepFlag(2, True,  "bad_tempo",       ["tempo_out_of_range"],   0.7, 1.5, "high"),
            RepFlag(3, True,  "unknown",          ["pose_low_confidence"],   0.7, 5.0, "low"),
        ]

    def test_total_reps(self):
        summary = summarize_flags(self._build_flags())
        assert summary["total_reps"] == 4

    def test_flagged_count(self):
        summary = summarize_flags(self._build_flags())
        assert summary["flagged_count"] == 3

    def test_good_count(self):
        summary = summarize_flags(self._build_flags())
        assert summary["good_count"] == 1

    def test_label_distribution(self):
        summary = summarize_flags(self._build_flags())
        ld = summary["label_distribution"]
        assert ld["good"] == 1
        assert ld["bad_rom_partial"] == 1
        assert ld["bad_tempo"] == 1
        assert ld["unknown"] == 1

    def test_reasons_distribution(self):
        summary = summarize_flags(self._build_flags())
        rd = summary["reasons_distribution"]
        assert rd["rom_below_cutoff"] == 1
        assert rd["tempo_out_of_range"] == 1
        assert rd["pose_low_confidence"] == 1

    def test_empty_flags(self):
        summary = summarize_flags([])
        assert summary["total_reps"] == 0
        assert summary["flagged_count"] == 0
        assert summary["good_count"] == 0
        assert summary["label_distribution"] == {}

    def test_all_good(self):
        flags = [RepFlag(i, False, "good", [], 0.7, 5.0, "high") for i in range(5)]
        summary = summarize_flags(flags)
        assert summary["flagged_count"] == 0
        assert summary["good_count"] == 5
        assert summary["label_distribution"] == {"good": 5}
        assert summary["reasons_distribution"] == {}


# ── save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoadFlags:
    def _sample_flags(self) -> list[RepFlag]:
        return [
            RepFlag(0, False, "good",           [],                          0.75, 5.1, "high"),
            RepFlag(1, True,  "bad_rom_partial", ["rom_below_cutoff"],       0.2,  5.0, "high"),
            RepFlag(2, True,  "bad_tempo",       ["tempo_out_of_range"],   0.7,  1.8, "high"),
        ]

    def test_round_trip(self, tmp_path):
        flags = self._sample_flags()
        out = tmp_path / "flags.json"
        save_flags(flags, out)
        assert out.exists()
        loaded = load_flags(out)
        assert len(loaded) == len(flags)
        for orig, reloaded in zip(flags, loaded):
            assert orig.rep_id == reloaded.rep_id
            assert orig.flagged == reloaded.flagged
            assert orig.predicted_label == reloaded.predicted_label
            assert orig.reasons == reloaded.reasons
            assert orig.rom_proxy_max == pytest.approx(reloaded.rom_proxy_max)
            assert orig.tempo_s == pytest.approx(reloaded.tempo_s)
            assert orig.confidence_level == reloaded.confidence_level

    def test_save_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "flags.json"
        save_flags([], out)
        assert out.exists()

    def test_empty_flags_round_trip(self, tmp_path):
        out = tmp_path / "flags.json"
        save_flags([], out)
        loaded = load_flags(out)
        assert loaded == []


# ── Integration: run on real sessions ────────────────────────────────────────

class TestIntegrationRealSessions:
    """Run flagger on real feature CSVs if available."""

    def _load_session(self, session_id: str):
        path = Path(f"data/features/{session_id}/features.csv")
        if not path.exists():
            pytest.skip(f"features.csv not found for {session_id}")
        return pd.read_csv(path)

    def test_good_session_mostly_unflagged(self):
        df = self._load_session("wall_slide_good_01")
        flags = flag_reps_baseline(df, "wall_slide")
        good_count = sum(1 for f in flags if f.predicted_label == "good")
        # Majority of good reps should pass
        assert good_count >= len(flags) // 2

    def test_bad_rom_session_has_rom_flags(self):
        df = self._load_session("wall_slide_bad_rom_partial_01")
        flags = flag_reps_baseline(df, "wall_slide")
        rom_flagged = [f for f in flags if f.predicted_label == "bad_rom_partial"]
        assert len(rom_flagged) > 0

    def test_bad_tempo_session_has_tempo_flags(self):
        df = self._load_session("wall_slide_bad_tempo_01")
        flags = flag_reps_baseline(df, "wall_slide")
        tempo_flagged = [f for f in flags if f.predicted_label == "bad_tempo"]
        assert len(tempo_flagged) > 0
