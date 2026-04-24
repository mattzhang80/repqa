"""Tests for src/ml/personalize.py — Phase 14."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.personalize import (
    _percentile_of,
    apply_personalization,
    fit_user_baseline,
    load_user_baseline,
    personalize_splits,
)


def _rep(session_id, rep_id, label, rom, tempo, user="u1", exercise="wall_slide"):
    return {
        "session_id": session_id,
        "rep_id": rep_id,
        "user_id": user,
        "exercise": exercise,
        "label_detail": label,
        "rom_proxy_max": rom,
        "rom_proxy_range": 0.4,
        "tempo_s": tempo,
        "tempo_deviation": 0.0,
        "conf_mean": 0.9,
        "conf_min": 0.8,
    }


def _make_features_df():
    rows = []
    # 3 good sessions with ROM clustered around 0.75, tempo around 5.0
    rng = np.random.default_rng(0)
    for sid in ["g1", "g2", "g3"]:
        for i in range(5):
            rows.append(_rep(sid, i, "good",
                             rom=float(rng.normal(0.75, 0.03)),
                             tempo=float(rng.normal(5.0, 0.3))))
    # 2 bad sessions with much smaller ROM (bad_rom_partial style)
    for sid in ["b1", "b2"]:
        for i in range(5):
            rows.append(_rep(sid, i, "bad_rom_partial",
                             rom=float(rng.normal(0.35, 0.03)),
                             tempo=float(rng.normal(5.0, 0.3))))
    return pd.DataFrame(rows)


class TestFitBaseline:
    def test_uses_good_reps_only_by_default(self, tmp_path, monkeypatch):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        b = fit_user_baseline(df, user_id="u1", exercise="wall_slide", save=True)
        # Median ROM should be ~0.75 (good), NOT between 0.75 and 0.35 which
        # would indicate bad reps polluted the baseline.
        assert 0.68 < b["rom_proxy_max_median"] < 0.82
        assert b["n_reps_used"] == 15  # 3 good sessions × 5 reps
        assert "b1" not in b["sessions"] and "b2" not in b["sessions"]

    def test_explicit_session_list_overrides_label_filter(self, tmp_path, monkeypatch):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        # Explicit list includes a bad session — must be honoured
        b = fit_user_baseline(
            df, user_id="u1", exercise="wall_slide",
            baseline_session_ids=["g1", "b1"],
            save=True,
        )
        assert set(b["sessions"]) == {"g1", "b1"}
        # Median pulled down by including bad_rom_partial reps
        assert 0.45 < b["rom_proxy_max_median"] < 0.65

    def test_unknown_session_raises(self, monkeypatch, tmp_path):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        with pytest.raises(ValueError):
            fit_user_baseline(
                df, user_id="u1", exercise="wall_slide",
                baseline_session_ids=["does_not_exist"],
                save=False,
            )

    def test_no_good_reps_raises(self, monkeypatch, tmp_path):
        df = _make_features_df()
        bad_only = df[df["label_detail"] != "good"]
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        with pytest.raises(ValueError):
            fit_user_baseline(bad_only, user_id="u1", exercise="wall_slide", save=False)

    def test_iqr_is_correct(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        # Construct known distribution: [1, 2, 3, 4, 5] → median=3, IQR=2
        rows = []
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            rows.append(_rep("g1", i, "good", rom=v, tempo=5.0))
        df = pd.DataFrame(rows)
        b = fit_user_baseline(df, user_id="u1", exercise="wall_slide", save=False)
        assert b["rom_proxy_max_median"] == pytest.approx(3.0)
        assert b["rom_proxy_max_iqr"] == pytest.approx(2.0)

    def test_n_sessions_cap(self, monkeypatch, tmp_path):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        b = fit_user_baseline(
            df, user_id="u1", exercise="wall_slide", n_sessions=2, save=False
        )
        assert len(b["sessions"]) == 2

    def test_saves_json_when_save_true(self, tmp_path, monkeypatch):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        fit_user_baseline(df, user_id="u1", exercise="wall_slide", save=True)
        out = tmp_path / "u1_wall_slide.json"
        assert out.exists()
        loaded = load_user_baseline("u1", "wall_slide")
        assert loaded is not None
        assert loaded["user_id"] == "u1"


class TestApplyPersonalization:
    def test_adds_z_and_pct_columns(self, monkeypatch, tmp_path):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        b = fit_user_baseline(df, user_id="u1", exercise="wall_slide", save=False)
        out = apply_personalization(df, b)
        for col in ["rom_proxy_max_z", "rom_proxy_max_pct",
                    "tempo_s_z", "tempo_s_pct"]:
            assert col in out.columns

    def test_z_correct_sign(self, monkeypatch, tmp_path):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        b = fit_user_baseline(df, user_id="u1", exercise="wall_slide", save=False)
        out = apply_personalization(df, b)
        # Bad (low-ROM) reps should have negative z-scores
        bad_rows = out[out["label_detail"] == "bad_rom_partial"]
        assert (bad_rows["rom_proxy_max_z"] < 0).all()

    def test_originals_preserved(self, monkeypatch, tmp_path):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        b = fit_user_baseline(df, user_id="u1", exercise="wall_slide", save=False)
        out = apply_personalization(df, b)
        for c in df.columns:
            assert c in out.columns
            assert out[c].equals(df[c])

    def test_pct_near_50_for_baseline_median(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        rows = [_rep("g1", i, "good", rom=float(v), tempo=5.0)
                for i, v in enumerate([1, 2, 3, 4, 5])]
        df = pd.DataFrame(rows)
        b = fit_user_baseline(df, user_id="u1", exercise="wall_slide", save=False)
        # Apply to a test rep whose ROM == baseline median
        test_df = pd.DataFrame([
            _rep("t1", 0, "good", rom=b["rom_proxy_max_median"], tempo=5.0)
        ])
        out = apply_personalization(test_df, b)
        # "mean" convention puts median at 50 for odd-N symmetric distribution
        assert 40.0 <= float(out["rom_proxy_max_pct"].iloc[0]) <= 60.0

    def test_missing_baseline_fills_nan(self, monkeypatch, tmp_path):
        df = _make_features_df()
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        empty_baseline = {
            "user_id": "u1", "exercise": "wall_slide",
            "rom_proxy_max_median": None, "rom_proxy_max_iqr": None,
            "rom_proxy_max_values": [],
            "tempo_s_median": None, "tempo_s_iqr": None, "tempo_s_values": [],
        }
        out = apply_personalization(df, empty_baseline)
        assert out["rom_proxy_max_z"].isna().all()
        assert out["rom_proxy_max_pct"].isna().all()


class TestPersonalizeSplits:
    def test_baseline_from_train_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "src.ml.personalize._BASELINES_DIR", tmp_path, raising=False
        )
        # Varied ROM so IQR > 0 in the train baseline.
        train = pd.DataFrame([
            _rep("g1", i, "good", 0.7 + 0.02 * i, 5.0) for i in range(5)
        ])
        test = pd.DataFrame([_rep("g2", i, "good", 0.3, 3.0) for i in range(5)])
        _, test_pers, baseline = personalize_splits(
            train, test, user_id="u1", exercise="wall_slide", save=False
        )
        # Test reps should receive strongly negative z-scores because their
        # ROM is way below the train baseline median (0.7).
        assert (test_pers["rom_proxy_max_z"] < -1).all()
        # Baseline sessions must only include train sessions
        assert "g2" not in baseline["sessions"]


class TestPercentileOf:
    def test_median_maps_to_50(self):
        assert _percentile_of([1.0, 2.0, 3.0, 4.0, 5.0], 3.0) == pytest.approx(50.0)

    def test_min_and_max_bounds(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert 0.0 <= _percentile_of(vals, 0.5) <= 10.0
        assert 90.0 <= _percentile_of(vals, 10.0) <= 100.0

    def test_empty_values_returns_nan(self):
        assert np.isnan(_percentile_of([], 1.0))

    def test_nan_input_returns_nan(self):
        assert np.isnan(_percentile_of([1.0, 2.0], float("nan")))
