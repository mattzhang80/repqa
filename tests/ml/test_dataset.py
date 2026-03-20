"""Tests for src/ml/dataset.py — Phase 12."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ml.dataset import assemble_dataset, save_splits, split_dataset


def _make_features_csv(tmp_path: Path, session_id: str, n_reps: int = 5) -> Path:
    rows = [
        {
            "session_id": session_id, "rep_id": i, "exercise": "wall_slide",
            "user_id": "test", "rom_proxy_max": 0.7, "rom_proxy_range": 0.4,
            "tempo_s": 5.0, "tempo_deviation": 0.0,
            "conf_mean": 0.9, "conf_min": 0.8,
        }
        for i in range(n_reps)
    ]
    d = tmp_path / session_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "features.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _make_labels_csv(tmp_path: Path, reps: list[tuple]) -> Path:
    """reps: list of (session_id, rep_id, label)"""
    rows = [{"session_id": s, "rep_id": r, "label": l} for s, r, l in reps]
    p = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


class TestAssembleDataset:
    def test_basic_merge(self, tmp_path):
        _make_features_csv(tmp_path, "s1", 3)
        labels_path = _make_labels_csv(tmp_path, [
            ("s1", 0, "good"), ("s1", 1, "bad_tempo"), ("s1", 2, "bad_rom_partial"),
        ])
        df = assemble_dataset(tmp_path, labels_path)
        assert len(df) == 3

    def test_y_bad_correct(self, tmp_path):
        _make_features_csv(tmp_path, "s1", 3)
        labels_path = _make_labels_csv(tmp_path, [
            ("s1", 0, "good"), ("s1", 1, "bad_tempo"), ("s1", 2, "bad_rom_partial"),
        ])
        df = assemble_dataset(tmp_path, labels_path)
        assert df[df["label_detail"] == "good"]["y_bad"].iloc[0] == 0
        assert df[df["label_detail"] == "bad_tempo"]["y_bad"].iloc[0] == 1
        assert df[df["label_detail"] == "bad_rom_partial"]["y_bad"].iloc[0] == 1

    def test_label_detail_preserved(self, tmp_path):
        _make_features_csv(tmp_path, "s1", 2)
        labels_path = _make_labels_csv(tmp_path, [("s1", 0, "good"), ("s1", 1, "bad_tempo")])
        df = assemble_dataset(tmp_path, labels_path)
        assert "label_detail" in df.columns
        assert set(df["label_detail"]) == {"good", "bad_tempo"}

    def test_no_nans_in_features(self, tmp_path):
        _make_features_csv(tmp_path, "s1", 3)
        labels_path = _make_labels_csv(tmp_path, [
            ("s1", 0, "good"), ("s1", 1, "good"), ("s1", 2, "good"),
        ])
        df = assemble_dataset(tmp_path, labels_path)
        feat_cols = ["rom_proxy_max", "tempo_s", "conf_mean"]
        for col in feat_cols:
            assert df[col].notna().all()

    def test_multiple_sessions(self, tmp_path):
        _make_features_csv(tmp_path, "s1", 3)
        _make_features_csv(tmp_path, "s2", 3)
        labels_path = _make_labels_csv(tmp_path, [
            ("s1", 0, "good"), ("s1", 1, "good"), ("s1", 2, "good"),
            ("s2", 0, "bad_tempo"), ("s2", 1, "bad_tempo"), ("s2", 2, "good"),
        ])
        df = assemble_dataset(tmp_path, labels_path)
        assert df["session_id"].nunique() == 2
        assert len(df) == 6

    def test_missing_labels_file_raises(self, tmp_path):
        _make_features_csv(tmp_path, "s1", 3)
        with pytest.raises(FileNotFoundError):
            assemble_dataset(tmp_path, tmp_path / "nonexistent.csv")

    def test_no_features_files_raises(self, tmp_path):
        labels_path = _make_labels_csv(tmp_path, [("s1", 0, "good")])
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError):
            assemble_dataset(empty_dir, labels_path)

    def test_deduplicates_labels(self, tmp_path):
        _make_features_csv(tmp_path, "s1", 1)
        # Two labels for same rep — last should win
        rows = [
            {"session_id": "s1", "rep_id": 0, "label": "good"},
            {"session_id": "s1", "rep_id": 0, "label": "bad_tempo"},
        ]
        labels_path = tmp_path / "labels.csv"
        pd.DataFrame(rows).to_csv(labels_path, index=False)
        df = assemble_dataset(tmp_path, labels_path)
        assert len(df) == 1
        assert df["label_detail"].iloc[0] == "bad_tempo"


class TestSplitDataset:
    def _build_df(self, tmp_path: Path) -> pd.DataFrame:
        sessions = [f"s{i:02d}" for i in range(6)]
        for s in sessions:
            _make_features_csv(tmp_path, s, 5)
        labels = [(s, i, "good") for s in sessions for i in range(5)]
        labels_path = _make_labels_csv(tmp_path, labels)
        return assemble_dataset(tmp_path, labels_path)

    def test_no_session_overlap(self, tmp_path):
        df = self._build_df(tmp_path)
        train, test = split_dataset(df, test_size=0.33)
        train_sessions = set(train["session_id"])
        test_sessions = set(test["session_id"])
        assert len(train_sessions & test_sessions) == 0

    def test_all_reps_accounted_for(self, tmp_path):
        df = self._build_df(tmp_path)
        train, test = split_dataset(df)
        assert len(train) + len(test) == len(df)

    def test_both_splits_nonempty(self, tmp_path):
        df = self._build_df(tmp_path)
        train, test = split_dataset(df)
        assert len(train) > 0
        assert len(test) > 0


class TestSaveSplits:
    def test_files_created(self, tmp_path):
        data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        save_splits(data, data, tmp_path)
        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "test.csv").exists()

    def test_creates_parent_dirs(self, tmp_path):
        data = pd.DataFrame({"a": [1]})
        out = tmp_path / "x" / "y"
        save_splits(data, data, out)
        assert (out / "train.csv").exists()
