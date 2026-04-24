"""Tests for src/api/labeling.py — Phase 11."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_label_fn():
    """Import _write_label lazily to avoid FastAPI import issues in test discovery."""
    from src.api.labeling import _write_label
    return _write_label


def _load_existing_labels_fn():
    from src.api.labeling import _load_existing_labels
    return _load_existing_labels


# ── _write_label / _load_existing_labels ─────────────────────────────────────

class TestWriteLabel:
    def test_creates_labels_csv(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "labels.csv"
        monkeypatch.setattr("src.api.labeling._LABELS_PATH", labels_path)

        _write_label_fn()("session_01", 0, "wall_slide", "good")
        assert labels_path.exists()

        with open(labels_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "session_01"
        assert rows[0]["rep_id"] == "0"
        assert rows[0]["label"] == "good"

    def test_deduplicates_on_same_rep(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "labels.csv"
        monkeypatch.setattr("src.api.labeling._LABELS_PATH", labels_path)

        write = _write_label_fn()
        write("session_01", 0, "wall_slide", "good")
        write("session_01", 0, "wall_slide", "bad_tempo")  # overwrite

        with open(labels_path) as fh:
            rows = list(csv.DictReader(fh))
        # Should have exactly one row for (session_01, rep 0)
        assert len(rows) == 1
        assert rows[0]["label"] == "bad_tempo"

    def test_multiple_reps_preserved(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "labels.csv"
        monkeypatch.setattr("src.api.labeling._LABELS_PATH", labels_path)

        write = _write_label_fn()
        write("session_01", 0, "wall_slide", "good")
        write("session_01", 1, "wall_slide", "bad_tempo")
        write("session_01", 2, "wall_slide", "bad_rom_partial")

        with open(labels_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3


class TestLoadExistingLabels:
    def test_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.api.labeling._LABELS_PATH", tmp_path / "nope.csv")
        assert _load_existing_labels_fn()() == {}

    def test_loads_written_labels(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "labels.csv"
        monkeypatch.setattr("src.api.labeling._LABELS_PATH", labels_path)

        _write_label_fn()("s1", 0, "wall_slide", "good")
        _write_label_fn()("s1", 1, "wall_slide", "bad_tempo")

        existing = _load_existing_labels_fn()()
        assert existing[("s1", 0)] == "good"
        assert existing[("s1", 1)] == "bad_tempo"


# ── CSV schema ───────────────────────────────────────────────────────────────

class TestLabelsCsvSchema:
    def test_csv_has_correct_columns(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "labels.csv"
        monkeypatch.setattr("src.api.labeling._LABELS_PATH", labels_path)

        _write_label_fn()("s1", 0, "wall_slide", "good")

        with open(labels_path) as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames or []) == {
                "session_id", "rep_id", "exercise", "label", "labeler", "timestamp"
            }

    def test_timestamp_is_iso_format(self, tmp_path, monkeypatch):
        labels_path = tmp_path / "labels.csv"
        monkeypatch.setattr("src.api.labeling._LABELS_PATH", labels_path)

        _write_label_fn()("s1", 0, "wall_slide", "good")

        with open(labels_path) as fh:
            rows = list(csv.DictReader(fh))
        ts = rows[0]["timestamp"]
        # ISO format includes T separator
        assert "T" in ts
