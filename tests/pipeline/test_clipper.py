"""Tests for src/pipeline/clipper.py — Phase 8."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.pipeline.baseline import RepFlag
from src.pipeline.clipper import clip_flagged_reps, extract_clip, extract_thumbnail
from src.pipeline.rep_segment import Rep


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rep(rep_id=0, start_s=1.0, end_s=4.0, fps=30) -> Rep:
    return Rep(
        rep_id=rep_id,
        start_frame=int(start_s * fps),
        end_frame=int(end_s * fps),
        start_time_s=start_s,
        end_time_s=end_s,
    )


def _make_flag(rep_id=0, flagged=True, label="bad_tempo") -> RepFlag:
    return RepFlag(
        rep_id=rep_id, flagged=flagged, predicted_label=label,
        reasons=["tempo_out_of_range"], rom_proxy_max=0.7,
        tempo_s=2.0, confidence_level="high",
    )


# ── extract_clip ──────────────────────────────────────────────────────────────

class TestExtractClip:
    def test_raises_if_video_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_clip(tmp_path / "missing.mp4", 0.0, 3.0, 0.3, tmp_path / "out.mp4")

    def test_clip_produced_from_real_video(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")
        out = tmp_path / "clip.mp4"
        result = extract_clip(video, 5.0, 8.0, 0.3, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 1000  # non-empty

    def test_creates_parent_dirs(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")
        out = tmp_path / "a" / "b" / "clip.mp4"
        extract_clip(video, 5.0, 8.0, 0.3, out)
        assert out.exists()


# ── extract_thumbnail ─────────────────────────────────────────────────────────

class TestExtractThumbnail:
    def test_raises_if_video_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_thumbnail(tmp_path / "missing.mp4", 1.0, tmp_path / "thumb.jpg")

    def test_thumbnail_produced_from_real_video(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")
        out = tmp_path / "thumb.jpg"
        result = extract_thumbnail(video, 5.0, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 500

    def test_creates_parent_dirs(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")
        out = tmp_path / "a" / "b" / "thumb.jpg"
        extract_thumbnail(video, 5.0, out)
        assert out.exists()


# ── clip_flagged_reps ─────────────────────────────────────────────────────────

class TestClipFlaggedReps:
    def test_only_flagged_reps_are_clipped(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")

        reps = [_make_rep(0, 5.0, 8.0), _make_rep(1, 10.0, 14.0)]
        flags = [_make_flag(0, flagged=True), _make_flag(1, flagged=False)]

        clips = clip_flagged_reps(video, reps, flags, tmp_path)
        assert len(clips) == 1
        assert clips[0]["rep_id"] == 0

    def test_output_dict_keys(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")

        reps = [_make_rep(0, 5.0, 8.0)]
        flags = [_make_flag(0, flagged=True)]
        clips = clip_flagged_reps(video, reps, flags, tmp_path)
        keys = {"rep_id", "clip_path", "thumbnail_path", "predicted_label",
                "reasons", "start_s", "end_s", "duration_s"}
        assert keys.issubset(clips[0].keys())

    def test_clip_and_thumb_files_created(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")

        reps = [_make_rep(0, 5.0, 8.0)]
        flags = [_make_flag(0, flagged=True)]
        clips = clip_flagged_reps(video, reps, flags, tmp_path)

        assert Path(clips[0]["clip_path"]).exists()
        assert Path(clips[0]["thumbnail_path"]).exists()

    def test_no_flagged_reps_returns_empty(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")

        reps = [_make_rep(0, 5.0, 8.0)]
        flags = [_make_flag(0, flagged=False)]
        clips = clip_flagged_reps(video, reps, flags, tmp_path)
        assert clips == []

    def test_duration_in_metadata(self, tmp_path):
        video = Path("data/processed/wall_slide_good_01/video.mp4")
        if not video.exists():
            pytest.skip("Real session video not available.")

        reps = [_make_rep(0, 5.0, 10.0)]
        flags = [_make_flag(0, flagged=True)]
        clips = clip_flagged_reps(video, reps, flags, tmp_path, padding_s=0.0)
        assert clips[0]["duration_s"] == pytest.approx(5.0)
