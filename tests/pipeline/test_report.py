"""Tests for src/pipeline/report.py — Phases 9 + 10."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.report import generate_report, generate_review_page


SESSION_DIR = Path("data/processed/wall_slide_good_01")


@pytest.fixture(scope="module")
def session_dir():
    if not SESSION_DIR.exists() or not (SESSION_DIR / "flags.json").exists():
        pytest.skip("Session artifacts not available.")
    return SESSION_DIR


class TestGenerateReport:
    def test_creates_html_file(self, session_dir, tmp_path):
        out = tmp_path / "report.html"
        result = generate_report(session_dir, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 100

    def test_html_contains_safety_note(self, session_dir, tmp_path):
        out = tmp_path / "report.html"
        generate_report(session_dir, out)
        html = out.read_text()
        assert "pain" in html.lower()

    def test_html_contains_exercise_display_name(self, session_dir, tmp_path):
        out = tmp_path / "report.html"
        generate_report(session_dir, out)
        html = out.read_text()
        assert "Wall Slide" in html

    def test_html_contains_session_id(self, session_dir, tmp_path):
        out = tmp_path / "report.html"
        generate_report(session_dir, out)
        html = out.read_text()
        assert "wall_slide_good_01" in html

    def test_flagged_reps_appear_in_report(self, session_dir, tmp_path):
        out = tmp_path / "report.html"
        generate_report(session_dir, out)
        html = out.read_text()
        # There should be at least one badge for a non-good label
        assert "badge" in html

    def test_default_output_path(self, session_dir, tmp_path):
        """When output_path is None, writes to session_dir/report.html."""
        import shutil
        session_copy = tmp_path / "session_copy"
        shutil.copytree(session_dir, session_copy)
        result = generate_report(session_copy)
        assert result == session_copy / "report.html"
        assert result.exists()


class TestGenerateReviewPage:
    def test_creates_html_file(self, session_dir, tmp_path):
        out = tmp_path / "review.html"
        result = generate_review_page(session_dir, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 100

    def test_html_contains_safety_note(self, session_dir, tmp_path):
        out = tmp_path / "review.html"
        generate_review_page(session_dir, out)
        html = out.read_text()
        assert "pain" in html.lower()

    def test_html_renders_flagged_reps_section(self, session_dir, tmp_path):
        """Review page renders without error and contains rep card structure."""
        out = tmp_path / "review.html"
        generate_review_page(session_dir, out)
        html = out.read_text()
        # The rep cards should always be present for this session (has flagged reps)
        assert "rep-card" in html or "no-flags" in html

    def test_html_contains_confirm_dismiss_buttons(self, session_dir, tmp_path):
        out = tmp_path / "review.html"
        generate_review_page(session_dir, out)
        html = out.read_text()
        assert "Confirm" in html or "confirm" in html.lower()
        assert "Dismiss" in html or "dismiss" in html.lower()
