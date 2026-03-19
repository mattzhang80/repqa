"""Tests for src/utils/config.py"""
import pytest
from src.utils.config import get_config, get_section, get_exercise_config


def test_get_config_returns_dict():
    config = get_config()
    assert isinstance(config, dict)


def test_get_config_cached():
    """Second call returns same object (cached)."""
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2


def test_config_has_required_sections():
    config = get_config()
    required = {"video", "pose", "segmentation", "features", "baseline",
                "personalization", "model", "clip", "report", "exercises"}
    assert required.issubset(config.keys())


def test_video_section():
    video = get_section("video")
    assert video["fps"] == 30
    assert video["width"] == 720


def test_pose_section():
    pose = get_section("pose")
    assert 0 < pose["confidence_threshold"] < 1
    assert 0 < pose["low_confidence_frame_pct"] < 1


def test_exercise_registry_has_both_exercises():
    exercises = get_section("exercises")
    assert "wall_slide" in exercises
    assert "band_er_side" in exercises


def test_exercise_registry_wall_slide():
    ex = get_exercise_config("wall_slide")
    assert ex["filming_angle"] == "side"
    assert ex["display_name"] == "Wall Slide (Forearms on Wall)"
    assert "good" in ex["labels"]
    assert "bad_tempo" in ex["labels"]
    assert "bad_rom_partial" in ex["labels"]


def test_exercise_registry_band_er_side():
    ex = get_exercise_config("band_er_side")
    assert ex["filming_angle"] == "front"
    assert ex["display_name"] == "Band External Rotation at Side (Elbow Tucked)"
    assert "good" in ex["labels"]
    assert "bad_tempo" in ex["labels"]
    assert "bad_rom_partial" in ex["labels"]
    assert "bad_elbow_drift_mild" in ex["labels"]


def test_segmentation_has_both_exercises():
    seg = get_section("segmentation")
    assert "wall_slide" in seg
    assert "band_er_side" in seg


def test_segmentation_wall_slide_bounds():
    seg = get_section("segmentation")
    bounds = seg["wall_slide"]["rep_duration_bounds_s"]
    assert len(bounds) == 2
    assert bounds[0] < bounds[1]


def test_segmentation_band_er_side_bounds():
    seg = get_section("segmentation")
    bounds = seg["band_er_side"]["rep_duration_bounds_s"]
    assert len(bounds) == 2
    assert bounds[0] < bounds[1]


def test_baseline_has_both_exercises():
    bl = get_section("baseline")
    assert "wall_slide" in bl["rom_cutoffs"]
    assert "band_er_side" in bl["rom_cutoffs"]
    assert "wall_slide" in bl["tempo_bounds_s"]
    assert "band_er_side" in bl["tempo_bounds_s"]


def test_model_c_values():
    model = get_section("model")
    assert len(model["C_values"]) > 0
    assert all(c > 0 for c in model["C_values"])


def test_report_has_safety_note():
    report = get_section("report")
    assert "safety_note" in report
    assert "pain" in report["safety_note"].lower()


def test_get_section_missing_raises():
    with pytest.raises(KeyError):
        get_section("nonexistent_section")


def test_get_exercise_config_missing_raises():
    with pytest.raises(KeyError):
        get_exercise_config("nonexistent_exercise")


def test_no_old_exercise_names():
    """Verify old exercise identifiers are fully removed."""
    exercises = get_section("exercises")
    assert "wall_slides" not in exercises
    assert "external_rotation" not in exercises
    seg = get_section("segmentation")
    assert "wall_slides" not in seg
    assert "external_rotation" not in seg
