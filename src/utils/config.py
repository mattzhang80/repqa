"""Load and access project configuration from config.yaml."""

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_config: dict | None = None


def get_config() -> dict:
    """Load and return the project config (cached after first call)."""
    global _config
    if _config is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
        with open(_CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    assert _config is not None
    return _config


def get_section(section: str) -> dict:
    """Return a top-level config section by name."""
    config = get_config()
    if section not in config:
        raise KeyError(f"Config section '{section}' not found. Available: {list(config.keys())}")
    return config[section]


def get_exercise_config(exercise: str) -> dict:
    """Return the exercise registry entry for a given exercise identifier."""
    exercises = get_config()["exercises"]
    if exercise not in exercises:
        raise KeyError(f"Unknown exercise '{exercise}'. Known: {list(exercises.keys())}")
    return exercises[exercise]
