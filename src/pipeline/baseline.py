"""Baseline threshold-based rep flagger for RepQA.

Applies simple, interpretable threshold rules to classify reps without ML.
The goal is *high-confidence flagging*: only flag when clearly out of range.
Ambiguous reps are left as 'good' (conservative / trust-over-completeness).

Rules (per rep):
  1. Low pose confidence → predicted_label = 'unknown', reason = 'pose_low_confidence'
  2. ROM below cutoff (with good confidence) → predicted_label = 'bad_rom_partial'
  3. Tempo deviation above threshold → predicted_label = 'bad_tempo'
  4. Otherwise → predicted_label = 'good'

Multiple rules can apply; the *first* applicable rule determines predicted_label
but *all* applicable reasons are recorded.

Usage:
    python src/pipeline/baseline.py \\
        --features data/features/<session>/features.csv \\
        --exercise wall_slide \\
        --output   data/processed/<session>/flags.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import get_config


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RepFlag:
    rep_id: int
    flagged: bool
    predicted_label: str          # 'good', 'bad_tempo', 'bad_rom_partial', or 'unknown'
    reasons: list[str]
    rom_proxy_max: float
    tempo_s: float
    confidence_level: str         # 'high', 'low'


# ── Thresholds ────────────────────────────────────────────────────────────────

def _get_tempo_bounds(exercise: str, config: dict) -> tuple[float, float]:
    """Return (min_s, max_s) acceptable tempo range for an exercise.

    Reps faster than min or slower than max are flagged as bad_tempo.
    Asymmetric bounds allow tighter control than a symmetric deviation threshold.
    """
    bounds = config["baseline"]["tempo_bounds_s"][exercise]
    return (float(bounds[0]), float(bounds[1]))


# ── Core flagger ──────────────────────────────────────────────────────────────

def flag_reps_baseline(
    features_df: pd.DataFrame,
    exercise: str,
    config: dict | None = None,
) -> list[RepFlag]:
    """Apply baseline threshold rules to flag low-quality reps.

    Args:
        features_df: DataFrame from extract_rep_features().  Must contain columns:
                     rep_id, rom_proxy_max, tempo_s, tempo_deviation,
                     conf_mean, conf_min.
        exercise:    Exercise identifier ('wall_slide' or 'band_er_side').
        config:      Project config dict.  If None, loaded from config.yaml.

    Returns:
        One RepFlag per row in features_df (same order).

    Raises:
        KeyError: If required config sections are missing.
    """
    if config is None:
        config = get_config()

    rom_cutoff: float = config["baseline"]["rom_cutoffs"][exercise]
    conf_threshold: float = config["pose"]["confidence_threshold"]
    tempo_min, tempo_max = _get_tempo_bounds(exercise, config)
    flag_unknown_on_low_conf: bool = config["baseline"].get(
        "flag_unknown_on_low_confidence", True
    )

    flags: list[RepFlag] = []

    for _, row in features_df.iterrows():
        rep_id = int(row["rep_id"])
        rom = float(row["rom_proxy_max"])
        tempo_s = float(row["tempo_s"])
        conf_mean = float(row["conf_mean"]) if pd.notna(row["conf_mean"]) else 0.0

        reasons: list[str] = []

        # ── Rule 1: low pose confidence ───────────────────────────────────────
        low_confidence = conf_mean < conf_threshold
        if low_confidence:
            reasons.append("pose_low_confidence")

        # ── Rule 2: partial ROM (only when confidence is sufficient) ──────────
        if not low_confidence and pd.notna(rom) and rom < rom_cutoff:
            reasons.append("rom_below_cutoff")

        # ── Rule 3: bad tempo (outside acceptable range) ──────────────────────
        if pd.notna(tempo_s) and (tempo_s < tempo_min or tempo_s > tempo_max):
            reasons.append("tempo_out_of_range")

        # ── Determine predicted label ─────────────────────────────────────────
        if flag_unknown_on_low_conf and low_confidence:
            predicted_label = "unknown"
        elif "tempo_out_of_range" in reasons:
            predicted_label = "bad_tempo"
        elif "rom_below_cutoff" in reasons:
            predicted_label = "bad_rom_partial"
        else:
            predicted_label = "good"

        flagged = predicted_label != "good"
        confidence_level = "low" if low_confidence else "high"

        flags.append(
            RepFlag(
                rep_id=rep_id,
                flagged=flagged,
                predicted_label=predicted_label,
                reasons=reasons,
                rom_proxy_max=rom,
                tempo_s=tempo_s,
                confidence_level=confidence_level,
            )
        )

    return flags


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize_flags(flags: list[RepFlag]) -> dict:
    """Compute summary statistics over a list of RepFlags.

    Args:
        flags: List of RepFlag dataclasses from flag_reps_baseline().

    Returns:
        Dict with keys:
            total_reps, flagged_count, good_count,
            label_distribution: {label: count},
            reasons_distribution: {reason: count}
    """
    label_dist: dict[str, int] = {}
    reasons_dist: dict[str, int] = {}

    for f in flags:
        label_dist[f.predicted_label] = label_dist.get(f.predicted_label, 0) + 1
        for r in f.reasons:
            reasons_dist[r] = reasons_dist.get(r, 0) + 1

    flagged_count = sum(1 for f in flags if f.flagged)
    return {
        "total_reps": len(flags),
        "flagged_count": flagged_count,
        "good_count": len(flags) - flagged_count,
        "label_distribution": label_dist,
        "reasons_distribution": reasons_dist,
    }


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_flags(flags: list[RepFlag], out_path: Path) -> None:
    """Save RepFlag list to JSON.

    Args:
        flags:    List of RepFlag dataclasses.
        out_path: Destination path; parent directories created if absent.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "rep_id": f.rep_id,
            "flagged": f.flagged,
            "predicted_label": f.predicted_label,
            "reasons": f.reasons,
            "rom_proxy_max": f.rom_proxy_max,
            "tempo_s": f.tempo_s,
            "confidence_level": f.confidence_level,
        }
        for f in flags
    ]
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)


def load_flags(path: Path) -> list[RepFlag]:
    """Load RepFlag list from JSON produced by save_flags().

    Args:
        path: Path to flags.json.

    Returns:
        List of RepFlag dataclasses.
    """
    with open(path) as fh:
        records = json.load(fh)
    return [
        RepFlag(
            rep_id=r["rep_id"],
            flagged=r["flagged"],
            predicted_label=r["predicted_label"],
            reasons=r["reasons"],
            rom_proxy_max=r["rom_proxy_max"],
            tempo_s=r["tempo_s"],
            confidence_level=r["confidence_level"],
        )
        for r in records
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply baseline threshold rules to flag low-quality reps."
    )
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument(
        "--exercise", required=True, choices=["wall_slide", "band_er_side"]
    )
    parser.add_argument("--output", required=True, help="Path for output flags.json")
    args = parser.parse_args()

    from src.pipeline.features import load_features

    features_df = load_features(Path(args.features))
    flags = flag_reps_baseline(features_df, args.exercise)
    save_flags(flags, Path(args.output))

    summary = summarize_flags(flags)
    print(f"Total reps  : {summary['total_reps']}")
    print(f"Flagged     : {summary['flagged_count']}")
    print(f"Good        : {summary['good_count']}")
    print(f"Labels      : {summary['label_distribution']}")
    print(f"Reasons     : {summary['reasons_distribution']}")
    print(f"Saved to    : {args.output}")
