"""Auto-label every detected rep in every processed session from its filename.

For this project, every video in data/raw/matthew is a pure session — all
reps share the label encoded in the filename (e.g. ``wall_slide_bad_tempo_04``
→ every detected rep is ``bad_tempo``).  This script walks every session in
``data/processed/``, parses the exercise + label from the session_id, and
writes one row per rep in ``reps.csv`` to ``data/labels/labels.csv``.

Sessions with patterns not matching an allowed label for their exercise are
skipped (e.g. the dropped ``band_er_side_bad_elbow_drift_mild_*`` sessions).
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config import get_config

LABELS_PATH = Path("data/labels/labels.csv")
LABELS_COLS = ["session_id", "rep_id", "exercise", "label", "labeler", "timestamp"]
PROCESSED_DIR = Path("data/processed")


# Known exercise prefixes sorted longest-first so ``band_er_side`` matches
# before ``band_er`` (not that we have the latter, but this is the safe order).
_EXERCISE_PREFIXES = ["band_er_side", "wall_slide"]


def parse_session_id(session_id: str) -> tuple[str, str] | None:
    """Return (exercise, label) parsed from a session_id, or None to skip.

    Session IDs follow ``<exercise>_<label>_<NN>`` where label may itself
    contain underscores (e.g. ``bad_rom_partial``).
    """
    ex = next((e for e in _EXERCISE_PREFIXES if session_id.startswith(e + "_")), None)
    if ex is None:
        return None
    remainder = session_id[len(ex) + 1 :]          # e.g. 'bad_tempo_04'
    parts = remainder.rsplit("_", 1)               # ['bad_tempo', '04']
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return ex, parts[0]


def main() -> int:
    cfg = get_config()
    allowed_by_exercise = {
        ex: set(meta["labels"]) for ex, meta in cfg["exercises"].items()
    }

    if not PROCESSED_DIR.exists():
        print("No data/processed/ directory found.")
        return 1

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    per_session_counts: list[tuple[str, str, int]] = []

    for session_dir in sorted(PROCESSED_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        reps_csv = session_dir / "reps.csv"
        if not reps_csv.exists():
            skipped.append((session_dir.name, "no reps.csv"))
            continue

        parsed = parse_session_id(session_dir.name)
        if parsed is None:
            skipped.append((session_dir.name, "unparseable session_id"))
            continue
        exercise, label = parsed

        allowed = allowed_by_exercise.get(exercise, set())
        if label not in allowed:
            skipped.append((session_dir.name, f"label '{label}' not in {sorted(allowed)}"))
            continue

        with open(reps_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            rep_ids = [int(r["rep_id"]) for r in reader]

        for rep_id in rep_ids:
            rows.append(
                {
                    "session_id": session_dir.name,
                    "rep_id": str(rep_id),
                    "exercise": exercise,
                    "label": label,
                    "labeler": "auto_from_filename",
                    "timestamp": timestamp,
                }
            )
        per_session_counts.append((session_dir.name, label, len(rep_ids)))

    # De-duplicate on (session_id, rep_id) — keep the last occurrence, which
    # for this script is the freshly-generated row.
    dedup: dict[tuple[str, int], dict] = {}
    for r in rows:
        dedup[(r["session_id"], int(r["rep_id"]))] = r
    final_rows = list(dedup.values())

    with open(LABELS_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABELS_COLS)
        writer.writeheader()
        writer.writerows(final_rows)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"Wrote {len(final_rows)} rows to {LABELS_PATH}")
    print("")
    print("Per-session rep counts:")
    for sid, label, n in per_session_counts:
        print(f"  {sid:<55} {label:<18} {n:>3} reps")
    print("")

    # Label distribution
    from collections import Counter
    by_ex: dict[str, Counter[str]] = {}
    for r in final_rows:
        by_ex.setdefault(r["exercise"], Counter())[r["label"]] += 1
    print("Label distribution:")
    for ex, counter in by_ex.items():
        total = sum(counter.values())
        good = counter.get("good", 0)
        bad = total - good
        print(f"  {ex}: total={total}  good={good}  bad={bad}  "
              f"breakdown={dict(counter)}")

    if skipped:
        print("")
        print(f"Skipped {len(skipped)} session(s):")
        for sid, reason in skipped:
            print(f"  {sid}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
