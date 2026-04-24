"""Batch-process every raw video in data/raw/matthew not yet in data/processed.

Uses the video stem as the session_id (lowercase) to match the existing
naming convention.  Skips anything already processed and the elbow-drift
videos (exercise dropped from the project scope).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_pipeline import run_pipeline

RAW_DIR = Path("data/raw/matthew")
PROCESSED_DIR = Path("data/processed")


def exercise_from_stem(stem: str) -> str | None:
    if stem.startswith("wall_slide_"):
        return "wall_slide"
    if stem.startswith("band_er_side_"):
        return "band_er_side"
    return None


def main() -> int:
    videos = sorted(p for p in RAW_DIR.iterdir()
                    if p.suffix.lower() in {".mov", ".mp4"})
    todo: list[tuple[Path, str, str]] = []
    skipped: list[tuple[Path, str]] = []
    for v in videos:
        stem = v.stem
        if "elbow_drift" in stem:
            skipped.append((v, "elbow_drift dropped"))
            continue
        ex = exercise_from_stem(stem)
        if ex is None:
            skipped.append((v, "unknown exercise prefix"))
            continue
        if (PROCESSED_DIR / stem).exists():
            skipped.append((v, "already processed"))
            continue
        todo.append((v, ex, stem))

    print(f"Videos found     : {len(videos)}")
    print(f"Already processed: {sum(1 for _, r in skipped if r == 'already processed')}")
    print(f"Skipped (drift)  : {sum(1 for _, r in skipped if r == 'elbow_drift dropped')}")
    print(f"To process       : {len(todo)}")
    print("")

    failures: list[tuple[str, str]] = []
    for i, (video, exercise, session_id) in enumerate(todo, 1):
        t0 = time.time()
        print(f"=== [{i}/{len(todo)}] {video.name} ({exercise}) ===")
        try:
            run_pipeline(
                video_path=video,
                exercise=exercise,
                user_id="matthew",
                session_id=session_id,
            )
            dt = time.time() - t0
            print(f"    [{session_id}] done in {dt:.1f}s\n")
        except Exception as e:
            traceback.print_exc()
            failures.append((session_id, str(e)))
            print(f"    [{session_id}] FAILED: {e}\n")

    print("=" * 60)
    print(f"Processed: {len(todo) - len(failures)}/{len(todo)}")
    if failures:
        print("Failures:")
        for sid, err in failures:
            print(f"  - {sid}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
