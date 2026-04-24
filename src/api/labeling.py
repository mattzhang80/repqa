"""Minimal rep labeling server.

Serves a browser UI for labeling detected reps with exercise-specific labels.
Labels are appended to data/labels/labels.csv in real time.

Start with:
    uvicorn src.api.labeling:app --reload

Endpoints:
    GET  /                         → labeling UI (redirect to first unlabeled session)
    GET  /sessions                 → list all processed sessions
    GET  /sessions/{id}/reps       → list reps + clips + allowed labels for session
    POST /sessions/{id}/reps/{rep} → save label {label: str}
    GET  /export                   → download labels.csv
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils.config import get_config

app = FastAPI(title="RepQA Labeling Tool")

_PROCESSED_DIR = Path("data/processed")
_LABELS_PATH = Path("data/labels/labels.csv")
_LABELS_COLS = ["session_id", "rep_id", "exercise", "label", "labeler", "timestamp"]

# Serve processed session files (clips, thumbnails, videos) as static files
if _PROCESSED_DIR.exists():
    app.mount("/data/processed", StaticFiles(directory=str(_PROCESSED_DIR)), name="processed")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_sessions() -> list[dict]:
    """Return all sessions in data/processed/ that have flags.json."""
    sessions = []
    if not _PROCESSED_DIR.exists():
        return sessions
    for d in sorted(_PROCESSED_DIR.iterdir()):
        if d.is_dir() and (d / "flags.json").exists() and (d / "meta.json").exists():
            with open(d / "meta.json") as fh:
                meta = json.load(fh)
            sessions.append(
                {
                    "session_id": d.name,
                    "exercise": meta.get("exercise", "unknown"),
                    "display_name": meta.get("display_name", d.name),
                    "user_id": meta.get("user_id", "unknown"),
                    "reps_detected": meta.get("reps_detected", 0),
                }
            )
    return sessions


def _load_existing_labels() -> dict[tuple[str, int], str]:
    """Return {(session_id, rep_id): label} from labels.csv."""
    existing: dict[tuple[str, int], str] = {}
    if not _LABELS_PATH.exists():
        return existing
    with open(_LABELS_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            existing[(row["session_id"], int(row["rep_id"]))] = row["label"]
    return existing


def _write_label(
    session_id: str,
    rep_id: int,
    exercise: str,
    label: str,
    labeler: str = "human",
) -> None:
    """Append a label to labels.csv, replacing any existing label for this rep."""
    _LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read all existing rows, replace matching (session_id, rep_id) if present
    rows: list[dict] = []
    if _LABELS_PATH.exists():
        with open(_LABELS_PATH, newline="") as fh:
            rows = list(csv.DictReader(fh))

    # Remove existing label for this rep (dedup)
    rows = [r for r in rows if not (r["session_id"] == session_id and int(r["rep_id"]) == rep_id)]

    rows.append(
        {
            "session_id": session_id,
            "rep_id": str(rep_id),
            "exercise": exercise,
            "label": label,
            "labeler": labeler,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    with open(_LABELS_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LABELS_COLS)
        writer.writeheader()
        writer.writerows(rows)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=RedirectResponse)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/sessions")


@app.get("/sessions")
async def list_sessions() -> list[dict]:
    return _list_sessions()


@app.get("/sessions/{session_id}/reps")
async def get_session_reps(session_id: str) -> dict:
    session_dir = _PROCESSED_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, f"Session '{session_id}' not found")

    with open(session_dir / "meta.json") as fh:
        meta = json.load(fh)
    exercise = meta["exercise"]

    cfg = get_config()
    allowed_labels: list[str] = cfg["exercises"][exercise]["labels"]

    import pandas as pd
    from src.pipeline.baseline import load_flags

    reps_df = pd.read_csv(session_dir / "reps.csv")
    flags = load_flags(session_dir / "flags.json")
    flag_by_id = {f.rep_id: f for f in flags}
    existing = _load_existing_labels()

    video_url = f"/data/processed/{session_id}/video.mp4"
    reps = []
    for _, row in reps_df.iterrows():
        rep_id = int(row["rep_id"])
        flag = flag_by_id.get(rep_id)
        clips_dir = session_dir / "clips"
        clip_rel = f"/data/processed/{session_id}/clips/rep_{rep_id:02d}.mp4"
        thumb_rel = f"/data/processed/{session_id}/clips/rep_{rep_id:02d}_thumb.jpg"
        reps.append(
            {
                "rep_id": rep_id,
                "start_time_s": float(row["start_time_s"]),
                "end_time_s": float(row["end_time_s"]),
                "predicted_label": flag.predicted_label if flag else None,
                "flagged": flag.flagged if flag else False,
                "existing_label": existing.get((session_id, rep_id)),
                "clip_url": clip_rel if (clips_dir / f"rep_{rep_id:02d}.mp4").exists() else None,
                "thumbnail_url": thumb_rel if (clips_dir / f"rep_{rep_id:02d}_thumb.jpg").exists() else None,
            }
        )

    return {
        "session_id": session_id,
        "exercise": exercise,
        "display_name": meta.get("display_name", exercise),
        "allowed_labels": allowed_labels,
        "video_url": video_url,
        "reps": reps,
    }


class LabelRequest(BaseModel):
    label: str
    labeler: str = "human"


@app.post("/sessions/{session_id}/reps/{rep_id}")
async def save_label(session_id: str, rep_id: int, body: LabelRequest) -> dict:
    session_dir = _PROCESSED_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, f"Session '{session_id}' not found")

    with open(session_dir / "meta.json") as fh:
        meta = json.load(fh)
    exercise = meta["exercise"]

    cfg = get_config()
    allowed = cfg["exercises"][exercise]["labels"]
    if body.label not in allowed:
        raise HTTPException(422, f"Label '{body.label}' not in allowed set: {allowed}")

    _write_label(session_id, rep_id, exercise, body.label, body.labeler)
    return {"status": "ok", "session_id": session_id, "rep_id": rep_id, "label": body.label}


@app.post("/sessions/{session_id}/reps/{rep_id}/review")
async def save_review_decision(session_id: str, rep_id: int, body: dict) -> dict:
    """Accept confirm/dismiss decisions from the review page."""
    return {"status": "ok", "decision": body.get("decision")}


@app.get("/export")
async def export_labels() -> FileResponse:
    if not _LABELS_PATH.exists():
        raise HTTPException(404, "No labels file yet — label some reps first.")
    return FileResponse(
        str(_LABELS_PATH),
        media_type="text/csv",
        filename="labels.csv",
    )


# ── Labeling UI ───────────────────────────────────────────────────────────────

@app.get("/ui/{session_id}", response_class=HTMLResponse)
async def labeling_ui(session_id: str) -> HTMLResponse:
    """Serve the in-browser labeling interface for a session."""
    session_dir = _PROCESSED_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, f"Session '{session_id}' not found")

    template_path = Path(__file__).parent.parent / "pipeline" / "templates" / "labeling.html"
    if not template_path.exists():
        raise HTTPException(500, "Labeling template not found")

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=True)
    tmpl = env.get_template("labeling.html")

    data = await get_session_reps(session_id)
    html = tmpl.render(session=data)
    return HTMLResponse(html)
