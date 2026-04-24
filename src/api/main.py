"""RepQA unified API server.

Includes all labeling routes plus:
  POST /upload               — accept video + metadata, run pipeline in background
  GET  /jobs/{id}/status     — poll pipeline job status
  GET  /sessions/{id}        — full session detail (meta + reps + features + flags)
  GET  /sessions/{id}/plot   — redirect to segmentation plot
  GET  /sessions/{id}/report — redirect to HTML report

Start with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.utils.config import get_config
from src.api.labeling import (
    _load_existing_labels,
    save_label,
    save_review_decision,
    list_sessions,
    get_session_reps,
    export_labels,
    labeling_ui,
)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RepQA API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROCESSED_DIR = Path("data/processed")
_RAW_UPLOAD_DIR = Path("data/raw/_uploads")

# Static files (clips, thumbnails, plots)
if _PROCESSED_DIR.exists():
    app.mount(
        "/data/processed",
        StaticFiles(directory=str(_PROCESSED_DIR)),
        name="processed",
    )

# ── Job registry (in-memory; single-process) ──────────────────────────────────

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


# ── Background pipeline task ──────────────────────────────────────────────────

def _run_pipeline(job_id: str, video_path: Path, exercise: str, user_id: str) -> None:
    """Run the full pipeline in a background thread and update job state."""
    try:
        _set_job(job_id, state="running", progress="starting pipeline")
        from scripts.run_pipeline import run_pipeline  # type: ignore[import]

        result = run_pipeline(video_path, exercise, user_id)
        session_id = result.get("session_id", "")
        _set_job(job_id, state="done", progress="complete", session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        _set_job(job_id, state="error", error=str(exc))


# ── Upload endpoint ───────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    exercise: str = Form(...),
    user_id: str = Form("user"),
) -> dict:
    """Accept a video file, save it, kick off the pipeline in the background.

    Returns a job_id for polling via GET /jobs/{job_id}/status.
    """
    from src.utils.config import get_config
    cfg = get_config()
    if exercise not in cfg.get("exercises", {}):
        raise HTTPException(422, f"Unknown exercise '{exercise}'")

    _RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(video_file.filename or "video.mp4").suffix or ".mp4"
    dest = _RAW_UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(dest, "wb") as fh:
        shutil.copyfileobj(video_file.file, fh)

    job_id = uuid.uuid4().hex
    _set_job(job_id, state="queued", progress="file received", session_id=None)

    background_tasks.add_task(_run_pipeline, job_id, dest, exercise, user_id)
    return {"job_id": job_id, "status": "queued"}


# ── Job status ────────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str) -> dict:
    """Poll pipeline job status."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return {"job_id": job_id, **job}


# ── Session detail ────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}")
async def session_detail(session_id: str) -> dict:
    """Full session detail: meta + reps + per-rep features + flags."""
    session_dir = _PROCESSED_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, f"Session '{session_id}' not found")

    with open(session_dir / "meta.json") as fh:
        meta = json.load(fh)

    # Reps + features
    import pandas as pd
    reps_df = pd.read_csv(session_dir / "reps.csv") if (session_dir / "reps.csv").exists() else pd.DataFrame()

    feat_path = session_dir / "features.csv"
    feat_df = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()

    from src.pipeline.baseline import load_flags
    flags = load_flags(session_dir / "flags.json") if (session_dir / "flags.json").exists() else []
    flag_by_id = {f.rep_id: f for f in flags}

    existing_labels = _load_existing_labels()
    clips_dir = session_dir / "clips"

    reps_out = []
    for _, row in reps_df.iterrows():
        rid = int(row["rep_id"])
        flag = flag_by_id.get(rid)

        feat_row = feat_df[feat_df["rep_id"] == rid].iloc[0].to_dict() if len(feat_df) > 0 and rid in feat_df["rep_id"].values else {}
        # Remove non-numeric / redundant keys already in meta
        for k in ("session_id", "exercise", "user_id"):
            feat_row.pop(k, None)

        reps_out.append({
            "rep_id": rid,
            "start_time_s": float(row["start_time_s"]),
            "end_time_s": float(row["end_time_s"]),
            "duration_s": round(float(row["end_time_s"]) - float(row["start_time_s"]), 2),
            "features": feat_row,
            "flagged": flag.flagged if flag else False,
            "predicted_label": flag.predicted_label if flag else "good",
            "reasons": flag.reasons if flag else [],
            "confidence_level": flag.confidence_level if flag else "high",
            "existing_label": existing_labels.get((session_id, rid)),
            "clip_url": f"/data/processed/{session_id}/clips/rep_{rid:02d}.mp4"
                if (clips_dir / f"rep_{rid:02d}.mp4").exists() else None,
            "thumbnail_url": f"/data/processed/{session_id}/clips/rep_{rid:02d}_thumb.jpg"
                if (clips_dir / f"rep_{rid:02d}_thumb.jpg").exists() else None,
        })

    flagged = [r for r in reps_out if r["flagged"]]

    cfg = get_config()
    allowed_labels: list[str] = cfg["exercises"][meta["exercise"]]["labels"]

    return {
        "session_id": session_id,
        "meta": meta,
        "reps": reps_out,
        "allowed_labels": allowed_labels,
        "video_url": f"/data/processed/{session_id}/video.mp4"
            if (session_dir / "video.mp4").exists() else None,
        "summary": {
            "total_reps": len(reps_out),
            "flagged_reps": len(flagged),
            "good_reps": len(reps_out) - len(flagged),
            "pct_flagged": round(100 * len(flagged) / max(len(reps_out), 1), 1),
        },
        "plot_url": f"/data/processed/{session_id}/segmentation_plot.png"
            if (session_dir / "segmentation_plot.png").exists() else None,
        "report_url": f"/data/processed/{session_id}/report.html"
            if (session_dir / "report.html").exists() else None,
    }


@app.get("/sessions/{session_id}/plot")
async def session_plot(session_id: str) -> FileResponse:
    p = _PROCESSED_DIR / session_id / "segmentation_plot.png"
    if not p.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(str(p), media_type="image/png")


@app.get("/sessions/{session_id}/report")
async def session_report(session_id: str) -> FileResponse:
    p = _PROCESSED_DIR / session_id / "report.html"
    if not p.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(str(p), media_type="text/html")


# ── Re-export labeling routes ─────────────────────────────────────────────────

app.get("/sessions", tags=["labeling"])(list_sessions)
app.get("/sessions/{session_id}/reps", tags=["labeling"])(get_session_reps)
app.post("/sessions/{session_id}/reps/{rep_id}", tags=["labeling"])(save_label)
app.post("/sessions/{session_id}/reps/{rep_id}/review", tags=["labeling"])(save_review_decision)
app.get("/export", tags=["labeling"])(export_labels)
app.get("/ui/{session_id}", tags=["labeling"])(labeling_ui)
