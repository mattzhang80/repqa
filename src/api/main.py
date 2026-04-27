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

# Force the non-interactive Agg backend before any pyplot import anywhere in
# the process tree. The pipeline runs in a background thread, and the default
# macOS backend can't create figures off the main thread.
import matplotlib
matplotlib.use("Agg")

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

# Optional ML — load lazily to keep the API importable even if models
# haven't been trained yet.
try:
    from src.ml.train_logreg import load_model as _load_ml_model  # type: ignore
except Exception:  # noqa: BLE001
    _load_ml_model = None  # type: ignore[assignment]

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
_REPORTS_DIR = Path("data/reports")
_MODELS_DIR = Path("data/models")

# Static files (clips, thumbnails, plots)
if _PROCESSED_DIR.exists():
    app.mount(
        "/data/processed",
        StaticFiles(directory=str(_PROCESSED_DIR)),
        name="processed",
    )

if _REPORTS_DIR.exists():
    app.mount(
        "/data/reports",
        StaticFiles(directory=str(_REPORTS_DIR)),
        name="reports",
    )


# ── Per-exercise model cache (lazy load) ──────────────────────────────────────

_model_cache: dict[str, dict | None] = {}
_model_cache_lock = threading.Lock()


def _get_model(exercise: str) -> dict | None:
    """Return loaded model artifacts for an exercise, or None if no model
    is saved.  Cached for the lifetime of the process; call
    ``_clear_model_cache()`` to force a reload after retraining."""
    if _load_ml_model is None:
        return None
    with _model_cache_lock:
        if exercise in _model_cache:
            return _model_cache[exercise]
        model_dir = _MODELS_DIR / exercise
        if not (model_dir / "model.pkl").exists():
            _model_cache[exercise] = None
            return None
        try:
            _model_cache[exercise] = _load_ml_model(model_dir)
        except Exception:  # noqa: BLE001 — bad pickle, missing deps, etc.
            _model_cache[exercise] = None
        return _model_cache[exercise]


def _clear_model_cache() -> None:
    """Drop cached models so the next request re-reads from disk.  Used
    primarily by tests that mutate ``data/models/``."""
    with _model_cache_lock:
        _model_cache.clear()


def _predict_on_features(
    exercise: str, feat_rows_by_rep: dict[int, dict], user_id: str | None = None
) -> dict[int, dict]:
    """Run the trained model on a set of per-rep feature rows.

    If the model was trained with personalization enabled (``feature_cols``
    contains ``*_z`` / ``*_pct`` columns) and a baseline JSON exists at
    ``data/models/baselines/<user_id>_<exercise>.json``, apply
    personalization on the fly so the model receives the feature shape
    it expects.  Gracefully returns ``{}`` when no model is trained, or
    a model requires personalization but no baseline is available for
    this user.
    """
    artifacts = _get_model(exercise)
    if artifacts is None:
        return {}

    feat_cols: list[str] = artifacts["feature_cols"]
    threshold: float = artifacts["threshold"]
    model = artifacts["model"]
    scaler = artifacts["scaler"]

    # If the model expects personalized features, synthesize them from the
    # saved baseline before building the feature matrix.
    needs_pers = any(c.endswith("_z") or c.endswith("_pct") for c in feat_cols)
    if needs_pers and user_id is not None:
        try:
            import pandas as pd
            from src.ml.personalize import (
                apply_personalization as _apply_pers,
                load_user_baseline as _load_baseline,
            )
            baseline = _load_baseline(user_id, exercise)
            if baseline is not None:
                rows_df = pd.DataFrame.from_dict(feat_rows_by_rep, orient="index")
                rows_df = _apply_pers(rows_df, baseline)
                feat_rows_by_rep = {
                    int(rid): row.to_dict()
                    for rid, row in rows_df.iterrows()
                }
        except Exception:  # noqa: BLE001 — missing deps, malformed baseline
            pass

    usable: list[int] = []
    rows: list[list[float]] = []
    for rid, feat in feat_rows_by_rep.items():
        try:
            rows.append([float(feat[c]) for c in feat_cols])
            usable.append(rid)
        except (KeyError, TypeError, ValueError):
            # Missing / non-numeric feature → skip this rep silently; the
            # frontend falls back to the baseline flag for it.
            continue

    if not usable:
        return {}

    import numpy as np
    X = np.asarray(rows, dtype=float)
    X_scaled = scaler.transform(X)
    prob = np.asarray(model.predict_proba(X_scaled))[:, 1]
    return {
        rid: {
            "prob_bad": float(prob[i]),
            "predicted_bad": bool(prob[i] >= threshold),
            "threshold": float(threshold),
        }
        for i, rid in enumerate(usable)
    }

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

    # Pre-build feature dicts keyed by rep_id, then batch-predict once.
    feat_rows_by_rep: dict[int, dict] = {}
    for _, frow in feat_df.iterrows():
        rid = int(frow["rep_id"])
        feat = frow.to_dict()
        for k in ("session_id", "exercise", "user_id"):
            feat.pop(k, None)
        feat_rows_by_rep[rid] = feat

    model_preds = _predict_on_features(
        meta["exercise"],
        feat_rows_by_rep,
        user_id=meta.get("user_id"),
    )

    reps_out = []
    for _, row in reps_df.iterrows():
        rid = int(row["rep_id"])
        flag = flag_by_id.get(rid)
        feat_row = feat_rows_by_rep.get(rid, {})

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
            "model_prediction": model_preds.get(rid),
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


# ── Health + reports ─────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Readiness probe for the API and its supporting artifacts.

    Reports whether each data directory exists and which trained models are
    currently loadable.  Never raises — a 200 with ``status: degraded`` is
    the expected response when pipelines haven't been run yet.
    """
    models_present: dict[str, bool] = {}
    if _MODELS_DIR.exists():
        for d in _MODELS_DIR.iterdir():
            if d.is_dir() and (d / "model.pkl").exists():
                models_present[d.name] = True

    n_sessions = (
        sum(1 for d in _PROCESSED_DIR.iterdir() if d.is_dir())
        if _PROCESSED_DIR.exists() else 0
    )
    status = "ok" if n_sessions > 0 else "degraded"
    return {
        "status": status,
        "sessions": n_sessions,
        "models_available": sorted(models_present.keys()),
        "reports_dir_present": _REPORTS_DIR.exists(),
        "processed_dir_present": _PROCESSED_DIR.exists(),
    }


@app.get("/reports/metrics")
async def list_reports() -> dict:
    """Return all saved per-exercise Phase 16 metrics JSONs, keyed by
    exercise.  Empty dict if the reports directory is missing."""
    out: dict[str, dict] = {}
    if not _REPORTS_DIR.exists():
        return {"metrics": out}
    for p in sorted(_REPORTS_DIR.glob("metrics_*.json")):
        exercise = p.stem.replace("metrics_", "", 1)
        try:
            with open(p) as fh:
                out[exercise] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
    return {"metrics": out}


@app.get("/reports/metrics/{exercise}")
async def report_for(exercise: str) -> dict:
    """Return Phase 16 metrics for a specific exercise."""
    p = _REPORTS_DIR / f"metrics_{exercise}.json"
    if not p.exists():
        raise HTTPException(404, f"No metrics for '{exercise}'")
    with open(p) as fh:
        return json.load(fh)


@app.get("/reports/figures")
async def list_figures() -> dict:
    """List available Phase 16 figure URLs, grouped by filename stem
    (roc_curve, pr_curve, …).  Missing figures are silently absent."""
    figs: dict[str, str] = {}
    figs_dir = _REPORTS_DIR / "figures"
    if figs_dir.exists():
        for p in sorted(figs_dir.glob("*.png")):
            figs[p.stem] = f"/data/reports/figures/{p.name}"
    return {"figures": figs}


# ── Re-export labeling routes ─────────────────────────────────────────────────

app.get("/sessions", tags=["labeling"])(list_sessions)
app.get("/sessions/{session_id}/reps", tags=["labeling"])(get_session_reps)
app.post("/sessions/{session_id}/reps/{rep_id}", tags=["labeling"])(save_label)
app.post("/sessions/{session_id}/reps/{rep_id}/review", tags=["labeling"])(save_review_decision)
app.get("/export", tags=["labeling"])(export_labels)
app.get("/ui/{session_id}", tags=["labeling"])(labeling_ui)
