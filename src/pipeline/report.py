"""Session summary report and flagged-rep review page generator.

Reads session artifacts (meta.json, features.csv, flags.json, clips/) and
renders Jinja2 templates into self-contained HTML files.

Usage:
    python src/pipeline/report.py --session-dir data/processed/<session>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.pipeline.baseline import load_flags
from src.utils.config import get_config

_TEMPLATES_DIR = Path(__file__).parent / "templates"


# ── Template rendering ────────────────────────────────────────────────────────

def render_report(template_path: Path, context: dict, output_path: Path) -> Path:
    """Render a Jinja2 template with the given context.

    Args:
        template_path: Path to the .html Jinja2 template file.
        context:       Dict of variables passed to the template.
        output_path:   Destination .html file.

    Returns:
        Path to the written file.
    """
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=True,
    )
    template = env.get_template(template_path.name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.render(**context), encoding="utf-8")
    return output_path


# ── Context builders ──────────────────────────────────────────────────────────

def _safe_float(val: object, default: float = float("nan")) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_session_context(session_dir: Path) -> dict:
    """Load all session artifacts and build a shared template context."""
    session_dir = Path(session_dir)

    with open(session_dir / "meta.json") as fh:
        meta = json.load(fh)

    features_df = pd.read_csv(session_dir / "features.csv")
    flags = load_flags(session_dir / "flags.json")
    cfg = get_config()

    # ── Reps CSV for timing info (features.csv has tempo but not start/end times) ──
    reps_df = pd.read_csv(session_dir / "reps.csv")
    # Merge timing into features context
    timing = {
        int(r["rep_id"]): (float(r["start_time_s"]), float(r["end_time_s"]))
        for _, r in reps_df.iterrows()
    }

    # Summary
    label_dist: dict[str, int] = {}
    for flag in flags:
        label_dist[flag.predicted_label] = label_dist.get(flag.predicted_label, 0) + 1
    flagged_count = sum(1 for f in flags if f.flagged)
    summary = {
        "total_reps": len(flags),
        "flagged_count": flagged_count,
        "good_count": len(flags) - flagged_count,
        "label_distribution": label_dist,
    }

    rom_vals = features_df["rom_proxy_max"].dropna()
    tempo_vals = features_df["tempo_s"].dropna()
    rom_median = float(rom_vals.median()) if len(rom_vals) else 0.0
    tempo_median = float(tempo_vals.median()) if len(tempo_vals) else 0.0
    flagged_pct = round(100 * flagged_count / max(len(flags), 1))

    # Build flagged reps context with timing
    feat_by_id = {int(row["rep_id"]): row for _, row in features_df.iterrows()}
    clips_dir = session_dir / "clips"
    flagged_reps = []
    for flag in flags:
        if not flag.flagged:
            continue
        feat = feat_by_id.get(flag.rep_id)
        start_s, end_s = timing.get(flag.rep_id, (None, None))
        clip_path = clips_dir / f"rep_{flag.rep_id:02d}.mp4"
        thumb_path = clips_dir / f"rep_{flag.rep_id:02d}_thumb.jpg"
        flagged_reps.append(
            {
                "rep_id": flag.rep_id,
                "predicted_label": flag.predicted_label,
                "reasons": flag.reasons,
                "rom_proxy_max": _safe_float(feat["rom_proxy_max"]) if feat is not None else float("nan"),
                "tempo_s": _safe_float(feat["tempo_s"]) if feat is not None else float("nan"),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": (end_s - start_s) if start_s is not None and end_s is not None else None,
                "clip_path": str(clip_path) if clip_path.exists() else None,
                "thumbnail_path": str(thumb_path) if thumb_path.exists() else None,
            }
        )

    return {
        "meta": meta,
        "summary": summary,
        "features_df": features_df,
        "flags": flags,
        "flagged_reps": flagged_reps,
        "rom_median": rom_median,
        "tempo_median": tempo_median,
        "flagged_pct": flagged_pct,
        "safety_note": cfg["report"]["safety_note"],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    session_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate HTML session summary report.

    Reads meta.json, features.csv, flags.json, and clips/ from session_dir.
    Writes report.html into session_dir (or output_path if given).

    Args:
        session_dir:  Processed session directory.
        output_path:  Override output path (optional).

    Returns:
        Path to generated report.html.
    """
    session_dir = Path(session_dir)
    output_path = Path(output_path) if output_path else session_dir / "report.html"

    ctx = _load_session_context(session_dir)
    template = _TEMPLATES_DIR / "session_report.html"
    return render_report(template, ctx, output_path)


def generate_review_page(
    session_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate interactive review page with flagged rep clips.

    Args:
        session_dir:  Processed session directory.
        output_path:  Override output path (optional).

    Returns:
        Path to generated review.html.
    """
    session_dir = Path(session_dir)
    output_path = Path(output_path) if output_path else session_dir / "review.html"

    ctx = _load_session_context(session_dir)
    template = _TEMPLATES_DIR / "review_page.html"
    return render_report(template, ctx, output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate session report + review page."
    )
    parser.add_argument("--session-dir", required=True,
                        help="Path to processed session directory")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--review-out", default=None)
    args = parser.parse_args()

    sd = Path(args.session_dir)
    r = generate_report(sd, Path(args.report_out) if args.report_out else None)
    rv = generate_review_page(sd, Path(args.review_out) if args.review_out else None)
    print(f"Report : {r}")
    print(f"Review : {rv}")
