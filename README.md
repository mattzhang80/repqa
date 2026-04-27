# RepQA

Smartphone video → rep counts, form flags, ROM trends, and a one-page session summary for home PT shoulder exercises.

Two exercises are supported end-to-end: **wall slide** (side view) and **band external rotation at side** (front view). The stack spans a Python pipeline (MediaPipe + scipy + scikit-learn), a FastAPI service, and a Next.js frontend.

> **Safety note** — shown on every user-facing screen: *stop if pain >3/10 or if you feel instability/apprehension.*

## What it does

1. **Preprocess** — FFmpeg normalizes to 30 fps / fixed width.
2. **Pose** — MediaPipe Tasks API extracts 33 2D landmarks per frame.
3. **Segment** — per-exercise 1-D signal + Savitzky-Golay + peak detection, with an **amplitude-ratio ghost-rep filter** and **active-window detection** that trim setup/teardown.
4. **Features** — per-rep ROM proxy, tempo deviation, pose confidence, elbow-drift.
5. **Flag (baseline)** — hand-tuned thresholds with human-readable reasons.
6. **Flag (model)** — per-exercise L2 logistic regression (StandardScaler + StratifiedGroupKFold CV + OOF threshold + class-balanced weights).
7. **Personalize** — per-user, per-exercise baseline from their good reps, producing z-score and percentile features.
8. **Evaluate** — cluster-bootstrap 95% CIs on AUC / precision / recall, plus a forest plot.
9. **Report** — one-page HTML per session; a reports page aggregating held-out test metrics + figures.

## Setup

```bash
# System: FFmpeg + Node (for the frontend)
brew install ffmpeg node

# Python
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Frontend
cd frontend && npm install && cd ..
```

## Quick start

```bash
make help                 # list everything
make api                  # FastAPI on :8000
make frontend             # Next.js on :3000
make dev                  # both, in parallel

make pipeline             # batch-process data/raw/matthew/*.mov
make train                # train per-exercise models (+ personalization)
make eval                 # held-out test metrics + figures
make test                 # full pytest suite
```

Then open http://localhost:3000:

- `/` — dashboard, one card per session
- `/upload` — upload a phone clip, watch the pipeline run
- `/sessions/[id]` — session detail: meta, segmentation plot, every rep
- `/sessions/[id]/review` — flagged reps with inline clips, baseline label + model probability side-by-side
- `/sessions/[id]/label` — keyboard-driven labeling UI (1–N for labels, arrows to navigate)
- `/reports` — held-out test metrics with 95% cluster-bootstrap CIs and all evaluation figures

## CLI-only flow

```bash
# 1. Process one video
python scripts/run_pipeline.py \
    --video data/raw/matthew/wall_slide_good_01.mov \
    --exercise wall_slide \
    --user-id matthew

# 2. Batch process every raw video
python scripts/batch_process.py

# 3. Auto-label every detected rep from filename → data/labels/labels.csv
python scripts/generate_labels.py

# 4. Assemble + split dataset (stratified group split by exercise × label)
python src/ml/dataset.py \
    --features-dir data/features \
    --labels data/labels/labels.csv \
    --out-dir data/features

# 5. Train per-exercise models with personalization
python scripts/train_model.py --personalize

# 6. Evaluate with bootstrap CIs and figures
python scripts/eval_all.py
```

## Filming

| Exercise | Camera | View | Frame |
|---|---|---|---|
| wall_slide | shoulder height, 6–10 ft away | side | head → hips, both arms, wall |
| band_er_side | chest height, 6–10 ft away | front | head → waist, band + towel visible |

Landscape for both. Stand still for ~3 s at start and end so active-window detection has a clean boundary to find.

## Layout

```
src/
  pipeline/      preprocess, pose, segment, features, baseline flags, clipper, report
  ml/            dataset split, training, personalization, bootstrap, evaluation
  api/           FastAPI (upload + jobs + sessions + reports + labeling)
  utils/         config, geometry, video, plotting
frontend/src/app/
  page.tsx       dashboard
  upload/        upload form + pipeline job tracker
  sessions/[id]/ detail, review, label
  reports/       held-out test metrics + figures
scripts/         run_pipeline, batch_process, generate_labels, train_model, eval_all
data/            raw videos, processed sessions, features, labels, models, reports
tests/           pipeline, ml, api — pytest suite
config.yaml      exercise registry, segmentation params, model hyperparams
docs/CAPTURE_GUIDE.md   per-exercise filming instructions
```

## Tests

```bash
make test         # or: python3 -m pytest tests -q
```

The pipeline, ML, and API layers each have their own test package. Fixtures include a short real video and synthetic pose/feature DataFrames.

## Not in scope

No diagnosis, no medical claims, no real-time feedback, no deep learning — the spec deliberately constrains this to interpretable features + a small linear model. Deployment (Docker, cloud) is not implemented.

## AI Usage Disclosure

This project was created with assistance from AI tools. The content has been reviewed and edited by a human. For more information on the extent and nature of AI usage, please contact the author.
