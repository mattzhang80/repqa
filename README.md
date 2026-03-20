# repqa

Video analysis for home PT shoulder exercises. Point it at a video, get back rep counts, form flags, and a session summary.

Supports two exercises right now: wall slides and band external rotation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

You'll also need FFmpeg installed (`brew install ffmpeg`).

## Usage

```bash
python scripts/run_pipeline.py \
  --video data/raw/<user>/your_video.mov \
  --exercise wall_slide \
  --user-id <user>
```

Output lands in `data/processed/<session_id>/` — poses, rep boundaries, features, and a flags file.

## Web UI

```bash
# Terminal 1 — API (port 8000)
source .venv/bin/activate
uvicorn src.api.main:app --reload

# Terminal 2 — Frontend (port 3000)
cd frontend && npm run dev
```

Open http://localhost:3000 — dashboard lists all processed sessions. Use the upload page to process new videos through the browser instead of the CLI.

## Filming

- **Wall slide**: side view, camera at shoulder height, 6–10 ft away
- **Band ER**: front view, camera at chest height, 6–10 ft away

Landscape orientation for both.
