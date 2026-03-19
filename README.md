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

## Filming

- **Wall slide**: side view, camera at shoulder height, 6–10 ft away
- **Band ER**: front view, camera at chest height, 6–10 ft away

Landscape orientation for both.
