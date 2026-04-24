.PHONY: help install api frontend dev test lint pipeline train eval clean

help:
	@echo "RepQA — common tasks"
	@echo ""
	@echo "  install     Install Python deps + frontend deps"
	@echo "  api         Run FastAPI on :8000 (hot reload)"
	@echo "  frontend    Run Next.js on :3000 (hot reload)"
	@echo "  dev         Run api + frontend together (needs GNU make's parallelism)"
	@echo "  test        Run full pytest suite"
	@echo "  pipeline    Run pipeline on all raw videos in data/raw/matthew/"
	@echo "  train       Train both models (+ personalization)"
	@echo "  eval        Run Phase 16 evaluation + build figures"
	@echo "  clean       Remove generated artefacts (models/, reports/, processed/)"

install:
	python3 -m pip install -e .
	cd frontend && npm install

api:
	uvicorn src.api.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

# Run both in parallel — each subtask prints to stdout.  Ctrl-C stops both.
dev:
	@$(MAKE) -j2 api frontend

test:
	python3 -m pytest tests -q

pipeline:
	python3 scripts/batch_process.py

train:
	python3 scripts/train_model.py --personalize

eval:
	python3 scripts/eval_all.py

clean:
	rm -rf data/models data/reports
	rm -rf data/processed/_pipeline_cache 2>/dev/null || true
	find data/processed -name '__pycache__' -type d -exec rm -rf {} +
	@echo "Cleaned models/ and reports/.  Processed sessions left intact."
