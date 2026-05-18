# AGENTS.md

## Quick orientation
- Pipelines live in `preprocessing/`; data products are written under `data/`.
- Reusable station logic: `preprocessing/stations/*pipeline.py`; CLI entrypoints: `preprocessing/stations/extract_*.py`.
- Alert extraction: `preprocessing/emergency_alerts/extract_alerts_data.py` (docling OCR + Gemini) writes `data/emergency_alerts/processed/alerts_data.csv`.
- Station downloader is a Tkinter GUI: run `python data/stations/raw/ucr_uema_data_downloader.py` directly (not `-m`).

## Setup
- `pip install -r requirements.txt`
- Optional dev tools: `pip install pytest black flake8 mypy`

## Data pipeline (run from repo root, order matters)
1. `python data/stations/raw/ucr_uema_data_downloader.py`
2. `python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY`
3. `python -m preprocessing.stations.extract_stations_data_silver_layer`
4. `python -m preprocessing.stations.extract_stations_data_gold_layer`
5. `python -m preprocessing.stations.extract_global_gold_layer`

## Alert extraction notes
- Input PDFs: `data/emergency_alerts/raw/`; output CSV: `data/emergency_alerts/processed/alerts_data.csv`.
- `--google-api-key` is required; never commit keys.
- OCR is CPU-only (`CUDA_VISIBLE_DEVICES` is disabled).

## Data/legal note
- Read `NOTICE.md` before using or sharing datasets.
