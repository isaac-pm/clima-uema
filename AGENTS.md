# AGENTS.md

## Commands (run from repo root)
- `pip install -r requirements.txt`
- **Station downloader** (Tkinter GUI): `python data/stations/raw/ucr_uema_data_downloader.py`
- **Alerts extraction** (requires key): `python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key KEY`
- **Station pipeline** (must run in order):
  `python -m preprocessing.stations.extract_stations_data_silver_layer` →
  `python -m preprocessing.stations.extract_stations_data_gold_layer` →
  `python -m preprocessing.stations.extract_global_gold_layer`
- **Experiments**: `python run_experiments.py`
- All commands run from repo root; the alerts pipeline disables CUDA (`CUDA_VISIBLE_DEVICES=""`) internally.

## No lint / typecheck / test infrastructure
This repo has no test files, no pyproject.toml, no mypy/ruff config, no pre-commit, no Makefile. `flake8` is in requirements but has no config. Do not assume any verification step exists.

## Pipeline outputs
- **Silver**: `data/stations/processed/silver/<station>.csv` — time-aligned CSVs with cyclical features + alert enrichment
- **Gold (per station)**: `data/stations/processed/gold/<station>_X_train.npy`, `*_X_test.npy`, `*_y_test.npy`, `*_X_calib.npy`
- **Gold (global)**: same dir, `global_X_train.npy`, `global_X_test.npy`, `global_y_test.npy`, `global_X_calib.npy`, plus `global_station_ids_{train,test,calib}.npy`
- **Alerts**: `data/emergency_alerts/processed/alerts_data.csv`

## Results
- `python run_experiments.py` runs 4 combos: Local/Global × Baseline/Augmented
- Outputs timestamped `results/<ts>_{experiment}_metrics.csv` and `results/*_best_model.pt`
- Station `recinto-guapiles` is excluded via `IGNORED_STATIONS` in `run_experiments.py`
- The augmented pipeline uses Gaussian noise + temporal masking on continuous features

## Key modules
- `preprocessing/stations/config.py` — 10 station names, region mappings, sensor cutoffs, alert buffer windows (pre: 48h, post: 120h)
- `preprocessing/stations/silver_pipeline.py` — CSV consolidation, 10-min resampling, missing data fill, cyclical encoding, alert merge_asof
- `preprocessing/stations/gold_pipeline.py` — scaler fitting on normal data only, sliding windows (144 steps × stride 6), temporal train/test split
- `src/models/lstm_ae.py` — `LSTMAutoencoder` (3-layer LSTM encoder/decoder) and `StationAwareLSTMAutoencoder` (with station embeddings)
- `src/utils/dataset.py` — `NpySequenceDataset` (mmap, augmentation) and `DeviceDataLoader` wrapper

## Constraints
- `--google-api-key` required for alerts pipeline; never commit keys
- Read `NOTICE.md` before using or sharing datasets (UCR property, not open data)
- `data/stations/raw/ucr_uema_data_downloader.py` connects to a Grafana/InfluxDB instance and is the data entry point
