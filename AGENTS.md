# AGENTS.md

## Commands (run from repo root)
- `pip install -r requirements.txt`
- Station downloader (Tkinter GUI; Grafana/InfluxDB, fetches in 30-day chunks): `python data/stations/raw/ucr_uema_data_downloader.py`
- Alerts extraction (requires key; script sets `CUDA_VISIBLE_DEVICES=""`): `python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY`
- Station pipeline (order matters): `python -m preprocessing.stations.extract_stations_data_silver_layer` → `python -m preprocessing.stations.extract_stations_data_gold_layer` → `python -m preprocessing.stations.extract_global_gold_layer`
- Experiments/ablation grid: `python run_experiments.py`

## Outputs / data layout
- Silver CSVs: `data/stations/processed/silver/<station>.csv`
- Gold per-station: `data/stations/processed/gold/<station>_{X_train,X_test,y_test,X_calib}.npy`
- Gold global: `data/stations/processed/gold/global_{X_train,X_test,y_test,X_calib}.npy` + `global_station_ids_{train,test,calib}.npy`
- Alerts CSV: `data/emergency_alerts/processed/alerts_data.csv`
- `run_experiments.py` writes `results/<ts>_ablation_dim{dim}_{pipeline}.csv` (partial per-config results) plus `results/<ts>_ablation_all_seeds_raw.csv`, `results/<ts>_ablation_aggregated.csv`, and `*_best.pt` checkpoints per run.

## Feature vector (7 dims, fixed order)
- indices 0–2: `pressure_hPa` (StandardScaler), `precipitation_mm` (MinMaxScaler), `luminous_intensity_lux` (MinMaxScaler)
- indices 3–6: `hour_sin/cos`, `day_of_year_sin/cos` (unscaled cyclical)
- `LSTMAutoencoder` outputs only the 3 continuous features (index 0–2). Training loss and eval reconstruction error are weighted MSE (`[1.5, 2.0, 1.0]`) on those indices only; the training weight vector zeroes the cyclical features.

## Pipeline / modeling constraints
- Alert buffers live in `preprocessing/stations/config.py` (48h pre, 120h post); gold pipelines apply the buffer when building anomaly masks.
- Scalers are fit on normal training data only; do not fit on anomalous or test windows.
- Per-station temporal split happens before concatenation in the global gold pipeline; do not concatenate then split.
- `run_experiments.py` always excludes station `recinto-guapiles`.
- Sweep = 5 seeds × 4 bottleneck dims (8/16/32/64) × 4 pipelines (Local/Global × Baseline/Augmented) = 80 runs. Local models sweep thresholds at percentiles [85, 90, 95] of validation reconstruction errors; Global uses a fixed 95th-percentile threshold from the calibration set.

## Repo constraints
- No test/lint/typecheck infrastructure; do not assume verification steps exist.
- Data is UCR property; read `NOTICE.md` before using or sharing datasets.
