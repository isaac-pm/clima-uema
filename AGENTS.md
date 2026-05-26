# AGENTS.md

## Commands (run from repo root)
- `pip install -r requirements.txt`
- Station downloader (Tkinter GUI; Grafana/InfluxDB): `python data/stations/raw/ucr_uema_data_downloader.py`
- Alerts extraction (requires key; script sets `CUDA_VISIBLE_DEVICES=""`): `python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY`
- Station pipeline (order matters): `python -m preprocessing.stations.extract_stations_data_silver_layer` → `python -m preprocessing.stations.extract_stations_data_gold_layer` → `python -m preprocessing.stations.extract_global_gold_layer`
- Experiments/ablation grid: `python run_experiments.py`

## Outputs / data layout
- Silver CSVs: `data/stations/processed/silver/<station>.csv`
- Gold per-station: `data/stations/processed/gold/<station>_{X_train,X_test,y_test,X_calib}.npy`
- Gold global: `data/stations/processed/gold/global_{X_train,X_test,y_test,X_calib}.npy` + `global_station_ids_{train,test,calib}.npy`
- Alerts CSV: `data/emergency_alerts/processed/alerts_data.csv`
- `run_experiments.py` writes `results/<ts>_ablation_dim{dim}_{pipeline}.csv` plus `results/<ts>_ablation_all_seeds_raw.csv` and `results/<ts>_ablation_aggregated.csv`

## Pipeline / modeling constraints
- Alert buffers live in `preprocessing/stations/config.py` (48h pre, 120h post); gold pipelines apply the buffer when building anomaly masks.
- Scalers are fit on normal training data only; do not fit on anomalous or test windows.
- Per-station temporal split happens before concatenation in the global gold pipeline; do not concatenate then split.
- `run_experiments.py` always excludes station `recinto-guapiles`.

## Repo constraints
- No test/lint/typecheck infrastructure; do not assume verification steps exist.
- Data is UCR property; read `NOTICE.md` before using or sharing datasets.
