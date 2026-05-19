# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLIMA-µEMA is a deep learning early warning system for meteorological anomaly detection. It trains LSTM-Autoencoders on sensor data from 10 Costa Rican micro-weather stations to detect conditions aligned with emergency alerts issued by the CNE (National Emergency Commission). Training uses an unsupervised reconstruction-error approach where anomalies are defined by labeled alert windows.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full preprocessing pipeline (run in order)
python -m preprocessing.stations.extract_stations_data_silver_layer
python -m preprocessing.stations.extract_stations_data_gold_layer
python -m preprocessing.stations.extract_global_gold_layer

# Run all 4 experiment configurations
python run_experiments.py

# Extract emergency alerts from PDFs (requires Google API key)
python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY

# Download raw station data (GUI)
python data/stations/raw/ucr_uema_data_downloader.py
```

There is no test suite, linter config, or Makefile.

## Architecture

### Data Flow

```
Raw CSVs (pressure, precipitation, lux per station)  +  Emergency Alert PDFs
         ↓ silver_pipeline.py                                  ↓ extract_alerts_data.py
Silver CSVs: 10-min resampled, cyclical time encoding,     alerts_data.csv
             missing-data handling, alert flags merged
         ↓ gold_pipeline.py
Gold .npy arrays: anomaly-masked, scaled, sliding windows (144 steps = 24h)
         ↓ run_experiments.py
4 experiment variants → metrics CSVs + model checkpoints
```

### Preprocessing (`preprocessing/`)

- **`config.py`** — Single source of truth: station names/regions, sensor cutoffs (e.g., hardware replacement dates), alert buffer constants (48h pre-alert, 120h post-alert), excluded station (`recinto-guapiles`).
- **`silver_pipeline.py`** — Consolidates 3 sensor CSVs per station, applies sensor cutoffs, resamples to 10 min, trims to the common time overlap, interpolates/zero-fills missing values, adds cyclical `hour_sin/cos` and `day_of_year_sin/cos` columns, and merges alert windows with `merge_asof`.
- **`gold_pipeline.py`** — Dilates alert masks (adds pre/post buffers), fits scalers **only on normal data** (StandardScaler for pressure, MinMaxScaler for precipitation/lux), extracts sliding windows (stride 6), and writes `X_train`, `X_test`, `y_test`, `X_calib` numpy arrays per station.
- **`extract_global_gold_layer.py`** — Concatenates per-station gold arrays into global arrays, also outputs `station_ids_train.npy` and `station_ids_test.npy` for station-aware models.
- **`extract_alerts_data.py`** — Runs PDF OCR via docling, then Gemini API to extract structured alert records validated with Pydantic. Requires `CUDA_VISIBLE_DEVICES=""` (set internally).

### Model & Training (`src/`)

- **`src/models/lstm_ae.py`** — Three model classes:
  - `LSTMAutoencoder` — primary model: 3-layer stacked LSTM encoder → bottleneck (last hidden state) → repeat → 3-layer LSTM decoder → Linear projection. Dropout 0.2.
  - `LSTMAutoencoderImproved` — symmetric variant.
  - `StationAwareLSTMAutoencoder` — adds learned station embeddings (not used in current experiments).
- **`src/utils/dataset.py`** — `NpySequenceDataset` loads `.npy` files with `mmap_mode='r'` for memory efficiency. Augmentation (when enabled) applies Gaussian noise to continuous features and temporal masking (drop 2–4 consecutive steps at 10% probability). Cyclical features are never augmented.

### Experiments (`run_experiments.py`)

Runs 4 configurations — Local/Global × Baseline/Augmented:

- **Local**: one model trained per station, evaluated per station.
- **Global**: one model trained on all stations combined.
- **Baseline/Augmented**: controls whether `NpySequenceDataset` applies noise+masking.

Key design decisions:
- Loss is weighted MSE (`[2.0, 1.0, 1.5]` for pressure, precip, lux) applied only to continuous features (indices 0–2 of the 7-feature vector).
- Reconstruction error at evaluation time is also computed only on continuous features.
- Threshold calibration searches the 70th–96th percentile of reconstruction errors on the calibration set, picking the value that maximizes F1.
- Optional 3-step rolling mean smoothing on reconstruction errors before thresholding.

### Feature Vector (7 dimensions, always in this order)

| Index | Feature | Scaler |
|-------|---------|--------|
| 0 | `pressure_hPa` | StandardScaler |
| 1 | `precipitation_mm` | MinMaxScaler |
| 2 | `luminous_intensity_lux` | MinMaxScaler |
| 3 | `hour_sin` | none |
| 4 | `hour_cos` | none |
| 5 | `day_of_year_sin` | none |
| 6 | `day_of_year_cos` | none |

### Data Directory Layout

```
data/
├── stations/
│   ├── raw/{pressure,precipitation,luminous_intensity}/*.csv
│   └── processed/
│       ├── silver/<station>.csv
│       └── gold/<station>_{X_train,X_test,y_test,X_calib}.npy
│             global_{X_train,X_test,y_test,X_calib,station_ids_*}.npy
└── emergency_alerts/
    ├── raw/{2024,2025,2026}/*.pdf
    └── processed/alerts_data.csv
results/<timestamp>_<scope>_<variant>_metrics.csv
```

## Important Constraints

- Station `recinto-guapiles` is always excluded; its data has known quality issues.
- Scalers are fit on normal (non-alert) windows only — never refit on the full dataset.
- Alert buffers (48h pre, 120h post) are defined in `config.py`; changing them invalidates existing gold arrays.
- The Google API key for alert extraction must not be committed; pass it via `--google-api-key` flag.
- Data is UCR property; see `NOTICE.md` before sharing or publishing.
