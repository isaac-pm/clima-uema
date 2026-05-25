# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLIMA-µEMA is a deep learning early warning system for meteorological anomaly detection. It trains LSTM-Autoencoders on sensor data from 10 Costa Rican micro-weather stations to detect conditions aligned with emergency alerts issued by the CNE (National Emergency Commission). Training uses an unsupervised reconstruction-error approach where anomalies are defined by labeled alert windows.

## Commands

```bash
# Install dependencies (virtualenv is at env/)
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

There is no test suite, linter config, or Makefile. `flake8` is in requirements but has no config.

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
- **`gold_pipeline.py`** — Dilates alert masks (adds pre/post buffers), fits scalers **only on normal training data** (StandardScaler for pressure, MinMaxScaler for precipitation/lux), extracts sliding windows (stride 6), and writes `X_train`, `X_test`, `y_test`, `X_calib` numpy arrays per station. `X_calib` contains normal windows near anomaly boundaries (not random normal windows) for threshold calibration.
- **`extract_global_gold_layer.py`** — Splits each station's history independently, then concatenates into global arrays. Also outputs `station_ids_train.npy`, `station_ids_test.npy`, and `station_ids_calib.npy` for station-aware models.
- **`extract_alerts_data.py`** — Runs PDF OCR via docling, then Gemini API to extract structured alert records validated with Pydantic. Requires `CUDA_VISIBLE_DEVICES=""` (set internally).

### Model & Training (`src/`)

- **`src/models/lstm_ae.py`** — `LSTMAutoencoder`: 3-layer stacked LSTM encoder (128→64→16) using the final hidden state as the bottleneck → repeat across `seq_len` → 3-layer LSTM decoder (16→16→64→128) → `Linear(128, 3)`. The output has **3 dimensions** (continuous features only), not 7. Dropout 0.2.
- **`src/utils/dataset.py`** — `NpySequenceDataset` loads `.npy` files via mmap for memory efficiency. When augmentation is enabled, applies Gaussian noise (sigma = 1% of per-feature std, computed from training indices only) to continuous features and temporal masking (2–4 consecutive steps zeroed at 10% probability). Cyclical features are never augmented. `DeviceDataLoader` wraps `DataLoader` to move batches to device.

### Experiments (`run_experiments.py`)

Runs 4 configurations — Local/Global × Baseline/Augmented:

- **Local**: one model per station; threshold swept at percentiles [85, 90, 95] of validation-set reconstruction errors; all three are reported.
- **Global**: one model for all stations; threshold fixed at 95th percentile of the calibration set.
- **Baseline/Augmented**: controls whether `NpySequenceDataset` applies noise+masking augmentation.

Key design decisions:
- Loss is weighted MSE (`[1.5, 2.0, 1.0]` for pressure, precip, lux) applied only to continuous features (indices 0–2). The full weight vector `[1.5, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]` zeroes out cyclical features during training.
- Reconstruction error at evaluation time is also computed only on continuous features using `FEATURE_WEIGHTS = [1.5, 2.0, 1.0]`.
- 3-step rolling mean smoothing on reconstruction errors before thresholding.
- Results include both raw metrics and Point Adjustment (PA) metrics: a true-anomaly segment is credited as detected if ≥10% of its windows are flagged. FPR outside true-anomaly blocks is unaffected by PA.

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

- Station `recinto-guapiles` is always excluded (hardcoded in `run_experiments.py`); its data has known quality issues.
- Scalers are fit on the **training portion's normal data only** — never on test data or anomalous windows.
- Alert buffers (48h pre, 120h post) are defined in `config.py`; changing them invalidates existing gold arrays.
- The temporal split is done per-station before concatenation in the global pipeline; concatenating then splitting would leak future data across stations.
- The Google API key for alert extraction must not be committed; pass it via `--google-api-key` flag.
- Data is UCR property; see `NOTICE.md` before sharing or publishing.
