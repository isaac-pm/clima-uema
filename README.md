# CLIMA-µEMA: Climate Localized Incident Monitoring with Autoencoders Using Automatic Micro Weather Stations in Costa Rica

> View the [NOTICE.md](NOTICE.md) file for important information regarding data ownership, usage rights, and legal disclaimers.

Deep Learning early warning system (LSTM-Autoencoders) for meteorological anomaly detection using micro-station data in Costa Rica.

## Repository Organization

```text
clima-uema/
  data/
    emergency_alerts/
      raw/                  # Input alert PDFs
      processed/            # Extracted alerts_data.csv output
    stations/
      raw/                  # Downloaded station CSVs (by feature)
      processed/
        silver/             # Time-aligned station CSVs with engineered features
        gold/               # Training-ready .npy arrays
  preprocessing/
    emergency_alerts/
      extract_alerts_data.py
    stations/
      config.py
      silver_pipeline.py
      gold_pipeline.py
      extract_stations_data_silver_layer.py
      extract_stations_data_gold_layer.py
      extract_global_gold_layer.py
  src/
    utils/
      dataset.py            # PyTorch dataset/dataloader helpers for .npy files
  logs/
```

Notes:
- `preprocessing/stations/*pipeline.py` contains reusable processing logic.
- `preprocessing/stations/extract_*.py` are thin CLI entrypoints.
- `data/stations/raw/ucr_uema_data_downloader.py` remains a standalone GUI downloader.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

## How To Run

Run commands from the repository root.

### 1) Download raw station data (standalone GUI)

```bash
python data/stations/raw/ucr_uema_data_downloader.py
```

This opens a Tkinter app where you configure credentials, date range, stations, and features.

### 2) Extract emergency alerts from PDFs

```bash
python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY
python -m preprocessing.emergency_alerts.extract_alerts_data -h
```

Input PDFs: `data/emergency_alerts/raw/`
Output CSV: `data/emergency_alerts/processed/alerts_data.csv`

### 3) Build station Silver layer

```bash
python -m preprocessing.stations.extract_stations_data_silver_layer --help
python -m preprocessing.stations.extract_stations_data_silver_layer
```

Reads raw station CSVs and writes processed station files to `data/stations/processed/silver/`.

### 4) Build station Gold layer (per station)

```bash
python -m preprocessing.stations.extract_stations_data_gold_layer --help
python -m preprocessing.stations.extract_stations_data_gold_layer
```

Writes per-station arrays into `data/stations/processed/gold/`:
- `<station>_X_train.npy`
- `<station>_X_test.npy`
- `<station>_y_test.npy`

### 5) Build global Gold layer (all stations combined)

```bash
python -m preprocessing.stations.extract_global_gold_layer --help
python -m preprocessing.stations.extract_global_gold_layer
```

Writes global arrays into `data/stations/processed/gold/`:
- `global_X_train.npy`
- `global_X_test.npy`
- `global_y_test.npy`

## Quality and Tests

```bash
pytest
flake8 .
mypy .
black .
```
