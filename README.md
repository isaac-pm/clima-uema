# CLIMA-μEMA: Evaluating LSTM-Autoencoder Viability for Meteorological Anomaly Detection in Costa Rica

> View the [NOTICE.md](NOTICE.md) file for important information regarding data ownership, usage rights, and legal disclaimers.

Deep Learning early warning system for meteorological anomaly detection. Trains LSTM-Autoencoders on sensor data from micro-weather stations in Costa Rica to detect conditions aligned with emergency alerts issued by the CNE (National Emergency Commission).

## Repository Structure

```text
clima-uema/
  data/
    emergency_alerts/
      raw/                  # Input alert PDFs
      processed/            # Extracted alerts_data.csv
    stations/
      raw/                  # Downloaded station CSVs (by feature)
      processed/
        silver/             # Time-aligned CSVs with engineered features
        gold/               # Training-ready .npy arrays
  preprocessing/
    emergency_alerts/
      extract_alerts_data.py
    stations/
      config.py             # Station names, sensor cutoffs, alert buffer constants
      silver_pipeline.py
      gold_pipeline.py
      extract_stations_data_silver_layer.py
      extract_stations_data_gold_layer.py
      extract_global_gold_layer.py
  src/
    models/lstm_ae.py       # LSTM Autoencoder
    utils/dataset.py        # PyTorch dataset / dataloader utilities
  run_experiments.py        # Ablation benchmark runner
  results/                  # Generated metrics CSVs and model checkpoints
```

## Running

All commands from the repository root.

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download raw station data** (GUI)

```bash
python data/stations/raw/ucr_uema_data_downloader.py
```

**3. Extract emergency alerts from PDFs** (requires Google API key)

```bash
python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY
```

**4. Build Silver layer**

```bash
python -m preprocessing.stations.extract_stations_data_silver_layer
```

**5. Build Gold layer (per station)**

```bash
python -m preprocessing.stations.extract_stations_data_gold_layer
```

**6. Build global Gold layer**

```bash
python -m preprocessing.stations.extract_global_gold_layer
```

Accepts CLI overrides: `--window-size`, `--stride`, `--boundary-stride-multiplier`, `--pre-buffer-hours`, `--post-buffer-hours`, `--train-ratio`.

**7. Run ablation benchmark**

```bash
python run_experiments.py
```

Sweeps **5 seeds × 4 bottleneck dimensions (8, 16, 32, 64) × 4 pipeline variants** (Local/Global × Baseline/Augmented) — 80 training runs total. Per-seed results and aggregated mean/variance metrics are written to `results/`. With multiple GPUs, seed runs are distributed across devices automatically.

## HPC (IRIS) Setup

### First-time setup

```bash
# 1) Install Miniforge
cd $HOME
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc
conda config --set auto_activate_base false

# 2) Log out and back in, then:
cd ~/clima-uema
conda create -n clima-313 python=3.13.3 pip -y
conda activate clima-313

# 3) Request a GPU node for CUDA-linked installs
salloc -N 1 --ntasks-per-node=1 --cpus-per-task=7 --gpus-per-task=1 -p gpu -q normal -t 04:00:00
conda activate clima-313
module load system/CUDA/12.6.0

# 4) Install PyTorch (CUDA 12.6) and remaining dependencies
python -m pip install --upgrade pip setuptools wheel
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt --ignore-requires-python
```

### Running on IRIS

```bash
cd ~/clima-uema
salloc -N 1 --ntasks-per-node=1 --cpus-per-task=7 --gpus-per-task=1 -p gpu -q normal -t 04:00:00
conda activate clima-313
module load system/CUDA/12.6.0
python run_experiments.py
```
