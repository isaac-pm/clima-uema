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

## How To Run

Run commands from the repository root.

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Download raw station data (GUI)

```bash
python data/stations/raw/ucr_uema_data_downloader.py
```

3. Extract emergency alerts from PDFs (requires Google API key)

```bash
python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY
```

4. Build station Silver layer

```bash
python -m preprocessing.stations.extract_stations_data_silver_layer
```

5. Build station Gold layer (per station)

```bash
python -m preprocessing.stations.extract_stations_data_gold_layer
```

6. Build global Gold layer (all stations combined)

```bash
python -m preprocessing.stations.extract_global_gold_layer
```

7. Run experiments and generate metrics

```bash
python run_experiments.py
```

## Data Processing Strategy

This project uses a layered data pipeline (Raw → Silver → Gold) to transform raw sensor data into training-ready arrays. Each layer serves a specific purpose: the raw layer captures unmodified data from source systems, the silver layer aligns and enriches data across sources, and the gold layer prepares model-ready features with proper normalization.

### Raw Layer

Raw data comes from two sources:

**Station Sensor Data** — A GUI downloader (`data/stations/raw/ucr_uema_data_downloader.py`) fetches meteorological readings from a Grafana/InfluxDB instance. It retrieves three sensor types across 10 micro-stations:

- **Luminous intensity** (lux): measured continuously, aggregated as 10-minute means
- **Precipitation** (mm): measured as cumulative buckets, aggregated as 10-minute sums, with a hardware calibration factor (0.2794) applied
- **Atmospheric pressure** (hPa): measured by two sensor models (BME and LPS), with station-specific calibration offsets applied to correct for sensor drift

Data is downloaded in 30-day chunks to avoid API timeouts, saved as per-feature per-station CSV files, and stored with Costa Rica local time (UTC-6).

**Emergency Alert PDFs** — Costa Rican National Emergency Commission (CNE) alert PDFs are processed through `preprocessing/emergency_alerts/extract_alerts_data.py`:

1. **OCR with docling**: PDFs are converted to text via docling's built-in OCR engine (CPU-only, CUDA disabled)
2. **Structured extraction with Gemini**: A fine-tuned prompt guides Gemini to extract alert metadata (number, category, issue date/time, affected regions) from Spanish text
3. **Validation with Pydantic**: Extracted fields are validated against a schema, ensuring consistent data types and required fields
4. **CSV export**: Valid alerts are appended to `data/emergency_alerts/processed/alerts_data.csv`

This pipeline is necessary because CNE alerts are published as PDFs with no machine-readable format, but we need structured alert data to label anomalous meteorological periods.

### Silver Layer

The silver layer (`preprocessing/stations/silver_pipeline.py`) transforms raw CSVs into time-aligned, feature-enriched dataframes. The goal is to produce a consistent schema across all stations and merge external context (alerts).

**Steps:**

1. **Consolidation**: Merge pressure, precipitation, and luminous intensity CSVs for each station into a single dataframe with a unified datetime index

2. **Sensor cutoff filtering**: Some sensors had hardware changes mid-deployment. The pipeline trims data before the sensor change date to avoid inconsistent readings (e.g., sede-central_finca-2's lux sensor was replaced in May 2025)

3. **10-minute resampling**: Raw readings are irregular; resampling to 10-minute intervals creates a regular grid for downstream processing

4. **Overlap trimming**: Each sensor started recording at different times. The pipeline trims to the common period where all three sensors have data

5. **Missing data handling**: Pressure gaps are interpolated linearly (neighboring values are reasonable approximations), precipitation gaps are filled with zeros (no rain = no accumulation), luminous gaps are filled with zeros (nighttime or sensor occlusion)

6. **Cyclical time features**: Sine/cosine encoding of hour-of-day and day-of-year preserves the cyclic nature of temporal patterns (e.g., 23:00 is close to 00:00, December is close to January)

7. **Alert enrichment**: Station data is merged with alert records using `merge_asof` backward-looking. An alert is considered "active" for 72 hours after issuance, after which it expires. Each row gets `is_active_alert`, `alert_severity`, and `alert_id` flags

### Gold Layer

The gold layer (`preprocessing/stations/gold_pipeline.py`) converts silver dataframes into training-ready numpy arrays for the LSTM-Autoencoder. This involves normalization, windowing, and train/test splitting.

**Steps:**

1. **Anomaly mask creation**: Alert-active periods are dilated with a rolling window (72-hour buffer each direction). This ensures the model has contextual data around known anomalies rather than just the exact alert moment

2. **Scaler fitting on normal data only**: Normalization parameters are learned exclusively from non-anomalous data. This prevents anomalous readings from distorting the scale parameters, which would make the model less sensitive to anomalies during training

3. **Feature scaling**:
   - Pressure uses StandardScaler (z-score normalization) because it has a roughly Gaussian distribution
   - Precipitation and luminous intensity use MinMaxScaler (0-1 range) because they are right-skewed and bounded at zero
   - Cyclical features are already in [-1, 1] range, so they're passed through unchanged

4. **Sliding windows**: The time series is converted into fixed-length sequences (144 timesteps = 24 hours at 10-min resolution). Windows with stride=6 create overlapping samples, increasing training data density

5. **Train/test split**:
   - All anomalous windows go to the test set (the model should detect these as anomalous)
   - 20% of normal windows are sampled for the test set (the model should not flag these)
   - The remaining 80% of normal windows form the training set

**Global vs. Per-Station Gold**: The per-station pipeline trains separate models per location (captures local patterns). The global pipeline combines all stations into one model (captures cross-regional patterns).
