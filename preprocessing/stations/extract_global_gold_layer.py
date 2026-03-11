"""Build a global Gold dataset from all Silver station CSV files.

Unlike the per-station Gold pipeline, this script fits one global scaler set
from normal rows across all stations, then creates one shared train/test split.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from preprocessing.stations.extract_stations_data_gold_layer import (
        FEATURES,
        apply_scalers,
        create_anomalous_mask,
        find_silver_files,
        read_station_csv,
        sliding_windows,
        split_and_sample,
    )
except ModuleNotFoundError:
    from extract_stations_data_gold_layer import (
        FEATURES,
        apply_scalers,
        create_anomalous_mask,
        find_silver_files,
        read_station_csv,
        sliding_windows,
        split_and_sample,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_station_data_with_masks(
    silver_files: List[Path],
    buffer_hours: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    """Load all station data and build anomalous masks.

    Args:
        silver_files: Station CSV file paths.
        buffer_hours: Hours to buffer before and after active alerts.

    Returns:
        Tuple with station DataFrames and anomalous masks keyed by station name.
    """
    station_dfs: Dict[str, pd.DataFrame] = {}
    station_masks: Dict[str, pd.Series] = {}

    for csv_path in silver_files:
        station_name = csv_path.stem
        df = read_station_csv(csv_path)
        mask = create_anomalous_mask(df, buffer_hours=buffer_hours)
        station_dfs[station_name] = df
        station_masks[station_name] = mask
        logger.info("Loaded %s with %d rows", station_name, len(df))

    return station_dfs, station_masks


def fit_global_scalers(
    station_dfs: Dict[str, pd.DataFrame],
    station_masks: Dict[str, pd.Series],
) -> Dict[str, object]:
    """Fit one scaler set on normal rows from all stations.

    Args:
        station_dfs: Station DataFrames keyed by station name.
        station_masks: Anomalous masks keyed by station name.

    Returns:
        Scaler dictionary compatible with `apply_scalers`.
    """
    scalers: Dict[str, object] = {}

    pressure_chunks: List[pd.DataFrame] = []
    mm_chunks: List[pd.DataFrame] = []
    mm_cols = ["precipitation_mm", "luminous_intensity_lux"]

    for station_name, df in station_dfs.items():
        normal_df = df.loc[~station_masks[station_name]]

        if "pressure_hPa" in normal_df.columns:
            pressure_chunks.append(normal_df[["pressure_hPa"]].astype(float).dropna())

        available_mm = [c for c in mm_cols if c in normal_df.columns]
        if available_mm:
            mm_chunk = normal_df[available_mm].astype(float)
            mm_chunk = mm_chunk.reindex(columns=available_mm).dropna()
            if not mm_chunk.empty:
                mm_chunks.append(mm_chunk)

    if pressure_chunks:
        pressure_data = pd.concat(pressure_chunks, axis=0, ignore_index=True)
        pressure_scaler = StandardScaler()
        pressure_scaler.fit(pressure_data)
        scalers["pressure"] = pressure_scaler
        logger.info("Fitted global pressure scaler on %d rows", len(pressure_data))

    global_mm_cols = [
        c for c in mm_cols if any(c in df.columns for df in station_dfs.values())
    ]
    if mm_chunks and global_mm_cols:
        aligned_mm_chunks = [
            chunk.reindex(columns=global_mm_cols) for chunk in mm_chunks
        ]
        mm_data = pd.concat(aligned_mm_chunks, axis=0, ignore_index=True).dropna()
        if not mm_data.empty:
            mm_scaler = MinMaxScaler()
            mm_scaler.fit(mm_data)
            scalers["minmax"] = (mm_scaler, global_mm_cols)
            logger.info("Fitted global minmax scaler on %d rows", len(mm_data))

    scalers["cyclical"] = [c for c in FEATURES if c.endswith(("_sin", "_cos"))]

    return scalers


def build_global_windows(
    station_dfs: Dict[str, pd.DataFrame],
    station_masks: Dict[str, pd.Series],
    scalers: Dict[str, object],
    window_size: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Scale each station and aggregate all normal/anomalous windows.

    Args:
        station_dfs: Station DataFrames keyed by station name.
        station_masks: Anomalous masks keyed by station name.
        scalers: Global scaler dictionary.
        window_size: Timesteps per window.
        stride: Window stride.

    Returns:
        Tuple of aggregated normal and anomalous window lists.
    """
    all_normal_windows: List[np.ndarray] = []
    all_anomalous_windows: List[np.ndarray] = []

    for station_name, df in station_dfs.items():
        scaled_df = apply_scalers(df, scalers)
        arr = scaled_df.values
        mask_arr = station_masks[station_name].values.astype(bool)

        normal_wins, anomalous_wins = sliding_windows(
            arr,
            mask_arr,
            window_size=window_size,
            stride=stride,
        )

        all_normal_windows.extend(normal_wins)
        all_anomalous_windows.extend(anomalous_wins)

        logger.info(
            "Station %s -> normal windows: %d | anomalous windows: %d",
            station_name,
            len(normal_wins),
            len(anomalous_wins),
        )

    return all_normal_windows, all_anomalous_windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build global Gold dataset from all Silver station CSVs",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("data/stations/processed/silver"),
        help="Directory with Silver CSV station files",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/stations/processed/gold"),
        help="Output directory for global Gold NumPy arrays",
    )
    parser.add_argument("--window-size", type=int, default=144)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--buffer-hours", type=int, default=72)
    parser.add_argument("--normal-sample-in-test", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    silver_files = find_silver_files(args.silver_dir)
    if not silver_files:
        logger.warning("No CSV files found in %s", args.silver_dir)
        return

    station_dfs, station_masks = load_station_data_with_masks(
        silver_files=silver_files,
        buffer_hours=args.buffer_hours,
    )

    scalers = fit_global_scalers(
        station_dfs=station_dfs,
        station_masks=station_masks,
    )

    all_normal_windows, all_anomalous_windows = build_global_windows(
        station_dfs=station_dfs,
        station_masks=station_masks,
        scalers=scalers,
        window_size=args.window_size,
        stride=args.stride,
    )

    logger.info(
        "Global windows -> normal: %d | anomalous: %d",
        len(all_normal_windows),
        len(all_anomalous_windows),
    )

    global_x_train, global_x_test, global_y_test = split_and_sample(
        normal_windows=all_normal_windows,
        anomalous_windows=all_anomalous_windows,
        normal_sample_in_test=args.normal_sample_in_test,
        random_state=args.random_state,
    )

    args.gold_dir.mkdir(parents=True, exist_ok=True)
    xtrain_path = args.gold_dir / "global_X_train.npy"
    xtest_path = args.gold_dir / "global_X_test.npy"
    ytest_path = args.gold_dir / "global_y_test.npy"

    np.save(xtrain_path, global_x_train)
    np.save(xtest_path, global_x_test)
    np.save(ytest_path, global_y_test)

    logger.info("Saved global_X_train shape: %s", global_x_train.shape)
    logger.info("Saved global_X_test shape: %s", global_x_test.shape)
    logger.info("Saved global_y_test shape: %s", global_y_test.shape)


if __name__ == "__main__":
    main()
