"""Build a global Gold dataset from all Silver station CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.stations.config import PROCESSED_GOLD_DIR, PROCESSED_SILVER_DIR
from preprocessing.stations.gold_pipeline import (
    build_global_windows,
    create_anomalous_mask,
    find_silver_files,
    fit_per_station_scalers,
    read_station_csv,
    split_df_temporally,
    split_windows_temporally,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_station_data_with_masks(
    silver_files: list[Path],
    buffer_hours: int,
    train_ratio: float = 0.8,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.Series],
    dict[str, pd.DataFrame],
    dict[str, pd.Series],
]:
    """Load station data, split temporally, and build anomalous masks.

    Returns:
        station_dfs: Full station DataFrames
        station_masks: Anomaly masks for full data
        station_train_dfs: Training portion of each station
        station_train_masks: Anomaly masks for training portions
    """
    station_dfs: dict[str, pd.DataFrame] = {}
    station_masks: dict[str, pd.Series] = {}
    station_train_dfs: dict[str, pd.DataFrame] = {}
    station_train_masks: dict[str, pd.Series] = {}

    for csv_path in silver_files:
        station_name = csv_path.stem
        df = read_station_csv(csv_path)
        train_df, _ = split_df_temporally(df, train_ratio=train_ratio)

        mask = create_anomalous_mask(df, buffer_hours=buffer_hours)
        train_mask = create_anomalous_mask(train_df, buffer_hours=buffer_hours)

        station_dfs[station_name] = df
        station_masks[station_name] = mask
        station_train_dfs[station_name] = train_df
        station_train_masks[station_name] = train_mask
        logger.info(
            "Loaded %s with %d rows (train: %d)",
            station_name,
            len(df),
            len(train_df),
        )

    return station_dfs, station_masks, station_train_dfs, station_train_masks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build global Gold dataset from all Silver station CSVs",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=PROCESSED_SILVER_DIR,
        help="Directory with Silver CSV station files",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=PROCESSED_GOLD_DIR,
        help="Output directory for global Gold NumPy arrays",
    )
    parser.add_argument("--window-size", type=int, default=144)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--buffer-hours", type=int, default=72)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data (earliest) to use for training (default: 0.8)",
    )

    args = parser.parse_args()

    silver_files = find_silver_files(args.silver_dir)
    if not silver_files:
        logger.warning("No CSV files found in %s", args.silver_dir)
        return

    (
        station_dfs,
        station_masks,
        station_train_dfs,
        station_train_masks,
    ) = load_station_data_with_masks(
        silver_files=silver_files,
        buffer_hours=args.buffer_hours,
        train_ratio=args.train_ratio,
    )

    station_scalers = fit_per_station_scalers(
        station_train_dfs=station_train_dfs,
        station_train_masks=station_train_masks,
    )

    all_normal_windows, all_anomalous_windows, all_indices = build_global_windows(
        station_dfs=station_dfs,
        station_masks=station_masks,
        station_scalers=station_scalers,
        window_size=args.window_size,
        stride=args.stride,
    )

    logger.info(
        "Global windows -> normal: %d | anomalous: %d",
        len(all_normal_windows),
        len(all_anomalous_windows),
    )

    global_x_train, global_x_test, global_y_test = split_windows_temporally(
        normal_windows=all_normal_windows,
        anomalous_windows=all_anomalous_windows,
        window_indices=all_indices,
        train_ratio=args.train_ratio,
    )

    args.gold_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.gold_dir / "global_X_train.npy", global_x_train)
    np.save(args.gold_dir / "global_X_test.npy", global_x_test)
    np.save(args.gold_dir / "global_y_test.npy", global_y_test)

    logger.info("Saved global_X_train shape: %s", global_x_train.shape)
    logger.info("Saved global_X_test shape: %s", global_x_test.shape)
    logger.info("Saved global_y_test shape: %s", global_y_test.shape)


if __name__ == "__main__":
    main()
