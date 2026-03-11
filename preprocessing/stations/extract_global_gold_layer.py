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
    fit_global_scalers,
    read_station_csv,
    split_and_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_station_data_with_masks(
    silver_files: list[Path],
    buffer_hours: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Load all station data and build anomalous masks."""
    station_dfs: dict[str, pd.DataFrame] = {}
    station_masks: dict[str, pd.Series] = {}

    for csv_path in silver_files:
        station_name = csv_path.stem
        df = read_station_csv(csv_path)
        mask = create_anomalous_mask(df, buffer_hours=buffer_hours)
        station_dfs[station_name] = df
        station_masks[station_name] = mask
        logger.info("Loaded %s with %d rows", station_name, len(df))

    return station_dfs, station_masks


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
    np.save(args.gold_dir / "global_X_train.npy", global_x_train)
    np.save(args.gold_dir / "global_X_test.npy", global_x_test)
    np.save(args.gold_dir / "global_y_test.npy", global_y_test)

    logger.info("Saved global_X_train shape: %s", global_x_train.shape)
    logger.info("Saved global_X_test shape: %s", global_x_test.shape)
    logger.info("Saved global_y_test shape: %s", global_y_test.shape)


if __name__ == "__main__":
    main()
