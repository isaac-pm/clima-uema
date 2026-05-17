"""Build Gold-layer NumPy datasets from Silver station CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from preprocessing.stations.config import PROCESSED_GOLD_DIR, PROCESSED_SILVER_DIR
from preprocessing.stations.gold_pipeline import (
    apply_scalers,
    create_anomalous_mask,
    find_silver_files,
    fit_strict_scalers,
    read_station_csv,
    sliding_windows,
    split_df_temporally,
    split_windows_temporally,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_station(
    csv_path: Path,
    gold_dir: Path,
    window_size: int = 144,
    stride: int = 6,
    buffer_hours: int = 72,
    train_ratio: float = 0.8,
) -> None:
    """Process single station CSV and save gold-layer NumPy arrays.

    Uses temporal split: earlier data for training, later for testing.
    Scalers are fit exclusively on training portion's normal data.
    """
    station_name = csv_path.stem
    logger.info("Processing station: %s", station_name)

    df = read_station_csv(csv_path)
    train_df, test_df = split_df_temporally(df, train_ratio=train_ratio)

    anomalous_mask = create_anomalous_mask(df, buffer_hours=buffer_hours)
    train_anomalous_mask = create_anomalous_mask(train_df, buffer_hours=buffer_hours)

    scalers = fit_strict_scalers(train_df, train_anomalous_mask)
    scaled = apply_scalers(df, scalers)

    arr = scaled.values
    mask_arr = anomalous_mask.values.astype(bool)
    normal_wins, anomalous_wins, all_indices = sliding_windows(
        arr,
        mask_arr,
        window_size=window_size,
        stride=stride,
    )

    x_train, x_test, y_test = split_windows_temporally(
        normal_wins,
        anomalous_wins,
        all_indices,
        train_ratio=train_ratio,
    )

    gold_dir.mkdir(parents=True, exist_ok=True)
    np.save(gold_dir / f"{station_name}_X_train.npy", x_train)
    np.save(gold_dir / f"{station_name}_X_test.npy", x_test)
    np.save(gold_dir / f"{station_name}_y_test.npy", y_test)

    logger.info("Saved %s X_train shape: %s", station_name, x_train.shape)
    logger.info("Saved %s X_test shape: %s", station_name, x_test.shape)
    logger.info("Saved %s y_test shape: %s", station_name, y_test.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold dataset from Silver CSVs")
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=PROCESSED_SILVER_DIR,
        help="Directory with silver CSV station files",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=PROCESSED_GOLD_DIR,
        help="Output directory for gold NumPy arrays",
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

    for file_path in silver_files:
        try:
            process_station(
                file_path,
                args.gold_dir,
                window_size=args.window_size,
                stride=args.stride,
                buffer_hours=args.buffer_hours,
                train_ratio=args.train_ratio,
            )
        except Exception as exc:
            logger.exception("Failed processing %s: %s", file_path, exc)


if __name__ == "__main__":
    main()
