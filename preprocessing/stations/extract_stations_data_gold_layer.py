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
    split_and_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_station(
    csv_path: Path,
    gold_dir: Path,
    window_size: int = 144,
    stride: int = 6,
    buffer_hours: int = 72,
    normal_sample_in_test: float = 0.2,
    random_state: int = 42,
) -> None:
    """Process single station CSV and save gold-layer NumPy arrays."""
    station_name = csv_path.stem
    logger.info("Processing station: %s", station_name)

    df = read_station_csv(csv_path)
    anomalous_mask = create_anomalous_mask(df, buffer_hours=buffer_hours)
    scalers = fit_strict_scalers(df, anomalous_mask)
    scaled = apply_scalers(df, scalers)

    arr = scaled.values
    mask_arr = anomalous_mask.values.astype(bool)
    normal_wins, anomalous_wins = sliding_windows(
        arr,
        mask_arr,
        window_size=window_size,
        stride=stride,
    )

    x_train, x_test, y_test = split_and_sample(
        normal_wins,
        anomalous_wins,
        normal_sample_in_test,
        random_state,
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
    parser.add_argument("--normal-sample-in-test", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

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
                normal_sample_in_test=args.normal_sample_in_test,
                random_state=args.random_state,
            )
        except Exception as exc:
            logger.exception("Failed processing %s: %s", file_path, exc)


if __name__ == "__main__":
    main()
