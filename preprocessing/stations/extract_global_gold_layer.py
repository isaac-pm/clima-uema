"""Build a global Gold dataset from all Silver station CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.stations.config import (
    ALERT_POST_BUFFER_HOURS,
    ALERT_PRE_BUFFER_HOURS,
    PROCESSED_GOLD_DIR,
    PROCESSED_SILVER_DIR,
)
from preprocessing.stations.gold_pipeline import (
    apply_scalers,
    build_global_calibration_windows,
    create_anomalous_mask,
    find_silver_files,
    fit_per_station_scalers,
    read_station_csv,
    sliding_windows,
    split_df_temporally,
    split_windows_temporally,
    FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_station_data_with_masks(
    silver_files: list[Path],
    pre_buffer_hours: int = ALERT_PRE_BUFFER_HOURS,
    post_buffer_hours: int = ALERT_POST_BUFFER_HOURS,
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

        mask = create_anomalous_mask(
            df, pre_buffer_hours=pre_buffer_hours, post_buffer_hours=post_buffer_hours
        )
        train_mask = create_anomalous_mask(
            train_df,
            pre_buffer_hours=pre_buffer_hours,
            post_buffer_hours=post_buffer_hours,
        )

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
    parser.add_argument(
        "--boundary-stride-multiplier",
        type=int,
        default=2,
        help=(
            "Include normal windows within this many strides of anomaly boundaries "
            "for calibration (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--pre-buffer-hours",
        type=int,
        default=ALERT_PRE_BUFFER_HOURS,
        help="Hours to dilate anomaly mask before alert issue time (default: %(default)s)",
    )
    parser.add_argument(
        "--post-buffer-hours",
        type=int,
        default=ALERT_POST_BUFFER_HOURS,
        help="Hours to dilate anomaly mask after alert issue time (default: %(default)s)",
    )
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
        pre_buffer_hours=args.pre_buffer_hours,
        post_buffer_hours=args.post_buffer_hours,
        train_ratio=args.train_ratio,
    )

    station_scalers = fit_per_station_scalers(
        station_train_dfs=station_train_dfs,
        station_train_masks=station_train_masks,
    )

    station_name_to_id = {
        name: idx for idx, name in enumerate(sorted(station_dfs.keys()))
    }

    # Per-station temporal split: each station's history is split independently
    # before concatenation so no station ends up entirely in train or test.
    x_train_list: list[np.ndarray] = []
    x_test_list: list[np.ndarray] = []
    y_test_list: list[np.ndarray] = []
    train_ids_list: list[int] = []
    test_ids_list: list[int] = []

    for station_name, df in station_dfs.items():
        sid = station_name_to_id[station_name]
        scalers = station_scalers[station_name]
        arr = apply_scalers(df, scalers).values
        mask_arr = station_masks[station_name].values.astype(bool)
        normal_wins, anomalous_wins, indices = sliding_windows(
            arr, mask_arr,
            window_size=args.window_size,
            stride=args.stride,
        )
        x_train_s, x_test_s, y_test_s = split_windows_temporally(
            normal_wins, anomalous_wins, indices,
            train_ratio=args.train_ratio,
        )
        if len(x_train_s):
            x_train_list.append(x_train_s)
            train_ids_list.extend([sid] * len(x_train_s))
        if len(x_test_s):
            x_test_list.append(x_test_s)
            test_ids_list.extend([sid] * len(x_test_s))
            y_test_list.append(y_test_s)

    _shape = (args.window_size, len(FEATURES))
    global_x_train = np.concatenate(x_train_list) if x_train_list else np.empty((0, *_shape))
    global_x_test = np.concatenate(x_test_list) if x_test_list else np.empty((0, *_shape))
    global_y_test = (
        np.concatenate(y_test_list) if y_test_list else np.empty((0,), dtype=np.int64)
    )
    global_train_station_ids = np.array(train_ids_list, dtype=np.int64)
    global_test_station_ids = np.array(test_ids_list, dtype=np.int64)

    global_calib_windows, global_calib_station_ids = build_global_calibration_windows(
        station_train_dfs=station_train_dfs,
        station_train_masks=station_train_masks,
        station_scalers=station_scalers,
        station_name_to_id=station_name_to_id,
        window_size=args.window_size,
        stride=args.stride,
        boundary_stride_multiplier=args.boundary_stride_multiplier,
    )

    if global_calib_windows:
        calib_array = np.stack(global_calib_windows)
    else:
        calib_array = np.empty((0, args.window_size, len(FEATURES)))

    logger.info(
        "Global split -> train: %d | test: %d | anomalous in test: %d",
        len(global_x_train),
        len(global_x_test),
        int(global_y_test.sum()),
    )

    args.gold_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.gold_dir / "global_X_train.npy", global_x_train)
    np.save(args.gold_dir / "global_X_test.npy", global_x_test)
    np.save(args.gold_dir / "global_y_test.npy", global_y_test)
    np.save(args.gold_dir / "global_station_ids_train.npy", global_train_station_ids)
    np.save(args.gold_dir / "global_station_ids_test.npy", global_test_station_ids)
    np.save(args.gold_dir / "global_X_calib.npy", calib_array)
    np.save(
        args.gold_dir / "global_station_ids_calib.npy",
        np.array(global_calib_station_ids),
    )

    logger.info("Saved global_X_train shape: %s", global_x_train.shape)
    logger.info("Saved global_X_test shape: %s", global_x_test.shape)
    logger.info("Saved global_y_test shape: %s", global_y_test.shape)
    logger.info("Saved global_X_calib shape: %s", calib_array.shape)
    logger.info(
        "Saved global_station_ids_train shape: %s", global_train_station_ids.shape
    )
    logger.info(
        "Saved global_station_ids_test shape: %s", global_test_station_ids.shape
    )
    logger.info(
        "Saved global_station_ids_calib shape: %s",
        np.array(global_calib_station_ids).shape,
    )


if __name__ == "__main__":
    main()
