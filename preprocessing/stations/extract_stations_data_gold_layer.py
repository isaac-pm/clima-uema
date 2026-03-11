"""Build Gold-layer NumPy datasets from Silver station CSV files.

For each station, this script creates buffered anomaly masks, scales features,
builds sliding windows, splits train/test sets, and saves ``.npy`` arrays.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


FEATURES = [
    "pressure_hPa",
    "precipitation_mm",
    "luminous_intensity_lux",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


def find_silver_files(silver_dir: Path) -> List[Path]:
    """Return list of CSV files in the silver directory.

    Args:
        silver_dir: directory with station CSV files.

    Returns:
        List of file paths.
    """
    return sorted(silver_dir.glob("*.csv"))


def read_station_csv(path: Path) -> pd.DataFrame:
    """Read a station CSV and parse `time` as datetime index.

    Args:
        path: CSV file path.

    Returns:
        DataFrame indexed by datetime.
    """
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df.set_index("time", inplace=True)
    return df


def create_anomalous_mask(df: pd.DataFrame, buffer_hours: int = 72) -> pd.Series:
    """Create boolean `is_anomalous` mask with buffer around alerts using vectorization.

    This implementation uses a rolling maximum (binary dilation) over the
    boolean `is_active_alert` series to efficiently mark the buffer before
    and after each alert without looping over timestamps.

    Args:
        df: input DataFrame indexed by datetime and containing `is_active_alert`.
        buffer_hours: hours to buffer before and after each alert.

    Returns:
        Boolean Series aligned with `df.index`.
    """
    if "is_active_alert" not in df.columns:
        raise KeyError("Input CSV must contain `is_active_alert` column")

    active = df["is_active_alert"].fillna(False).astype(str).str.lower()
    active_bool = active.isin(["true", "1", "t", "yes"]) | (active == "true")

    # 6 periods per hour at 10-minute resolution.
    periods = buffer_hours * 6
    window_size = (2 * periods) + 1

    # Rolling max acts as binary dilation around alert periods.
    mask = (
        active_bool.astype(int)
        .rolling(window=window_size, center=True, min_periods=1)
        .max()
        .astype(bool)
    )

    # Preserve original index and bool dtype.
    mask = pd.Series(mask.values, index=df.index, dtype=bool)
    return mask


def fit_strict_scalers(df: pd.DataFrame, normal_mask: pd.Series) -> Dict[str, object]:
    """Fit scalers on the normal portion of the dataset.

    Args:
        df: full DataFrame with feature columns present.
        normal_mask: boolean Series where True denotes anomalous; we will fit on ~False.

    Returns:
        Dictionary of fitted scaler objects keyed by feature name or group.
    """
    scalers: Dict[str, object] = {}
    normal_df = df.loc[~normal_mask]

    # Pressure uses StandardScaler.
    if "pressure_hPa" in normal_df:
        ps = StandardScaler()
        Xp = normal_df[["pressure_hPa"]].astype(float)
        ps.fit(Xp)
        scalers["pressure"] = ps

    # Precipitation and luminosity use MinMaxScaler.
    mm_cols = [
        c for c in ["precipitation_mm", "luminous_intensity_lux"] if c in normal_df
    ]
    if mm_cols:
        mms = MinMaxScaler()
        Xm = normal_df[mm_cols].astype(float)
        mms.fit(Xm)
        scalers["minmax"] = (mms, mm_cols)

    # Cyclical features are pass-through.
    scalers["cyclical"] = [
        c for c in FEATURES if c in df and c.endswith(("_sin", "_cos"))
    ]

    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, object]) -> pd.DataFrame:
    """Apply fitted scalers to entire DataFrame and return scaled features DataFrame.

    Args:
        df: original DataFrame.
        scalers: dictionary from `fit_strict_scalers`.

    Returns:
        DataFrame with columns ordered as `FEATURES` and scaled values (float).
    """
    out = pd.DataFrame(index=df.index)

    # Pressure.
    if "pressure" in scalers:
        ps: StandardScaler = scalers["pressure"]
        out["pressure_hPa"] = ps.transform(df[["pressure_hPa"]].astype(float)).ravel()

    # MinMax group.
    if "minmax" in scalers:
        mms, cols = scalers["minmax"]
        transformed = mms.transform(df[cols].astype(float))
        for i, c in enumerate(cols):
            out[c] = transformed[:, i]

    # Cyclical pass-through.
    cyclical = scalers.get("cyclical", [])
    for c in cyclical:
        if c in df.columns:
            out[c] = df[c].astype(float)

    # Enforce feature order and backfill missing columns.
    for c in FEATURES:
        if c not in out.columns:
            out[c] = 0.0

    return out[FEATURES].astype(float)


def sliding_windows(
    arr: np.ndarray, mask: np.ndarray, window_size: int = 144, stride: int = 6
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create sliding windows and separate normal vs anomalous windows.

    Args:
        arr: 2D array (T, F) of features.
        mask: 1D boolean array of length T where True indicates anomalous.
        window_size: number of timesteps per sequence.
        stride: step size between windows.

    Returns:
        Tuple (normal_windows, anomalous_windows) where each is a list of 2D arrays
        shaped (window_size, F).
    """
    T = arr.shape[0]
    normal_windows: List[np.ndarray] = []
    anomalous_windows: List[np.ndarray] = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        win = arr[start:end]
        win_mask = mask[start:end]
        if win_mask.any():
            anomalous_windows.append(win)
        else:
            normal_windows.append(win)

    return normal_windows, anomalous_windows


def split_and_sample(
    normal_windows: List[np.ndarray],
    anomalous_windows: List[np.ndarray],
    normal_sample_in_test: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/test arrays and labels from window lists.

    Args:
        normal_windows: list of purely-normal windows.
        anomalous_windows: list of windows containing anomalies.
        normal_sample_in_test: Fraction of normal windows added to X_test.
        random_state: RNG seed for reproducibility.

    Returns:
        X_train, X_test, y_test
    """
    rng = np.random.RandomState(random_state)
    n_norm = len(normal_windows)
    n_sample = int(round(n_norm * normal_sample_in_test)) if n_norm > 0 else 0

    sample_idx = (
        rng.choice(n_norm, size=n_sample, replace=False)
        if n_sample > 0
        else np.array([], dtype=int)
    )

    # X_test = all anomalous + sampled normal windows.
    X_test_list = []
    y_test_list = []

    for w in anomalous_windows:
        X_test_list.append(w)
        y_test_list.append(1)

    for i in sample_idx:
        X_test_list.append(normal_windows[i])
        y_test_list.append(0)

    # X_train = remaining normal windows.
    X_train_list = [
        w for idx, w in enumerate(normal_windows) if idx not in set(sample_idx.tolist())
    ]

    # Determine reference shape for empty outputs.
    if normal_windows:
        ref_shape = normal_windows[0].shape
    elif anomalous_windows:
        ref_shape = anomalous_windows[0].shape
    else:
        ref_shape = (0, 0)

    X_train = np.stack(X_train_list) if X_train_list else np.empty((0, *ref_shape))
    X_test = np.stack(X_test_list) if X_test_list else np.empty((0, *ref_shape))
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, X_test, y_test


def process_station(
    csv_path: Path,
    gold_dir: Path,
    window_size: int = 144,
    stride: int = 6,
    buffer_hours: int = 72,
    normal_sample_in_test: float = 0.2,
    random_state: int = 42,
):
    """Process single station CSV and save gold-layer NumPy arrays.

    Args:
        csv_path: path to the station CSV in silver layer.
        gold_dir: directory to write output npy files.
    """
    station_name = csv_path.stem
    logger.info("Processing station: %s", station_name)

    df = read_station_csv(csv_path)

    # Build anomalous mask.
    anomalous_mask = create_anomalous_mask(df, buffer_hours=buffer_hours)

    # Fit scalers on normal data only.
    scalers = fit_strict_scalers(df, anomalous_mask)

    # Scale full dataset.
    scaled = apply_scalers(df, scalers)

    arr = scaled.values  # shape (T, F)
    mask_arr = anomalous_mask.values.astype(bool)

    normal_wins, anomalous_wins = sliding_windows(
        arr, mask_arr, window_size=window_size, stride=stride
    )

    X_train, X_test, y_test = split_and_sample(
        normal_wins, anomalous_wins, normal_sample_in_test, random_state
    )

    gold_dir.mkdir(parents=True, exist_ok=True)

    xtrain_path = gold_dir / f"{station_name}_X_train.npy"
    xtest_path = gold_dir / f"{station_name}_X_test.npy"
    ytest_path = gold_dir / f"{station_name}_y_test.npy"

    np.save(xtrain_path, X_train)
    np.save(xtest_path, X_test)
    np.save(ytest_path, y_test)

    logger.info("Saved %s X_train shape: %s", station_name, X_train.shape)
    logger.info("Saved %s X_test shape: %s", station_name, X_test.shape)
    logger.info("Saved %s y_test shape: %s", station_name, y_test.shape)


def main():
    parser = argparse.ArgumentParser(description="Build Gold dataset from Silver CSVs")
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("data/stations/processed/silver"),
        help="Directory with silver CSV station files",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/stations/processed/gold"),
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

    for f in silver_files:
        try:
            process_station(
                f,
                args.gold_dir,
                window_size=args.window_size,
                stride=args.stride,
                buffer_hours=args.buffer_hours,
                normal_sample_in_test=args.normal_sample_in_test,
                random_state=args.random_state,
            )
        except Exception as e:
            logger.exception("Failed processing %s: %s", f, e)


if __name__ == "__main__":
    main()
