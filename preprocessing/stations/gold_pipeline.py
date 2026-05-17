"""Shared Gold-layer preprocessing helpers for station and global datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    """Return list of CSV files in the silver directory."""
    return sorted(silver_dir.glob("*.csv"))


def read_station_csv(path: Path) -> pd.DataFrame:
    """Read a station CSV and parse `time` as datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df.set_index("time", inplace=True)
    return df


def create_anomalous_mask(df: pd.DataFrame, buffer_hours: int = 72) -> pd.Series:
    """Create buffered boolean `is_anomalous` mask using rolling dilation."""
    if "is_active_alert" not in df.columns:
        raise KeyError("Input CSV must contain `is_active_alert` column")

    active = df["is_active_alert"].fillna(False).astype(str).str.lower()
    active_bool = active.isin(["true", "1", "t", "yes"]) | (active == "true")

    periods = buffer_hours * 6
    window_size = (2 * periods) + 1

    mask = (
        active_bool.astype(int)
        .rolling(window=window_size, center=True, min_periods=1)
        .max()
        .astype(bool)
    )

    return pd.Series(mask.values, index=df.index, dtype=bool)


def split_df_temporally(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame temporally into train (earlier) and test (later) portions."""
    n = len(df)
    split_idx = int(n * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def fit_strict_scalers(df: pd.DataFrame, normal_mask: pd.Series) -> Dict[str, Any]:
    """Fit scalers on the normal portion of the dataset."""
    scalers: Dict[str, Any] = {}
    normal_df = df.loc[~normal_mask]

    if "pressure_hPa" in normal_df:
        pressure_scaler = StandardScaler()
        pressure_scaler.fit(normal_df[["pressure_hPa"]].astype(float))
        scalers["pressure"] = pressure_scaler

    mm_cols = [
        c for c in ["precipitation_mm", "luminous_intensity_lux"] if c in normal_df
    ]
    if mm_cols:
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(normal_df[mm_cols].astype(float))
        scalers["minmax"] = (minmax_scaler, mm_cols)

    scalers["cyclical"] = [
        c for c in FEATURES if c in df and c.endswith(("_sin", "_cos"))
    ]
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, Any]) -> pd.DataFrame:
    """Apply fitted scalers to full dataframe and return ordered features."""
    out = pd.DataFrame(index=df.index)

    if "pressure" in scalers:
        pressure_scaler = cast(StandardScaler, scalers["pressure"])
        out["pressure_hPa"] = pressure_scaler.transform(
            df[["pressure_hPa"]].astype(float)
        ).ravel()

    if "minmax" in scalers:
        minmax_scaler, cols = cast(tuple[MinMaxScaler, list[str]], scalers["minmax"])
        transformed = minmax_scaler.transform(df[cols].astype(float))
        for i, col in enumerate(cols):
            out[col] = transformed[:, i]

    cyclical = scalers.get("cyclical", [])
    for col in cyclical:
        if col in df.columns:
            out[col] = df[col].astype(float)

    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0.0

    return out[FEATURES].astype(float)


def sliding_windows(
    arr: np.ndarray,
    mask: np.ndarray,
    window_size: int = 144,
    stride: int = 6,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Create sliding windows and split normal/anomalous sequences.

    Returns:
        Tuple of (normal_windows, anomalous_windows, window_start_indices)
    """
    normal_windows: List[np.ndarray] = []
    anomalous_windows: List[np.ndarray] = []
    normal_indices: List[int] = []
    anomalous_indices: List[int] = []

    for start in range(0, arr.shape[0] - window_size + 1, stride):
        end = start + window_size
        win = arr[start:end]
        win_mask = mask[start:end]
        if win_mask.any():
            anomalous_windows.append(win)
            anomalous_indices.append(start)
        else:
            normal_windows.append(win)
            normal_indices.append(start)

    return normal_windows, anomalous_windows, normal_indices + anomalous_indices


def split_windows_temporally(
    normal_windows: List[np.ndarray],
    anomalous_windows: List[np.ndarray],
    window_indices: List[int],
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split windows temporally into train/test sets.

    Uses window start indices to determine temporal order.
    All anomalous windows go to test set.
    """
    if not window_indices:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    sorted_indices = sorted(range(len(window_indices)), key=lambda i: window_indices[i])
    normal_count = len(normal_windows)

    train_size = int(normal_count * train_ratio)

    x_train_list: list[np.ndarray] = []
    x_test_list: list[np.ndarray] = []
    y_test_list: list[int] = []

    for i, sorted_i in enumerate(sorted_indices):
        is_normal = sorted_i < normal_count
        win = (
            normal_windows[sorted_i]
            if is_normal
            else anomalous_windows[sorted_i - normal_count]
        )

        if is_normal and sorted_i < train_size:
            x_train_list.append(win)
        else:
            x_test_list.append(win)
            y_test_list.append(0 if is_normal else 1)

    if normal_windows:
        ref_shape = normal_windows[0].shape
    elif anomalous_windows:
        ref_shape = anomalous_windows[0].shape
    else:
        ref_shape = (0, 0)

    x_train = np.stack(x_train_list) if x_train_list else np.empty((0, *ref_shape))
    x_test = np.stack(x_test_list) if x_test_list else np.empty((0, *ref_shape))
    y_test = np.array(y_test_list, dtype=np.int64)

    return x_train, x_test, y_test


def split_and_sample(
    normal_windows: List[np.ndarray],
    anomalous_windows: List[np.ndarray],
    normal_sample_in_test: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/test arrays and labels from window lists."""
    rng = np.random.RandomState(random_state)
    n_norm = len(normal_windows)
    n_sample = int(round(n_norm * normal_sample_in_test)) if n_norm > 0 else 0

    sample_idx = (
        rng.choice(n_norm, size=n_sample, replace=False)
        if n_sample > 0
        else np.array([], dtype=int)
    )

    x_test_list: list[np.ndarray] = []
    y_test_list: list[int] = []

    for win in anomalous_windows:
        x_test_list.append(win)
        y_test_list.append(1)

    for idx in sample_idx:
        x_test_list.append(normal_windows[idx])
        y_test_list.append(0)

    sample_idx_set = set(sample_idx.tolist())
    x_train_list = [
        win for idx, win in enumerate(normal_windows) if idx not in sample_idx_set
    ]

    if normal_windows:
        ref_shape = normal_windows[0].shape
    elif anomalous_windows:
        ref_shape = anomalous_windows[0].shape
    else:
        ref_shape = (0, 0)

    x_train = np.stack(x_train_list) if x_train_list else np.empty((0, *ref_shape))
    x_test = np.stack(x_test_list) if x_test_list else np.empty((0, *ref_shape))
    y_test = np.array(y_test_list, dtype=np.int64)
    return x_train, x_test, y_test


def fit_per_station_scalers(
    station_train_dfs: Dict[str, pd.DataFrame],
    station_train_masks: Dict[str, pd.Series],
) -> Dict[str, Dict[str, Any]]:
    """Fit scalers per station using only that station's training data."""
    station_scalers: Dict[str, Dict[str, Any]] = {}

    for station_name, train_df in station_train_dfs.items():
        scalers: Dict[str, Any] = {}
        train_mask = station_train_masks[station_name]
        normal_df = train_df.loc[~train_mask]

        if "pressure_hPa" in normal_df.columns:
            pressure_scaler = StandardScaler()
            pressure_scaler.fit(normal_df[["pressure_hPa"]].astype(float))
            scalers["pressure"] = pressure_scaler

        mm_cols = [
            c for c in ["precipitation_mm", "luminous_intensity_lux"] if c in normal_df
        ]
        if mm_cols:
            minmax_scaler = MinMaxScaler()
            minmax_scaler.fit(normal_df[mm_cols].astype(float))
            scalers["minmax"] = (minmax_scaler, mm_cols)

        scalers["cyclical"] = [
            c for c in FEATURES if c in train_df and c.endswith(("_sin", "_cos"))
        ]
        station_scalers[station_name] = scalers

    return station_scalers


def build_global_windows(
    station_dfs: Dict[str, pd.DataFrame],
    station_masks: Dict[str, pd.Series],
    station_scalers: Dict[str, Dict[str, Any]],
    window_size: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Scale each station with per-station scalers and aggregate windows."""
    all_normal_windows: List[np.ndarray] = []
    all_anomalous_windows: List[np.ndarray] = []
    all_indices: List[int] = []

    for station_name, df in station_dfs.items():
        scalers = station_scalers[station_name]
        scaled_df = apply_scalers(df, scalers)
        arr = scaled_df.values
        mask_arr = station_masks[station_name].values.astype(bool)
        normal_wins, anomalous_wins, indices = sliding_windows(
            arr,
            mask_arr,
            window_size=window_size,
            stride=stride,
        )
        all_normal_windows.extend(normal_wins)
        all_anomalous_windows.extend(anomalous_wins)
        all_indices.extend(indices)

    return all_normal_windows, all_anomalous_windows, all_indices
