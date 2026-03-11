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
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create sliding windows and split normal/anomalous sequences."""
    normal_windows: List[np.ndarray] = []
    anomalous_windows: List[np.ndarray] = []

    for start in range(0, arr.shape[0] - window_size + 1, stride):
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


def fit_global_scalers(
    station_dfs: Dict[str, pd.DataFrame],
    station_masks: Dict[str, pd.Series],
) -> Dict[str, Any]:
    """Fit one scaler set on normal rows from all stations."""
    scalers: Dict[str, Any] = {}

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

    scalers["cyclical"] = [c for c in FEATURES if c.endswith(("_sin", "_cos"))]
    return scalers


def build_global_windows(
    station_dfs: Dict[str, pd.DataFrame],
    station_masks: Dict[str, pd.Series],
    scalers: Dict[str, Any],
    window_size: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Scale each station and aggregate all normal/anomalous windows."""
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

    return all_normal_windows, all_anomalous_windows
