"""Silver-layer processing helpers for UEMA station datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from preprocessing.stations.config import (
    ALERTS_DATA_PATH,
    GENERAL_REGIONS,
    RAW_DATA_DIR,
    SENSOR_CUTOFFS,
    STATION_NAMES,
    STATION_REGIONS,
)


def get_station_file_paths(station_name: str) -> dict[str, Path]:
    """Locate raw CSV files for a station's three sensor types."""
    pressure_files = list((RAW_DATA_DIR / "pressure").glob(f"*{station_name}.csv"))
    precip_files = list((RAW_DATA_DIR / "precipitation").glob(f"*{station_name}.csv"))
    luminous_files = list(
        (RAW_DATA_DIR / "luminous_intensity").glob(f"*{station_name}.csv")
    )

    if not (pressure_files and precip_files and luminous_files):
        raise FileNotFoundError(f"Missing files for station: {station_name}")

    return {
        "pressure": pressure_files[0],
        "precipitation": precip_files[0],
        "luminous_intensity": luminous_files[0],
    }


def load_and_prepare_sensor_data(
    file_path: Path,
    value_column_name: str,
) -> pd.DataFrame:
    """Load a single sensor CSV and prepare it with datetime index."""
    df = pd.read_csv(file_path, parse_dates=["time"], index_col="time")
    existing_value_col = [c for c in df.columns if c.startswith("value_")][0]
    df = df.rename(columns={existing_value_col: value_column_name})
    return df


def consolidate_station_data(station_name: str) -> pd.DataFrame:
    """Load and merge pressure, precipitation, and luminous intensity."""
    file_paths = get_station_file_paths(station_name)

    pressure_df = load_and_prepare_sensor_data(file_paths["pressure"], "pressure_hPa")
    precipitation_df = load_and_prepare_sensor_data(
        file_paths["precipitation"], "precipitation_mm"
    )
    luminous_df = load_and_prepare_sensor_data(
        file_paths["luminous_intensity"], "luminous_intensity_lux"
    )

    if station_name in SENSOR_CUTOFFS:
        for sensor_col, cutoff_time in SENSOR_CUTOFFS[station_name].items():
            if sensor_col in luminous_df.columns:
                cutoff_timestamp = pd.Timestamp(cutoff_time)
                luminous_df = luminous_df[luminous_df.index >= cutoff_timestamp]

    merged_df = pd.concat([pressure_df, precipitation_df, luminous_df], axis=1)
    return merged_df


def resample_to_10min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample dataframe to strict 10-minute frequency."""
    return df.resample("10min").mean()


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to overlap period and fill intermittent gaps."""
    start_time = df.apply(lambda col: col.first_valid_index()).max()
    end_time = df.apply(lambda col: col.last_valid_index()).min()

    df = df.loc[start_time:end_time].copy()
    df["pressure_hPa"] = df["pressure_hPa"].interpolate(method="linear")
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0)
    df["luminous_intensity_lux"] = df["luminous_intensity_lux"].fillna(0)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sine/cosine features for hour and day of year."""
    df = df.copy()
    datetime_index = pd.DatetimeIndex(df.index)

    total_minutes = (datetime_index.hour * 60) + datetime_index.minute
    hour_fraction = total_minutes / (24 * 60)
    df["hour_sin"] = np.sin(2 * np.pi * hour_fraction)
    df["hour_cos"] = np.cos(2 * np.pi * hour_fraction)

    day_of_year = datetime_index.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    return df


def normalize_alert_id(alert_number: str) -> str:
    """Normalize alert ID to NNN-YYYY format."""
    match = re.match(r"(\d+)[-\s](\d+)", str(alert_number))
    if not match:
        return str(alert_number)

    num_part, year_part = match.groups()
    if len(year_part) == 2:
        year_part = "20" + year_part

    normalized_num = num_part.zfill(3)
    normalized_year = year_part.zfill(4)
    return f"{normalized_num}-{normalized_year}"


def _check_region_match(
    alert_row: pd.Series,
    station_regions: list[str],
) -> tuple[bool, Optional[str], Optional[str]]:
    """Return alert status tuple for a station/alert pair."""
    category = str(alert_row["alert_category"]).strip()
    if category == "Cancellation":
        return (False, None, None)

    severity_cols = [
        ("Red", alert_row.get("regions_red_alert")),
        ("Orange", alert_row.get("regions_orange_alert")),
        ("Yellow", alert_row.get("regions_yellow_alert")),
        ("Green", alert_row.get("regions_green_alert")),
    ]

    matched_severity = None
    for severity, region_col in severity_cols:
        if pd.isna(region_col) or str(region_col).strip() == "":
            continue

        region_str = str(region_col).lower()

        for station_region in station_regions:
            if station_region.lower() in region_str:
                matched_severity = severity
                break

        if matched_severity:
            break

        for general_region in GENERAL_REGIONS:
            if general_region.lower() in region_str:
                return (
                    True,
                    matched_severity,
                    normalize_alert_id(alert_row["alert_number"]),
                )

    if matched_severity:
        return (
            True,
            matched_severity,
            normalize_alert_id(alert_row["alert_number"]),
        )

    return (False, None, None)


def add_alert_features(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """Add meteorological emergency alert features to station data."""
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)
    alerts_df["alert_datetime"] = pd.to_datetime(
        alerts_df["issue_date"].astype(str) + " " + alerts_df["issue_time"].astype(str)
    )
    alerts_df = alerts_df.sort_values("alert_datetime").reset_index(drop=True)

    station_regions = STATION_REGIONS.get(station_name, [])
    events = []

    for _, alert_row in alerts_df.iterrows():
        is_active, severity, alert_id = _check_region_match(alert_row, station_regions)
        events.append(
            {
                "alert_datetime": alert_row["alert_datetime"],
                "is_active_alert": is_active,
                "alert_severity": severity,
                "alert_id": alert_id,
            }
        )

    events_df = pd.DataFrame(events)
    if events_df.empty:
        df["is_active_alert"] = False
        df["alert_severity"] = None
        df["alert_id"] = None
        return df

    events_df["matched_alert_time"] = events_df["alert_datetime"]
    events_df = events_df.set_index("alert_datetime")

    df = df.reset_index().rename(columns={"time": "timestamp"})
    merged_df = pd.merge_asof(
        df.sort_values("timestamp"),
        events_df.sort_index(),
        left_on="timestamp",
        right_index=True,
        direction="backward",
    )

    merged_df["is_active_alert"] = merged_df["is_active_alert"].fillna(False).astype(bool)

    if "matched_alert_time" in merged_df.columns:
        elapsed = merged_df["timestamp"] - merged_df["matched_alert_time"]
        expired_mask = elapsed > pd.Timedelta(days=3)
        merged_df.loc[expired_mask, "is_active_alert"] = False
        merged_df.loc[expired_mask, "alert_severity"] = None
        merged_df.loc[expired_mask, "alert_id"] = None
        merged_df = merged_df.drop(columns=["matched_alert_time"])

    merged_df = merged_df.rename(columns={"timestamp": "time"}).set_index("time")
    return merged_df


def process_station(station_name: str) -> pd.DataFrame:
    """Process a single station through all silver transformations."""
    df = consolidate_station_data(station_name)
    df = resample_to_10min(df)
    df = handle_missing_data(df)
    df = add_cyclical_time_features(df)
    df = add_alert_features(df, station_name)
    return df


def process_all_stations() -> dict[str, pd.DataFrame]:
    """Process all configured stations and return a mapping by name."""
    return {station_name: process_station(station_name) for station_name in STATION_NAMES}
