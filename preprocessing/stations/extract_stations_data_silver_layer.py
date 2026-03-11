"""Build Silver-layer station datasets from raw UEMA sensor CSV files.

Pipeline steps:
1. Merge pressure, precipitation, and luminous-intensity streams per station.
2. Resample to a strict 10-minute timeline.
3. Align shared time range and fill gaps.
4. Add cyclical time features.
5. Join emergency-alert features.

Output files are written to ``data/stations/processed/silver/``.
"""

import os
from glob import glob

import numpy as np
import pandas as pd

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

RAW_DATA_DIR = "data/stations/raw"
"""Path to raw sensor CSV files (organized by measurement type)."""

PROCESSED_SILVER_DIR = "data/stations/processed/silver"
"""Path to output silver-layer processed CSV files."""

ALERTS_DATA_PATH = "data/emergency_alerts/processed/alerts_data.csv"
"""Path to meteorological emergency alerts CSV from IMN/CNE."""


# =============================================================================
# STATION CONFIGURATION
# =============================================================================

STATION_NAMES: list[str] = [
    "sede-central_finca-1",
    "sede-central_finca-2",
    "sede-central_finca-3",
    "sede-atlantico_turrialba",
    "sede-caribe_limon",
    "sede-guanacaste_liberia",
    "sede-sur_golfito",
    "recinto-esparza",
    "recinto-guapiles",
    "recinto-santa-cruz",
]
"""List of all station identifiers in the UEMA network."""


# =============================================================================
# REGION MAPPING FOR ALERT MATCHING
# =============================================================================
# Station-to-region mapping used for alert matching.
# Keep specific regions before broad ones (e.g., "Caribe Norte" before "Caribe").

STATION_REGIONS: dict[str, list[str]] = {
    "sede-central_finca-1": ["Valle Central", "Región Central", "Central"],
    "sede-central_finca-2": ["Valle Central", "Región Central", "Central"],
    "sede-central_finca-3": ["Valle Central", "Región Central", "Central"],
    "sede-atlantico_turrialba": [
        "Región Central Este (Oreamuno, Paraíso, Alvarado, Jiménez, Turrialba)",
        "Valle Central",
        "Región Central",
        "Central",
    ],
    "sede-caribe_limon": ["Caribe Sur", "Región Caribe", "Huetar Caribe", "Caribe"],
    "sede-guanacaste_liberia": ["Pacífico Norte"],
    "sede-sur_golfito": ["Pacífico Sur", "Pacifico Sur"],
    "recinto-esparza": ["Pacífico Central"],
    "recinto-guapiles": ["Caribe Norte", "Región Caribe", "Huetar Caribe", "Caribe"],
    "recinto-santa-cruz": ["Pacífico Norte"],
}

# Region labels that apply to all stations as fallback matches.
GENERAL_REGIONS = [
    "Todo el país",
    "Resto del país",
    "Todo el territorio nacional",
    "Nacional",
]


# =============================================================================
# SENSOR-SPECIFIC CONFIGURATION
# =============================================================================
# Per-station sensor cutoffs for known bad periods.
# Data earlier than each cutoff timestamp is discarded.

SENSOR_CUTOFFS = {
    "sede-central_finca-2": {
        "luminous_intensity_lux": pd.Timestamp("2025-05-20 18:20:00"),
    },
}


def get_station_file_paths(station_name: str) -> dict[str, str]:
    """Locate raw CSV files for a station's three sensor types.

    Args:
        station_name: Station identifier (e.g., 'sede-central_finca-1')

    Returns:
        Dict with keys 'pressure', 'precipitation', 'luminous_intensity'
        mapping to file paths.

    Raises:
        FileNotFoundError: If any of the three required sensor files
            are missing for the station.
    """
    pressure_files = glob(f"{RAW_DATA_DIR}/pressure/*{station_name}.csv")
    precip_files = glob(f"{RAW_DATA_DIR}/precipitation/*{station_name}.csv")
    luminous_files = glob(f"{RAW_DATA_DIR}/luminous_intensity/*{station_name}.csv")

    if not (pressure_files and precip_files and luminous_files):
        raise FileNotFoundError(f"Missing files for station: {station_name}")

    return {
        "pressure": pressure_files[0],
        "precipitation": precip_files[0],
        "luminous_intensity": luminous_files[0],
    }


def load_and_prepare_sensor_data(
    file_path: str,
    value_column_name: str,
) -> pd.DataFrame:
    """Load a single sensor CSV and prepare it with datetime index.

    Args:
        file_path: Path to raw sensor CSV file.
        value_column_name: Name to rename the sensor value column to
            (e.g., 'pressure_hPa', 'precipitation_mm').

    Returns:
        DataFrame with datetime index and single value column.
    """
    df = pd.read_csv(file_path, parse_dates=["time"], index_col="time")
    existing_value_col = [c for c in df.columns if c.startswith("value_")][0]
    df = df.rename(columns={existing_value_col: value_column_name})
    return df


def consolidate_station_data(station_name: str) -> pd.DataFrame:
    """Load and merge pressure, precipitation, and luminous intensity for a station.

    This function performs Step 1 of the pipeline: loading raw sensor data
    from separate CSV files and merging them into a single DataFrame.

    The merge uses outer join to keep all timestamps - gaps will be filled
    in later steps via resampling.

    Args:
        station_name: Station identifier.

    Returns:
        DataFrame with all three sensor columns, indexed by datetime.
        Rows where any sensor has no data will have NaN values.
    """
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
                luminous_df = luminous_df[luminous_df.index >= cutoff_time]

    merged_df = pd.concat([pressure_df, precipitation_df, luminous_df], axis=1)

    return merged_df


def resample_to_10min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample dataframe to strict 10-minute frequency.

    This enforces a regular time grid where missing readings appear as
    explicit NaN rows rather than invisible gaps. Uses .mean() to handle
    minor clock drift in sensor timestamps without dropping rows.

    Args:
        df: DataFrame with datetime index.

    Returns:
        DataFrame resampled to 10-minute intervals.
    """
    return df.resample("10min").mean()


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to overlapping timeline and fill intermittent gaps.

    Keeps only the overlap where all sensors exist, then fills gaps:
    - pressure_hPa: linear interpolation
    - precipitation_mm and luminous_intensity_lux: fill with zeros

    Args:
        df: DataFrame with sensor columns and datetime index.

    Returns:
        DataFrame trimmed to overlapping period with gaps filled.
    """
    # Find strict overlap where all sensors have data.
    start_time = df.apply(lambda col: col.first_valid_index()).max()
    end_time = df.apply(lambda col: col.last_valid_index()).min()

    # Slice to overlap window.
    df = df.loc[start_time:end_time].copy()

    # Fill intermittent gaps within the valid window.
    df["pressure_hPa"] = df["pressure_hPa"].interpolate(method="linear")

    df["precipitation_mm"] = df["precipitation_mm"].fillna(0)
    df["luminous_intensity_lux"] = df["luminous_intensity_lux"].fillna(0)

    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sine/cosine features for hour and day of year.

    Encodes hour/day cyclicality to preserve temporal continuity.

    Args:
        df: DataFrame with datetime index.

    Returns:
        DataFrame with added columns:
        - hour_sin, hour_cos: Cyclical encoding of time of day
        - day_of_year_sin, day_of_year_cos: Cyclical encoding of season
    """
    df = df.copy()

    datetime_index = pd.DatetimeIndex(df.index)

    # Minute-level hour fraction.
    hour = datetime_index.hour
    minute = datetime_index.minute
    total_minutes = hour * 60 + minute
    hour_fraction = total_minutes / (24 * 60)

    df["hour_sin"] = np.sin(2 * np.pi * hour_fraction)
    df["hour_cos"] = np.cos(2 * np.pi * hour_fraction)

    day_of_year = datetime_index.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    return df


def normalize_alert_id(alert_number: str) -> str:
    """Normalize alert ID to NNN-YYYY format.

    Converts alert IDs like '01-2024', '002-24', '014-24' to consistent
    '001-2024' format with 3-digit number and 4-digit year.

    Args:
        alert_number: Original alert number string (e.g., '01-2024', '002-24')

    Returns:
        Normalized alert ID in format 'NNN-YYYY'
    """
    import re

    match = re.match(r"(\d+)[-\s](\d+)", str(alert_number))
    if not match:
        return str(alert_number)

    num_part, year_part = match.groups()

    if len(year_part) == 2:
        year_part = "20" + year_part

    normalized_num = num_part.zfill(3)
    normalized_year = year_part.zfill(4)

    return f"{normalized_num}-{normalized_year}"


def add_alert_features(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """Add meteorological emergency alert features to station data.

    This integrates IMN/CNE weather alerts into the time-series data.

    Region Matching:
        - Checks if station's mapped regions appear in alert's affected regions
        - Case-insensitive substring matching
        - General regions (nation-wide) are matched as fallback

    Severity Hierarchy:
        - Red > Orange > Yellow > Green (highest severity wins)

    Alert State Propagation:
        - Uses pd.merge_asof with direction='backward' to propagate
          the current alert state forward in time
        - Before first alert or if no match: is_active_alert = False

    Expiration Rule:
        - Alerts auto-expire after 3 days (72 hours) if no cancellation issued
        - This handles alerts that remain active but become stale

    Args:
        df: DataFrame with datetime index.
        station_name: Station identifier for region lookup.

    Returns:
        DataFrame with added alert columns:
        - is_active_alert: bool
        - alert_severity: str ('Green', 'Yellow', 'Orange', 'Red') or None
        - alert_id: str or None (format: NNN-YYYY with 3-digit number)
    """
    alerts_df = pd.read_csv(ALERTS_DATA_PATH)

    alerts_df["alert_datetime"] = pd.to_datetime(
        alerts_df["issue_date"].astype(str) + " " + alerts_df["issue_time"].astype(str)
    )
    alerts_df = alerts_df.sort_values("alert_datetime").reset_index(drop=True)

    station_regions = STATION_REGIONS.get(station_name, [])

    def check_region_match(alert_row: pd.Series) -> tuple[bool, str | None, str | None]:
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

    events = []
    for _, alert_row in alerts_df.iterrows():
        is_active, severity, alert_id = check_region_match(alert_row)
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

    # Keep matched alert time for expiration logic.
    events_df["matched_alert_time"] = events_df["alert_datetime"]
    events_df = events_df.set_index("alert_datetime")

    df = df.reset_index()
    df = df.rename(columns={"time": "timestamp"})

    merged_df = pd.merge_asof(
        df.sort_values("timestamp"),
        events_df.sort_index(),
        left_on="timestamp",
        right_index=True,
        direction="backward",
    )

    merged_df["is_active_alert"] = (
        merged_df["is_active_alert"].fillna(False).astype(bool)
    )

    if "matched_alert_time" in merged_df.columns:
        # Time elapsed since matched alert was issued.
        elapsed = merged_df["timestamp"] - merged_df["matched_alert_time"]

        # Expire alerts after 3 days.
        expired_mask = elapsed > pd.Timedelta(days=3)

        merged_df.loc[expired_mask, "is_active_alert"] = False
        merged_df.loc[expired_mask, "alert_severity"] = None
        merged_df.loc[expired_mask, "alert_id"] = None

        # Drop temporary helper column.
        merged_df = merged_df.drop(columns=["matched_alert_time"])

    merged_df = merged_df.rename(columns={"timestamp": "time"})
    merged_df = merged_df.set_index("time")

    return merged_df


def process_station(station_name: str) -> pd.DataFrame:
    """Process a single station through all transformation steps.

    Args:
        station_name: Station identifier.

    Returns:
        Processed DataFrame ready for downstream modeling.
    """
    df = consolidate_station_data(station_name)
    df = resample_to_10min(df)
    df = handle_missing_data(df)
    df = add_cyclical_time_features(df)
    df = add_alert_features(df, station_name)
    return df


def save_station_data(df: pd.DataFrame, station_name: str) -> None:
    """Save processed station data to silver folder as CSV."""
    output_file = os.path.join(PROCESSED_SILVER_DIR, f"{station_name}.csv")
    df.to_csv(output_file)


def process_all_stations() -> None:
    """Process all stations and save to silver folder."""
    os.makedirs(PROCESSED_SILVER_DIR, exist_ok=True)

    for station_name in STATION_NAMES:
        print(f"Processing station: {station_name}")
        df = process_station(station_name)
        save_station_data(df, station_name)
        print(f"  Saved {len(df)} rows to {station_name}.csv")


if __name__ == "__main__":
    process_all_stations()
