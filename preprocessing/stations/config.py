"""Shared configuration for station preprocessing pipelines."""

from __future__ import annotations

from pathlib import Path

RAW_DATA_DIR = Path("data/stations/raw")
PROCESSED_SILVER_DIR = Path("data/stations/processed/silver")
PROCESSED_GOLD_DIR = Path("data/stations/processed/gold")
ALERTS_DATA_PATH = Path("data/emergency_alerts/processed/alerts_data.csv")

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
    "sede-central_sabanilla",
    "sede-central_losic-norte-1",
    "sede-central_losic-norte-2",
]

# Keep specific regions before broader labels.
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
    # Region inferred from shared pressure calibration offset with the
    # existing sede-central_finca-* stations (136.3/138.3) — not yet
    # confirmed with station maintainers.
    "sede-central_sabanilla": ["Valle Central", "Región Central", "Central"],
    "sede-central_losic-norte-1": ["Valle Central", "Región Central", "Central"],
    "sede-central_losic-norte-2": ["Valle Central", "Región Central", "Central"],
}

GENERAL_REGIONS = [
    "Todo el país",
    "Resto del país",
    "Todo el territorio nacional",
    "Nacional",
]

SENSOR_CUTOFFS = {
    "sede-central_finca-2": {
        "luminous_intensity_lux": "2025-05-20 18:20:00",  # This sensor started working on this date. We can discard earlier data.
    },
}

# Anomaly labelling buffer: asymmetric around alert issue time.
# Pre-alert: captures developing conditions within the CNE forecast horizon.
# Post-alert: covers event duration + sensor recovery (pressure, precipitation).
# Reasoning: CNE alert lead time is typically 12-48h, so 48h pre captures the
# full forecast horizon without going too far back into genuinely normal data.
# Post window covers event duration + pressure/precipitation recovery.
ALERT_PRE_BUFFER_HOURS: int = 48
ALERT_POST_BUFFER_HOURS: int = 120
