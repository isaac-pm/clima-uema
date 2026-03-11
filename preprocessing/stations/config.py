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
}

GENERAL_REGIONS = [
    "Todo el país",
    "Resto del país",
    "Todo el territorio nacional",
    "Nacional",
]

SENSOR_CUTOFFS = {
    "sede-central_finca-2": {
        "luminous_intensity_lux": "2025-05-20 18:20:00",
    },
}
