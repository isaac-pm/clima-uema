"""Build Silver-layer station datasets from raw UEMA sensor CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing.stations.config import PROCESSED_SILVER_DIR, STATION_NAMES
from preprocessing.stations.silver_pipeline import process_station


def main() -> None:
    """Process stations into silver CSV files."""
    parser = argparse.ArgumentParser(
        description="Build Silver-layer station datasets from raw sensor CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_SILVER_DIR),
        help="Output directory for processed silver CSV files",
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        default=STATION_NAMES,
        help="Optional list of station names to process",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for station_name in args.stations:
        print(f"Processing station: {station_name}")
        df = process_station(station_name)
        output_file = output_dir / f"{station_name}.csv"
        df.to_csv(output_file)
        print(f"  Saved {len(df)} rows to {output_file.name}")


if __name__ == "__main__":
    main()
