"""
UCR-µEMA Weather Station Data Downloader.

A GUI application for downloading meteorological data from the University of
Costa Rica's micro-Environmental Monitoring Array (µEMA) network.

The tool connects to a Grafana/InfluxDB instance and retrieves:
    - Luminous intensity (lux)
    - Precipitation (mm)
    - Atmospheric pressure (hPa)

For 10 weather stations across Costa Rica.

Usage:
    Run the script directly to launch the GUI:
        $ python ucr_uema_data_downloader.py

Requirements:
    - requests
    - tkinter (included with Python standard library)
"""

import csv
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import requests
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText


# --- Core Configuration ---

CR_TZ = timezone(timedelta(hours=-6))

ALL_FEATURES = {
    "luminous_intensity": "VMEL",
    "precipitation": "lluv",
    "pressure": "LPS",
}
UNIT_MAP = {
    "luminous_intensity": "value_lux",
    "precipitation": "value_mm",
    "pressure": "value_hPa",
}

ALL_STATIONS = {
    "00": {
        "db_id": "UCREscFis",
        "filename": "sede-central_finca-1",
        "field_overrides": {"pressure": "BME"},
        "pressure_offset": 136.3,
    },
    "01": {
        "db_id": "UCRPuntarenas",
        "filename": "recinto-esparza",
        "field_overrides": {"pressure": "LPS"},
        "pressure_offset": 23.8,
    },
    "02": {
        "db_id": "UCRGolfito",
        "filename": "sede-sur_golfito",
        "field_overrides": {"pressure": "LPS"},
        "pressure_offset": 3.1,
    },
    "03": {
        "db_id": "UCRGuapiles",
        "filename": "recinto-guapiles",
        "field_overrides": {"pressure": "BME"},
        "pressure_offset": 32.6,
    },
    "04": {
        "db_id": "UCRLiberia",
        "filename": "sede-guanacaste_liberia",
        "field_overrides": {"pressure": "LPS"},
        "pressure_offset": 15.3,
    },
    "05": {
        "db_id": "UCRLimon",
        "filename": "sede-caribe_limon",
        "field_overrides": {"pressure": "BME"},
        "pressure_offset": 7.4,
    },
    "06": {
        "db_id": "UCRturrialba",
        "filename": "sede-atlantico_turrialba",
        "field_overrides": {"pressure": "LPS"},
        "pressure_offset": 75.1,
    },
    "07": {
        "db_id": "UCRCigefi",
        "filename": "sede-central_finca-2",
        "field_overrides": {"pressure": "LPS"},
        "pressure_offset": 138.3,
    },
    "08": {
        "db_id": "UCRLosic",
        "filename": "sede-central_finca-3",
        "field_overrides": {"pressure": "LPS"},
        "pressure_offset": 138.3,
    },
    "09": {
        "db_id": "UCRSantaCruz",
        "filename": "recinto-santa-cruz",
        "field_overrides": {"pressure": "LPS"},
        "pressure_offset": 5.9,
    },
}


# --- Download Logic ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_flux_payload(
    feature_name: str,
    db_field: str,
    db_station_id: str,
    start_epoch: int,
    end_epoch: int,
    datasource_uid: str,
    pressure_offset: float = 0.0,
) -> dict[str, Any]:
    """Build a Flux query payload for InfluxDB data retrieval.

    Args:
        feature_name: The name of the feature (e.g., "precipitation").
        db_field: The InfluxDB field name (e.g., "lluv").
        db_station_id: The station identifier in the database.
        start_epoch: Start timestamp in milliseconds since epoch.
        end_epoch: End timestamp in milliseconds since epoch.
        datasource_uid: The UID of the InfluxDB datasource in Grafana.
        pressure_offset: Calibration offset to apply to pressure readings.

    Returns:
        A dictionary containing the query payload for the Grafana API.
    """
    agg_function = "mean"
    map_logic = ""
    empty_logic = "false"
    fill_logic = ""

    if feature_name == "precipitation":
        agg_function = "sum"
        map_logic = "|> map(fn: (r) => ({ r with _value: r._value * 0.2794 }))"
        empty_logic = "true"
        fill_logic = "|> fill(value: 0.0)"
    elif feature_name == "pressure":
        map_logic = (
            f"|> map(fn: (r) => ({{ r with _value: r._value + {pressure_offset} }}))"
        )

    flux_query = f"""from(bucket: "MQTT")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["tipo"] == "{db_station_id}")
  |> filter(fn: (r) => exists r._value)
  |> filter(fn: (r) => r._field == "{db_field}")
  {map_logic}
  |> aggregateWindow(every: 10m, fn: {agg_function}, createEmpty: {empty_logic})
  {fill_logic}
  |> group()
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")"""

    return {
        "queries": [
            {
                "datasource": {"type": "influxdb", "uid": datasource_uid},
                "query": flux_query,
                "refId": "A",
                "format": "time_series",
                "intervalMs": 600000,
                "maxDataPoints": 100000,
            }
        ],
        "from": str(start_epoch),
        "to": str(end_epoch),
    }


def fetch_and_save_data(
    grafana_url: str,
    datasource_uid: str,
    username: str,
    password: str,
    start_date: datetime,
    end_date: datetime,
    selected_features: dict[str, str],
    selected_stations: dict[str, dict[str, str]],
    stop_event: Optional[threading.Event] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Fetch and save meteorological data from Grafana/InfluxDB.

    Downloads data for the specified features and stations, saving each
    station's data to a separate CSV file in a directory named after
    the feature.

    Args:
        grafana_url: Base URL of the Grafana instance.
        datasource_uid: UID of the InfluxDB datasource.
        username: Username for HTTP basic authentication.
        password: Password for HTTP basic authentication.
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.
        selected_features: Dict mapping feature names to database field names.
        selected_stations: Dict mapping station keys to station info dicts.
        stop_event: Optional threading.Event to signal download cancellation.
        log_callback: Optional callback function for logging messages.
    """

    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg + "\n")

    base_dir = SCRIPT_DIR

    for feature in selected_features.keys():
        os.makedirs(os.path.join(base_dir, feature), exist_ok=True)

    api_endpoint = f"{grafana_url}/api/ds/query"
    auth = (username, password)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    for feature_name, db_field in selected_features.items():
        if stop_event and stop_event.is_set():
            log("[!] Stop requested. Aborting remaining features.")
            return
        column_header = UNIT_MAP[feature_name]

        for file_prefix, station_info in selected_stations.items():
            if stop_event and stop_event.is_set():
                log("[!] Stop requested. Aborting remaining stations.")
                return
            station_db_id = station_info["db_id"]
            station_filename = station_info["filename"]
            actual_db_field = station_info.get("field_overrides", {}).get(
                feature_name, db_field
            )
            pressure_offset = station_info.get("pressure_offset", 0.0)
            filepath = os.path.join(
                base_dir,
                feature_name,
                f"{file_prefix}_{feature_name}_{station_filename}.csv",
            )

            log(f"\n--- Processing {feature_name} for {station_filename} ---")

            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["time", column_header])

                total_rows = 0
                current_start = start_date

                while current_start < end_date:
                    if stop_event and stop_event.is_set():
                        log("[!] Stop requested. Aborting current station download.")
                        break
                    current_end = current_start + timedelta(days=30)
                    if current_end > end_date:
                        current_end = end_date

                    start_epoch = int(current_start.timestamp() * 1000)
                    end_epoch = int(current_end.timestamp() * 1000)

                    log(
                        f"  Fetching: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}..."
                    )

                    payload = build_flux_payload(
                        feature_name,
                        actual_db_field,
                        station_db_id,
                        start_epoch,
                        end_epoch,
                        datasource_uid,
                        pressure_offset,
                    )

                    try:
                        response = requests.post(
                            api_endpoint,
                            auth=auth,
                            headers=headers,
                            data=json.dumps(payload),
                        )
                        response.raise_for_status()
                        data = response.json()

                        frames = data.get("results", {}).get("A", {}).get("frames", [])
                        if frames:
                            frame = frames[0]
                            if (
                                "data" in frame
                                and "values" in frame["data"]
                                and len(frame["data"]["values"]) >= 2
                            ):
                                time_array = frame["data"]["values"][0]
                                value_array = frame["data"]["values"][1]

                                for t, v in zip(time_array, value_array):
                                    dt = datetime.fromtimestamp(
                                        t / 1000.0, tz=timezone.utc
                                    ).astimezone(CR_TZ)
                                    writer.writerow(
                                        [dt.strftime("%Y-%m-%d %H:%M:%S"), v]
                                    )
                                    total_rows += 1
                            else:
                                log(
                                    "  [!] Frame found but no values present. Skipping."
                                )
                        else:
                            log("  [!] No frames returned. Skipping.")

                    except requests.exceptions.RequestException as e:
                        log(f"  [!] HTTP Error: {e}")

                    current_start = current_end

            log(f"Finished saving: {filepath} | Total rows written: {total_rows}")
    log("\n=== ALL DOWNLOADS COMPLETE ===")


# --- UI Application ---
class DownloaderApp:
    """GUI application for UCR-µEMA data downloading.

    Provides a Tkinter-based interface for configuring and executing
    data downloads from the UCR-µEMA weather station network.
    """

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the downloader application.

        Args:
            root: The root Tkinter window.
        """
        self.root = root
        self.root.title("UCR-µEMA Data Downloader")
        self.root.geometry("650x750")
        self.root.configure(padx=20, pady=20)

        # Server & Credentials
        settings_frame = ttk.LabelFrame(root, text="Server & Credentials", padding=10)
        settings_frame.pack(fill="x", pady=5)

        ttk.Label(settings_frame, text="Grafana URL:").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.url_entry = ttk.Entry(settings_frame, width=35)
        self.url_entry.insert(0, "http://relampagos.ucr.ac.cr:3000")
        self.url_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(settings_frame, text="Datasource UID:").grid(
            row=1, column=0, sticky="w", padx=5
        )
        self.uid_entry = ttk.Entry(settings_frame, width=35)
        self.uid_entry.insert(0, "ddvqh2e51ge0wd")
        self.uid_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(settings_frame, text="Username:").grid(
            row=2, column=0, sticky="w", padx=5
        )
        self.user_entry = ttk.Entry(settings_frame, width=35)
        self.user_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(settings_frame, text="Password:").grid(
            row=3, column=0, sticky="w", padx=5
        )
        self.pass_entry = ttk.Entry(settings_frame, show="*", width=35)
        self.pass_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        # Dates
        date_frame = ttk.LabelFrame(root, text="Date Range (YYYY-MM-DD)", padding=10)
        date_frame.pack(fill="x", pady=5)

        ttk.Label(date_frame, text="Start Date:").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.start_entry = ttk.Entry(date_frame)
        self.start_entry.insert(0, "2024-01-01")
        self.start_entry.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(date_frame, text="End Date:").grid(
            row=0, column=2, sticky="w", padx=5
        )
        self.end_entry = ttk.Entry(date_frame)
        self.end_entry.insert(0, "2026-03-01")
        self.end_entry.grid(row=0, column=3, sticky="w", padx=5)

        # Features
        feat_frame = ttk.LabelFrame(root, text="Features", padding=10)
        feat_frame.pack(fill="x", pady=5)
        self.feature_vars = {}
        for idx, (feat, _) in enumerate(ALL_FEATURES.items()):
            var = tk.BooleanVar(value=True)
            self.feature_vars[feat] = var
            ttk.Checkbutton(
                feat_frame, text=feat.replace("_", " ").title(), variable=var
            ).grid(row=0, column=idx, padx=10, sticky="w")

        # Stations
        stat_frame = ttk.LabelFrame(root, text="Stations (Select multiple)", padding=10)
        stat_frame.pack(fill="x", pady=5)

        # Use a scrollable frame of checkbuttons so selections are visually explicit
        self.station_canvas = tk.Canvas(stat_frame, height=160)
        self.station_scroll = ttk.Scrollbar(
            stat_frame, orient="vertical", command=self.station_canvas.yview
        )
        self.station_inner = ttk.Frame(self.station_canvas)
        self.station_inner.bind(
            "<Configure>",
            lambda e: self.station_canvas.configure(
                scrollregion=self.station_canvas.bbox("all")
            ),
        )
        self.station_canvas.create_window(
            (0, 0), window=self.station_inner, anchor="nw"
        )
        self.station_canvas.configure(yscrollcommand=self.station_scroll.set)
        self.station_canvas.pack(side="left", fill="both", expand=True)
        self.station_scroll.pack(side="right", fill="y")

        self.station_keys = list(ALL_STATIONS.keys())
        self.station_vars = {}
        for key in self.station_keys:
            var = tk.BooleanVar(value=True)
            self.station_vars[key] = var
            ttk.Checkbutton(
                self.station_inner,
                text=f"[{key}] {ALL_STATIONS[key]['filename']}",
                variable=var,
            ).pack(anchor="w", padx=5, pady=2)

        # Select/Clear buttons for convenience
        sel_btn_frame = ttk.Frame(stat_frame)
        sel_btn_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(
            sel_btn_frame, text="Select All", command=lambda: self.select_all_stations()
        ).pack(side="left", padx=5)
        ttk.Button(
            sel_btn_frame, text="Clear All", command=lambda: self.clear_all_stations()
        ).pack(side="left", padx=5)

        # Run / Stop Buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        self.run_btn = ttk.Button(
            btn_frame, text="Download Data", command=self.start_download
        )
        self.run_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.request_stop)
        self.stop_btn.pack(side="left", padx=5)
        self.stop_btn.config(state="disabled")

        # Log Output
        log_frame = ttk.LabelFrame(root, text="Progress Log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=5)
        self.log_text = ScrolledText(log_frame, state="disabled", height=10)
        self.log_text.pack(fill="both", expand=True)

    def write_log(self, msg: str) -> None:
        """Thread-safe logging via tkinter main loop.

        Args:
            msg: The message to log.
        """
        self.root.after(0, self._append_log, msg)

    def _append_log(self, msg: str) -> None:
        """Append a message to the log text widget.

        Args:
            msg: The message to append.
        """
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def start_download(self) -> None:
        """Validate inputs and initiate the data download process."""
        try:
            start_date = datetime.strptime(self.start_entry.get(), "%Y-%m-%d").replace(
                tzinfo=CR_TZ
            )
            end_date = datetime.strptime(self.end_entry.get(), "%Y-%m-%d").replace(
                tzinfo=CR_TZ
            )
        except ValueError:
            messagebox.showerror(
                "Invalid Date", "Please ensure dates are in YYYY-MM-DD format."
            )
            return

        selected_features = {
            k: ALL_FEATURES[k] for k, v in self.feature_vars.items() if v.get()
        }
        if not selected_features:
            messagebox.showwarning("No Features", "Please select at least one feature.")
            return

        selected_stations = {
            k: ALL_STATIONS[k] for k, v in self.station_vars.items() if v.get()
        }
        if not selected_stations:
            messagebox.showwarning("No Stations", "Please select at least one station.")
            return

        grafana_url = self.url_entry.get()
        datasource_uid = self.uid_entry.get()
        username = self.user_entry.get()
        password = self.pass_entry.get()

        if not username or not password:
            messagebox.showwarning(
                "Credentials Missing", "Please enter your username and password."
            )
            return

        self.run_btn.config(state="disabled")
        self.stop_event = threading.Event()
        self.stop_btn.config(state="normal")
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")

        thread = threading.Thread(
            target=self._run_task,
            args=(
                grafana_url,
                datasource_uid,
                username,
                password,
                start_date,
                end_date,
                selected_features,
                selected_stations,
            ),
        )
        thread.start()

    def request_stop(self) -> None:
        """Signal the download thread to stop."""
        if hasattr(self, "stop_event") and self.stop_event:
            self.stop_event.set()
            self.stop_btn.config(state="disabled")
            self.write_log("[!] Stop requested by user.")

    def select_all_stations(self) -> None:
        """Select all station checkboxes."""
        for var in getattr(self, "station_vars", {}).values():
            var.set(True)

    def clear_all_stations(self) -> None:
        """Clear all station checkboxes."""
        for var in getattr(self, "station_vars", {}).values():
            var.set(False)

    def _run_task(self, *args: Any) -> None:
        """Execute the download task in a separate thread.

        Args:
            *args: Arguments passed to fetch_and_save_data.
        """
        try:
            fetch_and_save_data(
                *args,
                stop_event=getattr(self, "stop_event", None),
                log_callback=self.write_log,
            )
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))


if __name__ == "__main__":
    root = tk.Tk()
    app = DownloaderApp(root)
    root.mainloop()
