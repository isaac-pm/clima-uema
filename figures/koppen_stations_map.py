#!/usr/bin/env python3
"""
Map of Costa Rica with approximate Köppen-Geiger climate zones and
CLIMA-µEMA meteorological station locations + operational timeline.

Usage:
    python figures/koppen_stations_map.py
Output:
    figures/koppen_stations_map.png
"""

from __future__ import annotations

import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Station data
# TODO: replace lat/lon with real sensor coordinates.
# "excluded" marks stations not used in model training (known data quality issues).
# "since" is derived at runtime from the first record in the silver-layer CSV.
# ─────────────────────────────────────────────────────────────────────────────
STATIONS: list[dict] = [
    # id                              label           lat      lon       excl   label_xy_offset
    {"id": "sede-central_finca-1",    "label": "SC Finca 1",  "lat":  9.937, "lon": -84.055, "excluded": False, "offset": ( 5,  4)},
    {"id": "sede-central_finca-2",    "label": "SC Finca 2",  "lat":  9.942, "lon": -84.051, "excluded": False, "offset": ( 5, -9)},
    {"id": "sede-central_finca-3",    "label": "SC Finca 3",  "lat":  9.947, "lon": -84.047, "excluded": False, "offset": ( 5,  4)},
    {"id": "sede-atlantico_turrialba","label": "Turrialba",   "lat":  9.900, "lon": -83.678, "excluded": False, "offset": ( 5,  4)},
    {"id": "sede-caribe_limon",       "label": "Limón",       "lat":  9.992, "lon": -83.040, "excluded": False, "offset": ( 5,  4)},
    {"id": "sede-guanacaste_liberia", "label": "Liberia",     "lat": 10.634, "lon": -85.434, "excluded": False, "offset": ( 5,  4)},
    {"id": "sede-sur_golfito",        "label": "Golfito",     "lat":  8.640, "lon": -83.173, "excluded": False, "offset": ( 5,  4)},
    {"id": "recinto-esparza",         "label": "Esparza",     "lat":  9.989, "lon": -84.670, "excluded": False, "offset": ( 5,  4)},
    {"id": "recinto-guapiles",        "label": "Guápiles",    "lat": 10.217, "lon": -83.793, "excluded": True,  "offset": ( 5,  4)},
    {"id": "recinto-santa-cruz",      "label": "Santa Cruz",  "lat": 10.267, "lon": -85.582, "excluded": False, "offset": ( 5,  4)},
]

_SILVER_DIR = Path(__file__).parent.parent / "data" / "stations" / "processed" / "silver"


def _load_operational_dates() -> None:
    """Populate each station's 'since' key from the first row of its silver CSV."""
    for s in STATIONS:
        csv_path = _SILVER_DIR / f"{s['id']}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Silver CSV not found for {s['id']}: {csv_path}\n"
                "Run the preprocessing pipeline first."
            )
        times = pd.read_csv(csv_path, usecols=["time"])["time"]
        s["since"] = pd.to_datetime(times.iloc[0])
        s["until"] = pd.to_datetime(times.iloc[-1])

# ─────────────────────────────────────────────────────────────────────────────
# Approximate Köppen-Geiger climate zones
# Polygons are rough geographic approximations — not derived from a raster.
# Colors follow the Beck et al. (2018) standard palette.
# Draw order: lowest priority first; higher-priority zones paint on top.
# ─────────────────────────────────────────────────────────────────────────────
_CHIRIPO  = Point(-83.490,  9.480).buffer(0.09)   # Cerro Chirripó  3 820 m
_IRAZU    = Point(-83.851,  9.979).buffer(0.06)   # Volcán Irazú    3 432 m
_TURRIALB = Point(-83.767,  9.993).buffer(0.04)   # Volcán Turrialba 3 340 m

KOPPEN_ZONES: list[dict] = [
    {
        # Tropical savanna: Guanacaste + Nicoya Peninsula + Pacific NW lowlands.
        # Distinct dry season (Nov–Apr). Roughly everything west of the cordilleras.
        "code": "Aw", "label": "Aw – Tropical savanna",
        "color": "#46AAFA", "priority": 1,
        "poly": Polygon([
            (-86.0, 9.5), (-85.6, 9.1), (-85.1, 8.9), (-84.6, 9.0),
            (-84.3, 9.5), (-84.2, 10.0), (-84.4, 10.5), (-84.5, 11.2),
            (-86.0, 11.2),
        ]),
    },
    {
        # Tropical monsoon: Central Pacific slope + southern transitional zone.
        # Very wet but with a slight dry season; bridges Aw and Af.
        "code": "Am", "label": "Am – Tropical monsoon",
        "color": "#0078FF", "priority": 2,
        "poly": Polygon([
            (-85.3, 8.2), (-84.6, 8.8), (-84.3, 9.5), (-83.7, 9.3),
            (-83.5, 8.9), (-83.8, 8.0), (-85.3, 8.0),
        ]),
    },
    {
        # Tropical rainforest: Caribbean lowlands + Osa Peninsula.
        # Rain year-round (no dry season). Eastern side of the cordillera.
        "code": "Af", "label": "Af – Tropical rainforest",
        "color": "#0000FF", "priority": 3,
        "poly": Polygon([
            (-82.5, 8.0), (-82.5, 11.2), (-84.5, 11.2), (-84.5, 10.5),
            (-84.2, 10.0), (-84.3, 9.5), (-84.6, 9.0), (-85.1, 8.9),
            (-84.1, 8.0),
        ]),
    },
    {
        # Oceanic temperate: Valle Central highlands (900–1 800 m) and
        # mountain slopes of the Cordillera Central and Talamanca.
        "code": "Cfb", "label": "Cfb – Oceanic temperate",
        "color": "#64FF00", "priority": 4,
        "poly": Polygon([
            (-84.6, 9.6), (-83.5, 9.5), (-83.3, 9.9), (-83.4, 10.4),
            (-83.7, 10.7), (-84.3, 10.7), (-84.6, 10.3),
        ]),
    },
    {
        # Alpine tundra: páramo above ~3 000 m (Chirripó, Irazú, Turrialba).
        "code": "ET", "label": "ET – Alpine tundra",
        "color": "#B2B2B2", "priority": 5,
        "poly": unary_union([_CHIRIPO, _IRAZU, _TURRIALB]),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Natural Earth data (cached locally to avoid repeated downloads)
# ─────────────────────────────────────────────────────────────────────────────
_AUX_DIR = Path(__file__).parent.parent / "data" / "aux"
_NE_ZIP  = _AUX_DIR / "ne_10m_admin_0_countries.zip"
_NE_URL  = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_0_countries.zip"
)


def _load_costa_rica() -> gpd.GeoDataFrame:
    _AUX_DIR.mkdir(parents=True, exist_ok=True)
    if not _NE_ZIP.exists():
        print("  Downloading Natural Earth country boundaries (one-time)...")
        urllib.request.urlretrieve(_NE_URL, _NE_ZIP)
    world = gpd.read_file(str(_NE_ZIP))
    return world[world["NAME"] == "Costa Rica"].to_crs("EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fill_geom(ax: plt.Axes, geom, **kw) -> None:
    """Fill a Shapely Polygon or MultiPolygon on a matplotlib Axes."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.fill(x, y, **kw)
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            _fill_geom(ax, part, **kw)


def _scale_bar(ax: plt.Axes, length_deg: float, x0: float, y0: float, label: str) -> None:
    """Draw a simple horizontal scale bar in data coordinates."""
    ax.plot([x0, x0 + length_deg], [y0, y0], "k-", linewidth=2, zorder=8)
    ax.plot([x0, x0], [y0 - 0.02, y0 + 0.02], "k-", linewidth=2, zorder=8)
    ax.plot([x0 + length_deg, x0 + length_deg], [y0 - 0.02, y0 + 0.02], "k-", linewidth=2, zorder=8)
    ax.text(x0 + length_deg / 2, y0 - 0.06, label, ha="center", va="top",
            fontsize=7.5, zorder=8)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(cr: gpd.GeoDataFrame) -> plt.Figure:
    cr_geom = cr.geometry.unary_union
    minx, miny, maxx, maxy = -86.0, 8.0, -82.5, 11.2
    pad = 0.30

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[3, 1], hspace=0.10,
        left=0.04, right=0.97, top=0.95, bottom=0.04,
    )
    ax_map = fig.add_subplot(gs[0])
    ax_tl  = fig.add_subplot(gs[1])

    # ── Ocean background ──────────────────────────────────────────────────────
    ax_map.set_facecolor("#cde5f0")

    # ── Land background (base country fill) ───────────────────────────────────
    cr.plot(ax=ax_map, color="#e8e4d8", edgecolor="none", zorder=1)

    # ── Köppen zones (clip each to Costa Rica boundary) ───────────────────────
    for zone in sorted(KOPPEN_ZONES, key=lambda z: z["priority"]):
        clipped = zone["poly"].intersection(cr_geom)
        _fill_geom(ax_map, clipped, color=zone["color"], alpha=0.72, zorder=2)

    # ── Country border ────────────────────────────────────────────────────────
    cr.plot(ax=ax_map, color="none", edgecolor="#2a2a2a", linewidth=1.3, zorder=5)

    # ── Stations ──────────────────────────────────────────────────────────────
    for s in STATIONS:
        ax_map.plot(s["lon"], s["lat"], "o", markersize=9,
                    markerfacecolor="white", markeredgecolor="#111111",
                    markeredgewidth=1.3, zorder=7)
        dx, dy = s["offset"]
        ax_map.annotate(
            s["label"],
            xy=(s["lon"], s["lat"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7.5, fontweight="bold", color="#1a1a1a",
            zorder=8,
        )

    # ── Neighbour labels ──────────────────────────────────────────────────────
    ax_map.text(-85.7, 11.05, "Nicaragua", fontsize=8, color="#555",
                fontstyle="italic", ha="center", zorder=6)
    ax_map.text(-82.8,  8.15, "Panama",    fontsize=8, color="#555",
                fontstyle="italic", ha="center", zorder=6)
    ax_map.text(-84.2,  9.70, "Pacífico",  fontsize=7.5, color="#4488aa",
                ha="center", zorder=6, alpha=0.8)
    ax_map.text(-83.5, 10.40, "Caribe",    fontsize=7.5, color="#4488aa",
                ha="center", zorder=6, alpha=0.8)

    # ── North arrow + scale bar (top-right corner) ───────────────────────────
    # Anchor point in the ocean area east of the country
    tr_x = maxx + pad - 0.18   # right edge of axes, slight inset
    tr_y = maxy + pad - 0.18   # top edge of axes, slight inset

    # North arrow: label above, arrowhead pointing up
    ax_map.annotate(
        "N",
        xy=(tr_x, tr_y - 0.10),               # arrowhead
        xytext=(tr_x, tr_y - 0.32),            # arrow base / label anchor
        fontsize=11, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        zorder=9,
    )

    # Scale bar (~55 km ≈ 0.5° at 10 °N), centred under the north arrow
    _scale_bar(ax_map, 0.5, tr_x - 0.25, tr_y - 0.50, "≈ 55 km")

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax_map.set_xlim(minx - pad, maxx + pad)
    ax_map.set_ylim(miny - pad, maxy + pad)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Longitude", labelpad=4, fontsize=9)
    ax_map.set_ylabel("Latitude",  labelpad=4, fontsize=9)
    ax_map.grid(True, linestyle=":", linewidth=0.4, alpha=0.55, zorder=0)
    ax_map.tick_params(labelsize=8)
    ax_map.set_title(
        "CLIMA-µEMA Meteorological Station Network — Costa Rica\n"
        "Köppen-Geiger Climate Classification (approximate)",
        fontsize=12, fontweight="bold", pad=8,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    zone_patches = [
        mpatches.Patch(
            facecolor=z["color"], alpha=0.72, edgecolor="#555555",
            linewidth=0.6, label=z["label"],
        )
        for z in sorted(KOPPEN_ZONES, key=lambda z: z["priority"])
    ]
    station_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="white", markeredgecolor="#111", markersize=8,
               markeredgewidth=1.2, label="Station"),
    ]
    ax_map.legend(
        handles=zone_patches + station_handles,
        loc="lower left", fontsize=8.0, framealpha=0.92,
        title="Climate zones & stations", title_fontsize=8.5,
        edgecolor="#aaaaaa",
    )

    # ── Timeline ──────────────────────────────────────────────────────────────
    ax_tl.set_facecolor("#f5f5f5")
    for spine in ["top", "right"]:
        ax_tl.spines[spine].set_visible(False)

    earliest  = min(s["since"] for s in STATIONS)
    latest    = max(s["until"] for s in STATIONS)
    t_start   = earliest - pd.DateOffset(months=2)
    t_end     = latest   + pd.DateOffset(months=2)
    # Sort stations by operational date for the timeline
    ordered = sorted(STATIONS, key=lambda s: s["until"] - s["since"])

    for i, s in enumerate(ordered):
        since = pd.Timestamp(s["since"])
        until = pd.Timestamp(s["until"])
        ax_tl.barh(
            i,
            mdates.date2num(until) - mdates.date2num(since),
            left=mdates.date2num(since),
            height=0.55, color="#1a6fb5", alpha=0.65, edgecolor="none",
        )
        ax_tl.plot(mdates.date2num(since), i, "o", markersize=6, color="#1a6fb5", zorder=5)
        ax_tl.plot(mdates.date2num(until), i, "o", markersize=6, color="#1a6fb5", zorder=5)

    # Data cutoff line at the latest record across all stations
    ax_tl.axvline(
        mdates.date2num(latest), color="#555555",
        linestyle="--", linewidth=0.9, zorder=4, label=f"Data cutoff ({latest.strftime('%b %Y')})",
    )

    ax_tl.set_yticks(range(len(ordered)))
    ax_tl.set_yticklabels([s["label"] for s in ordered], fontsize=8)
    ax_tl.set_xlim(mdates.date2num(t_start), mdates.date2num(t_end))
    ax_tl.set_ylim(-0.6, len(ordered) - 0.4)
    ax_tl.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_tl.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_tl.tick_params(axis="x", rotation=30, labelsize=8)
    ax_tl.tick_params(axis="y", labelsize=8)
    ax_tl.set_xlabel("Date", labelpad=4, fontsize=9)
    ax_tl.set_title("Station Operational Timeline", fontsize=10, pad=4)
    ax_tl.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
    ax_tl.legend(fontsize=7.5, loc="lower left", framealpha=0.85)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    out = Path(__file__).parent / "koppen_stations_map.png"
    print("Reading operational dates from silver layer...")
    _load_operational_dates()
    print("Loading Costa Rica boundary...")
    cr = _load_costa_rica()
    print("Building figure...")
    fig = build_figure(cr)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
