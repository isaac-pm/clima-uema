#!/usr/bin/env python3
"""
Map of Costa Rica with Köppen-Geiger climate zones (Beck et al. 2018) and
µEMA meteorological station locations + operational timeline.

Usage:
    python figures/koppen_stations_map.py
Output:
    figures/koppen_stations_map.png
"""

from __future__ import annotations

import urllib.request
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Station data
# "since" / "until" are derived at runtime from the silver-layer CSVs.
# ─────────────────────────────────────────────────────────────────────────────
STATIONS: list[dict] = [
    {
        "id": "sede-central_finca-1",
        "label": "SC Finca 1",
        "lat": 9.936395248289797,
        "lon": -84.05050707469347,
        "offset": (5, 4),
    },
    {
        "id": "sede-central_finca-2",
        "label": "SC Finca 2",
        "lat": 9.939605935474509,
        "lon": -84.042018599548331,
        "offset": (5, -9),
    },
    {
        "id": "sede-central_finca-3",
        "label": "SC Finca 3",
        "lat": 9.947039492729115,
        "lon": -84.04540844697792,
        "offset": (5, 4),
    },
    {
        "id": "sede-atlantico_turrialba",
        "label": "Turrialba",
        "lat": 9.901463239893761,
        "lon": -83.6719512035301,
        "offset": (5, 4),
    },
    {
        "id": "sede-caribe_limon",
        "label": "Limón",
        "lat": 9.98217415289136,
        "lon": -83.06166336120232,
        "offset": (5, 4),
    },
    {
        "id": "sede-guanacaste_liberia",
        "label": "Liberia",
        "lat": 10.61800662764171,
        "lon": -85.45882068782365,
        "offset": (5, 4),
    },
    {
        "id": "sede-sur_golfito",
        "label": "Golfito",
        "lat": 8.645150123273215,
        "lon": -83.1715836612104,
        "offset": (5, 4),
    },
    {
        "id": "recinto-esparza",
        "label": "Esparza",
        "lat": 9.99433791258938,
        "lon": -84.65200550352944,
        "offset": (5, 4),
    },
    {
        "id": "recinto-guapiles",
        "label": "Guápiles",
        "lat": 10.212752466981385,
        "lon": -83.7714868612007,
        "offset": (5, 4),
    },
    {
        "id": "recinto-santa-cruz",
        "label": "Santa Cruz",
        "lat": 10.285597544094973,
        "lon": -85.58961405934548,
        "offset": (5, 4),
    },
]

_SILVER_DIR = (
    Path(__file__).parent.parent / "data" / "stations" / "processed" / "silver"
)


def _load_operational_dates() -> None:
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
# Beck et al. (2018) Köppen-Geiger color table
# Source: https://doi.org/10.1038/sdata.2018.214  (Table S2 / Fig. 2)
# Integer pixel value → (hex color, short label)
# ─────────────────────────────────────────────────────────────────────────────
KOPPEN_TABLE: dict[int, tuple[str, str]] = {
    1: ("#0000FF", "Af – Tropical rainforest"),
    2: ("#0078FF", "Am – Tropical monsoon"),
    3: ("#46AAFA", "Aw – Tropical savanna"),
    4: ("#FF0000", "BWh – Hot desert"),
    5: ("#FF5464", "BWk – Cold desert"),
    6: ("#F5A500", "BSh – Hot semi-arid"),
    7: ("#FFDC64", "BSk – Cold semi-arid"),
    8: ("#FFFF00", "Csa – Hot-summer Mediterranean"),
    9: ("#C8C800", "Csb – Warm-summer Mediterranean"),
    10: ("#969600", "Csc – Cold-summer Mediterranean"),
    11: ("#96FF00", "Cwa – Monsoon humid subtropical"),
    12: ("#64C800", "Cwb – Subtropical highland"),
    13: ("#329600", "Cwc – Cold subtropical highland"),
    14: ("#C8FF50", "Cfa – Humid subtropical"),
    15: ("#64FF00", "Cfb – Oceanic temperate"),
    16: ("#32C800", "Cfc – Subpolar oceanic"),
    17: ("#FFFF00", "Dsa – Med. hot-summer continental"),
    18: ("#C8C800", "Dsb – Med. warm-summer continental"),
    19: ("#969600", "Dsc – Med. subarctic"),
    20: ("#5A6400", "Dsd – Med. extremely cold subarctic"),
    21: ("#96FF00", "Dwa – Monsoon hot-summer continental"),
    22: ("#64C800", "Dwb – Monsoon warm-summer continental"),
    23: ("#329600", "Dwc – Monsoon subarctic"),
    24: ("#1E6400", "Dwd – Monsoon extremely cold subarctic"),
    25: ("#C8FF50", "Dfa – Hot-summer humid continental"),
    26: ("#64FF00", "Dfb – Warm-summer humid continental"),
    27: ("#32C800", "Dfc – Subarctic"),
    28: ("#00AA00", "Dfd – Extremely cold subarctic"),
    29: ("#B2B2B2", "ET – Alpine tundra"),
    30: ("#666666", "EF – Polar ice cap"),
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


# ─────────────────────────────────────────────────────────────────────────────
# External data (cached locally)
# ─────────────────────────────────────────────────────────────────────────────
_AUX_DIR = Path(__file__).parent.parent / "data" / "aux"

_NE_ZIP = _AUX_DIR / "ne_10m_admin_0_countries.zip"
_NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"

_BECK_TIF = _AUX_DIR / "Beck_KG_V1_present_0p0083.tif"
_BECK_ZIP = _AUX_DIR / "Beck_KG_V1.zip"
_BECK_URL = "https://ndownloader.figshare.com/files/12407516"


def _load_costa_rica() -> gpd.GeoDataFrame:
    _AUX_DIR.mkdir(parents=True, exist_ok=True)
    if not _NE_ZIP.exists():
        print("  Downloading Natural Earth boundaries (one-time)...")
        urllib.request.urlretrieve(_NE_URL, _NE_ZIP)
    world = gpd.read_file(str(_NE_ZIP))
    return world[world["NAME"] == "Costa Rica"].to_crs("EPSG:4326")


def _load_koppen_raster(
    cr_geom, minx: float, miny: float, maxx: float, maxy: float
) -> tuple[np.ndarray, tuple[float, float, float, float], list[int]]:
    """
    Return (rgba_image, imshow_extent, present_codes) clipped to Costa Rica.
    rgba_image: H×W×4 uint8, transparent outside the country boundary.
    imshow_extent: (west, east, south, north) in degrees.
    present_codes: list of integer class codes found inside the country.
    """
    _AUX_DIR.mkdir(parents=True, exist_ok=True)
    if not _BECK_TIF.exists():
        if not _BECK_ZIP.exists():
            print(
                "  Downloading Beck et al. (2018) Köppen-Geiger dataset (one-time, ~70 MB)..."
            )
            urllib.request.urlretrieve(_BECK_URL, _BECK_ZIP)
        print("  Extracting raster from zip...")
        with zipfile.ZipFile(_BECK_ZIP) as zf:
            zf.extract("Beck_KG_V1_present_0p0083.tif", _AUX_DIR)

    buf = 0.2
    with rasterio.open(_BECK_TIF) as src:
        win = from_bounds(minx - buf, miny - buf, maxx + buf, maxy + buf, src.transform)
        data = src.read(1, window=win)
        win_transform = src.window_transform(win)

    rows, cols = data.shape

    # RGBA image: map each pixel to its zone color
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    for code, (hex_color, _) in KOPPEN_TABLE.items():
        r, g, b = _hex_to_rgb(hex_color)
        mask = data == code
        rgba[mask] = [r, g, b, 230]

    # Mask pixels outside Costa Rica boundary (set alpha=0)
    inside = geometry_mask(
        [mapping(cr_geom)],
        transform=win_transform,
        invert=True,
        out_shape=(rows, cols),
    )
    rgba[~inside, 3] = 0

    # imshow extent (west, east, south, north)
    west = win_transform.c
    north = win_transform.f
    east = west + cols * win_transform.a
    south = north + rows * win_transform.e  # e is negative

    present_codes = sorted(
        {int(v) for v in np.unique(data[inside]) if v in KOPPEN_TABLE}
    )

    return rgba, (west, east, south, north), present_codes


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(cr: gpd.GeoDataFrame) -> plt.Figure:
    cr_geom = cr.geometry.unary_union
    minx, miny, maxx, maxy = -86.0, 8.0, -82.5, 11.2
    pad = 0.30

    print("  Loading Köppen raster...")
    rgba, extent, present_codes = _load_koppen_raster(cr_geom, minx, miny, maxx, maxy)

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[3, 1],
        hspace=0.12,
        left=0.04,
        right=0.97,
        top=0.95,
        bottom=0.04,
    )
    ax_map = fig.add_subplot(gs[0])
    ax_tl = fig.add_subplot(gs[1])

    # ── Ocean background ──────────────────────────────────────────────────────
    ax_map.set_facecolor("#cde5f0")

    # ── Land background ───────────────────────────────────────────────────────
    cr.plot(ax=ax_map, color="#e8e4d8", edgecolor="none", zorder=1)

    # ── Köppen raster ─────────────────────────────────────────────────────────
    ax_map.imshow(
        rgba,
        extent=extent,
        origin="upper",
        zorder=2,
        interpolation="nearest",
        aspect="auto",
    )

    # ── Country border ────────────────────────────────────────────────────────
    cr.plot(ax=ax_map, color="none", edgecolor="#2a2a2a", linewidth=1.3, zorder=5)

    # ── Stations ──────────────────────────────────────────────────────────────
    FINCA_IDS = {
        "sede-central_finca-1",
        "sede-central_finca-2",
        "sede-central_finca-3",
    }
    finca_stations = [s for s in STATIONS if s["id"] in FINCA_IDS]
    finca_centroid_lon = np.mean([s["lon"] for s in finca_stations])
    finca_centroid_lat = np.mean([s["lat"] for s in finca_stations])

    for s in STATIONS:
        if s["id"] in FINCA_IDS:
            continue  # replaced by single centroid marker below
        ax_map.plot(s["lon"], s["lat"], "o", markersize=9,
                    markerfacecolor="white", markeredgecolor="#111111",
                    markeredgewidth=1.3, zorder=7)
        dx, dy = s["offset"]
        ax_map.annotate(s["label"], xy=(s["lon"], s["lat"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.5, fontweight="bold", color="#1a1a1a", zorder=8)

    # Single centroid marker + label for the finca cluster on the main map
    ax_map.plot(finca_centroid_lon, finca_centroid_lat, "o", markersize=9,
                markerfacecolor="white", markeredgecolor="#111111",
                markeredgewidth=1.3, zorder=7)
    ax_map.annotate(
        "SC Fincas 1–3", xy=(finca_centroid_lon, finca_centroid_lat),
        xytext=(5, 6), textcoords="offset points",
        fontsize=7.5, fontweight="bold", color="#1a1a1a", zorder=8,
    )

    # ── Valle Central inset (zoom for SC Finca 1, 2, 3) ──────────────────────
    zx0, zx1 = -84.063, -84.033
    zy0, zy1 =   9.928,   9.955

    axins = ax_map.inset_axes(
        [0.27, 0.04, 0.32, 0.28],   # [left, bottom, width, height] in axes fraction
        xlim=(zx0, zx1), ylim=(zy0, zy1),
    )
    axins.set_facecolor("#cde5f0")
    cr.plot(ax=axins, color="#e8e4d8", edgecolor="none", zorder=1)
    axins.imshow(rgba, extent=extent, origin="upper", zorder=2,
                 interpolation="nearest", aspect="auto")
    cr.plot(ax=axins, color="none", edgecolor="#2a2a2a", linewidth=0.8, zorder=5)

    for s in finca_stations:
        axins.plot(s["lon"], s["lat"], "o", markersize=8,
                   markerfacecolor="white", markeredgecolor="#111111",
                   markeredgewidth=1.2, zorder=7)
        axins.annotate(s["label"], xy=(s["lon"], s["lat"]),
                       xytext=(5, 4), textcoords="offset points",
                       fontsize=8.5, fontweight="bold", color="#1a1a1a", zorder=8)

    axins.set_xlim(zx0, zx1)
    axins.set_ylim(zy0, zy1)
    axins.set_aspect("equal")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Valle Central detail", fontsize=7.5, pad=3)
    for spine in axins.spines.values():
        spine.set_edgecolor("#cc2222")
        spine.set_linewidth(1.5)

    ax_map.indicate_inset_zoom(axins, edgecolor="#cc2222", linewidth=1.2, alpha=0.8)

    # ── North arrow ───────────────────────────────────────────────────────────
    ax_map.text(
        0.95,
        0.18,
        "N",
        transform=ax_map.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=9,
    )
    ax_map.annotate(
        "",
        xy=(0.95, 0.175),
        xycoords="axes fraction",
        xytext=(0.95, 0.11),
        textcoords="axes fraction",
        arrowprops=dict(
            facecolor="black",
            edgecolor="none",
            width=1.5,
            headwidth=8,
            headlength=12,
            shrink=0,
        ),
        zorder=9,
    )

    # ── Scale bar ─────────────────────────────────────────────────────────────
    ax_map.add_artist(
        ScaleBar(
            dx=111000,
            units="m",
            location="lower right",
            pad=0.5,
            border_pad=0.5,
            box_alpha=0.7,
            color="#111111",
        )
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax_map.set_xlim(minx - pad, maxx + pad)
    ax_map.set_ylim(miny - pad, maxy + pad)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Longitude", labelpad=4, fontsize=9)
    ax_map.set_ylabel("Latitude", labelpad=4, fontsize=9)
    ax_map.grid(True, linestyle=":", linewidth=0.4, alpha=0.55, zorder=0)
    ax_map.tick_params(labelsize=8)
    ax_map.set_title(
        "µEMA Meteorological Station Network — Costa Rica\n"
        "Köppen-Geiger Climate Classification (Beck et al. 2018)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    # ── Legend (only classes actually present in Costa Rica) ──────────────────
    zone_patches = [
        mpatches.Patch(
            facecolor=KOPPEN_TABLE[c][0],
            alpha=0.9,
            edgecolor="#555555",
            linewidth=0.5,
            label=KOPPEN_TABLE[c][1],
        )
        for c in present_codes
    ]
    station_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="#111",
        markersize=8,
        markeredgewidth=1.2,
        label="Station",
    )
    ax_map.legend(
        handles=zone_patches + [station_handle],
        loc="lower left",
        fontsize=8.0,
        framealpha=0.92,
        title="Climate zones & stations",
        title_fontsize=8.5,
        edgecolor="#aaaaaa",
    )

    # ── Timeline ──────────────────────────────────────────────────────────────
    ax_tl.set_facecolor("#f5f5f5")
    for spine in ["top", "right"]:
        ax_tl.spines[spine].set_visible(False)

    earliest = min(s["since"] for s in STATIONS)
    latest = max(s["until"] for s in STATIONS)
    t_start = earliest - pd.DateOffset(months=2)
    t_end = latest + pd.DateOffset(months=2)
    ordered = sorted(STATIONS, key=lambda s: s["until"] - s["since"])

    for i, s in enumerate(ordered):
        since = pd.Timestamp(s["since"])
        until = pd.Timestamp(s["until"])
        ax_tl.barh(
            i,
            mdates.date2num(until) - mdates.date2num(since),
            left=mdates.date2num(since),
            height=0.55,
            color="#1a6fb5",
            alpha=0.65,
            edgecolor="none",
        )
        ax_tl.plot(
            mdates.date2num(since), i, "o", markersize=6, color="#1a6fb5", zorder=5
        )
        ax_tl.plot(
            mdates.date2num(until), i, "o", markersize=6, color="#1a6fb5", zorder=5
        )

    ax_tl.axvline(
        mdates.date2num(latest),
        color="#555555",
        linestyle="--",
        linewidth=0.9,
        zorder=4,
        label=f"Data cutoff ({latest.strftime('%b %Y')})",
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
