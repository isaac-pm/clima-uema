#!/usr/bin/env python3
"""
Medallion pipeline diagram for the µEMA project.
Visual language matches koppen_stations_map.py (serif fonts, earth-tone palette).

Usage:
    python figures/pipeline_diagram.py
Output:
    figures/pipeline_diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "serif"

# ── Palette: two tones + ink (mirrors koppen_stations_map.py) ────────────────
C_INK    = "#2a2a2a"    # all text, borders, arrows
C_LAND   = "#e8e4d8"    # every content box (steps + artifacts)
C_BG     = "#f5f5f5"    # figure background
C_COL_BG = "#ebebeb"    # column area background
C_HEAD   = "#d6cfc4"    # single header tone for all three columns
C_BORDER = "#888888"    # box borders


def _box(ax, x, y, w, h, fc=C_LAND, ec=C_BORDER, lw=0.8, r=0.006):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2, clip_on=False,
    ))


def _arrow_v(ax, cx, y_top, y_bot):
    ax.annotate(
        "", xy=(cx, y_bot), xytext=(cx, y_top),
        arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.9),
        zorder=5,
    )


def _arrow_h(ax, x0, y, x1):
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.1),
        zorder=5,
    )


def _t(ax, x, y, s, **kw):
    ax.text(x, y, s, zorder=10, **kw)


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Figure title ──────────────────────────────────────────────────────────
    _t(ax, 0.5, 0.972, "µEMA Network Medallion Pipeline",
       ha="center", va="center", fontsize=14, fontweight="bold", color=C_INK)
    _t(ax, 0.5, 0.950, "Integrating Sensor Telemetry and Unstructured Emergency Alerts",
       ha="center", va="center", fontsize=9, fontstyle="italic", color="#555555")

    # ── Layout constants ──────────────────────────────────────────────────────
    PAD    = 0.012     # inner horizontal margin
    GAP    = 0.014     # gap: column header → first box
    ARR    = 0.028     # vertical arrow span (longer than before)
    SLOT   = 0.100     # vertical space per numbered item
    COL_TOP = 0.930
    COL_BOT = 0.030
    HEAD_H  = 0.040

    col_specs = [
        (0.022, 0.300, "RAW LAYER"),
        (0.355, 0.300, "SILVER LAYER"),
        (0.688, 0.300, "GOLD LAYER"),
    ]

    for cx, cw, label in col_specs:
        _box(ax, cx, COL_BOT, cw, COL_TOP - COL_BOT, fc=C_COL_BG, ec=C_BORDER, lw=0.5)
        _box(ax, cx, COL_TOP - HEAD_H, cw, HEAD_H, fc=C_HEAD, ec=C_BORDER, lw=0.9)
        _t(ax, cx + cw / 2, COL_TOP - HEAD_H / 2, label,
           ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_INK)

    # ─────────────────────────────────────────────────────────────────────────
    # RAW COLUMN
    # ─────────────────────────────────────────────────────────────────────────
    rx, rw = 0.022, 0.300
    rcx = rx + rw / 2
    ry = COL_TOP - HEAD_H - GAP

    def raw_box(h, fc=C_LAND):
        _box(ax, rx + PAD, ry - h, rw - 2 * PAD, h, fc=fc, lw=0.9)

    def raw_art(label, h=0.038):
        raw_box(h, fc=C_LAND)
        _t(ax, rcx, ry - h / 2, label,
           ha="center", va="center", fontsize=8.5, color=C_INK)

    def adv(h):
        nonlocal ry
        ry -= h

    def arr():
        nonlocal ry
        _arrow_v(ax, rcx, ry, ry - ARR)
        ry -= ARR

    # Source
    h = 0.058
    raw_box(h)
    _t(ax, rcx, ry - h / 2,
       "Grafana / InfluxDB Instance\n10 Stations  |  3 Sensor Types",
       ha="center", va="center", fontsize=8.5, color=C_INK)
    adv(h); arr()

    # Step 1
    h = 0.145
    raw_box(h)
    _t(ax, rx + PAD + 0.008, ry - 0.016, "1.  Download Station Data",
       va="top", fontsize=9, fontweight="bold", color=C_INK)
    _t(ax, rx + PAD + 0.008, ry - 0.046,
       "Standalone GUI fetches from the\n"
       "µEMA network via InfluxDB:\n"
       "  ·  Atmospheric pressure (hPa)\n"
       "  ·  Precipitation (mm)\n"
       "  ·  Luminous intensity (lux)",
       va="top", fontsize=8.5, color="#333333", linespacing=1.4)
    adv(h); arr()

    # Raw CSVs artifact
    raw_art("Raw CSVs"); adv(0.038)

    # Parallel divider
    _t(ax, rcx, ry - 0.014, "— parallel process —",
       ha="center", va="center", fontsize=7.5, color="#777777", fontstyle="italic")
    ry -= 0.030

    # CNE PDFs artifact
    raw_art("CNE Report PDFs"); adv(0.038); arr()

    # Step 2
    h = 0.155
    raw_box(h)
    _t(ax, rx + PAD + 0.008, ry - 0.016, "2.  Extract Emergency Alerts",
       va="top", fontsize=9, fontweight="bold", color=C_INK)
    _t(ax, rx + PAD + 0.008, ry - 0.046,
       "  ·  OCR: Docling (PDF → text)\n"
       "  ·  Extraction: structured records\n"
       "       via Google Gemini API\n"
       "  ·  Validation: Pydantic schema",
       va="top", fontsize=8.5, color="#333333", linespacing=1.4)
    adv(h); arr()

    # Alerts CSV artifact
    raw_art("Alerts CSV")

    # ─────────────────────────────────────────────────────────────────────────
    # SILVER COLUMN — numbered items
    # ─────────────────────────────────────────────────────────────────────────
    sx, sw = 0.355, 0.300
    scx = sx + sw / 2
    sy = COL_TOP - HEAD_H - GAP

    silver_items = [
        ("1.", "Consolidation:",           "Merge per-station CSVs into a\nunified DataFrame."),
        ("2.", "Sensor Cutoff Filtering:", "Trim data before known\ndeployment/failure dates."),
        ("3.", "10-Minute Resampling:",    "Convert irregular readings to\na uniform temporal grid."),
        ("4.", "Overlap Trimming:",        "Establish a common period\nacross all sensors."),
        ("5.", "Missing Data Handling:",   "Interpolate pressure gaps; fill\nprecipitation and lux with 0s."),
        ("6.", "Cyclical Time Features:",  "sin/cos on hour-of-day\nand day-of-year."),
        ("7.", "Alert Enrichment:",        "merge_asof (10-min tolerance) flags\nalert rows in Silver. Asymmetric\ndilation (48 h pre / 120 h post)\napplied later in Gold layer."),
    ]
    s3_h = 0.040 + len(silver_items) * SLOT + 0.018
    _box(ax, sx + PAD, sy - s3_h, sw - 2 * PAD, s3_h, lw=0.9)
    _t(ax, sx + PAD + 0.008, sy - 0.016, "3.  Build Station Silver Layer",
       va="top", fontsize=9, fontweight="bold", color=C_INK)

    by = sy - 0.048
    for num, bold, body in silver_items:
        _t(ax, sx + PAD + 0.010, by, num,
           va="top", fontsize=8.5, color=C_INK)
        _t(ax, sx + PAD + 0.028, by, bold,
           va="top", fontsize=8.5, fontweight="bold", color=C_INK)
        _t(ax, sx + PAD + 0.028, by - 0.024, body,
           va="top", fontsize=8.5, color="#333333", linespacing=1.35)
        by -= SLOT

    sy -= s3_h
    _arrow_v(ax, scx, sy, sy - ARR); sy -= ARR

    sil_h = 0.038
    _box(ax, sx + PAD, sy - sil_h, sw - 2 * PAD, sil_h, lw=0.9)
    _t(ax, scx, sy - sil_h / 2, "Silver Data (Cleaned CSVs)",
       ha="center", va="center", fontsize=8.5, color=C_INK)

    # ─────────────────────────────────────────────────────────────────────────
    # GOLD COLUMN — numbered items
    # ─────────────────────────────────────────────────────────────────────────
    gx, gw = 0.688, 0.300
    gcx = gx + gw / 2
    gy = COL_TOP - HEAD_H - GAP

    gold_items = [
        ("1.", "Anomaly Mask:",      "Asymmetric dilation — 48 h pre-alert,\n120 h (5 d) post-alert."),
        ("2.", "Train/Test Split:",  "All anomalous windows → test;\n20% of normal → test;\n80% of normal → train."),
        ("3.", "Calibration Split:", "Normal windows near anomaly\nboundaries → X_calib."),
        ("4.", "Scaler Fitting:",    "StandardScaler (pressure);\nMinMaxScaler (precip., lux).\nFit on training data only."),
        ("5.", "Sliding Windows:",   "144 timesteps (24 h),\n10-min resolution, stride 6 (60 min)."),
    ]
    dp_h = 0.040 + len(gold_items) * SLOT + 0.018
    _box(ax, gx + PAD, gy - dp_h, gw - 2 * PAD, dp_h, lw=0.9)
    _t(ax, gx + PAD + 0.008, gy - 0.016, "Data Preparation",
       va="top", fontsize=9, fontweight="bold", color=C_INK)

    by = gy - 0.048
    for num, bold, body in gold_items:
        _t(ax, gx + PAD + 0.010, by, num,
           va="top", fontsize=8.5, color=C_INK)
        _t(ax, gx + PAD + 0.028, by, bold,
           va="top", fontsize=8.5, fontweight="bold", color=C_INK)
        _t(ax, gx + PAD + 0.028, by - 0.024, body,
           va="top", fontsize=8.5, color="#333333", linespacing=1.35)
        by -= SLOT

    gy -= dp_h
    _arrow_v(ax, gcx, gy, gy - ARR); gy -= ARR

    # Step 4
    s4_h = 0.078
    _box(ax, gx + PAD, gy - s4_h, gw - 2 * PAD, s4_h, lw=0.9)
    _t(ax, gx + PAD + 0.008, gy - 0.016, "4.  Build Station Gold Layer",
       va="top", fontsize=9, fontweight="bold", color=C_INK)
    _t(ax, gx + PAD + 0.008, gy - 0.044,
       "Per-station .npy arrays\n(X_train, X_test, y_test, X_calib).",
       va="top", fontsize=8.5, color="#333333", linespacing=1.35)
    gy -= s4_h
    _arrow_v(ax, gcx, gy, gy - ARR); gy -= ARR

    # Step 5
    s5_h = 0.078
    _box(ax, gx + PAD, gy - s5_h, gw - 2 * PAD, s5_h, lw=0.9)
    _t(ax, gx + PAD + 0.008, gy - 0.016, "5.  Build Global Gold Layer",
       va="top", fontsize=9, fontweight="bold", color=C_INK)
    _t(ax, gx + PAD + 0.008, gy - 0.044,
       "Concatenated arrays across all stations\n+ station_ids_train/test.npy.",
       va="top", fontsize=8.5, color="#333333", linespacing=1.35)
    gy -= s5_h
    _arrow_v(ax, gcx, gy, gy - ARR); gy -= ARR

    # NPY Files artifact
    npy_h = 0.038
    _box(ax, gx + PAD, gy - npy_h, gw - 2 * PAD, npy_h, lw=0.9)
    _t(ax, gcx, gy - npy_h / 2, "Model-Ready NPY Files",
       ha="center", va="center", fontsize=8.5, color=C_INK)

    # ── Horizontal arrows RAW → SILVER → GOLD ────────────────────────────────
    h_arr_y = 0.50
    _arrow_h(ax, col_specs[0][0] + col_specs[0][1] + 0.004, h_arr_y,
             col_specs[1][0] - 0.004)
    _arrow_h(ax, col_specs[1][0] + col_specs[1][1] + 0.004, h_arr_y,
             col_specs[2][0] - 0.004)

    return fig


def main() -> None:
    out = Path(__file__).parent / "pipeline_diagram.png"
    print("Building pipeline diagram...")
    fig = build_figure()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
