# -*- coding: utf-8 -*-
r"""
ERPI 2001–2020 | FINAL corrected-candidate publication figure suite V7
=======================================================

This script regenerates the manuscript figures ONLY from the corrected
nighttime-light candidate and its refreshed validation tables.

Default candidate:
    F:\data\China\ERI_recalculated_2001_2020_NTLfix_candidate_v1

Default output:
    <candidate>\manuscript_figures_FINAL_v5

The script aborts if the fixed weights look like the obsolete pre-correction
product (HCI ≈ 0.152509).

Figures
-------
Fig. 1  Study domain, study boundary and South China Sea context with nine-dash line
Fig. 2  ERPI construction and validation workflow
Fig. 3  Spatial overview of corrected ERPI
Fig. 4  Output integrity and spatial support
Fig. 5  Weighting/component sensitivity
Fig. S1 Corrected component means and fixed weights
Fig. S2 Corrected NPP plausibility assessment

Dependencies
------------
numpy, pandas, rasterio, matplotlib
Optional for Fig. 1 South China Sea context:
geopandas, shapely

Important
---------
Fig. 1 becomes a submission-ready file only when a georeferenced South China
Sea context layer is available AND --cartographic-source is provided. Otherwise
the script writes a PREVIEW-labelled Fig. 1 but still builds all other figures.

Run
---
D:\conda\python.exe "C:\Users\Administrator\Downloads\make_ERPI_publication_figures_v7_1_layoutsafe.py"
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject, transform_bounds
from rasterio.transform import from_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from pyproj import Transformer, Geod

YEARS = list(range(2001, 2021))
IND = ["VCI", "LCN", "HCI", "TSI", "MSI"]

DATA_ROOT = Path(os.environ.get("ERPI_DATA_ROOT", "./data")).expanduser().resolve()
DEFAULT_ROOT = Path(os.environ.get("ERPI_PRODUCT_ROOT", str(DATA_ROOT / "ERI_recalculated_2001_2020_NTLfix_candidate_v1"))).expanduser().resolve()
OUTNAME = "manuscript_figures_FINAL_v7_1_layoutsafe"

EXPECTED_HCI = 0.155397
OLD_HCI = 0.152509
EXPECTED_COMMON = 9_055_311

# ---------------------------------------------------------------------
# Scientific Data / Nature-family publication style
# ---------------------------------------------------------------------
WIDTH = 7.20
DPI = 600

FONT_FAMILY = ["Arial", "DejaVu Sans"]
INK = "#24364B"
MUTED = "#6B7888"
BORDER = "#8A96A3"
FAINT = "#D9E0E6"
NODATA = "#E4E7E8"
SUPPORT = "#6F98AC"

BLUE = "#356FA4"
TEAL = "#1D7F78"
ORANGE = "#C9863A"

COMP = {
    "VCI":"#1B8479",
    "LCN":"#C88434",
    "HCI":"#3E73A5",
    "TSI":"#7A689A",
    "MSI":"#B05A54",
}

WF_INPUT = "#EAF0F6"
WF_GRID = "#E5EDF3"
WF_INDICATOR = "#EAF3EE"
WF_WEIGHT = "#F3EEE5"
WF_OUTPUT = "#EEF0F4"
WF_ARROW = "#7C8A99"

ERPI_COLORS = ["#08306B", "#46506B", "#8A8779", "#CDBF6C", "#F1D93B"]

def style():
    """One style definition for every manuscript and supplementary figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": 7.5,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "legend.fontsize": 6.9,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

def export(fig, stem, out):
    fig.savefig(out/f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out/f"{stem}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

def axes_clean(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=3, width=.7)

def panel(ax, letter, title):
    """Consistent Nature-family panel label + concise panel title."""
    ax.text(0.00, 1.075, letter, transform=ax.transAxes,
            fontsize=9.7, fontweight="bold", ha="left", va="bottom", color=INK)
    ax.text(0.075, 1.075, title, transform=ax.transAxes,
            fontsize=8.8, fontweight="normal", ha="left", va="bottom", color=INK)

def years_axis(ax):
    ax.set_xlim(2000.7,2020.3)
    ax.set_xticks([2001,2005,2010,2015,2020])

def must_csv(path, columns):
    if not path.exists():
        raise FileNotFoundError(path)
    df=pd.read_csv(path)
    miss=[x for x in columns if x not in df.columns]
    if miss:
        raise ValueError(f"{path.name}: missing columns {miss}")
    return df

def read_weights(root):
    df=must_csv(root/"entropy_weights_fixed_2001_2020.csv",["indicator","weight"])
    w=dict(zip(df.indicator.astype(str),df.weight.astype(float)))
    if any(k not in w for k in IND):
        raise ValueError("Weight file lacks one or more ERPI components.")
    if abs(sum(w.values())-1)>1e-6:
        raise ValueError("Corrected weights do not sum to one.")
    if abs(w["HCI"]-OLD_HCI) < 5e-5:
        raise RuntimeError("Obsolete pre-correction weights detected. Do not draw figures from the old product.")
    if abs(w["HCI"]-EXPECTED_HCI) > 5e-4:
        raise RuntimeError(f"Unexpected HCI weight {w['HCI']:.6f}; candidate identity should be checked.")
    return w

def discover_common_mask(root):
    choices=[
        root/"technical_validation"/"common_valid_mask_2001_2020.tif",
        root/"FINAL_ERPI_TECHNICAL_AUDIT_NTLfix"/"common_valid_mask_AUDITED_2001_2020.tif",
        root/"FINAL_ERPI_TECHNICAL_AUDIT"/"common_valid_mask_AUDITED_2001_2020.tif",
    ]
    for p in choices:
        if p.exists():
            return p
    raise FileNotFoundError("Corrected common-support mask not found.")

def common_count(path):
    n=0
    with rasterio.open(path) as s:
        for _,win in s.block_windows(1):
            a=s.read(1,window=win)
            n += int(np.count_nonzero(a==1))
    return n

def preview(src, max_side=1600, categorical=False):
    scale=max(src.width/max_side,src.height/max_side,1)
    h=max(1,int(round(src.height/scale)))
    w=max(1,int(round(src.width/scale)))
    rs=Resampling.nearest if categorical else Resampling.average
    a=src.read(1,out_shape=(h,w),masked=True,resampling=rs).astype(np.float64).filled(np.nan)
    return a,(src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top)

def common_mean_preview(root, common_path, max_side=1600):
    # Exact temporal mean first at source resolution is costly; for visualization,
    # average the 20 aligned annual previews only after applying a nearest-neighbour
    # preview of the strict common mask.
    with rasterio.open(common_path) as cm:
        mask,bounds=preview(cm,max_side,categorical=True)
    stack=[]
    for y in YEARS:
        with rasterio.open(root/"ERI"/f"ERI_{y}.tif") as s:
            a,b=preview(s,max_side,categorical=False)
        if b != bounds:
            # identical grids should have identical bounds; preview dimensions still match.
            pass
        stack.append(a)
    mean=np.nanmean(np.stack(stack,axis=0),axis=0)
    mean[~(mask>0.5)] = np.nan
    return mean,mask,bounds

def _degree_label(value, axis="lon"):
    value = float(value)
    if axis == "lon":
        return f"{abs(value):g}°{'E' if value >= 0 else 'W'}"
    return f"{abs(value):g}°{'N' if value >= 0 else 'S'}"


def make_wgs84_display_grid(ref, max_width=1600, pad_deg=0.15):
    """
    Build ONE geographic display grid (EPSG:4326) for publication maps.

    This is display-only reprojection. The released ERPI rasters and all
    numerical analyses remain on the native 1-km Albers grid.
    """
    west, south, east, north = transform_bounds(
        ref.crs, "EPSG:4326", *ref.bounds, densify_pts=41
    )
    west -= pad_deg
    east += pad_deg
    south -= pad_deg
    north += pad_deg

    lon_span = east - west
    lat_span = north - south
    if lon_span <= 0 or lat_span <= 0:
        raise ValueError("Invalid transformed display bounds.")

    if lon_span >= lat_span:
        width = int(max_width)
        height = max(1, int(round(max_width * lat_span / lon_span)))
    else:
        height = int(max_width)
        width = max(1, int(round(max_width * lon_span / lat_span)))

    transform = from_bounds(west, south, east, north, width, height)
    return {
        "crs": "EPSG:4326",
        "transform": transform,
        "width": width,
        "height": height,
        "extent": (west, east, south, north),
    }


def reproject_raster_to_wgs84(src_or_path, display_grid, categorical=False):
    """
    Reproject one raster to the fixed geographic display grid.

    Continuous ERPI layers use bilinear resampling for visual display;
    categorical / mask layers use nearest-neighbour resampling.
    """
    close_after = False
    if isinstance(src_or_path, (str, Path)):
        src = rasterio.open(src_or_path)
        close_after = True
    else:
        src = src_or_path

    try:
        dst = np.full(
            (display_grid["height"], display_grid["width"]),
            np.nan, dtype=np.float32
        )
        rs = Resampling.nearest if categorical else Resampling.bilinear
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=display_grid["transform"],
            dst_crs=display_grid["crs"],
            dst_nodata=np.nan,
            resampling=rs,
            num_threads=2,
        )
        dst[~np.isfinite(dst)] = np.nan
        return dst
    finally:
        if close_after:
            src.close()


def common_mean_wgs84(root, common_path, display_grid):
    """
    Build the long-term mean on the SAME geographic display grid.

    The strict common mask is reprojected by nearest neighbour. Annual ERPI
    rasters are reprojected only for visualization, accumulated on that
    display grid, and then masked by the strict common-support display mask.
    """
    common = reproject_raster_to_wgs84(
        common_path, display_grid, categorical=True
    )
    common_ok = np.isfinite(common) & (common > 0.5)

    total = np.zeros(common.shape, dtype=np.float64)
    count = np.zeros(common.shape, dtype=np.uint8)

    for y in YEARS:
        a = reproject_raster_to_wgs84(
            root/"ERI"/f"ERI_{y}.tif", display_grid, categorical=False
        )
        ok = common_ok & np.isfinite(a)
        total[ok] += a[ok]
        count[ok] += 1

    mean = np.full(common.shape, np.nan, dtype=np.float32)
    ok = common_ok & (count > 0)
    mean[ok] = (total[ok] / count[ok]).astype(np.float32)

    # Keep only places supported by the strict common mask in the display.
    common_display = np.where(common_ok, 1.0, np.nan).astype(np.float32)
    return mean, common_display


def nice_ticks(lo, hi, step):
    """Return regularly spaced ticks that stay inside the displayed extent."""
    start = math.ceil(lo / step) * step
    end = math.floor(hi / step) * step
    if end < start:
        return []
    n = int(round((end - start) / step)) + 1
    return [start + i*step for i in range(n)]


def draw_rectangular_graticule(ax, extent,
                               lon_step=10, lat_step=10,
                               label_size=5.8,
                               grid_color="#D7DEE4",
                               frame_color="#B7C1C9"):
    """
    Rectilinear longitude/latitude grid for EPSG:4326 publication maps.

    Style follows the user's reference:
      * light solid grid;
      * longitude labels on bottom only;
      * latitude labels on left only;
      * no top/right duplicate labels;
      * no visible tick marks.
    """
    west, east, south, north = extent
    xticks = nice_ticks(west, east, lon_step)
    yticks = nice_ticks(south, north, lat_step)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([_degree_label(v, "lon") for v in xticks])
    ax.set_yticklabels([_degree_label(v, "lat") for v in yticks])

    ax.grid(
        True, color=grid_color, lw=0.45, ls="solid",
        alpha=0.95, zorder=1
    )

    ax.tick_params(
        axis="x", which="both",
        bottom=True, top=False, labelbottom=True, labeltop=False,
        length=0, pad=2.0, labelsize=label_size, colors="#7A8792"
    )
    ax.tick_params(
        axis="y", which="both",
        left=True, right=False, labelleft=True, labelright=False,
        length=0, pad=2.0, labelsize=label_size, colors="#7A8792"
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(0.55)


def draw_scs_graticule(ax, xlim=(104, 126), ylim=(1, 27),
                       lon_ticks=(105, 110, 115, 120, 125),
                       lat_ticks=(5, 10, 15, 20, 25),
                       label_size=5.6):
    """Rectilinear graticule for the South China Sea inset."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([_degree_label(v, "lon") for v in lon_ticks])
    ax.set_yticklabels([_degree_label(v, "lat") for v in lat_ticks])
    ax.grid(True, color="#D7DEE4", lw=0.45, ls="solid", alpha=0.95, zorder=1)
    ax.tick_params(
        axis="x", bottom=True, top=False, labelbottom=True, labeltop=False,
        length=0, pad=2.0, labelsize=label_size, colors="#7A8792"
    )
    ax.tick_params(
        axis="y", left=True, right=False, labelleft=True, labelright=False,
        length=0, pad=2.0, labelsize=label_size, colors="#7A8792"
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B7C1C9")
        spine.set_linewidth(0.55)


def draw_geographic_north_arrow(ax, anchor=(0.070, 0.115), length=0.075):
    """
    North arrow for an EPSG:4326 display.
    Increasing latitude is true north in the geographic display plane.
    """
    x, y = anchor
    ax.annotate(
        "", xy=(x, y + length), xytext=(x, y),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>", lw=1.05, color=INK, mutation_scale=8.5
        ),
        zorder=20, annotation_clip=False
    )
    ax.text(
        x, y + length + 0.018, "N",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=6.5, fontweight="bold", color=INK,
        zorder=20, clip_on=False
    )


def _lon_delta_for_geodesic_distance(lon0, lat0, distance_km):
    """
    Longitude difference at fixed latitude whose WGS84 geodesic distance
    equals the requested distance. Bisection keeps the drawn scale bar
    horizontal in the rectilinear geographic display.
    """
    geod = Geod(ellps="WGS84")
    target = float(distance_km) * 1000.0
    low, high = 0.0, 30.0

    for _ in range(60):
        mid = (low + high) / 2.0
        _, _, dist = geod.inv(lon0, lat0, lon0 + mid, lat0)
        if dist < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def draw_geographic_scale_bar(ax, extent, length_km=1000,
                              location=(0.085, 0.070),
                              bar_height_deg=0.45):
    """
    Horizontal scale bar for EPSG:4326 display, calibrated geodesically at
    the bar latitude. It is for cartographic display only.
    """
    west, east, south, north = extent
    lon0 = west + location[0] * (east - west)
    lat0 = south + location[1] * (north - south)

    dlon = _lon_delta_for_geodesic_distance(lon0, lat0, length_km)
    half = dlon / 2.0

    ax.add_patch(
        patches.Rectangle(
            (lon0, lat0), half, bar_height_deg,
            fc=INK, ec=INK, lw=0.45, zorder=20
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (lon0 + half, lat0), half, bar_height_deg,
            fc="white", ec=INK, lw=0.45, zorder=20
        )
    )
    ax.text(
        lon0, lat0 - 0.55*bar_height_deg, "0",
        fontsize=5.6, ha="center", va="top", color=INK, zorder=20
    )
    ax.text(
        lon0 + dlon, lat0 - 0.55*bar_height_deg, f"{length_km:,} km",
        fontsize=5.6, ha="center", va="top", color=INK, zorder=20
    )


def raster_panel_wgs84(ax, a, extent, title, letter, cmap,
                       boundary_gdf=None, vmin=0, vmax=1,
                       lon_step=10, lat_step=10,
                       show_north=True, show_scale=False):
    """
    Publication map panel in EPSG:4326 with a true rectilinear graticule.
    """
    im = ax.imshow(
        a, extent=extent, origin="upper", cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest", zorder=0
    )

    # Plate Carrée / geographic display. This is intentionally the display CRS.
    ax.set_aspect("equal", adjustable="box")
    draw_rectangular_graticule(
        ax, extent, lon_step=lon_step, lat_step=lat_step
    )

    if boundary_gdf is not None:
        try:
            plot_context_geometry(
                boundary_gdf, ax, color="#74838E",
                linewidth=0.48, zorder=8
            )
        except Exception:
            pass

    ax.text(
        0.00, 1.018, letter, transform=ax.transAxes,
        fontsize=9.3, fontweight="bold",
        ha="left", va="bottom", color=INK, clip_on=False
    )
    ax.text(
        0.085, 1.018, title, transform=ax.transAxes,
        fontsize=8.45, ha="left", va="bottom",
        color=INK, clip_on=False
    )

    if show_north:
        draw_geographic_north_arrow(ax)
    if show_scale:
        draw_geographic_scale_bar(ax, extent, length_km=1000)

    return im



def draw_shared_cartographic_reference(ax, extent, length_km=1000):
    """
    Shared north arrow + scale bar for multi-panel maps.

    The axis is intentionally the same physical width as one map panel.
    The scale-bar length is converted to an axes fraction using the geographic
    map width, so its on-page length matches the map panels while remaining
    outside the data area.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    west, east, south, north = extent
    lon0 = west + 0.22 * (east - west)
    lat0 = south + 0.22 * (north - south)

    # WGS84 geodesic distance -> longitude span at the reference latitude.
    dlon = _lon_delta_for_geodesic_distance(lon0, lat0, length_km)
    frac = dlon / (east - west)
    frac = max(0.10, min(frac, 0.58))

    # North arrow occupies the far-left part of the shared reference band.
    nx = 0.075
    ax.annotate(
        "", xy=(nx, 0.78), xytext=(nx, 0.30),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>", lw=1.05, color=INK, mutation_scale=8.5
        ),
        annotation_clip=False, zorder=10
    )
    ax.text(
        nx, 0.84, "N",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=6.6, fontweight="bold", color=INK
    )

    # Scale bar to the right of the north arrow.
    x0 = 0.235
    y0 = 0.46
    h = 0.11
    half = frac / 2.0
    ax.add_patch(
        patches.Rectangle(
            (x0, y0), half, h,
            transform=ax.transAxes,
            fc=INK, ec=INK, lw=0.45, zorder=10
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (x0 + half, y0), half, h,
            transform=ax.transAxes,
            fc="white", ec=INK, lw=0.45, zorder=10
        )
    )
    ax.text(
        x0, y0 - 0.10, "0",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=5.6, color=INK
    )
    ax.text(
        x0 + frac, y0 - 0.10, f"{length_km:,} km",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=5.6, color=INK
    )
    ax.text(
        0.00, 0.02, "Shared cartographic reference for panels a–d",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=5.15, color=MUTED
    )


def _bbox_overlap(a, b, pad=1.5):
    """Return True when two display-coordinate bounding boxes overlap."""
    return not (
        a.x1 + pad <= b.x0 or b.x1 + pad <= a.x0 or
        a.y1 + pad <= b.y0 or b.y1 + pad <= a.y0
    )


def validate_figure3_layout(fig, main_axes, inset_ax, carto_ax, colorbar_ax,
                            extra_text_artists=None):
    """
    Fail loudly if the main panel axes or bottom-band axes overlap.

    This checks the axes rectangles in figure coordinates and selected
    figure-level text bounding boxes after rendering. It is deliberately
    conservative: a failed QA means the layout should be adjusted rather
    than silently exporting a crowded figure.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []

    # Axes rectangles must be disjoint.
    axes_to_check = list(main_axes) + [inset_ax, carto_ax, colorbar_ax]
    for i in range(len(axes_to_check)):
        bi = axes_to_check[i].get_position()
        for j in range(i + 1, len(axes_to_check)):
            bj = axes_to_check[j].get_position()
            # Convert figure-coordinate Bboxes to display-coordinate Bboxes.
            p0 = fig.transFigure.transform((bi.x0, bi.y0))
            p1 = fig.transFigure.transform((bi.x1, bi.y1))
            q0 = fig.transFigure.transform((bj.x0, bj.y0))
            q1 = fig.transFigure.transform((bj.x1, bj.y1))
            from matplotlib.transforms import Bbox
            bb_i = Bbox.from_extents(p0[0], p0[1], p1[0], p1[1])
            bb_j = Bbox.from_extents(q0[0], q0[1], q1[0], q1[1])
            if _bbox_overlap(bb_i, bb_j, pad=0.0):
                failures.append(
                    f"axes overlap: {i} vs {j}"
                )

    # Selected figure-level texts should not collide with bottom-band axes.
    if extra_text_artists:
        bottom_axes = [inset_ax, carto_ax, colorbar_ax]
        for txt in extra_text_artists:
            tb = txt.get_window_extent(renderer=renderer)
            for k, ax in enumerate(bottom_axes):
                ab = ax.get_window_extent(renderer=renderer)
                if _bbox_overlap(tb, ab, pad=0.5):
                    failures.append(
                        f"figure text overlaps bottom axis {k}: {txt.get_text()!r}"
                    )

    if failures:
        raise RuntimeError(
            "Fig. 3 layout QA failed: " + "; ".join(failures)
        )


def plot_nine_dash_geometry(gdf, ax, color=INK, linewidth=1.05, zorder=8):
    """
    Preserve the source nine-dash geometry.

    Line/polygon geometries are drawn SOLID because the source geometry itself
    already contains discrete dash segments. If the source is point geometry,
    horizontal "_" markers are used rather than circular points.
    """
    if gdf is None or len(gdf) == 0:
        return False

    try:
        gg = gdf.explode(index_parts=False, ignore_index=True)
    except TypeError:
        gg = gdf.explode(index_parts=False)

    types = gg.geometry.geom_type.astype(str)
    line_mask = types.isin(["LineString", "LinearRing", "MultiLineString"])
    poly_mask = types.isin(["Polygon", "MultiPolygon"])
    point_mask = types.isin(["Point", "MultiPoint"])

    rendered = False
    if line_mask.any():
        gg.loc[line_mask].plot(
            ax=ax, color=color, linewidth=linewidth,
            linestyle="solid", zorder=zorder
        )
        rendered = True

    if poly_mask.any():
        gg.loc[poly_mask].boundary.plot(
            ax=ax, color=color, linewidth=linewidth,
            linestyle="solid", zorder=zorder
        )
        rendered = True

    if point_mask.any():
        pts = []
        for geom in gg.loc[point_mask].geometry:
            if geom.geom_type == "Point":
                pts.append((geom.x, geom.y))
            elif geom.geom_type == "MultiPoint":
                pts.extend([(p.x, p.y) for p in geom.geoms])
        if pts:
            xx, yy = zip(*pts)
            ax.scatter(
                xx, yy, marker="_", s=38, linewidths=1.15,
                color=color, zorder=zorder
            )
            rendered = True

    return rendered


def plot_context_geometry(gdf, ax, color="#8798A5",
                          linewidth=0.45, zorder=4):
    """
    Plot polygons, lines, and points without discarding tiny island geometry.
    """
    if gdf is None or len(gdf) == 0:
        return False

    try:
        gg = gdf.explode(index_parts=False, ignore_index=True)
    except TypeError:
        gg = gdf.explode(index_parts=False)

    types = gg.geometry.geom_type.astype(str)
    poly = types.isin(["Polygon", "MultiPolygon"])
    line = types.isin(["LineString", "LinearRing", "MultiLineString"])
    point = types.isin(["Point", "MultiPoint"])
    rendered = False

    if poly.any():
        gg.loc[poly].boundary.plot(
            ax=ax, color=color, linewidth=linewidth, zorder=zorder
        )
        rendered = True

    if line.any():
        gg.loc[line].plot(
            ax=ax, color=color, linewidth=linewidth, zorder=zorder
        )
        rendered = True

    if point.any():
        pts = []
        for geom in gg.loc[point].geometry:
            if geom.geom_type == "Point":
                pts.append((geom.x, geom.y))
            elif geom.geom_type == "MultiPoint":
                pts.extend([(p.x, p.y) for p in geom.geoms])
        if pts:
            xx, yy = zip(*pts)
            ax.scatter(
                xx, yy, s=3.5, marker="o",
                color=color, linewidths=0, zorder=zorder
            )
            rendered = True

    return rendered

def _first_existing(paths):
    for p in paths:
        if not p:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None

def _read_vector_for_plot(path, target_crs=None, allow_lonlat_inference=True):
    """
    Read a local vector robustly. If CRS metadata is missing but bounds clearly
    look like longitude/latitude, assign EPSG:4326. Otherwise fail rather than
    silently invent a projection.
    """
    import geopandas as gpd
    g = gpd.read_file(path)
    if g.empty:
        raise ValueError(f"Empty vector: {path}")
    if g.crs is None:
        b = g.total_bounds
        lonlat_like = (
            np.all(np.isfinite(b))
            and -180 <= b[0] <= 180 and -180 <= b[2] <= 180
            and -90 <= b[1] <= 90 and -90 <= b[3] <= 90
        )
        if allow_lonlat_inference and lonlat_like:
            g = g.set_crs("EPSG:4326", allow_override=True)
        else:
            raise ValueError(
                f"Vector has no CRS and is not safely inferable as lon/lat: {path}. "
                "Assign the correct CRS to the source file before plotting."
            )
    if target_crs is not None:
        g = g.to_crs(target_crs)
    return g

def audit_local_vectors(data_root, out_dir):
    """
    Inspect exactly the local shapefiles available in F:\\data\\China\\MAP\\shp.

    The audit is written every time the figure script runs so that ambiguous
    filenames never have to be interpreted silently.
    """
    rows = []
    try:
        import geopandas as gpd
    except Exception:
        return None

    folder = data_root/"MAP"/"shp"
    for name in [
        "china.shp",
        "china_country.shp",
        "china_nine_dotted_line.shp",
        "china_without_islands.shp",
        "simplified_china_country.shp",
    ]:
        p = folder/name
        if not p.exists():
            continue
        try:
            g = gpd.read_file(p)
            crs = str(g.crs)
            b = g.total_bounds if len(g) else [np.nan]*4
            types = ",".join(sorted(set(g.geometry.geom_type.astype(str)))) if len(g) else ""
            rows.append(dict(
                file=name,
                exists=True,
                n_features=int(len(g)),
                geometry_types=types,
                crs=crs,
                minx=float(b[0]), miny=float(b[1]),
                maxx=float(b[2]), maxy=float(b[3]),
            ))
        except Exception as e:
            rows.append(dict(
                file=name, exists=True, n_features=np.nan,
                geometry_types=f"READ_ERROR: {e}", crs="",
                minx=np.nan, miny=np.nan, maxx=np.nan, maxy=np.nan
            ))

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir/"CARTOGRAPHY_VECTOR_AUDIT.csv", index=False,
                  encoding="utf-8-sig")
        return df
    return None


def _discover_fig1_vectors(data_root, scs_line=None, china_context=None,
                           boundary=None):
    """
    Roles tailored to the user's actual local files:

    - china_nine_dotted_line.shp : nine-dash cartographic geometry
    - china.shp                  : full China geographic context, including
                                   all island geometries present in that file
    - china_country.shp          : preferred national boundary for main panels
    - china_without_islands.shp  : fallback mainland boundary
    - simplified_china_country.shp : final fallback boundary/context
    """
    folder = data_root/"MAP"/"shp"

    scs = _first_existing([
        scs_line,
        folder/"china_nine_dotted_line.shp",
    ])

    ctx = _first_existing([
        china_context,
        folder/"china.shp",
        folder/"china_country.shp",
        folder/"simplified_china_country.shp",
    ])

    bnd = _first_existing([
        boundary,
        folder/"china_country.shp",
        data_root/"China_boudary.shp",
        data_root/"China_boundary.shp",
        folder/"china_without_islands.shp",
        folder/"simplified_china_country.shp",
        folder/"china.shp",
    ])

    mainland = _first_existing([
        folder/"china_without_islands.shp",
        folder/"china_country.shp",
        bnd,
    ])

    return scs, ctx, bnd, mainland

def fig1(root, out, common_path, scs_line=None, china_context=None,
         boundary=None, cartographic_source=None):
    """
    Figure 1 — study domain and spatial support.

    Publication DISPLAY CRS = EPSG:4326 so that the main map, national
    boundary, South China Sea inset, and longitude/latitude grid all share
    one rectilinear geographic coordinate system.

    The archived ERPI data remain on the original 1-km Albers grid.
    """
    data_root = root.parent
    scs_path, ctx_path, bnd_path, mainland_path = _discover_fig1_vectors(
        data_root, scs_line, china_context, boundary
    )
    audit_local_vectors(data_root, out)

    print("Fig.1 geographic display:")
    print("  national boundary :", bnd_path)
    print("  full China context :", ctx_path)
    print("  nine-dash line     :", scs_path)

    # One display grid derived from the corrected ERPI raster.
    with rasterio.open(common_path) as ref:
        display = make_wgs84_display_grid(ref, max_width=1600)

    support = reproject_raster_to_wgs84(
        common_path, display, categorical=True
    )
    land = np.where(
        np.isfinite(support) & (support > 0.5), 1.0, np.nan
    )

    extent = display["extent"]
    cmap = ListedColormap([SUPPORT])
    cmap.set_bad(NODATA)

    boundary_geo = None
    boundary_ok = False
    if bnd_path is not None:
        try:
            boundary_geo = _read_vector_for_plot(
                bnd_path, target_crs="EPSG:4326"
            )
            boundary_ok = True
        except Exception as e:
            print("  WARNING: national boundary failed:", e)

    fig = plt.figure(figsize=(WIDTH, 4.95), facecolor="white")

    # Panel a — main map in geographic coordinates.
    ax = fig.add_axes([0.060, 0.165, 0.595, 0.735])
    ax.imshow(
        land, extent=extent, origin="upper", cmap=cmap,
        vmin=0, vmax=1, interpolation="nearest", zorder=0
    )
    ax.set_aspect("equal", adjustable="box")
    draw_rectangular_graticule(
        ax, extent, lon_step=10, lat_step=10
    )

    if boundary_geo is not None:
        plot_context_geometry(
            boundary_geo, ax,
            color="#74838E", linewidth=0.50, zorder=8
        )

    panel(ax, "a", "ERPI terrestrial coverage")
    draw_geographic_north_arrow(
        ax, anchor=(0.075, 0.105), length=0.075
    )
    draw_geographic_scale_bar(
        ax, extent, length_km=1000,
        location=(0.095, 0.070), bar_height_deg=0.38
    )

    # Panel b — South China Sea context, also EPSG:4326.
    ax2 = fig.add_axes([0.720, 0.315, 0.245, 0.485])
    ax2.set_facecolor("#F7F9FA")
    panel(ax2, "b", "South China Sea context")

    ctx_ok = False
    dash_ok = False

    if ctx_path is not None:
        try:
            gc = _read_vector_for_plot(
                ctx_path, target_crs="EPSG:4326"
            )
            ctx_ok = plot_context_geometry(
                gc, ax2, color="#657984",
                linewidth=0.52, zorder=5
            )
        except Exception as e:
            print("  WARNING: full China context failed:", e)

    if scs_path is not None:
        try:
            gd = _read_vector_for_plot(
                scs_path, target_crs="EPSG:4326"
            )
            dash_ok = plot_nine_dash_geometry(
                gd, ax2, color=INK,
                linewidth=1.05, zorder=8
            )
        except Exception as e:
            print("  WARNING: nine-dash geometry failed:", e)

    draw_scs_graticule(ax2)

    if not dash_ok:
        ax2.text(
            0.50, 0.50, "Nine-dash geometry not rendered",
            transform=ax2.transAxes,
            ha="center", va="center",
            fontsize=6.3, color="#A24D3C",
            fontweight="bold"
        )

    # Compact legend.
    key = fig.add_axes([0.695, 0.070, 0.290, 0.185])
    key.set_axis_off()
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)

    key.add_patch(
        patches.Rectangle(
            (0.02, 0.72), 0.070, 0.13,
            fc=SUPPORT, ec="none"
        )
    )
    key.text(
        0.13, 0.785,
        "Strict common-valid ERPI land pixels",
        fontsize=6.25, va="center"
    )

    key.add_patch(
        patches.Rectangle(
            (0.02, 0.47), 0.070, 0.13,
            fc=NODATA, ec="#BCC7CE", lw=0.5
        )
    )
    key.text(
        0.13, 0.535,
        "Outside data coverage / NoData",
        fontsize=6.25, va="center"
    )

    key.plot(
        [0.025, 0.095], [0.285, 0.285],
        color=INK, lw=1.05, ls="solid"
    )
    key.text(
        0.13, 0.285,
        "Nine-dash cartographic geometry",
        fontsize=6.25, va="center"
    )

    key.text(
        0.02, 0.07,
        f"Native product: 1-km Albers · display: EPSG:4326 · "
        f"common support {EXPECTED_COMMON:,}",
        fontsize=5.55, color=MUTED
    )

    final_ready = (
        boundary_ok and ctx_ok and dash_ok and
        bool(cartographic_source and str(cartographic_source).strip())
    )

    if not final_ready:
        fig.text(
            0.50, 0.015,
            "PREVIEW ONLY · VERIFY CARTOGRAPHIC SOURCE BEFORE SUBMISSION",
            ha="center", va="center",
            fontsize=5.95, fontweight="bold",
            color="#A24D3C"
        )
        export(fig, "Fig1_study_area_FINAL_v7_PREVIEW", out)

        (out/"FIG1_CARTOGRAPHY_STATUS.txt").write_text(
            "Fig. 1 is PREVIEW only.\\n"
            f"National boundary: {bnd_path}\\n"
            f"Full China context: {ctx_path}\\n"
            f"Nine-dash geometry: {scs_path}\\n"
            f"Boundary rendered: {boundary_ok}\\n"
            f"Context rendered: {ctx_ok}\\n"
            f"Nine-dash rendered: {dash_ok}\\n"
            "Publication display CRS: EPSG:4326.\\n"
            "Native ERPI product remains on the 1-km Albers grid.\\n"
            "A verified cartographic-source statement is still required.\\n",
            encoding="utf-8"
        )
    else:
        export(fig, "Fig1_study_area_FINAL_v7", out)
        (out/"Fig1_cartographic_source.txt").write_text(
            str(cartographic_source), encoding="utf-8"
        )
        status = out/"FIG1_CARTOGRAPHY_STATUS.txt"
        if status.exists():
            status.unlink()


def fig2(out, weights):
    """
    Figure 2: five-layer, single-direction workflow.

    Design rules:
    - same-size boxes within each row;
    - no diagonal or crossing arrows;
    - no sentence-length box text;
    - layer colour encodes hierarchy only;
    - automatic containment QA catches future text overflow.
    """
    fig = plt.figure(figsize=(WIDTH, 5.35), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box_records = []

    def section_header(y, number, title):
        circ = patches.Circle((0.055, y), 0.020, facecolor=INK, edgecolor="none", zorder=5)
        ax.add_patch(circ)
        ax.text(0.055, y, str(number), color="white", fontsize=7.4,
                fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(0.087, y, title, fontsize=8.4, fontweight="bold",
                ha="left", va="center", color=INK)

    def workflow_box(x, y, w, h, title, subtitle,
                     facecolor, edgecolor=BORDER, linewidth=0.80,
                     title_fs=7.55, subtitle_fs=6.55):
        p = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0045,rounding_size=0.010",
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, zorder=2
        )
        ax.add_patch(p)
        title_obj = ax.text(
            x + w/2, y + h*0.62, title,
            fontsize=title_fs, fontweight="bold",
            ha="center", va="center", color=INK, zorder=4
        )
        sub_obj = ax.text(
            x + w/2, y + h*0.28, subtitle,
            fontsize=subtitle_fs, fontweight="normal",
            ha="center", va="center", color=MUTED, zorder=4
        )
        box_records.append((p, title_obj, sub_obj, title))
        return p

    def down_arrow(y0, y1, x=0.5):
        ax.annotate(
            "", xy=(x, y1), xytext=(x, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.0,
                            color=WF_ARROW, mutation_scale=10),
            zorder=3
        )

    def right_arrow(x0, x1, y):
        ax.annotate(
            "", xy=(x1, y), xytext=(x0, y),
            arrowprops=dict(arrowstyle="-|>", lw=1.0,
                            color=WF_ARROW, mutation_scale=10),
            zorder=3
        )

    # Layer 1 — annual input data
    section_header(0.955, 1, "Annual input data  |  2001–2020")
    x0, gap = 0.035, 0.014
    w = (0.93 - 4*gap) / 5.0
    xs = [x0 + i*(w+gap) for i in range(5)]
    inputs = [
        ("NDVI", "Vegetation condition"),
        ("CLCD", "Land-cover composition"),
        ("Night lights", "Human-pressure proxy"),
        ("2-m temperature", "Thermal conditions"),
        ("Soil water", "Moisture conditions"),
    ]
    for x, (title, sub) in zip(xs, inputs):
        workflow_box(x, 0.835, w, 0.088, title, sub, WF_INPUT)

    down_arrow(0.825, 0.786)

    # Layer 2 — spatial harmonization
    section_header(0.770, 2, "Spatial harmonization")
    workflow_box(
        0.145, 0.665, 0.710, 0.076,
        "Common 1-km grid",
        "Projection · alignment · resampling · valid-data masking",
        WF_GRID, title_fs=7.9, subtitle_fs=6.6
    )
    ax.text(
        0.5, 0.640,
        "Night-light fill value −32768 is masked before reprojection",
        fontsize=6.25, color=MUTED, ha="center", va="center"
    )

    down_arrow(0.624, 0.586)

    # Layer 3 — directed indicators
    section_header(0.570, 3, "Directed indicator construction")
    indicators = [
        ("VCI", "Relative NDVI signal"),
        ("LCN", "Natural-cover share"),
        ("HCI", "Low night-light pressure"),
        ("TSI", "Thermal reference"),
        ("MSI", "Moisture reference"),
    ]
    for x, (title, sub) in zip(xs, indicators):
        workflow_box(x, 0.450, w, 0.088, title, sub, WF_INDICATOR)

    down_arrow(0.438, 0.400)

    # Layer 4 — fixed integration
    section_header(0.384, 4, "Index integration")
    workflow_box(
        0.195, 0.285, 0.610, 0.070,
        "One pooled entropy-weight vector",
        "Estimated once from pooled 2001–2020 pixel-year observations",
        WF_WEIGHT, edgecolor=INK, linewidth=0.90,
        title_fs=7.75, subtitle_fs=6.45
    )
    weight_txt = " · ".join(f"{k} {weights[k]:.3f}" for k in IND)
    ax.text(0.5, 0.263, weight_txt, fontsize=6.15, color=MUTED,
            ha="center", va="center")

    down_arrow(0.248, 0.210)

    # Layer 5 — products and technical validation
    section_header(0.194, 5, "Products and technical validation")
    workflow_box(
        0.095, 0.070, 0.415, 0.078,
        "Annual ERPI products",
        "20 yearly maps + components + baseline and QA layers",
        WF_OUTPUT, edgecolor=INK, linewidth=0.90,
        title_fs=7.75, subtitle_fs=6.35
    )
    workflow_box(
        0.610, 0.070, 0.295, 0.078,
        "Technical validation",
        "Coverage · reconstruction · sensitivity",
        WF_OUTPUT, edgecolor=BORDER,
        title_fs=7.75, subtitle_fs=6.35
    )
    right_arrow(0.520, 0.600, 0.109)

    # Automated text containment QA.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    failures = []
    for patch, title_obj, sub_obj, label in box_records:
        px, py = patch.get_x(), patch.get_y()
        pw, ph = patch.get_width(), patch.get_height()
        for obj, role in ((title_obj, "title"), (sub_obj, "subtitle")):
            bb = obj.get_window_extent(renderer=renderer)
            pts = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            tx0, ty0 = pts[0]
            tx1, ty1 = pts[1]
            if (tx0 < px-0.006 or tx1 > px+pw+0.006 or
                    ty0 < py-0.006 or ty1 > py+ph+0.006):
                failures.append(f"{label} {role}")

    if failures:
        raise RuntimeError(
            "Fig. 2 layout QA failed; text escaped its box: " + ", ".join(failures)
        )

    export(fig, "Fig2_workflow_FINAL_v6", out)

def fig3(root, out, common_path, scs_line=None, china_context=None,
         boundary=None):
    """
    Figure 3 — corrected ERPI spatial overview, layout-safe publication version.

    Layout
    ------
    Top 72%:
        2 × 2 data panels with generous row/column spacing.
    Bottom 23%:
        left   = one shared north arrow + scale bar
        centre = South China Sea context inset
        right  = shared ERPI colour bar + map notes

    Repeated cartographic furniture is intentionally removed from individual
    panels. This prevents the overlap seen in earlier versions and creates a
    cleaner Scientific Data / Nature-family multi-panel map.
    """
    data_root = root.parent
    scs_path, ctx_path, bnd_path, mainland_path = _discover_fig1_vectors(
        data_root, scs_line, china_context, boundary
    )
    audit_local_vectors(data_root, out)

    with rasterio.open(root/"ERI"/"ERI_2001.tif") as ref:
        display = make_wgs84_display_grid(ref, max_width=1500)

    extent = display["extent"]

    # Display-only products on one exact WGS84 grid.
    mean, common_display = common_mean_wgs84(
        root, common_path, display
    )
    annual_2001 = reproject_raster_to_wgs84(
        root/"ERI"/"ERI_2001.tif", display, categorical=False
    )
    annual_2020 = reproject_raster_to_wgs84(
        root/"ERI"/"ERI_2020.tif", display, categorical=False
    )

    boundary_geo = None
    if bnd_path is not None:
        try:
            boundary_geo = _read_vector_for_plot(
                bnd_path, target_crs="EPSG:4326"
            )
        except Exception as e:
            print("Fig.3 boundary warning:", e)

    # Larger canvas than V7: space is deliberately reserved for row separation
    # and for a dedicated bottom cartographic band.
    fig = plt.figure(figsize=(WIDTH, 6.45), facecolor="white")

    # Main panels: slightly narrower/shorter than before, with a true vertical gap.
    rects = [
        (0.060, 0.640, 0.390, 0.255),  # a
        (0.550, 0.640, 0.390, 0.255),  # b
        (0.060, 0.325, 0.390, 0.255),  # c
        (0.550, 0.325, 0.390, 0.255),  # d
    ]
    axs = [fig.add_axes(r) for r in rects]

    cmap = LinearSegmentedColormap.from_list(
        "ERPI_seq", ERPI_COLORS, N=256
    )
    cmap.set_bad(NODATA)

    # No per-panel north arrows or scale bars. The graticule is sufficient
    # inside each panel; shared map furniture is placed below.
    im = raster_panel_wgs84(
        axs[0], mean, extent,
        "2001–2020 mean", "a", cmap,
        boundary_gdf=boundary_geo,
        lon_step=10, lat_step=10,
        show_north=False, show_scale=False
    )
    raster_panel_wgs84(
        axs[1], annual_2001, extent,
        "2001", "b", cmap,
        boundary_gdf=boundary_geo,
        lon_step=10, lat_step=10,
        show_north=False, show_scale=False
    )
    raster_panel_wgs84(
        axs[2], annual_2020, extent,
        "2020", "c", cmap,
        boundary_gdf=boundary_geo,
        lon_step=10, lat_step=10,
        show_north=False, show_scale=False
    )

    mc = ListedColormap([SUPPORT])
    mc.set_bad(NODATA)
    raster_panel_wgs84(
        axs[3], common_display, extent,
        "Common 20-year support", "d", mc,
        boundary_gdf=boundary_geo,
        lon_step=10, lat_step=10,
        show_north=False, show_scale=False
    )

    # ------------------------------------------------------------------
    # Bottom band — three independent zones, no element is allowed to enter
    # the four data panels.
    # ------------------------------------------------------------------

    # Left: shared north + geodesically calibrated scale bar.
    carto = fig.add_axes([0.060, 0.070, 0.265, 0.145])
    draw_shared_cartographic_reference(
        carto, extent, length_km=1000
    )

    # Centre: shared South China Sea inset.
    inset = fig.add_axes([0.370, 0.055, 0.180, 0.170])
    inset.set_facecolor("#F7F9FA")
    inset_title = inset.text(
        0.0, 1.050,
        "South China Sea context",
        transform=inset.transAxes,
        ha="left", va="bottom",
        fontsize=6.15, color=INK
    )

    ctx_ok = False
    if ctx_path is not None:
        try:
            gc = _read_vector_for_plot(
                ctx_path, target_crs="EPSG:4326"
            )
            ctx_ok = plot_context_geometry(
                gc, inset, color="#657984",
                linewidth=0.42, zorder=5
            )
        except Exception as e:
            print("Fig.3 context warning:", e)

    dash_ok = False
    if scs_path is not None:
        try:
            gd = _read_vector_for_plot(
                scs_path, target_crs="EPSG:4326"
            )
            dash_ok = plot_nine_dash_geometry(
                gd, inset, color=INK,
                linewidth=0.90, zorder=8
            )
        except Exception as e:
            print("Fig.3 nine-dash warning:", e)

    draw_scs_graticule(
        inset,
        lon_ticks=(110, 120),
        lat_ticks=(5, 15, 25),
        label_size=5.0
    )

    if not ctx_ok:
        inset.text(
            0.02, 0.02, "china.shp context not rendered",
            transform=inset.transAxes,
            fontsize=4.7, color="#A24D3C",
            ha="left", va="bottom"
        )

    if not dash_ok:
        inset.text(
            0.50, 0.50,
            "Nine-dash geometry\\nnot rendered",
            transform=inset.transAxes,
            fontsize=5.0, color="#A24D3C",
            ha="center", va="center"
        )

    # Right: shared colour bar and compact notes.
    legend_ax = fig.add_axes([0.615, 0.060, 0.325, 0.155])
    legend_ax.set_axis_off()

    cb = fig.add_axes([0.650, 0.165, 0.255, 0.014])
    cbar = fig.colorbar(
        im, cax=cb, orientation="horizontal",
        ticks=[0, 0.25, 0.5, 0.75, 1]
    )
    cbar.ax.tick_params(
        labelsize=6.05, length=2, pad=1
    )
    cbar.outline.set_linewidth(0.45)

    t1 = fig.text(
        0.650, 0.192,
        "ERPI (unitless)",
        fontsize=6.65, ha="left", va="bottom"
    )
    t2 = fig.text(
        0.650, 0.125,
        "Grey: NoData / outside coverage",
        fontsize=5.75, ha="left", color=MUTED
    )
    t3 = fig.text(
        0.650, 0.105,
        "Blue in panel d: strict common-valid pixels",
        fontsize=5.75, ha="left", color=MUTED
    )
    t4 = fig.text(
        0.650, 0.078,
        "Display CRS: EPSG:4326",
        fontsize=5.25, ha="left", color=MUTED
    )
    t5 = fig.text(
        0.650, 0.061,
        "Released rasters: native 1-km Albers grid",
        fontsize=5.25, ha="left", color=MUTED
    )

    # Figure-level overlap QA. This runs before export and prevents accidental
    # future edits from recreating the crowding seen in earlier versions.
    validate_figure3_layout(
        fig,
        main_axes=axs,
        inset_ax=inset,
        carto_ax=carto,
        colorbar_ax=cb,
        extra_text_artists=[t1, t2, t3, t4, t5]
    )

    export(fig, "Fig3_data_overview_FINAL_v7_1_layoutsafe", out)


def find_fullscan(root):
    choices=[
        root/"FINAL_ERPI_TECHNICAL_AUDIT_NTLfix"/"02_ERPI_year_fullscan_statistics.csv",
        root/"FINAL_ERPI_TECHNICAL_AUDIT"/"02_ERPI_year_fullscan_statistics.csv",
    ]
    for p in choices:
        if p.exists(): return p
    return None

def fig4(root,out):
    common=must_csv(root/"technical_validation"/"annual_eri_common_mask.csv",
                    ["year","ERI_entropy_common_mask_mean"])
    fs=find_fullscan(root)
    if fs:
        qa=must_csv(fs,["year","valid_pixels","q01_approx","mean","q99_approx"])
        low=qa.q01_approx.to_numpy(); high=qa.q99_approx.to_numpy()
    else:
        # Fallback: derive valid counts and deterministic 1%/99% sampled quantiles.
        rows=[]
        for y in YEARS:
            vals=[]
            n=0;s=0.0
            with rasterio.open(root/"ERI"/f"ERI_{y}.tif") as src:
                for _,win in src.block_windows(1):
                    a=src.read(1,window=win,masked=True).astype(np.float64).filled(np.nan)
                    x=a[np.isfinite(a)]
                    n+=x.size;s+=float(x.sum())
                    # deterministic thinning to control memory
                    if x.size: vals.append(x[::max(1,x.size//2000)])
            z=np.concatenate(vals)
            rows.append(dict(year=y,valid_pixels=n,mean=s/n,
                             q01_approx=float(np.quantile(z,.01)),
                             q99_approx=float(np.quantile(z,.99))))
        qa=pd.DataFrame(rows)
        low=qa.q01_approx.to_numpy();high=qa.q99_approx.to_numpy()

    fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.88),facecolor="white")
    fig.subplots_adjust(left=.095,right=.98,top=.77,bottom=.23,wspace=.32)
    x=qa.year.to_numpy()

    ax=axs[0]
    ax.plot(x,(qa.valid_pixels.to_numpy()-qa.valid_pixels.iloc[0])/1000,
            color=BLUE,marker="o",ms=2.6,lw=1.4)
    ax.axhline(0,lw=.65,color=FAINT);years_axis(ax);axes_clean(ax)
    ax.set_ylabel("Change in valid pixels (×10³)");ax.set_xlabel("Year")
    panel(ax,"a","Spatial coverage")

    ax=axs[1]
    ax.fill_between(x,low,high,facecolor="#D6E0E4",alpha=.58,lw=0,label="1st–99th percentiles")
    ax.plot(x,qa["mean"],color=TEAL,lw=1.5,label="Annual-mask mean")
    ax.plot(x,common.ERI_entropy_common_mask_mean,color=ORANGE,ls="--",lw=1.0,label="Common-mask mean")
    ax.set_ylim(0,1);years_axis(ax);axes_clean(ax);ax.set_xlabel("Year");ax.set_ylabel("ERPI (unitless)")
    ax.legend(loc="upper left",frameon=False,fontsize=6.5,handlelength=2.0,labelspacing=.3)
    panel(ax,"b","Annual distribution and spatial support")
    export(fig,"Fig4_output_quality_FINAL_v6",out)

def fig5(root,out):
    tech=root/"technical_validation"
    eq=must_csv(tech/"entropy_vs_equal_comparison.csv",
                ["year","pearson_r_full","spearman_rho_sample","equal_mean"])
    loo=must_csv(tech/"leave_one_out_comparison.csv",
                 ["year","omitted_indicator","pearson_r_full","leave_one_out_mean"])
    annual=must_csv(tech/"annual_eri_common_mask.csv",
                    ["year","ERI_entropy_common_mask_mean"])
    d=must_csv(tech/"change_decomposition_2019_2020.csv",
               ["indicator","weighted_change_2020_minus_2019"])

    fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.45),facecolor="white")
    fig.subplots_adjust(left=.105,right=.975,top=.88,bottom=.105,wspace=.34,hspace=.53)

    ax=axs[0,0]
    ax.plot(eq.year,eq.pearson_r_full,color=BLUE,marker="o",ms=2.3,label="Pearson r")
    ax.plot(eq.year,eq.spearman_rho_sample,color=ORANGE,ls="--",marker="s",ms=2.1,
            label="Spearman ρ (sample)")
    ax.set_ylim(.84,1.01);years_axis(ax);axes_clean(ax)
    ax.set_ylabel("Spatial correlation");ax.set_xlabel("Year")
    ax.legend(loc="lower left",frameon=False,fontsize=6.6,ncol=2,handlelength=2)
    panel(ax,"a","Weighting sensitivity")

    ax=axs[0,1]
    for j,k in enumerate(IND):
        z=loo[loo.omitted_indicator==k].sort_values("year")
        v=z.pearson_r_full.to_numpy()
        ax.plot([v.min(),v.max()],[j,j],color="#B9C3CC",lw=2,zorder=1)
        ax.scatter(v,j+np.linspace(-.11,.11,len(v)),s=6,color=COMP[k],alpha=.48,zorder=2)
        ax.scatter([v.mean()],[j],s=34,color=COMP[k],edgecolor="white",lw=.35,zorder=4)
    ax.set_yticks(range(5),[f"Without {k}" for k in IND]);ax.invert_yaxis()
    ax.set_xlim(.55,1.02);ax.set_xlabel("Pearson r with full ERPI");ax.axvline(1,color=FAINT,lw=.6)
    axes_clean(ax);panel(ax,"b","Leave-one-out sensitivity")

    ax=axs[1,0]
    full=annual.sort_values("year")
    vci=loo[loo.omitted_indicator=="VCI"].sort_values("year")
    curves=[
        (full.year,full.ERI_entropy_common_mask_mean,"Entropy weighted",TEAL,"-"),
        (eq.year,eq.equal_mean,"Equal weighted",BLUE,"--"),
        (vci.year,vci.leave_one_out_mean,"Without VCI",ORANGE,"-"),
    ]
    for y,v,l,c,ls in curves:
        vv=np.asarray(v,dtype=float); ax.plot(y,vv-vv[0],color=c,ls=ls,marker="o",ms=2.2,lw=1.35,label=l)
    ax.axhline(0,color=FAINT,lw=.7);years_axis(ax);axes_clean(ax)
    ax.set_xlabel("Year");ax.set_ylabel("Change relative to 2001")
    ax.legend(loc="upper left",frameon=False,fontsize=6.5,handlelength=1.7,labelspacing=.3)
    panel(ax,"c","Temporal sensitivity to VCI")

    ax=axs[1,1]
    d=d.set_index("indicator").loc[IND].reset_index()
    vals=d.weighted_change_2020_minus_2019.to_numpy()
    yy=np.arange(len(IND))
    ax.barh(yy,vals,height=.47,color=[COMP[k] for k in IND],edgecolor="none")
    ax.set_yticks(yy,IND);ax.invert_yaxis();ax.axvline(0,lw=.75,color=MUTED)
    for j,v in enumerate(vals):
        ax.annotate(f"{v:+.4f}",(v,j),xytext=(4 if v>=0 else -4,0),textcoords="offset points",
                    ha="left" if v>=0 else "right",va="center",fontsize=6.4)
    span=max(abs(vals))*1.35
    ax.set_xlim(min(-span*.45,float(vals.min()*1.6)),max(span,float(vals.max()*1.2)))
    ax.set_xlabel("Weighted change, 2020 − 2019");ax.xaxis.set_major_locator(MaxNLocator(4))
    axes_clean(ax);panel(ax,"d","Annual contribution decomposition")
    export(fig,"Fig5_robustness_FINAL_v6",out)

def figs1(root,out,weights):
    stat=must_csv(root/"technical_validation"/"annual_indicator_contributions_common_mask.csv",
                  ["year"]+[f"{k}_mean" for k in IND])
    fig,axs=plt.subplots(5,2,figsize=(WIDTH,5.65),facecolor="white",
                         gridspec_kw={"width_ratios":[3.6,1]})
    fig.subplots_adjust(left=.12,right=.975,top=.96,bottom=.09,wspace=.24,hspace=.22)
    for j,k in enumerate(IND):
        ax=axs[j,0]
        ax.plot(stat.year,stat[f"{k}_mean"],color=COMP[k],marker="o",ms=2.1,lw=1.25)
        ax.set_ylim(0,1);years_axis(ax);axes_clean(ax)
        ax.set_ylabel(k,rotation=0,labelpad=17,va="center",fontweight="bold")
        if j<4:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel("Year")
        ax2=axs[j,1]
        ax2.barh([0],[weights[k]],height=.46,color=COMP[k])
        ax2.set_xlim(0,.5);ax2.set_ylim(-.75,.75);ax2.set_yticks([]);ax2.set_xticks([0,.25,.5])
        ax2.spines[["top","right","left"]].set_visible(False);ax2.tick_params(axis="x",labelsize=6.7)
        ax2.text(weights[k]+.014,0,f"{weights[k]:.3f}",ha="left",va="center",fontsize=7)
        if j<4:ax2.tick_params(labelbottom=False)
        else:ax2.set_xlabel("Fixed weight")
    export(fig,"FigS1_components_and_weights_FINAL_v6",out)

def figs2(root,out):
    npp=root/"independent_NPP_only_validation"
    a=must_csv(npp/"NPP_annual_spatial_validation_and_means.csv",["year","pearson_r_full"])
    b=must_csv(npp/"NPP_long_term_spatial_validation.csv",
               ["group","comparison","pearson_r_full","spearman_rho_systematic_sample"])
    b=b[b.group=="all"].set_index("comparison")
    choices=[("Mean NPP","mean_ERI_vs_mean_NPP"),
             ("NPP stability","mean_ERI_vs_bounded_NPP_stability")]

    fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.85),facecolor="white")
    fig.subplots_adjust(left=.095,right=.975,top=.76,bottom=.23,wspace=.33)
    ax=axs[0]
    ax.plot(a.year,a.pearson_r_full,color=BLUE,lw=1.3,marker="o",ms=2.5)
    ax.axhline(0,color=MUTED,lw=.65);years_axis(ax);axes_clean(ax)
    ax.set_xlabel("Year");ax.set_ylabel("Pearson r");panel(ax,"a","Annual ERPI–NPP association")

    ax=axs[1];yy=np.arange(2)
    rr=np.array([b.loc[k,"pearson_r_full"] for _,k in choices],dtype=float)
    ss=np.array([b.loc[k,"spearman_rho_systematic_sample"] for _,k in choices],dtype=float)
    ax.scatter(rr,yy-.12,color=BLUE,s=30,zorder=4,label="Pearson r")
    ax.scatter(ss,yy+.12,facecolors="white",edgecolors=ORANGE,lw=1.2,s=30,zorder=4,
               label="Spearman ρ (sample)")
    ax.axvline(0,color=MUTED,lw=.65);ax.set_yticks(yy,[x for x,_ in choices]);ax.invert_yaxis()
    ax.set_xlim(-.34,.25);ax.set_xlabel("Spatial correlation")
    ax.legend(loc="lower left",frameon=False,fontsize=6.6);axes_clean(ax)
    panel(ax,"b","Long-term NPP plausibility")
    export(fig,"FigS2_NPP_plausibility_FINAL_v6",out)

CAPTIONS = """# Final corrected figure legends | publication style V6

**Figure 1 | Study domain and spatial support of ERPI.**
(a) Mainland-China terrestrial cells valid throughout 2001–2020, displayed in geographic coordinates (EPSG:4326) for cartographic consistency; the released raster product remains on its native 1-km Albers grid. Grey denotes outside coverage or NoData. (b) South China Sea geographic context, shown for orientation only and not as ERPI coverage.

**Figure 2 | Workflow for constructing the annual ERPI dataset.**
Annual NDVI, CLCD land cover, night-light, temperature and soil-water data were harmonized to a common 1-km grid and converted into five directed indicators (VCI, LCN, HCI, TSI and MSI). A pooled entropy-weight vector estimated from 2001–2020 pixel-year observations was applied uniformly to generate annual ERPI products, followed by coverage, reconstruction and sensitivity analyses.

**Figure 3 | Spatial overview of the corrected ERPI product.**
Panels show the 2001–2020 mean, annual ERPI in 2001 and 2020, and the strict 20-year common support. For publication display, raster and vector layers are reprojected to geographic coordinates (EPSG:4326), producing a rectilinear longitude–latitude grid consistent with the South China Sea inset; the distributed ERPI rasters remain on the native 1-km Albers grid. ERPI panels share a 0–1 sequential scale; grey is NoData, not ERPI = 0.

**Figure 4 | Output integrity and spatial support.**
Annual valid-cell counts are shown relative to 2001, together with annual ERPI distributions and means calculated on year-specific and fixed common support. Agreement of the two mean series indicates limited sensitivity to small annual coverage changes.

**Figure 5 | Sensitivity to index construction.**
Panels compare entropy and equal weighting, five leave-one-out alternatives, common-support temporal trajectories, and weighted 2019–2020 component contributions. The analyses identify dependence on weighting and component inclusion rather than causal ecological effects.

**Supplementary Figure S1 | Corrected component means and fixed weights.**
Annual common-support means of VCI, LCN, HCI, TSI and MSI are shown alongside the corrected fixed pooled entropy weights.

**Supplementary Figure S2 | NPP plausibility assessment.**
Annual spatial ERPI–NPP correlations and long-term associations of mean ERPI with mean NPP and bounded NPP stability are calculated on the strict corrected ERPI–NPP common support. NPP is not treated as resilience ground truth.
"""

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--out",type=Path,default=None)
    ap.add_argument("--scs-line",type=Path,default=DATA_ROOT/"MAP"/"shp"/"china_nine_dotted_line.shp")
    ap.add_argument("--china-context",type=Path,default=DATA_ROOT/"MAP"/"shp"/"china.shp",
                    help="China/coastline context vector for the South China Sea panel.")
    ap.add_argument("--boundary",type=Path,default=DATA_ROOT/"MAP"/"shp"/"china_country.shp",
                    help="Mainland study-boundary vector for panel a.")
    ap.add_argument("--cartographic-source",default=None,
                    help="Provider/product/version/licence for the South China Sea context layer.")
    ap.add_argument("--only",default="fig1,fig2,fig3,fig4,fig5,figS1,figS2")
    args=ap.parse_args()

    root=args.root
    out=args.out or root/OUTNAME
    out.mkdir(parents=True,exist_ok=True)
    style()
    weights=read_weights(root)
    common=discover_common_mask(root)
    ncommon=common_count(common)
    if ncommon != EXPECTED_COMMON:
        raise RuntimeError(f"Common-support identity mismatch: found {ncommon:,}, expected {EXPECTED_COMMON:,}.")
    (out/"FIGURE_CAPTIONS_FINAL_v7_1.md").write_text(CAPTIONS,encoding="utf-8")
    (out/"FIGURE_VERSION_IDENTITY.json").write_text(json.dumps({
        "root":str(root),
        "strict_common_pixels":ncommon,
        "weights":weights,
        "product":"corrected NTL candidate",
        "figure_version":"FINAL_v7_1_layoutsafe"
    },ensure_ascii=False,indent=2),encoding="utf-8")

    jobs={
        "fig1":lambda:fig1(root,out,common,args.scs_line,args.china_context,args.boundary,args.cartographic_source),
        "fig2":lambda:fig2(out,weights),
        "fig3":lambda:fig3(root,out,common,args.scs_line,args.china_context,args.boundary),
        "fig4":lambda:fig4(root,out),
        "fig5":lambda:fig5(root,out),
        "figS1":lambda:figs1(root,out,weights),
        "figS2":lambda:figs2(root,out),
    }
    selected=[x.strip() for x in args.only.split(",") if x.strip()]
    unknown=set(selected)-set(jobs)
    if unknown:
        ap.error("Unknown figure code(s): "+", ".join(sorted(unknown)))
    for k in selected:
        print("Building",k)
        jobs[k]()
    print("Finished:",out)
    print("All data-dependent figures were drawn from the corrected candidate only.")

if __name__=="__main__":
    main()
