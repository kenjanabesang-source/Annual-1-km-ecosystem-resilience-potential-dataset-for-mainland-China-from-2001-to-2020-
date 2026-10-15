# -*- coding: utf-8 -*-
r"""
Final tailored workflow for recalculating China's Ecosystem Resilience Index (ERI)
=================================================================================

This script is tailored to the file structure confirmed in this project:

CLCD annual land cover:
    F:\data\China\CLCD\CLCD_v01_2001_albert.tif
    ...
    F:\data\China\CLCD\CLCD_v01_2020_albert.tif

Annual NDVI:
    F:\data\China\NDVI\2001_NDVI.tif
    ...
    F:\data\China\NDVI\2020_NDVI.tif

Annual Nighttime Light:
    F:\data\China\NightLight\PANDA_China_2001.tif
    ...
    F:\data\China\NightLight\PANDA_China_2020.tif

2-m temperature, one 26-band GeoTIFF:
    F:\data\China\temperature_2m_2000_2025.tif
    band 1 = 2000
    ...
    band 20 = 2019
    ...
    band 26 = 2025

Volumetric soil water layer 1, one 26-band GeoTIFF:
    F:\data\China\volumetric_soil_water_layer_1_2000_2025.tif
    band 1 = 2000
    ...
    band 20 = 2019
    ...
    band 26 = 2025

China boundary:
    edit CHINA_BOUNDARY below to match your actual shapefile.

Main outputs
------------
1. VCI: vegetation condition index
2. LCN: land-cover naturalness
3. HCI: low human-disturbance condition
4. TSI: thermal stability index
5. MSI: soil-moisture stability index
6. One fixed pooled entropy-weight vector for 2001-2020
7. ERI_2001.tif ... ERI_2020.tif
8. QA tables

Core design
-----------
- Only existing data are used.
- Output grid is 1 km.
- Target CRS defaults to the CRS of CLCD_2000, preserving its equal-area Albers grid concept.
- NoData remains NoData; it is never converted to zero.
- Fixed 2001-2020 entropy weights are used for all years to improve temporal comparability.
- NDVI annual files are treated as already-aggregated annual products; no P90 calculation is attempted.
- Temperature and soil moisture are read directly by year from the 26-band GeoTIFFs.

Recommended environment
-----------------------
conda create -n eri python=3.11
conda activate eri
conda install -c conda-forge rasterio geopandas numpy pandas shapely pyproj tqdm

"""

from __future__ import annotations

import math
import json
import warnings
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.transform import from_origin
import geopandas as gpd
from tqdm import tqdm


# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

ROOT = Path(os.environ.get("ERPI_DATA_ROOT", "./data")).expanduser().resolve()

def _find_case_insensitive_child(root: Path, wanted_name: str, want_dir: bool = False) -> Path:
    """Find a child of root by case-insensitive name."""
    wanted = wanted_name.lower()
    if root.exists():
        for p in root.iterdir():
            if p.name.lower() == wanted and (p.is_dir() if want_dir else p.is_file()):
                return p
    return root / wanted_name

def _find_boundary_shp(root: Path) -> Path:
    """Prefer China_boundary.shp; otherwise find a plausible China boundary shp."""
    exact = _find_case_insensitive_child(root, "China_boundary.shp", want_dir=False)
    if exact.exists():
        return exact
    if root.exists():
        shp_files = sorted(root.glob("*.shp"))
        for p in shp_files:
            s = p.stem.lower()
            if "china" in s and ("boundary" in s or "bound" in s):
                return p
        if len(shp_files) == 1:
            return shp_files[0]
    return root / "China_boundary.shp"

# Exact structure visible in your F:\data\China screenshot
CHINA_BOUNDARY = _find_boundary_shp(ROOT)
CLCD_DIR = _find_case_insensitive_child(ROOT, "clcd", want_dir=True)
NDVI_DIR = _find_case_insensitive_child(ROOT, "NDVI", want_dir=True)
NTL_DIR = _find_case_insensitive_child(ROOT, "NightLight", want_dir=True)

T2M_FILE = _find_case_insensitive_child(ROOT, "temperature_2m_2000_2025.tif", want_dir=False)
SM1_FILE = _find_case_insensitive_child(ROOT, "volumetric_soil_water_layer_1_2000_2025.tif", want_dir=False)

OUT_DIR = ROOT / "ERI_recalculated_2001_2020"

YEARS = list(range(2001, 2021))
RESOLUTION_M = 1000

OUT_NODATA = -9999.0
BLOCK = 512
COMPRESS = "deflate"
NUM_THREADS = 4  # rasterio requires an integer here; increase if desired

# NDVI handling
NDVI_SCALE_IF_RAW = 0.0001
NDVI_VALID_MIN = -0.2
NDVI_VALID_MAX = 1.0
VCI_Q_LOW = 5.0
VCI_Q_HIGH = 95.0

# CLCD classes
CLCD_CODES = {
    "cropland": 1,
    "forest": 2,
    "shrub": 3,
    "grassland": 4,
    "water": 5,
    "snow_ice": 6,
    "barren": 7,
    "impervious": 8,
    "wetland": 9,
}

# If water + snow/ice fraction exceeds this threshold, exclude the 1-km cell.
WATER_SNOW_MASK_THRESHOLD = 0.50

# NTL normalization
NTL_Q_LOW = 1.0
NTL_Q_HIGH = 99.0
NTL_SAMPLE_PER_YEAR = 250_000

# Stability indices
MAD_SCALE = 1.4826
EPS = 1e-6


# =============================================================================
# 1. SMALL UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def valid(a):
    return np.isfinite(a) & (a != OUT_NODATA)


def iter_windows(width, height, block=BLOCK):
    for row in range(0, height, block):
        h = min(block, height - row)
        for col in range(0, width, block):
            w = min(block, width - col)
            yield Window(col, row, w, h)


def output_profile(grid, dtype="float32", nodata=OUT_NODATA):
    return {
        "driver": "GTiff",
        "width": grid["width"],
        "height": grid["height"],
        "count": 1,
        "dtype": dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": COMPRESS,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }


def write_float(path: Path, arr: np.ndarray, grid):
    with rasterio.open(path, "w", **output_profile(grid)) as dst:
        dst.write(arr.astype(np.float32), 1)


def resolve_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(str(p) for p in candidates))


# =============================================================================
# 2. INPUT FILE RESOLUTION
# =============================================================================

def clcd_path(year: int) -> Path:
    return resolve_existing(
        CLCD_DIR / f"CLCD_v01_{year}_albert.tif",
        CLCD_DIR / f"CLCD_v01_{year}_albers.tif",
        CLCD_DIR / f"CLCD_v01_{year}.tif",
    )


def ndvi_path(year: int) -> Path:
    return resolve_existing(
        NDVI_DIR / f"{year}_NDVI.tif",
        NDVI_DIR / f"{year}_NDVI.TIF",
    )


def ntl_path(year: int) -> Path:
    return resolve_existing(
        NTL_DIR / f"PANDA_China_{year}.tif",
        NTL_DIR / f"PANDA_China_{year}.TIF",
    )


def year_to_band(year: int) -> int:
    # 2001 -> 2, 2020 -> 21 (source band 1 is year 2000)
    return year - 2000 + 1


# =============================================================================
# 3. BUILD FIXED 1-KM GRID FROM CLCD PROJECTION
# =============================================================================

def build_target_grid():
    print("[1/10] Building 1-km target grid from CLCD projection ...")

    ref_path = clcd_path(2001)
    with rasterio.open(ref_path) as ref:
        if ref.crs is None:
            raise ValueError("CLCD 2001 has no CRS.")
        target_crs = ref.crs

    gdf = gpd.read_file(CHINA_BOUNDARY)
    if gdf.crs is None:
        raise ValueError("China boundary shapefile has no CRS.")
    gdf = gdf.to_crs(target_crs)

    xmin, ymin, xmax, ymax = gdf.total_bounds

    xmin = math.floor(xmin / RESOLUTION_M) * RESOLUTION_M
    ymin = math.floor(ymin / RESOLUTION_M) * RESOLUTION_M
    xmax = math.ceil(xmax / RESOLUTION_M) * RESOLUTION_M
    ymax = math.ceil(ymax / RESOLUTION_M) * RESOLUTION_M

    width = int(round((xmax - xmin) / RESOLUTION_M))
    height = int(round((ymax - ymin) / RESOLUTION_M))
    transform = from_origin(xmin, ymax, RESOLUTION_M, RESOLUTION_M)

    grid = {
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "bounds": (xmin, ymin, xmax, ymax),
    }

    geoms = [g.__geo_interface__ for g in gdf.geometry if g is not None and not g.is_empty]
    inside = geometry_mask(
        geoms,
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )

    return grid, inside


# =============================================================================
# 4. GENERIC REPROJECTION
# =============================================================================

def reproject_band_to_grid(
    src_path: Path,
    grid,
    band=1,
    resampling=Resampling.bilinear,
    src_nodata_override=None,
):
    out = np.full((grid["height"], grid["width"]), OUT_NODATA, dtype=np.float32)

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src_nodata_override is None else src_nodata_override

        reproject(
            source=rasterio.band(src, band),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=OUT_NODATA,
            resampling=resampling,
            num_threads=NUM_THREADS,
        )

    return out


# =============================================================================
# 5. NDVI: ANNUAL NDVI -> 1 KM -> VCI INPUT
# =============================================================================

def preprocess_ndvi(grid, inside, out_dir):
    print("[2/10] Reprojecting annual NDVI to 1-km grid ...")
    d = ensure_dir(out_dir / "intermediate" / "NDVI")

    for year in YEARS:
        out = d / f"NDVI_{year}.tif"
        if out.exists():
            continue

        src = ndvi_path(year)
        arr = reproject_band_to_grid(src, grid, resampling=Resampling.average)

        m = valid(arr)

        # Automatic detection:
        # If data look like MODIS integer DN, apply scale factor.
        if np.any(m):
            probe = arr[m]
            probe = probe[::max(1, probe.size // 100000)]
            med_abs = float(np.nanmedian(np.abs(probe)))
            max_abs = float(np.nanmax(np.abs(probe)))

            if med_abs > 2.0 or max_abs > 10.0:
                arr[m] *= NDVI_SCALE_IF_RAW

        m = valid(arr)
        bad = m & ((arr < NDVI_VALID_MIN) | (arr > NDVI_VALID_MAX))
        arr[bad] = OUT_NODATA
        arr[~inside] = OUT_NODATA

        write_float(out, arr, grid)


# =============================================================================
# 6. CLCD 30 m -> 1-km LCN
# =============================================================================

def create_clcd_binary(src_path: Path, out_path: Path, mode: str):
    """
    Native-resolution temporary byte mask.

    modified:
        cropland or impervious = 1
    water_snow:
        water or snow/ice = 1
    valid:
        classes 1-9 = 1
    """
    codes = list(CLCD_CODES.values())

    with rasterio.open(src_path) as src:
        p = src.profile.copy()
        p.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=255,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(out_path, "w", **p) as dst:
            for _, win in src.block_windows(1):
                a = src.read(1, window=win)
                is_valid = np.isin(a, codes)

                if mode == "modified":
                    x = np.isin(a, [CLCD_CODES["cropland"], CLCD_CODES["impervious"]])
                elif mode == "water_snow":
                    x = np.isin(a, [CLCD_CODES["water"], CLCD_CODES["snow_ice"]])
                elif mode == "valid":
                    x = is_valid
                else:
                    raise ValueError(mode)

                b = np.where(is_valid, x.astype(np.uint8), 255).astype(np.uint8)
                dst.write(b, 1, window=win)


def reproject_fraction(mask_path: Path, grid):
    arr = np.full((grid["height"], grid["width"]), OUT_NODATA, dtype=np.float32)

    with rasterio.open(mask_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=255,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=OUT_NODATA,
            resampling=Resampling.average,
            num_threads=NUM_THREADS,
        )

    return arr


def preprocess_clcd(grid, inside, out_dir):
    print("[3/10] Aggregating CLCD 30 m to 1-km LCN ...")
    out_lcn = ensure_dir(out_dir / "intermediate" / "LCN")
    temp = ensure_dir(out_dir / "_tmp_clcd")

    for year in YEARS:
        dst_path = out_lcn / f"LCN_{year}.tif"
        if dst_path.exists():
            continue

        src = clcd_path(year)

        mpath = temp / f"modified_{year}.tif"
        wspath = temp / f"water_snow_{year}.tif"
        vpath = temp / f"valid_{year}.tif"

        create_clcd_binary(src, mpath, "modified")
        create_clcd_binary(src, wspath, "water_snow")
        create_clcd_binary(src, vpath, "valid")

        modified = reproject_fraction(mpath, grid)
        water_snow = reproject_fraction(wspath, grid)
        validfrac = reproject_fraction(vpath, grid)

        lcn = np.full(modified.shape, OUT_NODATA, dtype=np.float32)

        m = inside & valid(modified) & valid(water_snow) & valid(validfrac)
        landfrac = validfrac - water_snow

        m &= water_snow < WATER_SNOW_MASK_THRESHOLD
        m &= landfrac > EPS

        lcn[m] = 1.0 - np.clip(modified[m] / landfrac[m], 0.0, 1.0)
        lcn[m] = np.clip(lcn[m], 0.0, 1.0)

        write_float(dst_path, lcn, grid)

        for p in [mpath, wspath, vpath]:
            try:
                p.unlink()
            except Exception:
                pass


# =============================================================================
# 7. NIGHTTIME LIGHT -> 1 KM RAW
# =============================================================================

def preprocess_ntl(grid, inside, out_dir):
    print("[4/10] Reprojecting annual Nighttime Light to 1-km grid ...")
    d = ensure_dir(out_dir / "intermediate" / "NTL")

    for year in YEARS:
        out = d / f"NTL_{year}.tif"
        if out.exists():
            continue

        arr = reproject_band_to_grid(
            ntl_path(year),
            grid,
            resampling=Resampling.bilinear,
        )

        m = valid(arr)
        arr[m & (arr < 0)] = 0.0
        arr[~inside] = OUT_NODATA
        write_float(out, arr, grid)


# =============================================================================
# 8. 26-BAND TEMPERATURE AND SOIL MOISTURE -> ANNUAL 1 KM
# =============================================================================

def inspect_multiband_file(path: Path, expected_min_bands=20):
    with rasterio.open(path) as src:
        if src.count < expected_min_bands:
            raise ValueError(f"{path.name} has only {src.count} bands; expected >= {expected_min_bands}.")
        if src.crs is None:
            raise ValueError(f"{path.name} has no CRS.")


def preprocess_multiband_annual(src_path: Path, prefix: str, grid, inside, out_dir):
    d = ensure_dir(out_dir / "intermediate" / prefix)

    with rasterio.open(src_path) as src:
        for year in YEARS:
            out = d / f"{prefix}_{year}.tif"
            if out.exists():
                continue

            band = year_to_band(year)
            arr = np.full((grid["height"], grid["width"]), OUT_NODATA, dtype=np.float32)

            reproject(
                source=rasterio.band(src, band),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                dst_nodata=OUT_NODATA,
                resampling=Resampling.bilinear,
                num_threads=NUM_THREADS,
            )

            # Uploaded files use NaN internally; remove non-finite values explicitly.
            arr[~np.isfinite(arr)] = OUT_NODATA
            arr[~inside] = OUT_NODATA

            write_float(out, arr, grid)


# =============================================================================
# 9. TEMPORAL BASELINES
# =============================================================================

def temporal_percentile(paths, q, out_path, grid):
    srcs = [rasterio.open(p) for p in paths]

    try:
        with rasterio.open(out_path, "w", **output_profile(grid)) as dst:
            for win in tqdm(iter_windows(grid["width"], grid["height"]), desc=out_path.stem):
                stack = []
                for src in srcs:
                    a = src.read(1, window=win).astype(np.float32)
                    a[a == OUT_NODATA] = np.nan
                    stack.append(a)

                x = np.stack(stack, axis=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    o = np.nanpercentile(x, q, axis=0)

                o = np.where(np.isfinite(o), o, OUT_NODATA).astype(np.float32)
                dst.write(o, 1, window=win)
    finally:
        for src in srcs:
            src.close()


def temporal_median(paths, out_path, grid):
    srcs = [rasterio.open(p) for p in paths]

    try:
        with rasterio.open(out_path, "w", **output_profile(grid)) as dst:
            for win in tqdm(iter_windows(grid["width"], grid["height"]), desc=out_path.stem):
                stack = []
                for src in srcs:
                    a = src.read(1, window=win).astype(np.float32)
                    a[a == OUT_NODATA] = np.nan
                    stack.append(a)

                x = np.stack(stack, axis=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    o = np.nanmedian(x, axis=0)

                o = np.where(np.isfinite(o), o, OUT_NODATA).astype(np.float32)
                dst.write(o, 1, window=win)
    finally:
        for src in srcs:
            src.close()


def temporal_mad(paths, median_path, out_path, grid):
    srcs = [rasterio.open(p) for p in paths]
    medsrc = rasterio.open(median_path)

    try:
        with rasterio.open(out_path, "w", **output_profile(grid)) as dst:
            for win in tqdm(iter_windows(grid["width"], grid["height"]), desc=out_path.stem):
                med = medsrc.read(1, window=win).astype(np.float32)
                med[med == OUT_NODATA] = np.nan

                diffs = []
                for src in srcs:
                    a = src.read(1, window=win).astype(np.float32)
                    a[a == OUT_NODATA] = np.nan
                    diffs.append(np.abs(a - med))

                x = np.stack(diffs, axis=0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    o = np.nanmedian(x, axis=0)

                o = np.where(np.isfinite(o), o, OUT_NODATA).astype(np.float32)
                dst.write(o, 1, window=win)
    finally:
        medsrc.close()
        for src in srcs:
            src.close()


def compute_baselines(out_dir, grid):
    print("[6/10] Computing long-term 2001-2020 pixel baselines ...")
    bdir = ensure_dir(out_dir / "baselines")

    ndvi = [out_dir / "intermediate" / "NDVI" / f"NDVI_{y}.tif" for y in YEARS]
    t2m = [out_dir / "intermediate" / "T2M" / f"T2M_{y}.tif" for y in YEARS]
    sm1 = [out_dir / "intermediate" / "SM1" / f"SM1_{y}.tif" for y in YEARS]

    ndvi_lo = bdir / "NDVI_Q05_2001_2020.tif"
    ndvi_hi = bdir / "NDVI_Q95_2001_2020.tif"
    t_med = bdir / "T2M_median_2001_2020.tif"
    t_mad = bdir / "T2M_MAD_2001_2020.tif"
    sm_med = bdir / "SM1_median_2001_2020.tif"
    sm_mad = bdir / "SM1_MAD_2001_2020.tif"

    if not ndvi_lo.exists():
        temporal_percentile(ndvi, VCI_Q_LOW, ndvi_lo, grid)
    if not ndvi_hi.exists():
        temporal_percentile(ndvi, VCI_Q_HIGH, ndvi_hi, grid)

    if not t_med.exists():
        temporal_median(t2m, t_med, grid)
    if not t_mad.exists():
        temporal_mad(t2m, t_med, t_mad, grid)

    if not sm_med.exists():
        temporal_median(sm1, sm_med, grid)
    if not sm_mad.exists():
        temporal_mad(sm1, sm_med, sm_mad, grid)


# =============================================================================
# 10. POOLED NTL QUANTILES
# =============================================================================

def regular_sample(path: Path, max_n: int):
    with rasterio.open(path) as src:
        vals = []
        for _, win in src.block_windows(1):
            a = src.read(1, window=win)
            m = valid(a)
            if np.any(m):
                vals.append(a[m].astype(np.float64))

        if not vals:
            return np.array([], dtype=np.float64)

        x = np.concatenate(vals)
        x = x[np.isfinite(x)]

        if x.size <= max_n:
            return x

        step = max(1, x.size // max_n)
        return x[::step][:max_n]


def compute_ntl_quantiles(out_dir):
    print("[7/10] Computing pooled 2001-2020 NTL quantiles ...")
    qpath = out_dir / "baselines" / "NTL_pooled_quantiles.json"

    if qpath.exists():
        with open(qpath, "r", encoding="utf-8") as f:
            return json.load(f)

    vals = []

    for year in YEARS:
        path = out_dir / "intermediate" / "NTL" / f"NTL_{year}.tif"
        x = regular_sample(path, NTL_SAMPLE_PER_YEAR)
        x = x[x >= 0]
        vals.append(np.log1p(x))

    x = np.concatenate(vals)

    result = {
        "log1p_q01": float(np.percentile(x, NTL_Q_LOW)),
        "log1p_q99": float(np.percentile(x, NTL_Q_HIGH)),
    }

    write_json(result, qpath)
    return result


# =============================================================================
# 11. BUILD FIVE INDICATORS
# =============================================================================

def build_indicators(out_dir, grid, ntl_q):
    print("[8/10] Building VCI / LCN / HCI / TSI / MSI ...")

    idir = ensure_dir(out_dir / "indicators")
    bdir = out_dir / "baselines"

    base = {
        "ndvi_lo": rasterio.open(bdir / "NDVI_Q05_2001_2020.tif"),
        "ndvi_hi": rasterio.open(bdir / "NDVI_Q95_2001_2020.tif"),
        "tmed": rasterio.open(bdir / "T2M_median_2001_2020.tif"),
        "tmad": rasterio.open(bdir / "T2M_MAD_2001_2020.tif"),
        "smmed": rasterio.open(bdir / "SM1_median_2001_2020.tif"),
        "smmad": rasterio.open(bdir / "SM1_MAD_2001_2020.tif"),
    }

    try:
        for year in YEARS:
            outs = {k: idir / f"{k}_{year}.tif" for k in ["VCI", "LCN", "HCI", "TSI", "MSI"]}

            if all(p.exists() for p in outs.values()):
                continue

            raw = {
                "ndvi": rasterio.open(out_dir / "intermediate" / "NDVI" / f"NDVI_{year}.tif"),
                "lcn": rasterio.open(out_dir / "intermediate" / "LCN" / f"LCN_{year}.tif"),
                "ntl": rasterio.open(out_dir / "intermediate" / "NTL" / f"NTL_{year}.tif"),
                "t2m": rasterio.open(out_dir / "intermediate" / "T2M" / f"T2M_{year}.tif"),
                "sm1": rasterio.open(out_dir / "intermediate" / "SM1" / f"SM1_{year}.tif"),
            }

            dst = {k: rasterio.open(p, "w", **output_profile(grid)) for k, p in outs.items()}

            try:
                for win in tqdm(iter_windows(grid["width"], grid["height"]), desc=f"Indicators {year}", leave=False):
                    ndvi = raw["ndvi"].read(1, window=win)
                    lcn = raw["lcn"].read(1, window=win)
                    ntl = raw["ntl"].read(1, window=win)
                    t2m = raw["t2m"].read(1, window=win)
                    sm1 = raw["sm1"].read(1, window=win)

                    nlo = base["ndvi_lo"].read(1, window=win)
                    nhi = base["ndvi_hi"].read(1, window=win)
                    tmed = base["tmed"].read(1, window=win)
                    tmad = base["tmad"].read(1, window=win)
                    smmed = base["smmed"].read(1, window=win)
                    smmad = base["smmad"].read(1, window=win)

                    m = (
                        valid(ndvi) & valid(lcn) & valid(ntl) &
                        valid(t2m) & valid(sm1) &
                        valid(nlo) & valid(nhi) &
                        valid(tmed) & valid(tmad) &
                        valid(smmed) & valid(smmad)
                    )

                    O = {k: np.full(ndvi.shape, OUT_NODATA, dtype=np.float32) for k in outs}

                    # VCI
                    den = nhi - nlo
                    mv = m & (den > EPS)
                    O["VCI"][mv] = np.clip((ndvi[mv] - nlo[mv]) / den[mv], 0.0, 1.0)

                    # Nearly invariant NDVI time series -> neutral score.
                    mc = m & (den <= EPS)
                    O["VCI"][mc] = 0.5

                    # LCN already [0,1]
                    O["LCN"][m] = np.clip(lcn[m], 0.0, 1.0)

                    # HCI
                    q01 = ntl_q["log1p_q01"]
                    q99 = ntl_q["log1p_q99"]

                    z = np.log1p(np.maximum(ntl[m], 0.0))
                    z = np.clip((z - q01) / max(q99 - q01, EPS), 0.0, 1.0)
                    O["HCI"][m] = 1.0 - z

                    # TSI: robust absolute anomaly relative to own long-term climate
                    at = np.abs(t2m[m] - tmed[m]) / (MAD_SCALE * tmad[m] + EPS)
                    O["TSI"][m] = 1.0 / (1.0 + at)

                    # MSI: robust absolute anomaly relative to own long-term moisture state
                    asm = np.abs(sm1[m] - smmed[m]) / (MAD_SCALE * smmad[m] + EPS)
                    O["MSI"][m] = 1.0 / (1.0 + asm)

                    # Common-valid mask
                    cm = np.ones(ndvi.shape, dtype=bool)
                    for k in O:
                        cm &= valid(O[k])
                        cm &= np.isfinite(O[k])
                        cm &= (O[k] >= 0.0) & (O[k] <= 1.0)

                    for k in O:
                        O[k][~cm] = OUT_NODATA
                        dst[k].write(O[k], 1, window=win)

            finally:
                for s in raw.values():
                    s.close()
                for s in dst.values():
                    s.close()

    finally:
        for s in base.values():
            s.close()


# =============================================================================
# 12. FIXED POOLED ENTROPY WEIGHTS
# =============================================================================

def compute_entropy_weights(out_dir):
    print("[9/10] Computing fixed pooled entropy weights ...")

    names = ["VCI", "LCN", "HCI", "TSI", "MSI"]
    idir = out_dir / "indicators"

    sum_x = {k: 0.0 for k in names}
    sum_xlnx = {k: 0.0 for k in names}
    n = {k: 0 for k in names}

    for year in YEARS:
        src = {k: rasterio.open(idir / f"{k}_{year}.tif") for k in names}

        try:
            for win in iter_windows(src[names[0]].width, src[names[0]].height):
                a = {k: src[k].read(1, window=win) for k in names}

                cm = np.ones(a[names[0]].shape, dtype=bool)
                for k in names:
                    cm &= valid(a[k])

                if not np.any(cm):
                    continue

                for k in names:
                    x = np.clip(a[k][cm].astype(np.float64), 0.0, 1.0)
                    sum_x[k] += float(np.sum(x))

                    pos = x > 0
                    if np.any(pos):
                        sum_xlnx[k] += float(np.sum(x[pos] * np.log(x[pos])))

                    n[k] += x.size

        finally:
            for s in src.values():
                s.close()

    entropy = {}
    divergence = {}

    for k in names:
        S = sum_x[k]
        N = n[k]

        if S <= 0 or N <= 1:
            e = 1.0
        else:
            # p_i = x_i / S
            # Σ p_i ln(p_i) = Σ(x_i ln x_i)/S - ln(S)
            splp = sum_xlnx[k] / S - math.log(S)
            e = -splp / math.log(N)
            e = float(np.clip(e, 0.0, 1.0))

        entropy[k] = e
        divergence[k] = max(0.0, 1.0 - e)

    dsum = sum(divergence.values())

    if dsum <= 0:
        weights = {k: 1.0 / len(names) for k in names}
    else:
        weights = {k: divergence[k] / dsum for k in names}

    table = pd.DataFrame({
        "indicator": names,
        "entropy": [entropy[k] for k in names],
        "divergence": [divergence[k] for k in names],
        "weight": [weights[k] for k in names],
        "n_valid": [n[k] for k in names],
    })

    table.to_csv(out_dir / "entropy_weights_fixed_2001_2020.csv", index=False)

    print("\nFixed pooled entropy weights:")
    print(table.to_string(index=False))

    return weights


# =============================================================================
# 13. ERI + QA
# =============================================================================

def update_stats(state, x):
    x = x[np.isfinite(x)]

    if x.size == 0:
        return

    n2 = x.size
    mean2 = float(np.mean(x))
    M2_2 = float(np.sum((x - mean2) ** 2))

    if state["n"] == 0:
        state["n"] = n2
        state["mean"] = mean2
        state["M2"] = M2_2
    else:
        n1 = state["n"]
        m1 = state["mean"]
        delta = mean2 - m1
        nt = n1 + n2

        state["mean"] = m1 + delta * n2 / nt
        state["M2"] = state["M2"] + M2_2 + delta * delta * n1 * n2 / nt
        state["n"] = nt

    state["min"] = min(state["min"], float(np.min(x)))
    state["max"] = max(state["max"], float(np.max(x)))


def compute_eri(out_dir, grid, weights):
    print("[10/10] Computing ERI and QA ...")

    names = ["VCI", "LCN", "HCI", "TSI", "MSI"]
    idir = out_dir / "indicators"
    edir = ensure_dir(out_dir / "ERI")

    rows = []

    for year in YEARS:
        src = {k: rasterio.open(idir / f"{k}_{year}.tif") for k in names}

        state = {
            "n": 0,
            "mean": 0.0,
            "M2": 0.0,
            "min": np.inf,
            "max": -np.inf,
        }

        samples = []

        try:
            with rasterio.open(edir / f"ERI_{year}.tif", "w", **output_profile(grid)) as dst:
                for win in tqdm(iter_windows(grid["width"], grid["height"]), desc=f"ERI {year}", leave=False):
                    a = {k: src[k].read(1, window=win) for k in names}

                    cm = np.ones(a[names[0]].shape, dtype=bool)
                    for k in names:
                        cm &= valid(a[k])

                    out = np.full(a[names[0]].shape, OUT_NODATA, dtype=np.float32)

                    if np.any(cm):
                        r = np.zeros(np.count_nonzero(cm), dtype=np.float64)

                        for k in names:
                            r += weights[k] * a[k][cm]

                        # Only floating point protection; values should already be [0,1].
                        r = np.clip(r, 0.0, 1.0)

                        out[cm] = r.astype(np.float32)

                        update_stats(state, r)

                        step = max(1, r.size // 2000)
                        samples.append(r[::step][:2000])

                    dst.write(out, 1, window=win)

        finally:
            for s in src.values():
                s.close()

        if state["n"] > 1:
            std = math.sqrt(state["M2"] / (state["n"] - 1))
        else:
            std = np.nan

        if samples:
            q = np.percentile(np.concatenate(samples), [1, 50, 99])
            p01, median, p99 = map(float, q)
        else:
            p01 = median = p99 = np.nan

        rows.append({
            "year": year,
            "valid_pixels": state["n"],
            "min": state["min"],
            "p01_sample": p01,
            "mean": state["mean"],
            "median_sample": median,
            "p99_sample": p99,
            "max": state["max"],
            "std": std,
            "range_ok": bool(state["min"] >= -1e-6 and state["max"] <= 1.0 + 1e-6),
        })

    qa = pd.DataFrame(rows)
    qa.to_csv(out_dir / "ERI_QA_2001_2020.csv", index=False)

    print("\nERI QA table:")
    print(qa.to_string(index=False))


# =============================================================================
# 14. INPUT VALIDATION
# =============================================================================

def validate_inputs():
    print("\nResolved input paths:")
    print(f"  ROOT           = {ROOT}")
    print(f"  Boundary       = {CHINA_BOUNDARY}   exists={CHINA_BOUNDARY.exists()}")
    print(f"  CLCD_DIR       = {CLCD_DIR}   exists={CLCD_DIR.exists()}")
    print(f"  NDVI_DIR       = {NDVI_DIR}   exists={NDVI_DIR.exists()}")
    print(f"  NightLight_DIR = {NTL_DIR}   exists={NTL_DIR.exists()}")
    print(f"  T2M_FILE       = {T2M_FILE}   exists={T2M_FILE.exists()}")
    print(f"  SM1_FILE       = {SM1_FILE}   exists={SM1_FILE.exists()}")
    required = [
        CHINA_BOUNDARY,
        CLCD_DIR,
        NDVI_DIR,
        NTL_DIR,
        T2M_FILE,
        SM1_FILE,
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "These inputs are missing. Edit the CONFIG paths first:\n" +
            "\n".join(str(p) for p in missing)
        )

    for y in YEARS:
        clcd_path(y)
        ndvi_path(y)
        ntl_path(y)

    inspect_multiband_file(T2M_FILE)
    inspect_multiband_file(SM1_FILE)


# =============================================================================
# 15. MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("China ERI recalculation, 2001-2020")
    print("=" * 78)

    validate_inputs()
    out_dir = ensure_dir(OUT_DIR)

    grid, inside = build_target_grid()

    write_json(
        {
            "years": YEARS,
            "target_crs": str(grid["crs"]),
            "resolution_m": RESOLUTION_M,
            "width": grid["width"],
            "height": grid["height"],
            "bounds": grid["bounds"],
            "transform": tuple(grid["transform"]),
            "clcd_definition": "LCN = 1 - (cropland + impervious) / non-water-snow valid land fraction",
            "vci_definition": "pixel-relative NDVI position between 2001-2020 temporal Q05 and Q95",
            "hci_definition": "1 - robust pooled normalized log1p(NTL)",
            "tsi_definition": "1 / (1 + abs(T-median)/(1.4826*MAD+eps))",
            "msi_definition": "1 / (1 + abs(SM1-median)/(1.4826*MAD+eps))",
            "weighting": "single pooled entropy-weight vector across all valid pixel-years, 2001-2020",
        },
        out_dir / "method_config.json"
    )

    # Raw preprocessing
    preprocess_ndvi(grid, inside, out_dir)
    preprocess_clcd(grid, inside, out_dir)
    preprocess_ntl(grid, inside, out_dir)

    print("[5/10] Reading annual bands from 26-band temperature and soil-water rasters ...")
    preprocess_multiband_annual(T2M_FILE, "T2M", grid, inside, out_dir)
    preprocess_multiband_annual(SM1_FILE, "SM1", grid, inside, out_dir)

    # Long-term baseline
    compute_baselines(out_dir, grid)

    # NTL pooled robust scale
    ntl_q = compute_ntl_quantiles(out_dir)

    # Five indicators
    build_indicators(out_dir, grid, ntl_q)

    # Fixed entropy weights
    weights = compute_entropy_weights(out_dir)

    # Final ERI
    compute_eri(out_dir, grid, weights)

    print("\nFinished.")
    print("Output:", out_dir)
    print("\nMain files:")
    print("  ERI\\ERI_2001.tif ... ERI\\ERI_2020.tif")
    print("  entropy_weights_fixed_2001_2020.csv")
    print("  ERI_QA_2001_2020.csv")
    print("  indicators\\")
    print("  baselines\\")
    print("  intermediate\\")
    print("\nIMPORTANT:")
    print("Before manuscript use, inspect ERI_QA_2001_2020.csv.")
    print("Every year should have range_ok=True and ERI min/max within [0,1].")


if __name__ == "__main__":
    main()
