# -*- coding: utf-8 -*-
r"""
ERPI 2001-2020 | Final-raster technical audit
==============================================

This script audits the FINAL annual ERPI rasters and, where available, verifies
them against the archived component rasters and fixed pooled weights.

Default project root
--------------------
F:\data\China\ERI_recalculated_2001_2020

Expected key files
------------------
ERI\ERI_2001.tif ... ERI\ERI_2020.tif
indicators\VCI_2001.tif ... MSI_2020.tif
entropy_weights_fixed_2001_2020.csv

The script does NOT modify any existing ERPI raster.

Main audit dimensions
---------------------
1. File completeness and readability
2. CRS / transform / resolution / extent / shape consistency
3. GeoTIFF encoding, NoData, NaN/Inf and [0,1] range checks
4. Exact valid-pixel counts and common/union spatial support
5. Year-to-year support gains/losses and Jaccard overlap
6. Approximate full-raster quantiles from a high-resolution histogram
7. SHA256 hashes and accidental exact-duplicate detection
8. Reconstruction of final ERPI from VCI/LCN/HCI/TSI/MSI + fixed weights
9. Mask-consistency checks between final ERPI and component rasters
10. Fixed-support annual statistics and year-to-year change diagnostics
11. Row/column stripe diagnostics (diagnostic only, not an automatic failure)
12. Machine-readable CSVs + a human-readable Markdown audit report

IMPORTANT LIMITATION
--------------------
A final-raster audit cannot prove that hidden fill values in the ORIGINAL NDVI
or nighttime-light inputs were correctly excluded before resampling, and it
cannot by itself prove native-resolution CLCD source coverage. Those require a
separate source-input audit. This script explicitly reports that limitation.

Dependencies
------------
numpy, pandas, rasterio, matplotlib

Run
---
D:\conda\python.exe C:\Users\Administrator\Downloads\audit_ERPI_final_rasters_2001_2020.py

Optional:
D:\conda\python.exe audit_ERPI_final_rasters_2001_2020.py --root "F:\data\China\ERI_recalculated_2001_2020"
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEARS = list(range(2001, 2021))
INDICATORS = ("VCI", "LCN", "HCI", "TSI", "MSI")

DEFAULT_ROOT = Path(os.environ.get("ERPI_PRODUCT_ROOT", str(Path(os.environ.get("ERPI_DATA_ROOT", "./data")) / "ERI_recalculated_2001_2020_NTLfix_candidate_v1"))).expanduser().resolve()
OUTDIR_NAME = "FINAL_ERPI_TECHNICAL_AUDIT"

# Numeric tolerances are deliberately strict for a weighted sum of float32 rasters.
RANGE_TOL = 1e-6
RECON_WARN = 1e-6
RECON_FAIL = 2e-5
MASK_MISMATCH_FAIL = 0

# Histogram for approximate quantiles of [0,1] ERPI.
NBINS = 10000

# A coverage shift larger than this fraction of 2001 valid pixels is highlighted.
# It is a REVIEW flag, not an automatic data failure.
COVERAGE_REVIEW_FRAC = 0.005

# Unexpectedly large annual national-mean jump: robust-z threshold.
TEMPORAL_REVIEW_Z = 3.5


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_windows(width: int, height: int, block: int = 1024):
    for r0 in range(0, height, block):
        h = min(block, height - r0)
        for c0 in range(0, width, block):
            w = min(block, width - c0)
            yield Window(c0, r0, w, h)


def valid_mask(arr: np.ndarray, nodata):
    m = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        m &= arr != nodata
    return m


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hist_quantile(hist: np.ndarray, q: float):
    total = int(hist.sum())
    if total == 0:
        return np.nan
    target = q * (total - 1)
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, target + 1, side="left"))
    idx = min(max(idx, 0), len(hist)-1)
    # midpoint of the selected bin
    return (idx + 0.5) / len(hist)


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.full_like(x, np.nan, dtype=float)
    return 0.67448975 * (x - med) / mad


@dataclass
class YearStats:
    year: int
    valid_pixels: int
    nodata_pixels: int
    nan_pixels: int
    posinf_pixels: int
    neginf_pixels: int
    below_zero_pixels: int
    above_one_pixels: int
    min: float
    q01_approx: float
    q05_approx: float
    mean: float
    median_approx: float
    q95_approx: float
    q99_approx: float
    max: float
    std: float
    file_bytes: int
    sha256: str


def load_weights(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    # Common form from the ERPI script: indicator, entropy, divergence, weight, n_valid
    if {"indicator", "weight"}.issubset(df.columns):
        d = dict(zip(df["indicator"].astype(str), df["weight"].astype(float)))
    else:
        # fallback: one row, columns named indicators
        d = {}
        for ind in INDICATORS:
            if ind in df.columns:
                d[ind] = float(df[ind].iloc[0])

    missing = [k for k in INDICATORS if k not in d]
    if missing:
        raise ValueError(f"Could not resolve weights for: {missing}")

    s = sum(d.values())
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(f"Fixed weights sum to {s}, not 1")

    return d, df


def structural_audit(eri_paths, out_dir):
    rows = []
    ref_sig = None
    critical = []
    review = []

    for y, p in eri_paths.items():
        if not p.exists():
            critical.append(f"Missing ERPI raster: {p}")
            continue

        try:
            with rasterio.open(p) as src:
                sig = (
                    src.driver,
                    src.count,
                    str(src.crs),
                    tuple(src.transform),
                    src.width,
                    src.height,
                    tuple(src.bounds),
                    tuple(src.res),
                )
                if ref_sig is None:
                    ref_sig = sig
                elif sig[2:] != ref_sig[2:]:
                    critical.append(f"Grid mismatch in ERI_{y}.tif")

                rows.append({
                    "year": y,
                    "path": str(p),
                    "driver": src.driver,
                    "bands": src.count,
                    "dtype": src.dtypes[0] if src.count else None,
                    "nodata": src.nodata,
                    "crs": str(src.crs),
                    "width": src.width,
                    "height": src.height,
                    "res_x": src.res[0],
                    "res_y": src.res[1],
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                    "tiled": src.is_tiled,
                    "block_shapes": str(src.block_shapes),
                    "compression": str(src.compression),
                    "overview_levels": str(src.overviews(1)) if src.count == 1 else "",
                    "file_bytes": p.stat().st_size,
                })

                if src.count != 1:
                    critical.append(f"ERI_{y}.tif has {src.count} bands; expected 1")
                if src.crs is None:
                    critical.append(f"ERI_{y}.tif has no CRS")
                if src.width <= 0 or src.height <= 0:
                    critical.append(f"ERI_{y}.tif has invalid dimensions")
                if src.nodata is None:
                    review.append(f"ERI_{y}.tif has no explicit NoData metadata")
        except Exception as e:
            critical.append(f"Unreadable raster {p}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "01_ERPI_file_structure_audit.csv", index=False)
    return df, critical, review


def scan_year_stats(eri_paths, out_dir):
    rows = []
    valid_full = {}
    row_mean_profiles = {}
    col_mean_profiles = {}

    for y in YEARS:
        p = eri_paths[y]
        with rasterio.open(p) as src:
            hist = np.zeros(NBINS, dtype=np.int64)
            n = 0
            s = 0.0
            ss = 0.0
            mn = np.inf
            mx = -np.inf
            nodata_count = 0
            nan_count = 0
            posinf_count = 0
            neginf_count = 0
            below = 0
            above = 0

            # row/column diagnostic accumulators
            row_s = np.zeros(src.height, dtype=np.float64)
            row_n = np.zeros(src.height, dtype=np.int64)
            col_s = np.zeros(src.width, dtype=np.float64)
            col_n = np.zeros(src.width, dtype=np.int64)

            full_valid = np.zeros((src.height, src.width), dtype=bool)

            for win in iter_windows(src.width, src.height):
                a = src.read(1, window=win).astype(np.float64)

                nan_count += int(np.count_nonzero(np.isnan(a)))
                posinf_count += int(np.count_nonzero(np.isposinf(a)))
                neginf_count += int(np.count_nonzero(np.isneginf(a)))

                if src.nodata is not None and np.isfinite(src.nodata):
                    nodata_here = a == src.nodata
                    nodata_count += int(nodata_here.sum())
                else:
                    nodata_here = np.zeros_like(a, dtype=bool)

                m = np.isfinite(a) & ~nodata_here
                rr0 = int(win.row_off)
                cc0 = int(win.col_off)
                full_valid[rr0:rr0+a.shape[0], cc0:cc0+a.shape[1]] = m

                if not np.any(m):
                    continue

                x = a[m]
                n += x.size
                s += float(x.sum())
                ss += float(np.dot(x, x))
                mn = min(mn, float(x.min()))
                mx = max(mx, float(x.max()))
                below += int(np.count_nonzero(x < -RANGE_TOL))
                above += int(np.count_nonzero(x > 1.0 + RANGE_TOL))

                in01 = (x >= 0) & (x <= 1)
                if np.any(in01):
                    idx = np.floor(np.clip(x[in01], 0, 1 - np.finfo(float).eps) * NBINS).astype(np.int64)
                    idx = np.clip(idx, 0, NBINS-1)
                    hist += np.bincount(idx, minlength=NBINS)

                # row/column means
                b = np.where(m, a, 0.0)
                row_s[rr0:rr0+a.shape[0]] += b.sum(axis=1)
                row_n[rr0:rr0+a.shape[0]] += m.sum(axis=1)
                col_s[cc0:cc0+a.shape[1]] += b.sum(axis=0)
                col_n[cc0:cc0+a.shape[1]] += m.sum(axis=0)

            mean = s / n if n else np.nan
            var = ss / n - mean * mean if n else np.nan
            std = math.sqrt(max(var, 0.0)) if np.isfinite(var) else np.nan

            rows.append(asdict(YearStats(
                year=y,
                valid_pixels=n,
                nodata_pixels=nodata_count,
                nan_pixels=nan_count,
                posinf_pixels=posinf_count,
                neginf_pixels=neginf_count,
                below_zero_pixels=below,
                above_one_pixels=above,
                min=mn if n else np.nan,
                q01_approx=hist_quantile(hist, 0.01),
                q05_approx=hist_quantile(hist, 0.05),
                mean=mean,
                median_approx=hist_quantile(hist, 0.50),
                q95_approx=hist_quantile(hist, 0.95),
                q99_approx=hist_quantile(hist, 0.99),
                max=mx if n else np.nan,
                std=std,
                file_bytes=p.stat().st_size,
                sha256=sha256_file(p),
            )))

            valid_full[y] = full_valid
            row_mean_profiles[y] = np.divide(row_s, row_n, out=np.full_like(row_s, np.nan), where=row_n>0)
            col_mean_profiles[y] = np.divide(col_s, col_n, out=np.full_like(col_s, np.nan), where=col_n>0)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "02_ERPI_year_fullscan_statistics.csv", index=False)
    return df, valid_full, row_mean_profiles, col_mean_profiles


def support_audit(valid_full, ref_path, out_dir):
    common = None
    union = None
    for y in YEARS:
        m = valid_full[y]
        common = m.copy() if common is None else (common & m)
        union = m.copy() if union is None else (union | m)

    trans_rows = []
    baseline_n = int(valid_full[YEARS[0]].sum())
    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        a = valid_full[y0]
        b = valid_full[y1]
        inter = int((a & b).sum())
        uni = int((a | b).sum())
        lost = int((a & ~b).sum())
        gained = int((~a & b).sum())
        trans_rows.append({
            "year_from": y0,
            "year_to": y1,
            "valid_from": int(a.sum()),
            "valid_to": int(b.sum()),
            "lost_pixels": lost,
            "gained_pixels": gained,
            "net_change": int(b.sum()) - int(a.sum()),
            "jaccard": inter / uni if uni else np.nan,
            "lost_frac_of_2001": lost / baseline_n if baseline_n else np.nan,
            "gained_frac_of_2001": gained / baseline_n if baseline_n else np.nan,
        })

    trans = pd.DataFrame(trans_rows)
    trans.to_csv(out_dir / "03_ERPI_coverage_transitions.csv", index=False)

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
        with rasterio.open(out_dir / "common_valid_mask_AUDITED_2001_2020.tif", "w", **profile) as dst:
            dst.write(common.astype(np.uint8), 1)
        with rasterio.open(out_dir / "union_valid_mask_AUDITED_2001_2020.tif", "w", **profile) as dst:
            dst.write(union.astype(np.uint8), 1)

    support_summary = pd.DataFrame([{
        "common_valid_pixels": int(common.sum()),
        "union_valid_pixels": int(union.sum()),
        "common_fraction_of_union": float(common.sum()/union.sum()) if union.sum() else np.nan,
        "valid_2001": int(valid_full[2001].sum()),
        "valid_2020": int(valid_full[2020].sum()),
        "net_valid_change_2001_2020": int(valid_full[2020].sum()-valid_full[2001].sum()),
        "net_valid_change_fraction_2001": float(
            (valid_full[2020].sum()-valid_full[2001].sum()) / valid_full[2001].sum()
        ),
    }])
    support_summary.to_csv(out_dir / "04_ERPI_support_summary.csv", index=False)

    return common, union, trans, support_summary


def exact_duplicate_audit(stats_df, out_dir):
    hash_groups = stats_df.groupby("sha256")["year"].apply(list)
    duplicate_groups = [yrs for yrs in hash_groups if len(yrs) > 1]
    rows = []
    for yrs in duplicate_groups:
        rows.append({"years_identical_file_hash": ",".join(map(str, yrs)), "n_years": len(yrs)})
    df = pd.DataFrame(rows, columns=["years_identical_file_hash", "n_years"])
    df.to_csv(out_dir / "05_ERPI_exact_duplicate_hashes.csv", index=False)
    return df


def reconstruction_audit(root, eri_paths, weights, out_dir):
    rows = []
    unavailable = []
    all_components_available = True

    for y in YEARS:
        for ind in INDICATORS:
            p = root / "indicators" / f"{ind}_{y}.tif"
            if not p.exists():
                unavailable.append(str(p))
                all_components_available = False

    if not all_components_available:
        pd.DataFrame({"missing_component_file": unavailable}).to_csv(
            out_dir / "06_RECONSTRUCTION_NOT_RUN_missing_components.csv", index=False
        )
        return None, unavailable

    for y in YEARS:
        eri_p = eri_paths[y]
        ind_paths = {ind: root / "indicators" / f"{ind}_{y}.tif" for ind in INDICATORS}

        with rasterio.open(eri_p) as eri, \
             rasterio.open(ind_paths["VCI"]) as vci, \
             rasterio.open(ind_paths["LCN"]) as lcn, \
             rasterio.open(ind_paths["HCI"]) as hci, \
             rasterio.open(ind_paths["TSI"]) as tsi, \
             rasterio.open(ind_paths["MSI"]) as msi:

            srcs = dict(VCI=vci, LCN=lcn, HCI=hci, TSI=tsi, MSI=msi)

            # Exact grid check first
            for ind, src in srcs.items():
                if (
                    src.crs != eri.crs or
                    src.transform != eri.transform or
                    src.width != eri.width or
                    src.height != eri.height
                ):
                    raise ValueError(f"{ind}_{y}.tif grid differs from ERI_{y}.tif")

            n_comp_valid = 0
            n_eri_valid = 0
            mask_mismatch = 0
            n_compare = 0
            abs_sum = 0.0
            sq_sum = 0.0
            max_abs = 0.0
            gt1e6 = 0
            gt1e5 = 0
            sample_err = []

            for win in iter_windows(eri.width, eri.height):
                final = eri.read(1, window=win).astype(np.float64)
                m_final = valid_mask(final, eri.nodata)

                component_arrays = {}
                m_all = np.ones(final.shape, dtype=bool)
                for ind, src in srcs.items():
                    a = src.read(1, window=win).astype(np.float64)
                    m = valid_mask(a, src.nodata)
                    component_arrays[ind] = a
                    m_all &= m

                n_comp_valid += int(m_all.sum())
                n_eri_valid += int(m_final.sum())
                mask_mismatch += int(np.count_nonzero(m_all ^ m_final))

                m = m_all & m_final
                if not np.any(m):
                    continue

                rec = np.zeros(final.shape, dtype=np.float64)
                for ind in INDICATORS:
                    rec += weights[ind] * component_arrays[ind]

                err = np.abs(final[m] - rec[m])
                n_compare += err.size
                abs_sum += float(err.sum())
                sq_sum += float(np.dot(err, err))
                max_abs = max(max_abs, float(err.max()))
                gt1e6 += int(np.count_nonzero(err > 1e-6))
                gt1e5 += int(np.count_nonzero(err > 1e-5))

                # deterministic compact sample for empirical percentiles
                take = min(3000, err.size)
                if take:
                    step = max(1, err.size // take)
                    sample_err.append(err[::step][:take])

            sample = np.concatenate(sample_err) if sample_err else np.array([])
            rows.append({
                "year": y,
                "component_valid_pixels": n_comp_valid,
                "ERI_valid_pixels": n_eri_valid,
                "mask_mismatch_pixels": mask_mismatch,
                "compared_pixels": n_compare,
                "mean_abs_error": abs_sum / n_compare if n_compare else np.nan,
                "rmse": math.sqrt(sq_sum / n_compare) if n_compare else np.nan,
                "p99_abs_error_sample": float(np.quantile(sample, .99)) if sample.size else np.nan,
                "max_abs_error": max_abs,
                "pixels_abs_error_gt_1e-6": gt1e6,
                "pixels_abs_error_gt_1e-5": gt1e5,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "06_ERPI_reconstruction_from_components.csv", index=False)
    return df, []


def common_mask_temporal_audit(eri_paths, common, out_dir):
    rows = []

    for y in YEARS:
        with rasterio.open(eri_paths[y]) as src:
            s = 0.0
            ss = 0.0
            n = 0
            hist = np.zeros(NBINS, dtype=np.int64)

            for win in iter_windows(src.width, src.height):
                r0, c0 = int(win.row_off), int(win.col_off)
                cm = common[r0:r0+int(win.height), c0:c0+int(win.width)]
                if not np.any(cm):
                    continue
                a = src.read(1, window=win).astype(np.float64)
                m = cm & valid_mask(a, src.nodata)
                x = a[m]
                if x.size == 0:
                    continue
                n += x.size
                s += float(x.sum())
                ss += float(np.dot(x, x))
                idx = np.floor(np.clip(x, 0, 1 - np.finfo(float).eps) * NBINS).astype(np.int64)
                hist += np.bincount(np.clip(idx, 0, NBINS-1), minlength=NBINS)

            mean = s/n if n else np.nan
            var = ss/n - mean*mean if n else np.nan
            rows.append({
                "year": y,
                "common_valid_pixels_used": n,
                "common_mean": mean,
                "common_median_approx": hist_quantile(hist, .5),
                "common_q05_approx": hist_quantile(hist, .05),
                "common_q95_approx": hist_quantile(hist, .95),
                "common_std": math.sqrt(max(var,0)) if np.isfinite(var) else np.nan,
            })

    df = pd.DataFrame(rows)
    df["delta_mean_from_previous_year"] = df["common_mean"].diff()
    z = robust_z(df["delta_mean_from_previous_year"].to_numpy()[1:])
    df["delta_mean_robust_z"] = np.nan
    df.loc[1:, "delta_mean_robust_z"] = z
    df["temporal_jump_review_flag"] = (
        np.abs(df["delta_mean_robust_z"].fillna(0)) > TEMPORAL_REVIEW_Z
    )
    df.to_csv(out_dir / "07_ERPI_common_support_temporal_diagnostics.csv", index=False)

    # Pixelwise consecutive-year changes on common support
    diff_rows = []
    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        with rasterio.open(eri_paths[y0]) as s0, rasterio.open(eri_paths[y1]) as s1:
            n=0; sm=0.; ss=0.; abs_s=0.; gt02=0; gt04=0; mn=np.inf; mx=-np.inf
            for win in iter_windows(s0.width, s0.height):
                r0,c0=int(win.row_off),int(win.col_off)
                cm=common[r0:r0+int(win.height), c0:c0+int(win.width)]
                if not np.any(cm):
                    continue
                a=s0.read(1,window=win).astype(float)
                b=s1.read(1,window=win).astype(float)
                m=cm & valid_mask(a,s0.nodata) & valid_mask(b,s1.nodata)
                d=b[m]-a[m]
                if d.size==0: continue
                n += d.size
                sm += float(d.sum())
                ss += float(np.dot(d,d))
                abs_s += float(np.abs(d).sum())
                gt02 += int(np.count_nonzero(np.abs(d) > .2))
                gt04 += int(np.count_nonzero(np.abs(d) > .4))
                mn=min(mn,float(d.min())); mx=max(mx,float(d.max()))
            mean=sm/n if n else np.nan
            var=ss/n - mean*mean if n else np.nan
            diff_rows.append({
                "year_from": y0, "year_to": y1, "n": n,
                "mean_pixel_change": mean,
                "mean_abs_pixel_change": abs_s/n if n else np.nan,
                "std_pixel_change": math.sqrt(max(var,0)) if np.isfinite(var) else np.nan,
                "min_pixel_change": mn if n else np.nan,
                "max_pixel_change": mx if n else np.nan,
                "fraction_abs_change_gt_0_2": gt02/n if n else np.nan,
                "fraction_abs_change_gt_0_4": gt04/n if n else np.nan,
            })

    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(out_dir / "08_ERPI_pixelwise_interannual_change.csv", index=False)
    return df, diffs


def stripe_diagnostic(row_profiles, col_profiles, out_dir):
    """
    Diagnostic only. Sharp row/column mean discontinuities can arise from real
    geography as well as processing seams. We report robust z-scores but never
    auto-fail a raster solely on this basis.
    """
    rows = []
    for y in YEARS:
        rp = row_profiles[y]
        cp = col_profiles[y]
        dr = np.diff(rp)
        dc = np.diff(cp)
        rz = robust_z(dr[np.isfinite(dr)])
        cz = robust_z(dc[np.isfinite(dc)])

        rows.append({
            "year": y,
            "max_abs_adjacent_row_mean_jump": float(np.nanmax(np.abs(dr))),
            "max_abs_adjacent_col_mean_jump": float(np.nanmax(np.abs(dc))),
            "max_abs_row_jump_robust_z": float(np.nanmax(np.abs(rz))) if np.any(np.isfinite(rz)) else np.nan,
            "max_abs_col_jump_robust_z": float(np.nanmax(np.abs(cz))) if np.any(np.isfinite(cz)) else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "09_ERPI_row_column_seam_diagnostics.csv", index=False)
    return df


def build_flags(struct_df, stats_df, trans_df, dup_df, recon_df, temporal_df, out_dir):
    flags = []

    def add(severity, code, message):
        flags.append({"severity": severity, "code": code, "message": message})

    # Final range / finite checks
    bad_range = stats_df[(stats_df["below_zero_pixels"] > 0) | (stats_df["above_one_pixels"] > 0)]
    if len(bad_range):
        add("FAIL", "ERPI_RANGE", f"{len(bad_range)} annual rasters contain valid values outside [0,1] beyond tolerance.")

    bad_inf = stats_df[(stats_df["posinf_pixels"] > 0) | (stats_df["neginf_pixels"] > 0)]
    if len(bad_inf):
        add("FAIL", "INF_VALUES", f"{len(bad_inf)} annual rasters contain ±Inf values.")

    # Coverage
    base = float(stats_df.loc[stats_df.year==2001, "valid_pixels"].iloc[0])
    max_abs_frac = float(np.max(np.abs(stats_df["valid_pixels"] - base) / base))
    if max_abs_frac > COVERAGE_REVIEW_FRAC:
        add("REVIEW", "COVERAGE_CHANGE",
            f"Maximum annual valid-pixel deviation from 2001 is {max_abs_frac:.3%}; inspect spatial support changes.")

    # Duplicates
    if dup_df is not None and len(dup_df):
        add("REVIEW", "IDENTICAL_YEAR_FILES",
            "At least two annual GeoTIFF files have identical whole-file SHA256 hashes; confirm this is intentional.")

    # Reconstruction
    if recon_df is not None:
        mismatch = int(recon_df["mask_mismatch_pixels"].sum())
        mx = float(recon_df["max_abs_error"].max())
        if mismatch > MASK_MISMATCH_FAIL:
            add("FAIL", "COMPONENT_MASK_MISMATCH",
                f"Final ERPI/component validity masks disagree at {mismatch:,} pixel-years.")
        if mx > RECON_FAIL:
            add("FAIL", "RECONSTRUCTION_ERROR",
                f"Maximum ERPI reconstruction error is {mx:.3g}, above fail threshold {RECON_FAIL}.")
        elif mx > RECON_WARN:
            add("REVIEW", "RECONSTRUCTION_ROUNDING",
                f"Maximum ERPI reconstruction error is {mx:.3g}; likely rounding/encoding, inspect before release.")
    else:
        add("REVIEW", "RECONSTRUCTION_NOT_RUN",
            "Component rasters were not all available, so final ERPI could not be independently reconstructed.")

    # Temporal
    flagged = temporal_df[temporal_df["temporal_jump_review_flag"]]
    if len(flagged):
        yrs = ",".join(map(str, flagged["year"].tolist()))
        add("REVIEW", "TEMPORAL_JUMP", f"Robust national-mean jump diagnostic flags year(s): {yrs}.")

    # Structural metadata notes
    if not struct_df.empty:
        if any(struct_df["nodata"].isna()):
            add("REVIEW", "NODATA_METADATA", "At least one annual raster lacks explicit NoData metadata.")

    # Always add final-raster limitation
    add("INFO", "SOURCE_AUDIT_LIMITATION",
        "Final-raster checks cannot prove pre-resampling exclusion of hidden source fill values or native-resolution CLCD coverage.")

    flags_df = pd.DataFrame(flags)
    flags_df.to_csv(out_dir / "10_ERPI_audit_flags.csv", index=False)
    return flags_df


def plot_audit(stats_df, trans_df, recon_df, temporal_df, out_dir):
    # Figure A: coverage and fixed-support mean
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    axs[0].plot(stats_df["year"], stats_df["valid_pixels"], marker="o")
    axs[0].set_title("Annual valid pixels")
    axs[0].set_xlabel("Year")
    axs[0].set_ylabel("Pixels")
    axs[0].tick_params(axis="x", rotation=45)

    axs[1].plot(temporal_df["year"], temporal_df["common_mean"], marker="o")
    axs[1].set_title("Common-support annual ERPI mean")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("ERPI")
    axs[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "audit_plot_coverage_and_mean.png", dpi=220)
    plt.close(fig)

    if recon_df is not None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.semilogy(recon_df["year"], np.maximum(recon_df["max_abs_error"], 1e-12), marker="o", label="max abs error")
        ax.semilogy(recon_df["year"], np.maximum(recon_df["mean_abs_error"], 1e-12), marker="s", label="mean abs error")
        ax.axhline(RECON_WARN, linestyle="--", linewidth=1, label="review threshold")
        ax.axhline(RECON_FAIL, linestyle=":", linewidth=1, label="fail threshold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Absolute reconstruction error")
        ax.set_title("Final ERPI reconstruction audit")
        ax.legend(frameon=False)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / "audit_plot_reconstruction_error.png", dpi=220)
        plt.close(fig)


def write_report(root, struct_df, stats_df, support_summary, trans_df, dup_df,
                 recon_df, temporal_df, diffs_df, stripe_df, flags_df, weights,
                 out_dir):

    sev = set(flags_df["severity"].tolist()) if not flags_df.empty else set()
    if "FAIL" in sev:
        status = "FAIL — do not archive/release until critical findings are resolved."
    elif "REVIEW" in sev:
        status = "REVIEW — no critical final-raster failure detected, but flagged items require inspection."
    else:
        status = "PASS — no critical or review-level finding in the final-raster audit."

    ss = support_summary.iloc[0]
    max_coverage_dev = float(np.max(np.abs(stats_df["valid_pixels"] - stats_df["valid_pixels"].iloc[0]) / stats_df["valid_pixels"].iloc[0]))

    lines = []
    lines.append("# ERPI 2001–2020 final-raster technical audit")
    lines.append("")
    lines.append(f"**Project root:** `{root}`")
    lines.append(f"**Audit status:** **{status}**")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append("This audit checks the final annual ERPI GeoTIFFs and, when present, independently reconstructs ERPI from the archived five indicator rasters and the fixed pooled entropy weights.")
    lines.append("")
    lines.append("It does **not** prove that hidden fill values in the original NDVI/nighttime-light inputs were excluded before resampling, and it does **not** independently quantify native-resolution CLCD source coverage. Those are source-input audit questions.")
    lines.append("")
    lines.append("## 2. Final-raster structure")
    lines.append("")
    lines.append(f"- Annual rasters found and scanned: **{len(stats_df)}/20**.")
    if len(struct_df):
        lines.append(f"- Grid size: **{int(struct_df.iloc[0].width):,} × {int(struct_df.iloc[0].height):,} pixels**.")
        lines.append(f"- Resolution: **{struct_df.iloc[0].res_x:g} × {abs(struct_df.iloc[0].res_y):g} map units**.")
        lines.append(f"- CRS: `{struct_df.iloc[0].crs}`.")
        lines.append(f"- NoData metadata: `{struct_df.iloc[0].nodata}`.")
    lines.append("")
    lines.append("## 3. Numeric integrity")
    lines.append("")
    lines.append(f"- Valid-value range across annual products: **{stats_df['min'].min():.6f} to {stats_df['max'].max():.6f}**.")
    lines.append(f"- Total valid pixel-years: **{int(stats_df['valid_pixels'].sum()):,}**.")
    lines.append(f"- ±Inf valid-cell findings: **{int(stats_df['posinf_pixels'].sum()+stats_df['neginf_pixels'].sum()):,}**.")
    lines.append(f"- Values below 0 beyond tolerance: **{int(stats_df['below_zero_pixels'].sum()):,}**.")
    lines.append(f"- Values above 1 beyond tolerance: **{int(stats_df['above_one_pixels'].sum()):,}**.")
    lines.append("")
    lines.append("## 4. Spatial support")
    lines.append("")
    lines.append(f"- 2001 valid pixels: **{int(ss.valid_2001):,}**.")
    lines.append(f"- 2020 valid pixels: **{int(ss.valid_2020):,}**.")
    lines.append(f"- Net 2001→2020 coverage change: **{int(ss.net_valid_change_2001_2020):,} pixels ({ss.net_valid_change_fraction_2001:.3%})**.")
    lines.append(f"- Strict common-valid pixels across all 20 years: **{int(ss.common_valid_pixels):,}**.")
    lines.append(f"- Union-valid pixels across all 20 years: **{int(ss.union_valid_pixels):,}**.")
    lines.append(f"- Common/union support ratio: **{ss.common_fraction_of_union:.3%}**.")
    lines.append(f"- Maximum annual valid-pixel deviation from 2001: **{max_coverage_dev:.3%}**.")
    lines.append("")
    lines.append("## 5. Fixed weights")
    lines.append("")
    for ind in INDICATORS:
        lines.append(f"- {ind}: **{weights[ind]:.6f}**")
    lines.append("")
    lines.append("## 6. Independent reconstruction")
    lines.append("")
    if recon_df is None:
        lines.append("- Reconstruction was **not run** because at least one component raster was unavailable.")
    else:
        lines.append(f"- Total mask mismatches between final ERPI and the five-component validity intersection: **{int(recon_df['mask_mismatch_pixels'].sum()):,} pixel-years**.")
        lines.append(f"- Maximum annual mean absolute reconstruction error: **{recon_df['mean_abs_error'].max():.3g}**.")
        lines.append(f"- Maximum absolute reconstruction error over sampled annual blocks: **{recon_df['max_abs_error'].max():.3g}**.")
        lines.append(f"- Pixel-years with absolute reconstruction error >1e-5: **{int(recon_df['pixels_abs_error_gt_1e-5'].sum()):,}**.")
    lines.append("")
    lines.append("## 7. Temporal diagnostics")
    lines.append("")
    lines.append(f"- Common-support annual mean range: **{temporal_df['common_mean'].min():.6f}–{temporal_df['common_mean'].max():.6f}**.")
    if temporal_df["temporal_jump_review_flag"].any():
        yrs = ", ".join(map(str, temporal_df.loc[temporal_df.temporal_jump_review_flag, "year"].tolist()))
        lines.append(f"- Robust national-mean change diagnostic flags: **{yrs}**.")
    else:
        lines.append("- No annual national-mean change exceeded the configured robust jump-review threshold.")
    lines.append("")
    lines.append("## 8. Audit flags")
    lines.append("")
    if flags_df.empty:
        lines.append("- None.")
    else:
        for _, r in flags_df.iterrows():
            lines.append(f"- **{r.severity} — {r.code}:** {r.message}")
    lines.append("")
    lines.append("## 9. Release decision rule")
    lines.append("")
    lines.append("- Any **FAIL** finding should be resolved before manuscript submission or public archive freeze.")
    lines.append("- **REVIEW** findings require visual/data inspection and a documented decision.")
    lines.append("- A clean final-raster audit does not replace the separate source-input audit described above.")
    lines.append("")
    lines.append("## 10. Files generated by this audit")
    lines.append("")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            lines.append(f"- `{p.name}`")

    (out_dir / "ERPI_FINAL_RASTER_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Technical audit of final ERPI 2001-2020 rasters.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = args.root
    out_dir = ensure_dir(root / OUTDIR_NAME)

    eri_paths = {y: root / "ERI" / f"ERI_{y}.tif" for y in YEARS}
    missing_eri = [str(p) for p in eri_paths.values() if not p.exists()]
    if missing_eri:
        raise FileNotFoundError(
            "Missing final annual ERPI rasters:\n" + "\n".join(missing_eri)
        )

    weights_path = root / "entropy_weights_fixed_2001_2020.csv"
    weights, weights_df = load_weights(weights_path)
    weights_df.to_csv(out_dir / "00_fixed_weights_copy.csv", index=False)

    print("="*80)
    print("FINAL ERPI TECHNICAL AUDIT | 2001-2020")
    print("="*80)
    print(f"Root: {root}")
    print(f"Audit output: {out_dir}")

    print("\n[1/8] Structural GeoTIFF audit ...")
    struct_df, structural_critical, structural_review = structural_audit(eri_paths, out_dir)

    if structural_critical:
        pd.DataFrame({"critical_structural_issue": structural_critical}).to_csv(
            out_dir / "CRITICAL_STRUCTURAL_ISSUES.csv", index=False
        )
        print("Critical structural findings detected; continuing scans where possible.")

    print("[2/8] Full-value scan, range, quantiles, SHA256 ...")
    stats_df, valid_full, row_profiles, col_profiles = scan_year_stats(eri_paths, out_dir)

    print("[3/8] Common/union support and annual transitions ...")
    common, union, trans_df, support_summary = support_audit(
        valid_full, eri_paths[2001], out_dir
    )

    print("[4/8] Exact duplicate-file audit ...")
    dup_df = exact_duplicate_audit(stats_df, out_dir)

    print("[5/8] Reconstructing ERPI from five components + fixed weights ...")
    recon_df, missing_components = reconstruction_audit(root, eri_paths, weights, out_dir)

    print("[6/8] Fixed-support temporal and interannual pixel-change diagnostics ...")
    temporal_df, diffs_df = common_mask_temporal_audit(eri_paths, common, out_dir)

    print("[7/8] Row/column seam diagnostics ...")
    stripe_df = stripe_diagnostic(row_profiles, col_profiles, out_dir)

    print("[8/8] Compiling flags, report and audit plots ...")
    flags_df = build_flags(
        struct_df, stats_df, trans_df, dup_df, recon_df, temporal_df, out_dir
    )

    # Add structural issues gathered before full scan
    extra_flags = []
    for s in structural_critical:
        extra_flags.append({"severity":"FAIL","code":"STRUCTURE","message":s})
    for s in structural_review:
        extra_flags.append({"severity":"REVIEW","code":"STRUCTURE_METADATA","message":s})
    if extra_flags:
        flags_df = pd.concat([flags_df, pd.DataFrame(extra_flags)], ignore_index=True)
        flags_df.to_csv(out_dir / "10_ERPI_audit_flags.csv", index=False)

    plot_audit(stats_df, trans_df, recon_df, temporal_df, out_dir)
    write_report(
        root, struct_df, stats_df, support_summary, trans_df, dup_df,
        recon_df, temporal_df, diffs_df, stripe_df, flags_df, weights, out_dir
    )

    fail_n = int((flags_df["severity"]=="FAIL").sum()) if len(flags_df) else 0
    review_n = int((flags_df["severity"]=="REVIEW").sum()) if len(flags_df) else 0

    print("\n" + "="*80)
    print("AUDIT FINISHED")
    print("="*80)
    print(f"FAIL flags   : {fail_n}")
    print(f"REVIEW flags : {review_n}")
    print(f"Report       : {out_dir / 'ERPI_FINAL_RASTER_AUDIT_REPORT.md'}")
    print("\nPlease send back these four files first:")
    print("  ERPI_FINAL_RASTER_AUDIT_REPORT.md")
    print("  10_ERPI_audit_flags.csv")
    print("  06_ERPI_reconstruction_from_components.csv   (if created)")
    print("  07_ERPI_common_support_temporal_diagnostics.csv")


if __name__ == "__main__":
    main()
