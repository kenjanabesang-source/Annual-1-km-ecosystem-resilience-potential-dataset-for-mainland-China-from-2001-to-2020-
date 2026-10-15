# -*- coding: utf-8 -*-
r"""Independent NPP-only ecological plausibility check for China's 2001-2020 ERI.

Uses ONLY existing local ERI and NPP rasters. Does not read GPP, and does not
recalculate or modify any main ERI outputs.

Expected defaults:
  F:\data\China\NPP\NPP_2001.tif ... NPP_2020.tif
  F:\data\China\ERI_recalculated_2001_2020\ERI\ERI_2001.tif ... ERI_2020.tif

Outputs:
  F:\data\China\ERI_recalculated_2001_2020\independent_NPP_only_validation\

Python packages: numpy, pandas, rasterio, tqdm.
Run: python ERI_NPP_only_validation_2001_2020.py
For naturalness-stratified supplementary results:
     python ERI_NPP_only_validation_2001_2020.py --lcn-strata

NOTE: NPP is an independent ecological-function proxy, not ground-truth
observations of resistance or disturbance recovery. Weak/negative associations
are legitimate findings, not errors to be tuned away.
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
from rasterio.warp import reproject
from rasterio.windows import Window
from tqdm import tqdm

YEARS = tuple(range(2001, 2021))
NODATA = -9999.0
TILE = 512
VERSION = "NPP_only_2026-09-24_v1"


def windows(src):
    for row in range(0, src.height, TILE):
        for col in range(0, src.width, TILE):
            yield Window(col, row, min(TILE, src.width-col), min(TILE, src.height-row))


def read_nan(src, win):
    # Reads GDAL internal masks as well as declared NoData values.
    a = src.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
    a[~np.isfinite(a)] = np.nan
    return a


def profile_like(ref, dtype="float32", nodata=NODATA):
    p = ref.profile.copy()
    # Source ERI may use small strip/block sizes; do not inherit invalid tiled blocks.
    p.pop("blockxsize", None)
    p.pop("blockysize", None)
    p.update(driver="GTiff", dtype=dtype, count=1, nodata=nodata,
             compress="deflate", tiled=True, blockxsize=256, blockysize=256,
             BIGTIFF="IF_SAFER")
    return p


def same_grid(a, b):
    return (a.width == b.width and a.height == b.height and a.crs == b.crs
            and a.transform.almost_equals(b.transform))


def checked_grid(path, ref):
    with rasterio.open(path) as src:
        if not same_grid(src, ref):
            raise ValueError(f"Grid/CRS is not identical to ERI: {path}")


def write_atomic(path, profile, writer):
    """Avoid treating an interrupted TIFF as a completed result."""
    partial = path.with_name(path.stem + ".partial.tif")
    if partial.exists():
        partial.unlink()
    try:
        with rasterio.open(partial, "w", **profile) as dst:
            writer(dst)
        os.replace(partial, path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def path_for(folder, basename):
    direct = folder / basename
    if direct.exists():
        return direct
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    matches = [p for p in folder.iterdir() if p.is_file()
               and p.name.lower() == basename.lower()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Expected {basename} in {folder}")


def sample_mask_for_window(win, step):
    rr = (np.arange(int(win.row_off), int(win.row_off+win.height)) % step) == 0
    cc = (np.arange(int(win.col_off), int(win.col_off+win.width)) % step) == 0
    return rr[:, None] & cc[None, :]


class Paired:
    """Exact full-pixel Pearson; spatially systematic sample for Spearman."""
    def __init__(self, sample_step):
        self.step = sample_step
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.sample_x, self.sample_y = [], []

    def add(self, x, y, valid, win):
        valid = valid & np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            return
        xx, yy = x[valid], y[valid]
        self.n += xx.size
        self.sx += float(xx.sum())
        self.sy += float(yy.sum())
        self.sxx += float(np.dot(xx, xx))
        self.syy += float(np.dot(yy, yy))
        self.sxy += float(np.dot(xx, yy))
        sm = valid & sample_mask_for_window(win, self.step)
        if sm.any():
            self.sample_x.append(x[sm].astype(np.float32))
            self.sample_y.append(y[sm].astype(np.float32))

    def finish(self):
        nx = self.n * self.sxx - self.sx * self.sx
        ny = self.n * self.syy - self.sy * self.sy
        den = math.sqrt(max(nx, 0.0) * max(ny, 0.0))
        r = (self.n * self.sxy - self.sx * self.sy) / den if den > 0 else np.nan
        if self.sample_x:
            x, y = np.concatenate(self.sample_x), np.concatenate(self.sample_y)
            rx = pd.Series(x).rank(method="average").to_numpy()
            ry = pd.Series(y).rank(method="average").to_numpy()
            rho = (float(np.corrcoef(rx, ry)[0, 1])
                   if len(rx) >= 3 and np.std(rx) > 0 and np.std(ry) > 0
                   else np.nan)
            sample_n = len(rx)
        else:
            rho, sample_n = np.nan, 0
        return dict(n_pixels=int(self.n), pearson_r_full=float(r),
                    spearman_rho_systematic_sample=float(rho),
                    spearman_sample_n=int(sample_n))


def source_metadata(paths, args, out):
    rows = []
    print("[1/7] Check original NPP source metadata (no unverified fill-value assumptions)")
    for year in YEARS:
        path = paths[year]
        with rasterio.open(path) as s:
            if s.crs is None or s.count != 1:
                raise ValueError(f"NPP must be one band with a CRS: {path}")
            raw_scale = float(s.scales[0] if s.scales[0] is not None else 1.0)
            raw_offset = float(s.offsets[0] if s.offsets[0] is not None else 0.0)
            scale = raw_scale if args.npp_scale is None else args.npp_scale
            offset = raw_offset if args.npp_offset is None else args.npp_offset
            nodata = s.nodata if args.raw_nodata is None else args.raw_nodata
            # The preview is solely a warning/QA tool; it does not set validity rules.
            h, w = min(s.height, 256), min(s.width, 256)
            preview = s.read(1, out_shape=(h, w), masked=True,
                             resampling=Resampling.nearest)
            raw = np.asarray(preview.data, dtype=np.float64)
            ok = ~np.ma.getmaskarray(preview) & np.isfinite(raw)
            if nodata is not None:
                ok &= raw != nodata
            vals = raw[ok]
            raw_min = float(vals.min()) if vals.size else np.nan
            raw_max = float(vals.max()) if vals.size else np.nan
            negative_sample = int(np.count_nonzero(vals < 0))
            special = int(np.count_nonzero(np.isin(vals, [32767, 65534, 65535])))
            rows.append(dict(year=year, source=str(path), dtype=s.dtypes[0],
                             width=s.width, height=s.height, crs=str(s.crs),
                             declared_nodata=s.nodata, effective_nodata=nodata,
                             band_scale=raw_scale, band_offset=raw_offset,
                             effective_scale=scale, effective_offset=offset,
                             units=(s.units[0] if s.units else None),
                             band_tags=json.dumps(s.tags(1), ensure_ascii=False),
                             sample_raw_min=raw_min, sample_raw_max=raw_max,
                             sample_negative_count=negative_sample,
                             sample_32767_65534_65535_count=special))
            if nodata is None and negative_sample:
                print(f"  WARNING {year}: negative sampled raw values but no NoData metadata."
                      " If these are fill codes, rerun with --raw-nodata VALUE.")
            if special:
                print(f"  WARNING {year}: sampled 32767/65534/65535 raw values;"
                      " verify the product's fill codes before interpreting results.")
            if year in (2001, 2020):
                print(f"  {year}: dtype={s.dtypes[0]}, nodata={s.nodata}, "
                      f"scale={raw_scale}, offset={raw_offset}, sample max={raw_max}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "NPP_original_metadata_2001_2020.csv", index=False,
              encoding="utf-8-sig")
    return df


def align_npp(paths, eri_paths, args, out):
    print("[2/7] Align NPP only to the exact 1-km ERI grid")
    folder = out / "aligned_NPP_1km"
    folder.mkdir(parents=True, exist_ok=True)
    aligned = {}
    with rasterio.open(eri_paths[2001]) as ref:
        profile = profile_like(ref)
        for year in tqdm(YEARS, desc="Align NPP"):
            dest = folder / f"NPP_1km_{year}.tif"
            aligned[year] = dest
            if dest.exists() and not args.force:
                checked_grid(dest, ref)
                continue
            with rasterio.open(paths[year]) as src:
                src_nodata = src.nodata if args.raw_nodata is None else args.raw_nodata
                scale = (float(src.scales[0] if src.scales[0] is not None else 1.0)
                         if args.npp_scale is None else args.npp_scale)
                offset = (float(src.offsets[0] if src.offsets[0] is not None else 0.0)
                          if args.npp_offset is None else args.npp_offset)
                arr = np.full((ref.height, ref.width), NODATA, dtype=np.float32)
                reproject(source=rasterio.band(src, 1), destination=arr,
                          src_transform=src.transform, src_crs=src.crs,
                          src_nodata=src_nodata,
                          dst_transform=ref.transform, dst_crs=ref.crs,
                          dst_nodata=NODATA, resampling=Resampling.average,
                          num_threads=args.threads)
                valid = np.isfinite(arr) & (arr != NODATA)
                arr[valid] = arr[valid] * scale + offset
                valid = np.isfinite(arr) & (arr != NODATA)
                if args.exclude_negative:
                    arr[valid & (arr < 0)] = NODATA
                arr[~np.isfinite(arr)] = NODATA
                write_atomic(dest, profile, lambda d: d.write(arr, 1))
    return aligned


def aligned_qa(aligned, out):
    print("[3/7] Inspect NPP range/zero rate after alignment")
    rows = []
    for year in YEARS:
        n = zero = 0
        total = mn = mx = 0.0
        mn, mx = np.inf, -np.inf
        with rasterio.open(aligned[year]) as src:
            for win in windows(src):
                a = read_nan(src, win)
                x = a[np.isfinite(a)]
                if x.size:
                    n += len(x)
                    zero += int((x == 0).sum())
                    total += float(x.sum())
                    mn = min(mn, float(x.min()))
                    mx = max(mx, float(x.max()))
        rows.append(dict(year=year, valid_pixels=n, min=(mn if n else np.nan),
                         mean=(total/n if n else np.nan),
                         max=(mx if n else np.nan),
                         zero_fraction=(zero/n if n else np.nan)))
    qa = pd.DataFrame(rows)
    qa.to_csv(out / "NPP_aligned_QA_2001_2020.csv", index=False)
    print(qa.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    return qa


def common_mask(eri_paths, aligned, args, out):
    print("[4/7] Build 20-year strict ERI + NPP common-valid mask (NO GPP)")
    path = out / "common_valid_mask_ERI_NPP_2001_2020.tif"
    with rasterio.open(eri_paths[2001]) as ref:
        profile = profile_like(ref, dtype="uint8", nodata=0)
        with _open_year_sets(eri_paths, aligned) as datasets:
            def writer(dst):
                total = 0
                for win in tqdm(windows(ref), desc="Common mask"):
                    valid = np.ones((int(win.height), int(win.width)), dtype=bool)
                    for year in YEARS:
                        valid &= np.isfinite(read_nan(datasets["eri"][year], win))
                        valid &= np.isfinite(read_nan(datasets["npp"][year], win))
                    total += int(valid.sum())
                    dst.write(valid.astype(np.uint8), 1, window=win)
                print(f"  Common-valid pixels across ALL 20 years: {total:,}")
                if total == 0:
                    raise RuntimeError("No common pixels. Check NPP NoData/CRS/scale.")
            write_atomic(path, profile, writer)
    return path


class _open_year_sets:
    def __init__(self, eri, npp):
        self.paths = {"eri": eri, "npp": npp}
        self.opened = {"eri": {}, "npp": {}}
    def __enter__(self):
        try:
            for typ in ("eri", "npp"):
                for y in YEARS:
                    self.opened[typ][y] = rasterio.open(self.paths[typ][y])
            return self.opened
        except Exception:
            self.__exit__(None, None, None)
            raise
    def __exit__(self, *args):
        for typ in ("eri", "npp"):
            for src in self.opened[typ].values():
                src.close()


def temporal_maps(eri_paths, aligned, mask_path, lcn_paths, out):
    print("[5/7] Compute mean ERI, mean NPP, NPP CV and bounded NPP stability")
    folder = out / "temporal_maps"
    folder.mkdir(parents=True, exist_ok=True)
    products = {k: folder / f"{k}_2001_2020.tif" for k in (
        "ERI_mean", "NPP_mean", "NPP_SD", "NPP_CV", "NPP_stability_1_over_1_plus_CV")}
    if lcn_paths:
        products["LCN_mean"] = folder / "LCN_mean_2001_2020.tif"

    with rasterio.open(eri_paths[2001]) as ref, rasterio.open(mask_path) as mask, \
         _open_year_sets(eri_paths, aligned) as ds:
        from contextlib import ExitStack
        with ExitStack() as stack:
            d = {name: stack.enter_context(rasterio.open(path, "w", **profile_like(ref)))
                 for name, path in products.items()}
            lcns = [stack.enter_context(rasterio.open(lcn_paths[y])) for y in YEARS] \
                if lcn_paths else []
            for win in tqdm(windows(ref), desc="Temporal maps"):
                v = mask.read(1, window=win) == 1
                shape = (int(win.height), int(win.width))
                arrays = {key: np.full(shape, NODATA, dtype=np.float32)
                          for key in products}
                if v.any():
                    e = np.stack([read_nan(ds["eri"][y], win) for y in YEARS], axis=0)
                    n = np.stack([read_nan(ds["npp"][y], win) for y in YEARS], axis=0)
                    em = np.mean(e[:, v], axis=0)
                    nm = np.mean(n[:, v], axis=0)
                    sd = np.std(n[:, v], axis=0, ddof=1)
                    arrays["ERI_mean"][v] = em.astype(np.float32)
                    arrays["NPP_mean"][v] = nm.astype(np.float32)
                    arrays["NPP_SD"][v] = sd.astype(np.float32)
                    good = (nm > 0) & (sd >= 0) & np.isfinite(nm) & np.isfinite(sd)
                    if good.any():
                        cv = sd[good] / nm[good]
                        iv = np.flatnonzero(v)[good]
                        arrays["NPP_CV"].flat[iv] = cv.astype(np.float32)
                        arrays["NPP_stability_1_over_1_plus_CV"].flat[iv] = (
                            1.0 / (1.0 + cv)).astype(np.float32)
                    if lcns:
                        l = np.stack([read_nan(s, win) for s in lcns], axis=0)
                        ok = v & np.all(np.isfinite(l), axis=0)
                        if ok.any():
                            arrays["LCN_mean"][ok] = np.mean(l[:, ok], axis=0).astype(np.float32)
                for key, dst in d.items():
                    dst.write(arrays[key], 1, window=win)
    return products


def compare_rasters(x_path, y_path, mask_path, step, lcn_path=None, group="all"):
    stats = Paired(step)
    from contextlib import ExitStack
    with ExitStack() as stack:
        x = stack.enter_context(rasterio.open(x_path))
        y = stack.enter_context(rasterio.open(y_path))
        m = stack.enter_context(rasterio.open(mask_path))
        lcn = stack.enter_context(rasterio.open(lcn_path)) if lcn_path else None
        for win in windows(x):
            v = m.read(1, window=win) == 1
            if lcn is not None:
                z = read_nan(lcn, win)
                if group == "natural_dominant":
                    v &= z >= 0.5
                elif group == "other":
                    v &= z < 0.5
            if v.any():
                stats.add(read_nan(x, win), read_nan(y, win), v, win)
    return stats.finish()


def long_term_comparison(products, mask_path, step, out, strata):
    print("[6/7] Long-term spatial ecological plausibility comparisons")
    combos = [
        ("mean_ERI_vs_mean_NPP", products["ERI_mean"], products["NPP_mean"]),
        ("mean_ERI_vs_bounded_NPP_stability", products["ERI_mean"],
         products["NPP_stability_1_over_1_plus_CV"]),
        ("mean_ERI_vs_NPP_CV", products["ERI_mean"], products["NPP_CV"]),
    ]
    rows = []
    groups = ("all", "natural_dominant", "other") if strata else ("all",)
    for group in groups:
        lcn = products.get("LCN_mean") if group != "all" else None
        for name, x, y in combos:
            result = compare_rasters(x, y, mask_path, step, lcn, group)
            rows.append(dict(group=group, comparison=name, **result))
            print(f"  {group:16s} {name:40s} r={result['pearson_r_full']:.4f}, "
                  f"rho={result['spearman_rho_systematic_sample']:.4f}, "
                  f"n={result['n_pixels']:,}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "NPP_long_term_spatial_validation.csv", index=False)
    return df


def annual_comparison(eri_paths, aligned, mask_path, step, out):
    print("[7/7] Annual spatial correlations and fixed-area national means")
    rows = []
    with rasterio.open(mask_path) as m:
        for year in tqdm(YEARS, desc="Annual NPP check"):
            stat = Paired(step)
            se = sn = 0.0
            with rasterio.open(eri_paths[year]) as e, rasterio.open(aligned[year]) as n:
                for win in windows(e):
                    v = m.read(1, window=win) == 1
                    if not v.any():
                        continue
                    x, y = read_nan(e, win), read_nan(n, win)
                    v &= np.isfinite(x) & np.isfinite(y)
                    if v.any():
                        stat.add(x, y, v, win)
                        se += float(np.sum(x[v]))
                        sn += float(np.sum(y[v]))
            r = stat.finish()
            rows.append(dict(year=year, **r, ERI_mean_common_mask=(se/r["n_pixels"]
                             if r["n_pixels"] else np.nan),
                             NPP_mean_common_mask=(sn/r["n_pixels"]
                             if r["n_pixels"] else np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "NPP_annual_spatial_validation_and_means.csv", index=False)
    return df


def temporal_associations(annual, out):
    y = annual["year"].to_numpy(dtype=float)
    a = annual["ERI_mean_common_mask"].to_numpy(dtype=float)
    b = annual["NPP_mean_common_mask"].to_numpy(dtype=float)
    def corr(x, z, reference_x=None, reference_z=None):
        # Avoid reporting a correlation of float32 rounding noise if a perfectly
        # linear national series has (almost) zero detrended/first-difference SD.
        rx = x if reference_x is None else reference_x
        rz = z if reference_z is None else reference_z
        tol_x = 1e-7 * max(1.0, float(np.nanmax(np.abs(rx))))
        tol_z = 1e-7 * max(1.0, float(np.nanmax(np.abs(rz))))
        if len(x) < 3 or np.std(x) <= tol_x or np.std(z) <= tol_z:
            return np.nan
        return float(np.corrcoef(x, z)[0, 1])
    a_det = a - np.polyval(np.polyfit(y, a, 1), y)
    b_det = b - np.polyval(np.polyfit(y, b, 1), y)
    rows = [
        dict(method="national_annual_means_level", n_years=len(y), pearson_r=corr(a,b)),
        dict(method="national_annual_means_linear_detrended", n_years=len(y),
             pearson_r=corr(a_det,b_det, a, b)),
        dict(method="national_annual_means_first_difference", n_years=len(y)-1,
             pearson_r=corr(np.diff(a),np.diff(b), a, b)),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "NPP_national_temporal_associations.csv", index=False)
    return df


def create_summary(meta, qa, long_df, annual, temporal, mask_path, out, strata):
    with rasterio.open(mask_path) as s:
        n_mask = sum(int(np.count_nonzero(s.read(1,window=w))) for w in windows(s))
    lines = [
        "Independent NPP-only ecological plausibility assessment of ERI",
        "Period: 2001-2020; grid: the existing 1-km ERI grid.",
        "NPP-only strict common mask across all 20 years; GPP is never used.",
        f"Common-valid ERI+NPP pixels: {n_mask:,}",
        f"NPP original scale factors: {sorted(meta.effective_scale.unique().tolist())}",
        f"NPP aligned mean annual range: {qa['mean'].min():.6g} to {qa['mean'].max():.6g}",
        f"NPP aligned zero-fraction annual range: {qa.zero_fraction.min():.4%} "
        f"to {qa.zero_fraction.max():.4%}",
        "",
        "Long-term spatial correlations (all common pixels):",
    ]
    for _, r in long_df[long_df.group == "all"].iterrows():
        lines.append(f"  {r.comparison}: Pearson r={r.pearson_r_full:.6f}; "
                     f"systematic-sample Spearman rho={r.spearman_rho_systematic_sample:.6f}; "
                     f"n={int(r.n_pixels):,}")
    lines += ["", "Annual spatial ERI-NPP Pearson correlations (20 maps):",
              f"  mean={annual.pearson_r_full.mean():.6f}; "
              f"min={annual.pearson_r_full.min():.6f}; "
              f"max={annual.pearson_r_full.max():.6f}",
              "", "Fixed-area national temporal associations (descriptive only):"]
    for _, r in temporal.iterrows():
        lines.append(f"  {r.method}: Pearson r={r.pearson_r:.6f}; n={int(r.n_years)}")
    if strata:
        lines.extend(["", "Supplementary LCN naturalness-stratified analysis:",
                      "  See NPP_long_term_spatial_validation.csv; thresholds are "
                      "mean LCN >=0.5 and <0.5, NOT vegetation classes."])
    lines.extend(["", "INTERPRETATION:",
                  "  These associations assess ecological plausibility; NPP is not a direct",
                  "  observation of disturbance resistance or post-disturbance recovery.",
                  "  Spatially autocorrelated pixels are not independent replicates;",
                  "  do not interpret pixel count as statistical sample size for p-values.",
                  "  NPP=0 is retained where the input marks it as valid; check product metadata.",
                  "  NPP might share satellite/algorithm inputs with NDVI; independence",
                  "  here means not used to compute ERI, not independent instrument lineage.",
                  "  Bounded stability = 1/(1 + SD/mean); only defined for mean NPP > 0."])
    path = out / "NPP_only_validation_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=os.environ.get("ERPI_DATA_ROOT", "./data"), help="Parent directory of NPP/ and ERPI products")
    p.add_argument("--erpi-root", default=os.environ.get("ERPI_PRODUCT_ROOT"), help="Corrected ERPI product directory (default: <root>/ERI_recalculated_2001_2020_NTLfix_candidate_v1")
    p.add_argument("--threads", type=int, default=4, help="rasterio warp threads, integer >=1")
    p.add_argument("--sample-cap", type=int, default=200_000,
                   help="Approximate maximum systematic Spearman sample per comparison")
    p.add_argument("--raw-nodata", type=float, default=None,
                   help="ONLY if source documentation confirms an unmarked raw fill code")
    p.add_argument("--npp-scale", type=float, default=None,
                   help="Override NPP band scale ONLY if the product documentation requires it")
    p.add_argument("--npp-offset", type=float, default=None,
                   help="Override NPP band offset ONLY if the product documentation requires it")
    p.add_argument("--exclude-negative", action="store_true",
                   help="Mask negative NPP after resampling, if scientifically appropriate")
    p.add_argument("--lcn-strata", action="store_true",
                   help="Supplementary analyses by mean LCN >=0.5 vs <0.5")
    p.add_argument("--force", action="store_true",
                   help="Regenerate aligned NPP rasters after any metadata/scale change")
    args = p.parse_args()
    if args.threads < 1 or args.sample_cap < 1:
        p.error("--threads and --sample-cap must be positive integers")
    return args


def main():
    args = parse_args()
    root = Path(args.root)
    eriroot = Path(args.erpi_root).expanduser().resolve() if args.erpi_root else root / "ERI_recalculated_2001_2020_NTLfix_candidate_v1"
    out = eriroot / "independent_NPP_only_validation"
    out.mkdir(parents=True, exist_ok=True)
    npp_dir = root / "NPP"
    eri_dir = eriroot / "ERI"
    npp = {y: path_for(npp_dir, f"NPP_{y}.tif") for y in YEARS}
    eri = {y: path_for(eri_dir, f"ERI_{y}.tif") for y in YEARS}
    lcn = ({y: path_for(eriroot / "indicators", f"LCN_{y}.tif") for y in YEARS}
           if args.lcn_strata else None)
    with rasterio.open(eri[2001]) as ref:
        for year in YEARS:
            checked_grid(eri[year], ref)
            if lcn:
                checked_grid(lcn[year], ref)
        sample_step = max(1, math.ceil(math.sqrt(
            (ref.width * ref.height) / args.sample_cap)))
        print(f"ERI grid: {ref.width} x {ref.height}; sample spacing: "
              f"every {sample_step}th row/column")

    print("=" * 75)
    print("ERI 2001-2020: NPP-ONLY independent validation (NO GPP)")
    print("=" * 75)
    print(f"ERI root: {eriroot}\nNPP root: {npp_dir}\nOutput: {out}")
    meta = source_metadata(npp, args, out)
    aligned = align_npp(npp, eri, args, out)
    qa = aligned_qa(aligned, out)
    mask = common_mask(eri, aligned, args, out)
    products = temporal_maps(eri, aligned, mask, lcn, out)
    long_df = long_term_comparison(products, mask, sample_step, out, args.lcn_strata)
    annual = annual_comparison(eri, aligned, mask, sample_step, out)
    temporal = temporal_associations(annual, out)
    create_summary(meta, qa, long_df, annual, temporal, mask, out, args.lcn_strata)
    print(f"\nFinished. Results: {out}")
    print("Please share NPP_only_validation_summary.txt, "
          "NPP_original_metadata_2001_2020.csv, and "
          "NPP_aligned_QA_2001_2020.csv")


if __name__ == "__main__":
    main()
