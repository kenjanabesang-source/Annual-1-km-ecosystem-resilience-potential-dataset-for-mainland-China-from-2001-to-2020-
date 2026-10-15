# -*- coding: utf-8 -*-
r"""
Assess CLCD native valid-area coverage on the *corrected* NTL-fix ERPI candidate.

Read-only inputs; never overwrites the corrected ERPI/LCN/weights.
By default, writes to F:\data\China\ERI_recalculated_2001_2020_NTLfix_candidate_v1\CLCD_COVERAGE_IMPACT_AUDIT.

First execution requires reading CLCD native rasters; coverage GeoTIFFs are
cached, so an interrupted run can be restarted without repeating completed
native-area warps. No coverage threshold is applied to the released candidate.
Thresholds 50/90/95/99% are *sensitivity scenarios*, not ecological thresholds.

Run:
D:\conda\python.exe "C:\Users\Administrator\Downloads\assess_CLCD_coverage_impact_NTLfix_2001_2020.py"

Dependencies: numpy, rasterio. The target/source CRSs must match and both grids
must be north-up, axis aligned; otherwise this script stops rather than using
an invalid area approximation.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

YEARS = range(2001, 2021)
THRESHOLDS = (0.50, 0.90, 0.95, 0.99)
ROOT = Path(r"F:\data\China")
CANDIDATE_NAME = "ERI_recalculated_2001_2020_NTLfix_candidate_v1"
NODATA = -9999.0
VALID_CLASSES = np.arange(1, 10, dtype=np.uint8)


def csv_write(path: Path, rows: list[dict], names=None):
    if not rows and names is None:
        return
    fields = names or list(rows[0].keys())
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(temp, path)


def signature(path: Path):
    s = path.stat()
    return {"path": str(path.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def src_file(root, year):
    for folder in (root / "clcd", root / "CLCD"):
        for stem in (f"CLCD_v01_{year}_albert.tif", f"CLCD_v01_{year}_albers.tif", f"CLCD_v01_{year}.tif"):
            p = folder / stem
            if p.is_file():
                return p
    raise FileNotFoundError(f"CLCD raster not found for {year} under {root}")


def check_grid(source, target, year):
    s, t = source.transform, target.transform
    if source.crs != target.crs or source.crs is None:
        raise ValueError(f"{year}: source/target CRS mismatch")
    if any((abs(s.b) > 1e-10, abs(s.d) > 1e-10, abs(t.b) > 1e-10,
            abs(t.d) > 1e-10, s.a <= 0, s.e >= 0, t.a <= 0, t.e >= 0)):
        raise ValueError(f"{year}: unsupported rotated or south-up grid")
    if abs(t.a - 1000.0) > 1e-7 or abs(t.e + 1000.0) > 1e-7:
        raise ValueError(f"{year}: target grid is not the expected 1 km grid")


def mask_profile(target):
    p = target.profile.copy()
    p.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA,
             tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
    p.pop("predictor", None)
    return p


def get_coverage(src_path, eri_path, coverage_path, tempdir, year):
    """Full 1-km valid-class area / full 1-km cell; explicit 0 for invalid native code.

    GDAL average normalizes to source footprint. A per-axis exact rectangle
    overlap factor converts that fraction to the fraction of the whole target
    cell. For any atypical CLCD warp, a separate exact native-cell sample audit
    remains necessary; the supplied previous audit already performed it.
    """
    if coverage_path.is_file():
        with rasterio.open(coverage_path) as cached, rasterio.open(eri_path) as target:
            if (cached.crs != target.crs or cached.transform != target.transform or
                    cached.shape != target.shape or cached.count != 1 or cached.nodata != NODATA):
                raise ValueError(f"Invalid cached CLCD coverage raster for {year}")
            return cached.read(1)

    native_tmp = tempdir / f"CLCD_valid_native_{year}.tif"
    pending = coverage_path.with_name(coverage_path.stem + ".partial.tif")
    if native_tmp.exists():
        native_tmp.unlink()
    if pending.exists():
        pending.unlink()

    try:
        with rasterio.open(src_path) as source, rasterio.open(eri_path) as target:
            check_grid(source, target, year)
            profile = source.profile.copy()
            profile.pop("predictor", None)
            profile.update(driver="GTiff", dtype="uint8", count=1,
                           nodata=None, compress="deflate", tiled=True,
                           BIGTIFF="IF_SAFER")
            if source.width < 512 or source.height < 512:
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
                profile["tiled"] = False
            else:
                profile.update(blockxsize=512, blockysize=512)
            with rasterio.open(native_tmp, "w", **profile) as binary:
                for _, win in source.block_windows(1):
                    arr = source.read(1, window=win)
                    binary.write(np.isin(arr, VALID_CLASSES).astype("uint8"), 1, window=win)

            coverage = np.full(target.shape, NODATA, dtype="float32")
            with rasterio.open(native_tmp) as native:
                reproject(source=rasterio.band(native, 1), destination=coverage,
                          src_transform=native.transform, src_crs=native.crs,
                          src_nodata=None, dst_transform=target.transform,
                          dst_crs=target.crs, dst_nodata=NODATA,
                          resampling=Resampling.average, num_threads=2,
                          warp_mem_limit=128)
            dt = target.transform
            xx = dt.c + np.arange(target.width, dtype="float64") * dt.a
            xf = np.clip((np.minimum(xx + dt.a, source.bounds.right) -
                         np.maximum(xx, source.bounds.left)) / dt.a, 0, 1)
            yt = dt.f + np.arange(target.height, dtype="float64") * dt.e
            yb = yt + dt.e
            yf = np.clip((np.minimum(yt, source.bounds.top) -
                         np.maximum(yb, source.bounds.bottom)) / (-dt.e), 0, 1)
            finite = coverage != NODATA
            # Scale in place without materializing a full-size broadcast array.
            coverage[~finite] = 0.0
            coverage *= xf[None, :].astype("float32")
            coverage *= yf[:, None].astype("float32")
            coverage[~finite] = NODATA
            if np.any(coverage[finite] > 1.0001) or np.any(coverage[finite] < -1e-6):
                raise ValueError(f"{year}: coverage outside [0,1] after area correction")
            coverage[finite] = np.clip(coverage[finite], 0, 1)
            with rasterio.open(pending, "w", **mask_profile(target)) as dest:
                dest.write(coverage, 1)
            os.replace(pending, coverage_path)
            return coverage
    finally:
        if native_tmp.exists():
            native_tmp.unlink()
        if pending.exists():
            pending.unlink()


def validity(arr, nodata):
    m = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        m &= arr != nodata
    return m


def support_edge(mask):
    """A final-valid cell touching an invalid 1-km neighbor (8 connectivity)."""
    interior = mask.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r0, r1 = max(0, -dr), min(h, h-dr)
            c0, c1 = max(0, -dc), min(w, w-dc)
            interior[r0:r1, c0:c1] &= mask[r0+dr:r1+dr, c0+dc:c1+dc]
    return mask & ~interior


def common_mean(path, mask):
    with rasterio.open(path) as src:
        total, n = 0.0, 0
        for _, win in src.block_windows(1):
            r, c = int(win.row_off), int(win.col_off)
            m = mask[r:r+int(win.height), c:c+int(win.width)]
            if not m.any():
                continue
            a = src.read(1, window=win)
            good = m & validity(a, src.nodata)
            n += int(good.sum())
            total += float(a[good].astype("float64").sum())
        return total/n if n else math.nan, n


def main():
    ap = argparse.ArgumentParser(description="CLCD 1-km source-coverage sensitivity for corrected ERPI")
    ap.add_argument("--root", type=Path, default=ROOT, help="Parent F:\\data\\China")
    ap.add_argument("--candidate", type=Path, default=None, help="NTLfix candidate folder")
    args = ap.parse_args()
    root = args.root
    candidate = args.candidate or (root / CANDIDATE_NAME)
    if candidate.name != CANDIDATE_NAME:
        raise ValueError("Refusing to analyze non-NTLfix candidate; check --candidate")
    if not (candidate / "CORRECTION_INPUT_MANIFEST.json").is_file():
        raise FileNotFoundError("Corrected candidate manifest missing")
    if not (candidate / "entropy_weights_fixed_2001_2020.csv").is_file():
        raise FileNotFoundError("Corrected fixed-weight table missing")
    out = candidate / "CLCD_COVERAGE_IMPACT_AUDIT"
    out.mkdir(parents=True, exist_ok=True)
    coverage_dir = out / "coverage"
    coverage_dir.mkdir(exist_ok=True)
    tmp = out / "_temporary_native_mask"
    tmp.mkdir(exist_ok=True)

    eri_paths = {y: candidate / "ERI" / f"ERI_{y}.tif" for y in YEARS}
    clcd_paths = {y: src_file(root, y) for y in YEARS}
    missing = [str(p) for p in eri_paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Corrected ERPI file(s) missing:\n" + "\n".join(missing))
    source_signatures = {str(y): signature(clcd_paths[y]) for y in YEARS}
    source_signatures["candidate_manifest"] = signature(candidate / "CORRECTION_INPUT_MANIFEST.json")
    source_signatures["candidate_weights"] = signature(candidate / "entropy_weights_fixed_2001_2020.csv")
    for y in YEARS:
        source_signatures[f"ERI_{y}"] = signature(eri_paths[y])
    manifest_path = out / "CLCD_AUDIT_INPUTS.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != source_signatures:
            raise RuntimeError("Input signatures changed. Do not reuse cached coverage: move old audit folder first")
    else:
        if list(coverage_dir.glob("CLCD_native_valid_coverage_*.tif")):
            raise RuntimeError("Unverified cached coverage exists without audit manifest")
        manifest_path.write_text(json.dumps(source_signatures, indent=2, ensure_ascii=False), encoding="utf-8")

    with rasterio.open(eri_paths[2001]) as reference:
        refsig = (reference.crs, reference.transform, reference.shape)
        shape = reference.shape
        transform = reference.transform
        profile = reference.profile.copy()
    common = np.ones(shape, dtype=bool)
    screened_common = {thr: np.ones(shape, dtype=bool) for thr in THRESHOLDS}
    annual, locations = [], []
    # Optional independent area-overlap spot-check from the previous full audit.
    prior_exact = root / "ERI_recalculated_2001_2020" / "SOURCE_PROCESSING_AUDIT" / "clcd_exact_area_samples.csv"
    exact_by_year = {y: [] for y in YEARS}
    if prior_exact.is_file():
        with prior_exact.open(newline="", encoding="utf-8-sig") as stream:
            for sample in csv.DictReader(stream):
                yr = int(sample["year"])
                if yr in exact_by_year:
                    exact_by_year[yr].append(sample)
    exact_comparisons = []
    print("CLCD coverage sensitivity: corrected ERPI candidate", flush=True)
    for y in YEARS:
        print(f"[{y}] full native-area coverage (cached if already complete)", flush=True)
        eri_path = eri_paths[y]
        with rasterio.open(eri_path) as src:
            if (src.crs, src.transform, src.shape) != refsig:
                raise ValueError(f"{y}: candidate ERPI grid mismatch")
            eri = src.read(1)
            is_valid = validity(eri, src.nodata)
            coverage = get_coverage(clcd_paths[y], eri_path,
                                    coverage_dir / f"CLCD_native_valid_coverage_{y}.tif", tmp, y)
            cov_valid = np.isfinite(coverage) & (coverage != NODATA)
            # Check GDAL full-grid coverage against exact native-pixel/target-cell
            # overlap for independently sampled cells, where source footprint exists.
            for sample in exact_by_year[y]:
                r, c = int(sample["row"]), int(sample["col"])
                expected = float(sample["valid_coverage"])
                footprint = float(sample["source_footprint_coverage"])
                observed = float(coverage[r, c]) if cov_valid[r, c] else math.nan
                if footprint > 0:
                    exact_comparisons.append({
                        "year": y, "row": r, "col": c,
                        "stratum": sample["stratum"],
                        "native_area_exact": expected,
                        "full_grid_warp_coverage": observed,
                        "abs_error": abs(observed-expected) if np.isfinite(observed) else math.nan,
                        "comparison_valid": bool(np.isfinite(observed)),
                    })
            missing = is_valid & ~cov_valid
            edge = support_edge(is_valid)
            common &= is_valid
            n = int(is_valid.sum())
            old_mean = float(eri[is_valid].astype("float64").mean())
            row = {"year": y, "candidate_valid_pixels": n,
                   "candidate_annual_mean": old_mean,
                   "candidate_valid_missing_CLCD_coverage": int(missing.sum())}
            for thr in THRESHOLDS:
                kept = is_valid & cov_valid & (coverage >= thr)
                removed = is_valid & ~kept
                screened_common[thr] &= kept
                k = int(removed.sum())
                kept_n = int(kept.sum())
                new_mean = float(eri[kept].astype("float64").mean()) if kept_n else math.nan
                tag = f"{int(thr*100)}pct"
                row[f"removed_{tag}"] = k
                row[f"removed_fraction_{tag}"] = k/n if n else math.nan
                row[f"screened_annual_mean_{tag}"] = new_mean
                row[f"screened_minus_candidate_mean_{tag}"] = new_mean-old_mean
                row[f"analytic_upper_bound_abs_mean_shift_{tag}"] = k/n if n else math.nan
                row[f"removed_touching_support_edge_{tag}"] = int((removed & edge).sum())
                row[f"removed_in_support_interior_{tag}"] = int((removed & ~edge).sum())
            low = is_valid & (missing | (coverage < 0.9))
            rr, cc = np.nonzero(low)
            for r, c in zip(rr.tolist(), cc.tolist()):
                x, py = transform * (c + .5, r + .5)
                locations.append({"year": y, "row": r, "col": c,
                                  "center_x_m": round(x, 2), "center_y_m": round(py, 2),
                                  "native_valid_area_fraction": float(coverage[r, c]) if cov_valid[r,c] else math.nan,
                                  "candidate_ERPI": float(eri[r, c]),
                                  "touches_8_neighbour_final_NoData": bool(edge[r, c])})
            annual.append(row)
            csv_write(out / "CLCD_coverage_impact_by_year.csv", annual)
            csv_write(out / "CLCD_below90pct_locations.csv", locations,
                      ["year", "row", "col", "center_x_m", "center_y_m", "native_valid_area_fraction", "candidate_ERPI", "touches_8_neighbour_final_NoData"])
            if exact_comparisons:
                csv_write(out / "CLCD_warp_vs_exact_native_overlap.csv", exact_comparisons)
            print(f"  candidate {n:,}; <90% removed {row['removed_90pct']:,}; "
                  f"annual mean Δ {row['screened_minus_candidate_mean_90pct']:+.8f}", flush=True)
            del eri, coverage, is_valid, cov_valid, edge

    common_rows = []
    masks = [("candidate", common)] + [(f"screened_{int(thr*100)}pct", screened_common[thr]) for thr in THRESHOLDS]
    for label, mask in masks:
        p = out / f"common_support_{label}.tif"
        output_profile = profile.copy()
        output_profile.pop("predictor", None)
        output_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0,
                              tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
        pending = p.with_name(p.stem + ".partial.tif")
        with rasterio.open(pending, "w", **output_profile) as dst:
            dst.write(mask.astype("uint8"), 1)
        os.replace(pending, p)
        common_rows.append({"scenario": label, "common_support_pixels": int(mask.sum()),
                            "common_support_loss_from_candidate": int(common.sum()-mask.sum())})
    csv_write(out / "CLCD_common_support_by_scenario.csv", common_rows)

    temporal = []
    for y in YEARS:
        row = {"year": y}
        for label, mask in masks:
            mean, n = common_mean(eri_paths[y], mask)
            row[f"common_mean_{label}"] = mean
            row[f"n_{label}"] = n
        for thr in THRESHOLDS:
            tag = f"screened_{int(thr*100)}pct"
            row[f"delta_common_mean_{tag}"] = row[f"common_mean_{tag}"]-row["common_mean_candidate"]
        temporal.append(row)
    csv_write(out / "CLCD_common_support_temporal_sensitivity.csv", temporal)

    total = sum(a["candidate_valid_pixels"] for a in annual)
    lines = ["# CLCD native coverage: impact on corrected ERPI", "",
             f"**Candidate:** `{candidate}`", "",
             "This sensitivity test changes spatial support only. It does not change the stored LCN values, "
             "NTL-corrected weights or ERPI values; no screening threshold is silently adopted.", "",
             f"Corrected candidate pixel-years audited: **{total:,}**.",
             f"Corrected candidate strict common pixels: **{int(common.sum()):,}**.", "",
             "## Coverage screening scenarios", ""]
    for thr in THRESHOLDS:
        tag = f"{int(thr*100)}pct"
        n = sum(a[f"removed_{tag}"] for a in annual)
        edge_n = sum(a[f"removed_touching_support_edge_{tag}"] for a in annual)
        max_delta = max(abs(a[f"screened_minus_candidate_mean_{tag}"]) for a in annual)
        cc = int(screened_common[thr].sum())
        max_common_delta = max(abs(a[f"delta_common_mean_screened_{int(thr*100)}pct"]) for a in temporal)
        lines += [f"- Coverage ≥{int(thr*100)}%: remove {n:,} pixel-years ({n/total:.6%}); "
                  f"{edge_n:,} removed cells touch an annual-support edge; "
                  f"strict common support {cc:,} pixels (loss {int(common.sum())-cc:,}); "
                  f"max |annual mean shift| {max_delta:.9f}; "
                  f"max |fixed-support mean shift| {max_common_delta:.9f}."]
    errors = [r["abs_error"] for r in exact_comparisons if r["comparison_valid"]]
    n_unmatched = sum(not r["comparison_valid"] for r in exact_comparisons)
    lines += ["", "## Independent native-area overlap check", ""]
    if prior_exact.is_file():
        n_large = sum(e > 0.005 for e in errors)
        lines += [f"- Compared {len(errors):,} valid sampled cells with exact native-area intersections; "
                  f"max |coverage difference| = {max(errors) if errors else math.nan:.8f}; "
                  f"comparisons >0.005 = {n_large:,}; unavailable full-grid coverage at sampled footprint cells = {n_unmatched:,}."]
        if n_large or n_unmatched:
            lines += ["- **REVIEW:** resolve large warp-versus-exact overlap discrepancies before treating full-grid coverage as area-exact."]
    else:
        lines += ["- The previous `clcd_exact_area_samples.csv` was unavailable. "
                  "Repeat the independent exact-overlap spot check before asserting area accuracy."]
    lines += ["", "## Interpretation", "",
              "- A full-cell source-area cutoff can exclude legitimate partially covered coastline/boundary cells. "
              "Do not treat 90% as a biological threshold or automatically discard these pixels.",
              "- This is an exact support-screening comparison for the current candidate. "
              "If a screened product is to replace it, normalization and entropy weights must be reconsidered and a new version audited.",
              "- The previous stratified exact native-area LCN reconstruction addresses arithmetic agreement only at sampled cells. "
              "It does not prove coverage adequacy in every low-coverage cell.", "",
              "Outputs: `CLCD_coverage_impact_by_year.csv`, `CLCD_common_support_by_scenario.csv`, "
              "`CLCD_common_support_temporal_sensitivity.csv`, `CLCD_below90pct_locations.csv`, "
              "and the coverage/ and common-support GeoTIFFs."]
    (out / "CLCD_COVERAGE_IMPACT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nFINISHED:", out / "CLCD_COVERAGE_IMPACT_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
