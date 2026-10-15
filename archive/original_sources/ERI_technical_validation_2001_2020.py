# -*- coding: utf-8 -*-
r"""
Technical validation and sensitivity analysis for the recalculated ERI
======================================================================

This script DOES NOT redo preprocessing. It uses the outputs already produced by:

    F:\data\China\ERI_recalculated_2001_2020

Required existing files:
    indicators\VCI_2001.tif ... VCI_2020.tif
    indicators\LCN_2001.tif ... LCN_2020.tif
    indicators\HCI_2001.tif ... HCI_2020.tif
    indicators\TSI_2001.tif ... TSI_2020.tif
    indicators\MSI_2001.tif ... MSI_2020.tif
    ERI\ERI_2001.tif ... ERI_2020.tif
    entropy_weights_fixed_2001_2020.csv

Outputs:
    technical_validation\
        common_valid_mask_2001_2020.tif
        annual_indicator_contributions_common_mask.csv
        annual_eri_common_mask.csv
        trend_summary.csv
        change_decomposition_2019_2020.csv
        entropy_vs_equal_comparison.csv
        leave_one_out_comparison.csv
        equal_weight\
            ERI_equal_2001.tif ... ERI_equal_2020.tif
        leave_one_out\     (optional, controlled by SAVE_LOO_RASTERS)

Analyses:
1) A fixed common-valid mask across ALL five indicators and ALL 20 years.
2) Annual means of VCI/LCN/HCI/TSI/MSI on the same spatial support.
3) Annual weighted contribution of each indicator.
4) Equal-weight ERI sensitivity:
       ERI_equal = 0.2*(VCI+LCN+HCI+TSI+MSI)
   Full-resolution Pearson r and MAE against entropy-weighted ERI.
   Spearman rho is estimated from a deterministic spatial sample.
5) Leave-one-out sensitivity:
       remove one indicator, renormalize the remaining entropy weights to sum to 1.
   Full-resolution Pearson r and MAE against the full entropy-weighted ERI.
   Spearman rho is estimated from a deterministic spatial sample.
6) 2019 -> 2020 change decomposition by weighted indicator contributions.

No new external data are used.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


# =============================================================================
# 0. CONFIG
# =============================================================================

ROOT = Path(r"F:\data\China")
ERI_ROOT = ROOT / "ERI_recalculated_2001_2020"

INDICATOR_DIR = ERI_ROOT / "indicators"
ERI_DIR = ERI_ROOT / "ERI"
WEIGHTS_CSV = ERI_ROOT / "entropy_weights_fixed_2001_2020.csv"

OUT_DIR = ERI_ROOT / "technical_validation"

YEARS = list(range(2001, 2021))
INDICATORS = ["VCI", "LCN", "HCI", "TSI", "MSI"]

BLOCK = 512

# Save 100 additional leave-one-out rasters only if you actually want them.
SAVE_LOO_RASTERS = False

# Save entropy-minus-equal difference rasters for mapping.
SAVE_EQUAL_DIFF_RASTERS = False

# Deterministic sample cap used only for Spearman rank correlation.
# Pearson and MAE are computed on all common-valid pixels.
SPEARMAN_SAMPLE_MAX = 200_000


# =============================================================================
# 1. UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def indicator_path(indicator: str, year: int) -> Path:
    return INDICATOR_DIR / f"{indicator}_{year}.tif"


def eri_path(year: int) -> Path:
    return ERI_DIR / f"ERI_{year}.tif"


def iter_windows(width, height, block=BLOCK):
    for row in range(0, height, block):
        h = min(block, height - row)
        for col in range(0, width, block):
            w = min(block, width - col)
            yield Window(col, row, w, h)


def read_as_nan(src, win):
    a = src.read(1, window=win).astype(np.float64)
    nodata = src.nodata
    bad = ~np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        bad |= (a == nodata)
    a[bad] = np.nan
    return a


def read_mask(mask_src, win):
    return mask_src.read(1, window=win) == 1


def float_profile_from_ref(ref):
    p = ref.profile.copy()
    p.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    return p


def rank_average_ties(x):
    """
    Average ranks, equivalent to scipy.stats.rankdata(method='average'),
    implemented with pandas to avoid a SciPy dependency.
    """
    return pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)


def spearman_from_samples(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan
    rx = rank_average_ties(x)
    ry = rank_average_ties(y)
    sx = np.std(rx)
    sy = np.std(ry)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


class PairStats:
    """Streaming Pearson correlation and MAE."""
    def __init__(self):
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0
        self.sabs = 0.0

    def update(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size == 0:
            return
        self.n += x.size
        self.sx += float(np.sum(x))
        self.sy += float(np.sum(y))
        self.sxx += float(np.sum(x * x))
        self.syy += float(np.sum(y * y))
        self.sxy += float(np.sum(x * y))
        self.sabs += float(np.sum(np.abs(x - y)))

    def pearson(self):
        if self.n < 3:
            return np.nan
        num = self.n * self.sxy - self.sx * self.sy
        denx = self.n * self.sxx - self.sx * self.sx
        deny = self.n * self.syy - self.sy * self.sy
        den = math.sqrt(max(denx, 0.0) * max(deny, 0.0))
        return np.nan if den == 0 else float(num / den)

    def mae(self):
        return np.nan if self.n == 0 else float(self.sabs / self.n)


class RunningMean:
    def __init__(self):
        self.n = 0
        self.s = 0.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        self.n += x.size
        self.s += float(np.sum(x))

    def mean(self):
        return np.nan if self.n == 0 else float(self.s / self.n)


def deterministic_sample_append(x, y, xs, ys, max_total):
    """
    Add a spatially systematic sample from the current block.
    Total size is capped approximately at max_total.
    """
    if len(xs) > 0:
        current = sum(len(v) for v in xs)
    else:
        current = 0
    remain = max_total - current
    if remain <= 0:
        return

    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return

    # Take at most 2000 values from each block and never exceed remaining cap.
    take = min(2000, remain, x.size)
    step = max(1, x.size // take)
    xx = x[::step][:take]
    yy = y[::step][:take]

    xs.append(xx.astype(np.float32))
    ys.append(yy.astype(np.float32))


# =============================================================================
# 2. INPUT CHECK
# =============================================================================

def validate_inputs():
    missing = []

    if not WEIGHTS_CSV.exists():
        missing.append(WEIGHTS_CSV)

    for y in YEARS:
        if not eri_path(y).exists():
            missing.append(eri_path(y))
        for ind in INDICATORS:
            p = indicator_path(ind, y)
            if not p.exists():
                missing.append(p)

    if missing:
        msg = "\n".join(str(p) for p in missing[:30])
        raise FileNotFoundError(
            f"Missing required outputs ({len(missing)} files). First examples:\n{msg}"
        )


def load_weights():
    df = pd.read_csv(WEIGHTS_CSV)
    weights = dict(zip(df["indicator"], df["weight"]))
    missing = [x for x in INDICATORS if x not in weights]
    if missing:
        raise ValueError(f"Missing indicators in weights CSV: {missing}")
    s = sum(weights[x] for x in INDICATORS)
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"Entropy weights do not sum to 1: {s}")
    return weights


# =============================================================================
# 3. COMMON VALID MASK
# =============================================================================

def build_common_mask(out_dir):
    print("[1/6] Building 2001-2020 common-valid mask ...")

    out_path = out_dir / "common_valid_mask_2001_2020.tif"
    if out_path.exists():
        print("      Common mask already exists; reusing it.")
        return out_path

    ref_path = indicator_path(INDICATORS[0], YEARS[0])
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        width, height = ref.width, ref.height

    # Open all 100 indicator rasters once.
    srcs = {}
    try:
        for y in YEARS:
            for ind in INDICATORS:
                srcs[(y, ind)] = rasterio.open(indicator_path(ind, y))

        with rasterio.open(out_path, "w", **profile) as dst:
            total_common = 0

            for win in tqdm(iter_windows(width, height), desc="Common mask"):
                cm = np.ones((int(win.height), int(win.width)), dtype=bool)

                for y in YEARS:
                    for ind in INDICATORS:
                        src = srcs[(y, ind)]
                        a = src.read(1, window=win)
                        nodata = src.nodata

                        cm &= np.isfinite(a)
                        if nodata is not None and np.isfinite(nodata):
                            cm &= (a != nodata)

                out = cm.astype(np.uint8)
                total_common += int(np.count_nonzero(cm))
                dst.write(out, 1, window=win)

        print(f"      Common-valid pixels: {total_common:,}")

    finally:
        for src in srcs.values():
            src.close()

    return out_path


# =============================================================================
# 4. ANNUAL INDICATOR MEANS + CONTRIBUTIONS + COMMON-MASK ERI
# =============================================================================

def annual_indicator_statistics(common_mask_path, out_dir, weights):
    print("[2/6] Computing annual indicator means and weighted contributions ...")

    rows = []
    eri_rows = []

    with rasterio.open(common_mask_path) as msrc:
        width, height = msrc.width, msrc.height

        for y in YEARS:
            srcs = {ind: rasterio.open(indicator_path(ind, y)) for ind in INDICATORS}
            esrc = rasterio.open(eri_path(y))

            means = {ind: RunningMean() for ind in INDICATORS}
            eri_mean = RunningMean()

            try:
                for win in iter_windows(width, height):
                    cm = read_mask(msrc, win)

                    if not np.any(cm):
                        continue

                    for ind in INDICATORS:
                        a = read_as_nan(srcs[ind], win)
                        means[ind].update(a[cm])

                    e = read_as_nan(esrc, win)
                    eri_mean.update(e[cm])

            finally:
                for s in srcs.values():
                    s.close()
                esrc.close()

            row = {"year": y}
            for ind in INDICATORS:
                mu = means[ind].mean()
                row[f"{ind}_mean"] = mu
                row[f"{ind}_weighted_contribution"] = weights[ind] * mu

            contribution_sum = sum(
                row[f"{ind}_weighted_contribution"] for ind in INDICATORS
            )

            row["contribution_sum"] = contribution_sum
            rows.append(row)

            eri_rows.append({
                "year": y,
                "ERI_entropy_common_mask_mean": eri_mean.mean(),
                "sum_of_weighted_indicator_means": contribution_sum,
            })

            print(
                f"      {y}: ERI(common mask)={eri_mean.mean():.6f}"
            )

    stats = pd.DataFrame(rows)
    stats.to_csv(
        out_dir / "annual_indicator_contributions_common_mask.csv",
        index=False
    )

    eri_stats = pd.DataFrame(eri_rows)
    eri_stats.to_csv(
        out_dir / "annual_eri_common_mask.csv",
        index=False
    )

    return stats, eri_stats


# =============================================================================
# 5. EQUAL-WEIGHT SENSITIVITY
# =============================================================================

def equal_weight_analysis(common_mask_path, out_dir):
    print("[3/6] Equal-weight sensitivity analysis ...")

    eq_dir = ensure_dir(out_dir / "equal_weight")
    diff_dir = ensure_dir(out_dir / "entropy_minus_equal") if SAVE_EQUAL_DIFF_RASTERS else None

    rows = []

    with rasterio.open(common_mask_path) as msrc:
        width, height = msrc.width, msrc.height

        for y in YEARS:
            srcs = {ind: rasterio.open(indicator_path(ind, y)) for ind in INDICATORS}
            esrc = rasterio.open(eri_path(y))

            profile = float_profile_from_ref(esrc)
            equal_path = eq_dir / f"ERI_equal_{y}.tif"

            stats = PairStats()
            equal_mean = RunningMean()
            entropy_mean = RunningMean()
            xs, ys = [], []

            with rasterio.open(equal_path, "w", **profile) as edst:
                ddst = None
                if SAVE_EQUAL_DIFF_RASTERS:
                    ddst = rasterio.open(
                        diff_dir / f"ERI_entropy_minus_equal_{y}.tif",
                        "w",
                        **profile
                    )

                try:
                    for win in tqdm(iter_windows(width, height), desc=f"Equal {y}", leave=False):
                        cm = read_mask(msrc, win)
                        out = np.full(
                            (int(win.height), int(win.width)),
                            profile["nodata"],
                            dtype=np.float32
                        )

                        if np.any(cm):
                            arrs = [read_as_nan(srcs[ind], win) for ind in INDICATORS]
                            stack = np.stack(arrs, axis=0)
                            eq = np.nanmean(stack, axis=0)

                            entropy = read_as_nan(esrc, win)

                            # Common mask is already strict; still guard finite values.
                            v = cm & np.isfinite(eq) & np.isfinite(entropy)

                            out[v] = eq[v].astype(np.float32)

                            stats.update(entropy[v], eq[v])
                            entropy_mean.update(entropy[v])
                            equal_mean.update(eq[v])
                            deterministic_sample_append(
                                entropy[v], eq[v], xs, ys, SPEARMAN_SAMPLE_MAX
                            )

                            if ddst is not None:
                                d = np.full_like(out, profile["nodata"], dtype=np.float32)
                                d[v] = (entropy[v] - eq[v]).astype(np.float32)
                                ddst.write(d, 1, window=win)

                        edst.write(out, 1, window=win)

                finally:
                    if ddst is not None:
                        ddst.close()

            for s in srcs.values():
                s.close()
            esrc.close()

            sx = np.concatenate(xs) if xs else np.array([])
            sy = np.concatenate(ys) if ys else np.array([])

            rows.append({
                "year": y,
                "n_common_pixels": stats.n,
                "entropy_mean": entropy_mean.mean(),
                "equal_mean": equal_mean.mean(),
                "pearson_r_full": stats.pearson(),
                "spearman_rho_sample": spearman_from_samples(sx, sy),
                "spearman_sample_n": len(sx),
                "MAE_full": stats.mae(),
                "mean_difference_entropy_minus_equal":
                    entropy_mean.mean() - equal_mean.mean(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "entropy_vs_equal_comparison.csv", index=False)
    return df


# =============================================================================
# 6. LEAVE-ONE-OUT SENSITIVITY
# =============================================================================

def leave_one_out_analysis(common_mask_path, out_dir, weights):
    print("[4/6] Leave-one-out sensitivity analysis ...")

    loo_dir = ensure_dir(out_dir / "leave_one_out")
    rows = []

    with rasterio.open(common_mask_path) as msrc:
        width, height = msrc.width, msrc.height

        for y in YEARS:
            srcs = {ind: rasterio.open(indicator_path(ind, y)) for ind in INDICATORS}
            esrc = rasterio.open(eri_path(y))
            profile = float_profile_from_ref(esrc)

            # One stats accumulator per omitted indicator.
            stats = {omit: PairStats() for omit in INDICATORS}
            means = {omit: RunningMean() for omit in INDICATORS}
            samples_x = {omit: [] for omit in INDICATORS}
            samples_y = {omit: [] for omit in INDICATORS}

            loo_dsts = {}
            if SAVE_LOO_RASTERS:
                for omit in INDICATORS:
                    od = ensure_dir(loo_dir / f"without_{omit}")
                    loo_dsts[omit] = rasterio.open(
                        od / f"ERI_without_{omit}_{y}.tif",
                        "w",
                        **profile
                    )

            try:
                for win in tqdm(iter_windows(width, height), desc=f"LOO {y}", leave=False):
                    cm = read_mask(msrc, win)
                    if not np.any(cm):
                        if SAVE_LOO_RASTERS:
                            blank = np.full(
                                (int(win.height), int(win.width)),
                                profile["nodata"],
                                dtype=np.float32
                            )
                            for dst in loo_dsts.values():
                                dst.write(blank, 1, window=win)
                        continue

                    arrs = {ind: read_as_nan(srcs[ind], win) for ind in INDICATORS}
                    entropy = read_as_nan(esrc, win)

                    # Common mask + finite guard.
                    base_valid = cm & np.isfinite(entropy)
                    for ind in INDICATORS:
                        base_valid &= np.isfinite(arrs[ind])

                    for omit in INDICATORS:
                        remaining = [ind for ind in INDICATORS if ind != omit]
                        denom = sum(weights[ind] for ind in remaining)

                        loo = np.zeros_like(entropy, dtype=np.float64)
                        for ind in remaining:
                            loo += (weights[ind] / denom) * arrs[ind]

                        v = base_valid & np.isfinite(loo)

                        stats[omit].update(entropy[v], loo[v])
                        means[omit].update(loo[v])
                        deterministic_sample_append(
                            entropy[v], loo[v],
                            samples_x[omit], samples_y[omit],
                            SPEARMAN_SAMPLE_MAX
                        )

                        if SAVE_LOO_RASTERS:
                            out = np.full(
                                entropy.shape,
                                profile["nodata"],
                                dtype=np.float32
                            )
                            out[v] = loo[v].astype(np.float32)
                            loo_dsts[omit].write(out, 1, window=win)

            finally:
                if SAVE_LOO_RASTERS:
                    for dst in loo_dsts.values():
                        dst.close()

                for s in srcs.values():
                    s.close()
                esrc.close()

            for omit in INDICATORS:
                sx = np.concatenate(samples_x[omit]) if samples_x[omit] else np.array([])
                sy = np.concatenate(samples_y[omit]) if samples_y[omit] else np.array([])

                rows.append({
                    "year": y,
                    "omitted_indicator": omit,
                    "n_common_pixels": stats[omit].n,
                    "leave_one_out_mean": means[omit].mean(),
                    "pearson_r_full": stats[omit].pearson(),
                    "spearman_rho_sample": spearman_from_samples(sx, sy),
                    "spearman_sample_n": len(sx),
                    "MAE_full": stats[omit].mae(),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "leave_one_out_comparison.csv", index=False)
    return df


# =============================================================================
# 7. TREND AND 2019->2020 CHANGE DIAGNOSTICS
# =============================================================================

def trend_and_change_tables(stats_df, eri_df, equal_df, loo_df, out_dir, weights):
    print("[5/6] Writing trend and 2019->2020 change diagnostics ...")

    years = np.array(YEARS, dtype=np.float64)
    trend_rows = []

    def add_trend(name, values):
        values = np.asarray(values, dtype=np.float64)
        m = np.isfinite(values)
        if np.count_nonzero(m) < 3:
            slope = r = np.nan
        else:
            slope = float(np.polyfit(years[m], values[m], 1)[0])
            r = float(np.corrcoef(years[m], values[m])[0, 1])
        trend_rows.append({
            "series": name,
            "linear_slope_per_year": slope,
            "pearson_r_with_year": r,
            "first_year_value": float(values[0]),
            "last_year_value": float(values[-1]),
            "change_2001_2020": float(values[-1] - values[0]),
        })

    add_trend(
        "ERI_entropy_common_mask",
        eri_df["ERI_entropy_common_mask_mean"].to_numpy()
    )
    add_trend(
        "ERI_equal_common_mask",
        equal_df["equal_mean"].to_numpy()
    )

    for ind in INDICATORS:
        add_trend(ind, stats_df[f"{ind}_mean"].to_numpy())
        add_trend(
            f"{ind}_weighted_contribution",
            stats_df[f"{ind}_weighted_contribution"].to_numpy()
        )

    # Leave-one-out annual means by omitted indicator.
    for omit in INDICATORS:
        z = (
            loo_df[loo_df["omitted_indicator"] == omit]
            .sort_values("year")["leave_one_out_mean"]
            .to_numpy()
        )
        add_trend(f"ERI_without_{omit}", z)

    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(out_dir / "trend_summary.csv", index=False)

    # 2019 -> 2020 decomposition.
    r18 = stats_df.loc[stats_df["year"] == 2019].iloc[0]
    r19 = stats_df.loc[stats_df["year"] == 2020].iloc[0]

    change_rows = []
    for ind in INDICATORS:
        m18 = float(r18[f"{ind}_mean"])
        m19 = float(r19[f"{ind}_mean"])
        c18 = float(r18[f"{ind}_weighted_contribution"])
        c19 = float(r19[f"{ind}_weighted_contribution"])

        change_rows.append({
            "indicator": ind,
            "weight": weights[ind],
            "mean_2019": m18,
            "mean_2020": m19,
            "indicator_change_2020_minus_2019": m19 - m18,
            "weighted_contribution_2019": c18,
            "weighted_contribution_2020": c19,
            "weighted_change_2020_minus_2019": c19 - c18,
        })

    change_df = pd.DataFrame(change_rows)
    total_change = float(change_df["weighted_change_2020_minus_2019"].sum())
    change_df["share_of_total_ERI_change"] = np.where(
        abs(total_change) > 1e-12,
        change_df["weighted_change_2020_minus_2019"] / total_change,
        np.nan
    )
    change_df.to_csv(out_dir / "change_decomposition_2019_2020.csv", index=False)

    return trend_df, change_df


# =============================================================================
# 8. SUMMARY
# =============================================================================

def write_validation_summary(equal_df, loo_df, trend_df, change_df, out_dir):
    print("[6/6] Validation summary ...")

    summary_lines = []

    eq_r = equal_df["pearson_r_full"]
    eq_s = equal_df["spearman_rho_sample"]
    eq_mae = equal_df["MAE_full"]

    summary_lines.append("Entropy vs equal-weight sensitivity (20 annual maps):")
    summary_lines.append(
        f"  Pearson r full-resolution: mean={eq_r.mean():.6f}, "
        f"min={eq_r.min():.6f}, max={eq_r.max():.6f}"
    )
    summary_lines.append(
        f"  Spearman rho sampled: mean={eq_s.mean():.6f}, "
        f"min={eq_s.min():.6f}, max={eq_s.max():.6f}"
    )
    summary_lines.append(
        f"  MAE full-resolution: mean={eq_mae.mean():.6f}, "
        f"min={eq_mae.min():.6f}, max={eq_mae.max():.6f}"
    )

    summary_lines.append("")
    summary_lines.append("Leave-one-out sensitivity by omitted indicator:")

    for omit in INDICATORS:
        z = loo_df[loo_df["omitted_indicator"] == omit]
        summary_lines.append(
            f"  without {omit}: "
            f"mean Pearson r={z['pearson_r_full'].mean():.6f}, "
            f"mean sampled Spearman rho={z['spearman_rho_sample'].mean():.6f}, "
            f"mean MAE={z['MAE_full'].mean():.6f}"
        )

    summary_lines.append("")
    summary_lines.append("2019 -> 2020 weighted contribution changes:")
    for _, r in change_df.iterrows():
        summary_lines.append(
            f"  {r['indicator']}: {r['weighted_change_2020_minus_2019']:+.6f}"
        )

    p = out_dir / "validation_summary.txt"
    p.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSummary saved to: {p}")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("ERI technical validation and sensitivity analysis (2001-2020)")
    print("=" * 80)

    validate_inputs()
    weights = load_weights()
    out_dir = ensure_dir(OUT_DIR)

    print("\nEntropy weights:")
    for k in INDICATORS:
        print(f"  {k}: {weights[k]:.6f}")

    common_mask = build_common_mask(out_dir)

    stats_df, eri_df = annual_indicator_statistics(
        common_mask, out_dir, weights
    )

    equal_df = equal_weight_analysis(
        common_mask, out_dir
    )

    loo_df = leave_one_out_analysis(
        common_mask, out_dir, weights
    )

    trend_df, change_df = trend_and_change_tables(
        stats_df, eri_df, equal_df, loo_df, out_dir, weights
    )

    write_validation_summary(
        equal_df, loo_df, trend_df, change_df, out_dir
    )

    print("\nFinished.")
    print(f"Outputs: {out_dir}")
    print("\nMost important files to send back for interpretation:")
    print("  annual_indicator_contributions_common_mask.csv")
    print("  annual_eri_common_mask.csv")
    print("  entropy_vs_equal_comparison.csv")
    print("  leave_one_out_comparison.csv")
    print("  trend_summary.csv")
    print("  change_decomposition_2019_2020.csv")
    print("  validation_summary.txt")


if __name__ == "__main__":
    main()
