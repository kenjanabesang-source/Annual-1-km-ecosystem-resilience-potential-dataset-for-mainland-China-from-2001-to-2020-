# -*- coding: utf-8 -*-
r"""
Targeted full-grid PANDA nighttime-light preprocessing recheck for ERPI 2001-2020.

PURPOSE: Resolve the source audit's NTL ambiguity. The previous sampled-window
warp cannot serve as a faithful counterfactual until the uncleaned full-grid
warp reproduces the archived NTL intermediate. This script runs the original
full-grid warp and a matched source-NoData=-32768 warp with exactly the same
transform/resampling/threads. It reads all existing ERPI products only.

Run:
 D:\conda\python.exe recheck_ERPI_NTL_exact_full_grid.py --root "F:\data\China"
Optional: --years 2001,2010,2020 --save-cleaned

Output:
 <root>/ERI_recalculated_2001_2020/SOURCE_PROCESSING_AUDIT/NTL_EXACT_RECHECK/

It does not alter the original rasters or fix the entropy weights/baselines.
"""

from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

NODATA = -9999.0
FILL = -32768
NTHREADS = 4             # Original 2001-2020 ERPI computation script
TOL = 1e-4
SIGNIFICANT = 1e-2
BLOCK = 1024


def windows(src):
    for r in range(0,src.height,BLOCK):
        for c in range(0,src.width,BLOCK):
            yield Window(c,r,min(BLOCK,src.width-c),min(BLOCK,src.height-r))


def good(a,nodata):
    v=np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):v &= a != nodata
    return v


def warp_ntl(source,target,source_nodata):
    """Match reproject_band_to_grid() and preprocess_ntl() in original code."""
    arr=np.full((target.height,target.width),NODATA,dtype=np.float32)
    reproject(
        source=rasterio.band(source,1),destination=arr,
        src_transform=source.transform,src_crs=source.crs,
        src_nodata=source_nodata,
        dst_transform=target.transform,dst_crs=target.crs,
        dst_nodata=NODATA,resampling=Resampling.bilinear,
        num_threads=NTHREADS,
    )
    v=good(arr,NODATA)
    arr[v & (arr<0)] = 0.0
    return arr


def count_diff(a,b,mask):
    if not np.any(mask):return (0,0,0.,0.)
    d=np.abs(a[mask].astype(np.float64)-b[mask].astype(np.float64))
    return (int(np.count_nonzero(d>TOL)),int(np.count_nonzero(d>SIGNIFICANT)),
            float(d.max()),float(d.mean()))


def calc_hci(z,q01,q99):
    return (1.0-np.clip((np.log1p(np.maximum(z,0.0))-q01) /
                        max(q99-q01,1e-6),0.0,1.0))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(r'F:\data\China'))
    p.add_argument('--years',default='2001-2020',help='2001-2020 or comma-separated years')
    p.add_argument('--save-cleaned',action='store_true',help='Save corrected NTL candidates in NEW audit folder only')
    args=p.parse_args(argv)
    years=list(range(2001,2021)) if args.years=='2001-2020' else [int(s) for s in args.years.split(',')]
    if not years or any(y<2001 or y>2020 for y in years):p.error('Years must be within 2001–2020')

    product=args.root/'ERI_recalculated_2001_2020'
    out=product/'SOURCE_PROCESSING_AUDIT'/'NTL_EXACT_RECHECK'
    out.mkdir(parents=True,exist_ok=True)
    qfile=product/'baselines'/'NTL_pooled_quantiles.json'
    q=json.loads(qfile.read_text(encoding='utf-8')) if qfile.exists() else None
    q01=float(q['log1p_q01']) if q is not None else None
    q99=float(q['log1p_q99']) if q is not None else None
    results=[]
    for year in years:
        srcpath=args.root/'NightLight'/f'PANDA_China_{year}.tif'
        archivedpath=product/'intermediate'/'NTL'/f'NTL_{year}.tif'
        finalpath=product/'ERI'/f'ERI_{year}.tif'
        hcipath=product/'indicators'/f'HCI_{year}.tif'
        for path in (srcpath,archivedpath,finalpath):
            if not path.exists():raise FileNotFoundError(path)
        print(f'[{year}] two native-to-FULL-GRID warps; unchanged and explicit -32768 NoData...',flush=True)
        with rasterio.open(srcpath) as source,rasterio.open(archivedpath) as archived,rasterio.open(finalpath) as final:
            if archived.crs!=final.crs or archived.transform!=final.transform or archived.shape!=final.shape:
                raise ValueError(f'Intermediate/final grid mismatch: {year}')
            if source.nodata is not None and source.nodata!=FILL:
                raise ValueError(f'Unexpected source NoData for {year}: {source.nodata}; inspect before continuing')
            old=warp_ntl(source,final,source.nodata)
            clean=warp_ntl(source,final,FILL)
            if args.save_cleaned:
                profile=final.profile.copy()
                profile.update(dtype='float32',count=1,nodata=NODATA,compress='deflate')
                # Written to the separate audit folder; outside final territorial
                # support remains NoData in the *candidate* saved output.
                saved=out/f'NTL_{year}_cleaned_candidate.tif'
                with rasterio.open(saved,'w',**profile) as dst:
                    for win in windows(final):
                        r,c,h,w=int(win.row_off),int(win.col_off),int(win.height),int(win.width)
                        support=good(final.read(1,window=win),final.nodata)
                        z=clean[r:r+h,c:c+w].copy()
                        z[~support]=NODATA
                        dst.write(z,1,window=win)
            stats=dict(year=year,source_nodata=str(source.nodata),source_fill_assumed=FILL,
                       final_valid_pixels=0,archived_valid_pixels_on_final=0,
                       legacy_archive_mask_mismatch=0,legacy_archive_changed_gt_1e_4=0,
                       legacy_archive_changed_gt_0_01=0,legacy_archive_max_abs_error=0.,
                       legacy_archive_sum_abs_error=0.,legacy_archive_compared_pixels=0,
                       cleaned_invalid_archived_valid=0,cleaned_valid_archived_invalid=0,
                       cleaned_archive_changed_gt_1e_4=0,cleaned_archive_changed_gt_0_01=0,
                       cleaned_archive_max_abs_error=0.,cleaned_archive_compared_pixels=0,
                       archive_HCI_legacy_max_abs_error=0.,archive_HCI_cleaned_changed_gt_1e_4=0,
                       archive_HCI_cleaned_changed_gt_0_01=0,
                       archive_HCI_cleaned_max_abs_change=0.,HCI_compared_pixels=0)
            hc = rasterio.open(hcipath) if (q is not None and hcipath.exists()) else None
            try:
                for win in windows(final):
                    r,c,h,w=int(win.row_off),int(win.col_off),int(win.height),int(win.width)
                    mfinal=good(final.read(1,window=win),final.nodata)
                    if not np.any(mfinal):continue
                    a=archived.read(1,window=win)
                    ov=old[r:r+h,c:c+w]; cv=clean[r:r+h,c:c+w]
                    m_a=good(a,archived.nodata);m_old=good(ov,NODATA);m_clean=good(cv,NODATA)
                    stats['final_valid_pixels']+=int(mfinal.sum())
                    stats['archived_valid_pixels_on_final']+=int(np.count_nonzero(mfinal&m_a))
                    stats['legacy_archive_mask_mismatch']+=int(np.count_nonzero(mfinal&(m_a^m_old)))
                    stats['cleaned_invalid_archived_valid']+=int(np.count_nonzero(mfinal&m_a&~m_clean))
                    stats['cleaned_valid_archived_invalid']+=int(np.count_nonzero(mfinal&~m_a&m_clean))
                    m0=mfinal&m_a&m_old
                    n1,n2,mx,mae=count_diff(a,ov,m0)
                    stats['legacy_archive_changed_gt_1e_4']+=n1
                    stats['legacy_archive_changed_gt_0_01']+=n2
                    stats['legacy_archive_max_abs_error']=max(stats['legacy_archive_max_abs_error'],mx)
                    stats['legacy_archive_sum_abs_error']+=mae*int(m0.sum())
                    stats['legacy_archive_compared_pixels']+=int(m0.sum())
                    m1=mfinal&m_a&m_clean
                    n1,n2,mx,_=count_diff(a,cv,m1)
                    stats['cleaned_archive_changed_gt_1e_4']+=n1
                    stats['cleaned_archive_changed_gt_0_01']+=n2
                    stats['cleaned_archive_max_abs_error']=max(stats['cleaned_archive_max_abs_error'],mx)
                    stats['cleaned_archive_compared_pixels']+=int(m1.sum())
                    if hc is not None:
                        archived_hci=hc.read(1,window=win)
                        mhci=good(archived_hci,hc.nodata)
                        mh=mfinal&mhci&m_old&m_clean
                        if np.any(mh):
                            pred_old=calc_hci(ov[mh],q01,q99)
                            pred_clean=calc_hci(cv[mh],q01,q99)
                            herr=np.abs(archived_hci[mh].astype(float)-pred_old)
                            hd=np.abs(archived_hci[mh].astype(float)-pred_clean)
                            stats['archive_HCI_legacy_max_abs_error']=max(stats['archive_HCI_legacy_max_abs_error'],float(herr.max()))
                            stats['archive_HCI_cleaned_changed_gt_1e_4']+=int(np.count_nonzero(hd>TOL))
                            stats['archive_HCI_cleaned_changed_gt_0_01']+=int(np.count_nonzero(hd>SIGNIFICANT))
                            stats['archive_HCI_cleaned_max_abs_change']=max(stats['archive_HCI_cleaned_max_abs_change'],float(hd.max()))
                            stats['HCI_compared_pixels']+=int(mh.sum())
            finally:
                if hc is not None:hc.close()
            stats['legacy_archive_mae']=(stats.pop('legacy_archive_sum_abs_error') / stats['legacy_archive_compared_pixels']
                                         if stats['legacy_archive_compared_pixels'] else float('nan'))
            if stats['legacy_archive_mask_mismatch']>0 or stats['legacy_archive_max_abs_error']>TOL:
                stats['decision']='REVIEW_ARCHIVED_PREPROCESSING_NOT_REPRODUCED'
            elif (stats['cleaned_invalid_archived_valid']>0 or
                  stats['cleaned_archive_changed_gt_1e_4']>0):
                stats['decision']='CONFIRMED_NATIVE_FILL_AFFECTS_INTERMEDIATE'
            else:
                stats['decision']='NO_MEASURABLE_FILL_EFFECT_ON_FINAL_SUPPORT'
            if hc is not None and stats['archive_HCI_legacy_max_abs_error']>TOL:
                stats['decision']='REVIEW_HCI_NOT_REPRODUCED'
            results.append(stats)
            print(f"  {stats['decision']}; old/archive max={stats['legacy_archive_max_abs_error']:.7g}; "
                  f"clean/archive max={stats['cleaned_archive_max_abs_error']:.7g}; "
                  f"changed >0.01={stats['cleaned_archive_changed_gt_0_01']:,}",flush=True)
        del old,clean
        gc.collect()
        pd.DataFrame(results).to_csv(out/'NTL_exact_full_grid_recheck.csv',index=False)

    result=pd.DataFrame(results)
    report=["# PANDA NTL full-grid preprocessing recheck","",
            "This audit reproduces the original full-grid legacy warp before interpreting the explicit -32768 NoData counterfactual.",
            "Stratified-window counts from the earlier audit are not treated as independent geographic pixel counts.","",
            f"Years checked: {len(result)}; full-grid source warps per year: 2; source fill tested: {FILL}.","",
            "## Yearly decisions",""]
    for _,r in result.iterrows():
        report.append(f"- {int(r.year)} — {r.decision}; archive vs original maximum absolute error "
                      f"{r.legacy_archive_max_abs_error:.7g}; cleaned vs archive maximum "
                      f"{r.cleaned_archive_max_abs_error:.7g}; >0.01 changed target pixels "
                      f"{int(r.cleaned_archive_changed_gt_0_01):,}; cleaned-NoData within final support "
                      f"{int(r.cleaned_invalid_archived_valid):,}.")
    report += ["","## Interpretation","",
               "- If original full-grid warp fails to reproduce archived NTL, identify the exact archived preprocessing first. Do not infer contamination from the earlier sampled-window counterfactual.",
               "- If original warp reproduces archived NTL and explicit -32768 masking changes target pixels, the source fill has a measured effect on NTL. The HCI comparison uses the ORIGINAL stored pooled quantiles only; corrected quantiles and entropy weights must be recomputed before assigning final ERPI impact.",
               "- If masking creates NoData in a final-valid cell, the final ERPI support must also be revisited.",
               "- No file in the existing ERPI product is modified. The optional --save-cleaned outputs are independent candidates, not a corrected final release.",""]
    (out/'NTL_exact_full_grid_report.md').write_text('\n'.join(report),encoding='utf-8')
    print('\nFinished. Upload NTL_exact_full_grid_recheck.csv and NTL_exact_full_grid_report.md',flush=True)

if __name__=='__main__':
    main()
