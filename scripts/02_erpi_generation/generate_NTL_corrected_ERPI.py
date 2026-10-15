# -*- coding: utf-8 -*-
r"""
Safely create a 2001–2020 ERPI candidate corrected for PANDA NTL -32768 fill.

Based on the uploaded original 2000–2019 script and its 2001–2020 counterpart.
The confirmed change is masking PANDA source -32768 BEFORE bilinear reprojection.
All former ERPI rasters are read-only; no original output is overwritten.

Typical Windows invocation:
  D:\conda\python.exe ERPI_2001_2020_NTL_corrected_candidate.py --root "F:\data\China"

To resume a partial candidate after interruption, repeat with --resume.
Outputs under F:\data\China\ERI_recalculated_2001_2020_NTLfix_candidate_v1

The existing VCI/LCN/TSI/MSI indicator files are reused without recalculating
unrelated upstream sources. They already embody the original common-valid mask.
Thus the corrected product deliberately RETAINS the original valid footprint or
shrinks it where corrected NTL becomes NoData; it cannot add new pixels. This is
an explicit conservative support choice, not a claim that new-pixel inputs are
absent. Its rationale and counts are logged.

LCN is held fixed; the separate CLCD low-source-coverage question remains open.
Do not claim this script resolves CLCD coverage or NDVI source coding questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

YEARS = tuple(range(2001, 2021))
OTHER = ('VCI', 'LCN', 'TSI', 'MSI')
NAMES = ('VCI', 'LCN', 'HCI', 'TSI', 'MSI')
FILL = -32768.0
NODATA = -9999.0
BLOCK = 512
THREADS = 4
NTL_SAMPLES_PER_YEAR = 250_000
EPS = 1e-6
OUTPUT_NAME = 'ERI_recalculated_2001_2020_NTLfix_candidate_v1'
ORIG_NAME = 'ERI_recalculated_2001_2020'


def win_iter(width, height, step=BLOCK):
    for r in range(0, height, step):
        for c in range(0, width, step):
            yield Window(c, r, min(step, width-c), min(step, height-r))


def good(a, nodata):
    m = np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        m &= a != nodata
    return m


def jdump(data, p):
    tmp = p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, p)


def write_csv_atomic(df, p):
    tmp = p.with_name(p.name+'.partial')
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def sha256(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def exact_grid(src, ref, name):
    if src.crs != ref.crs or src.transform != ref.transform or src.width != ref.width or src.height != ref.height or src.count != 1:
        raise ValueError(f'Grid or band count mismatch: {name}')


def new_profile(ref):
    p=ref.profile.copy()
    p.update(driver='GTiff', count=1, dtype='float32', nodata=NODATA,
             tiled=True, blockxsize=512, blockysize=512, compress='deflate', BIGTIFF='IF_SAFER')
    return p


def atomic_tiff(path, ref, writer):
    """A partial file is never regarded as a finished output."""
    part=path.with_name(path.stem+'.partial.tif')
    if part.exists():
        part.unlink()
    try:
        with rasterio.open(part, 'w', **new_profile(ref)) as dst:
            writer(dst)
        os.replace(part, path)
    except BaseException:
        if part.exists():
            part.unlink()
        raise


def signature(path):
    stat=path.stat()
    return {'path':str(path.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def source_inputs(root, orig):
    # Exact input paths are derived from the uploaded main workflow.
    paths={}
    for y in YEARS:
        paths[f'NTL_source_{y}']=root/'NightLight'/f'PANDA_China_{y}.tif'
        paths[f'NTL_old_{y}']=orig/'intermediate'/'NTL'/f'NTL_{y}.tif'
        paths[f'ERPI_old_{y}']=orig/'ERI'/f'ERI_{y}.tif'
        for ind in OTHER:
            paths[f'{ind}_old_{y}']=orig/'indicators'/f'{ind}_{y}.tif'
    paths['old_weights']=orig/'entropy_weights_fixed_2001_2020.csv'
    paths['old_pooled_ntl_quantiles']=orig/'baselines'/'NTL_pooled_quantiles.json'
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required inputs missing; cannot safely reconstruct candidate:\n'+'\n'.join(missing[:50]))
    return paths


def validate_previous_audit(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f'Previous full-grid exact recheck CSV is required:\n{csv_path}\nUse --recheck-csv if it is elsewhere.')
    d=pd.read_csv(csv_path)
    needed={'year','legacy_archive_mask_mismatch','legacy_archive_max_abs_error','decision'}
    if not needed.issubset(d.columns) or set(d.year.astype(int)) != set(YEARS):
        raise ValueError('Exact full-grid recheck has incomplete/malformed 2001–2020 records')
    if d['legacy_archive_mask_mismatch'].fillna(1).astype(int).sum() != 0 or d['legacy_archive_max_abs_error'].fillna(1).max() > 1e-6:
        raise ValueError('Legacy full-grid reprojection did not reproduce archived inputs; aborting correction')
    if not d['decision'].astype(str).str.startswith('CONFIRMED_NATIVE_FILL_AFFECTS_INTERMEDIATE').all():
        raise ValueError('Unexpected previous recheck decision; inspect CSV before proceeding')
    return d


def establish_candidate(root, out, manifest, resume):
    manifest_path=out/'CORRECTION_INPUT_MANIFEST.json'
    if out.exists():
        if not resume:
            raise FileExistsError(f'Candidate folder already exists: {out}\nUse --resume ONLY for the same unchanged inputs, or choose a different --output.')
        if not manifest_path.is_file():
            raise RuntimeError('Cannot resume without original input manifest')
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old != manifest:
            raise RuntimeError('Input files, settings or audit record changed since candidate began; refusing unsafe resume')
    else:
        out.mkdir(parents=True, exist_ok=False)
        jdump(manifest, manifest_path)
    for d in ('intermediate/NTL','indicators','ERI','QA'):
        (out/d).mkdir(parents=True, exist_ok=True)


def check_complete_raster(p, ref):
    if not p.is_file():
        return False
    with rasterio.open(p) as f:
        exact_grid(f,ref,str(p))
        if f.nodata != NODATA or f.dtypes[0] != 'float32':
            raise ValueError(f'Wrong encoding in completed candidate {p}')
    return True


def cleaned_ntl_one(src_path, archived_path, out_path, ref, stats):
    """Full-grid warp exactly as original except source -32768 is masked."""
    with rasterio.open(src_path) as source, rasterio.open(archived_path) as archived:
        exact_grid(archived,ref,str(archived_path))
        if source.crs is None or source.count<1:
            raise ValueError(f'Invalid NTL source metadata {src_path}')
        if source.nodata is not None and source.nodata != FILL:
            raise ValueError(f'Unexpected NTL source NoData {source.nodata} for {src_path}')
        arr=np.full((ref.height,ref.width), NODATA,dtype=np.float32)
        reproject(source=rasterio.band(source,1),destination=arr,
                  src_transform=source.transform,src_crs=source.crs,
                  src_nodata=FILL,dst_transform=ref.transform,dst_crs=ref.crs,
                  dst_nodata=NODATA,resampling=Resampling.bilinear,num_threads=THREADS)
        arr[~np.isfinite(arr)]=NODATA
        m=good(arr,NODATA)
        arr[m & (arr<0)]=0.0  # retain original post-warp treatment of other negatives
        # Original NTL includes original study boundary; preserve its domain.
        count={'year':int(stats),'old_ntl_valid':0,'corrected_ntl_valid':0,
               'old_valid_now_missing':0,'cleaned_values_changed_gt_0_01':0,
               'new_valid_outside_old_removed':0}
        def writer(dst):
            for win in win_iter(ref.width,ref.height):
                r,c,h,w=map(int,(win.row_off,win.col_off,win.height,win.width))
                old=archived.read(1,window=win)
                o=good(old,archived.nodata)
                v=arr[r:r+h,c:c+w].copy()
                m0=good(v,NODATA)
                count['old_ntl_valid']+=int(o.sum())
                count['old_valid_now_missing']+=int(np.count_nonzero(o & ~m0))
                count['new_valid_outside_old_removed']+=int(np.count_nonzero(~o & m0))
                cmp=o & m0
                count['cleaned_values_changed_gt_0_01']+=int(np.count_nonzero(np.abs(old[cmp].astype(float)-v[cmp].astype(float))>0.01))
                v[~o]=NODATA
                count['corrected_ntl_valid']+=int(good(v,NODATA).sum())
                dst.write(v,1,window=win)
        atomic_tiff(out_path,ref,writer)
        return count


def streaming_regular_sample(src,max_n):
    """Exactly match original regular_sample(): concatenate valid block arrays,
    sample every floor(N/max_n)-th element, cap to max_n, then log1p.
    Avoid allocating full annual raster observations at once.
    """
    n=0
    for _,win in src.block_windows(1):
        a=src.read(1,window=win)
        n+=int(good(a,src.nodata).sum())
    if n==0:
        return np.array([],dtype=np.float64)
    if n<=max_n:
        picked=np.arange(n,dtype=np.int64)
    else:
        picked=np.arange(0,n,max(1,n//max_n),dtype=np.int64)[:max_n]
    out=np.empty(len(picked),dtype=np.float64)
    total=0; written=0
    for _,win in src.block_windows(1):
        a=src.read(1,window=win)
        x=a[good(a,src.nodata)]
        start=int(np.searchsorted(picked,total,side='left'))
        stop=int(np.searchsorted(picked,total+x.size,side='left'))
        if stop>start:
            out[start:stop]=x[picked[start:stop]-total]
            written+=(stop-start)
        total+=x.size
    if written != len(picked) or total != n:
        raise RuntimeError('Streaming pooled-NTL sampler failed exact count check')
    return out


def calculate_quantiles(out):
    quant_path=out/'baselines'/'NTL_pooled_quantiles.json'
    quant_path.parent.mkdir(parents=True,exist_ok=True)
    if quant_path.is_file():
        q=json.loads(quant_path.read_text(encoding='utf-8'))
        if not (np.isfinite(q['log1p_q01']) and np.isfinite(q['log1p_q99']) and q['log1p_q99']>q['log1p_q01']):
            raise ValueError('Saved pooled NTL quantiles invalid')
        return q
    vals=[]
    for y in YEARS:
        with rasterio.open(out/'intermediate'/'NTL'/f'NTL_{y}.tif') as s:
            x=streaming_regular_sample(s,NTL_SAMPLES_PER_YEAR)
            x=x[x>=0]
            vals.append(np.log1p(x))
    x=np.concatenate(vals)
    q={'log1p_q01':float(np.percentile(x,1.0)),
       'log1p_q99':float(np.percentile(x,99.0)),
       'sample_count':int(len(x)),
       'method':'original block-major regular sample, max 250000/year, pooled log1p 1/99 percentile'}
    if not (q['log1p_q99']>q['log1p_q01']):
        raise ValueError('Corrected nighttime-light pooled quantiles degenerate')
    jdump(q,quant_path)
    return q


def make_hci_one(y,orig,out,q,ref):
    paths={ind:orig/'indicators'/f'{ind}_{y}.tif' for ind in OTHER}
    with rasterio.open(out/'intermediate'/'NTL'/f'NTL_{y}.tif') as ntl, \
         rasterio.open(paths['VCI']) as vci, rasterio.open(paths['LCN']) as lcn, \
         rasterio.open(paths['TSI']) as tsi, rasterio.open(paths['MSI']) as msi:
        srcs=(vci,lcn,tsi,msi)
        for s in (*srcs,ntl):exact_grid(s,ref,s.name)
        n={'year':y,'valid_hci':0,'cleaned_ntl_valid_but_other_components_invalid':0}
        def writer(dst):
            for win in win_iter(ref.width,ref.height):
                a=ntl.read(1,window=win)
                m=good(a,ntl.nodata)
                for s in srcs:
                    x=s.read(1,window=win)
                    m &= good(x,s.nodata)
                z=np.full(a.shape,NODATA,dtype=np.float32)
                if m.any():
                    zz=np.log1p(np.maximum(a[m].astype(float),0))
                    zz=np.clip((zz-q['log1p_q01'])/max(q['log1p_q99']-q['log1p_q01'],EPS),0,1)
                    z[m]=(1-zz).astype(np.float32)
                n['valid_hci']+=int(m.sum())
                n['cleaned_ntl_valid_but_other_components_invalid']+=int(np.count_nonzero(good(a,ntl.nodata)&~m))
                dst.write(z,1,window=win)
        atomic_tiff(out/'indicators'/f'HCI_{y}.tif',ref,writer)
        return n


def entropy_weights(orig,out,ref):
    weight_file=out/'entropy_weights_fixed_2001_2020.csv'
    if weight_file.exists():
        d=pd.read_csv(weight_file)
        return dict(zip(d['indicator'].astype(str),d['weight'].astype(float)))
    sx={i:0.0 for i in NAMES}; sxlog={i:0.0 for i in NAMES}; n={i:0 for i in NAMES}
    for y in YEARS:
        srcs={ind:rasterio.open((out if ind=='HCI' else orig)/'indicators'/f'{ind}_{y}.tif') for ind in NAMES}
        try:
            for s in srcs.values():exact_grid(s,ref,s.name)
            for win in win_iter(ref.width,ref.height):
                a={ind:s.read(1,window=win) for ind,s in srcs.items()}
                cm=np.ones_like(a['VCI'],dtype=bool)
                for ind in NAMES:cm &= good(a[ind],srcs[ind].nodata)
                if not cm.any():continue
                for ind in NAMES:
                    x=np.clip(a[ind][cm].astype(np.float64),0,1)
                    sx[ind]+=float(x.sum())
                    pos=x>0
                    if pos.any():sxlog[ind]+=float(np.sum(x[pos]*np.log(x[pos])))
                    n[ind]+=int(len(x))
        finally:
            for s in srcs.values():s.close()
    entropy={}; divergence={}
    for ind in NAMES:
        S,N=sx[ind],n[ind]
        if S<=0 or N<=1:
            e=1.0
        else:
            e=-(sxlog[ind]/S-math.log(S))/math.log(N)
            e=float(np.clip(e,0,1))
        entropy[ind]=e;divergence[ind]=max(0,1-e)
    ds=sum(divergence.values())
    if ds<=0:raise RuntimeError('All five pooled entropy divergences are zero')
    wt={i:divergence[i]/ds for i in NAMES}
    df=pd.DataFrame([{'indicator':i,'entropy':entropy[i],'divergence':divergence[i],
                      'weight':wt[i],'n_valid':n[i]} for i in NAMES])
    write_csv_atomic(df,weight_file)
    return wt


def make_erpi_one(y,orig,out,weights,ref):
    sources={i:rasterio.open((out if i=='HCI' else orig)/'indicators'/f'{i}_{y}.tif') for i in NAMES}
    old=rasterio.open(orig/'ERI'/f'ERI_{y}.tif')
    try:
        for s in sources.values():exact_grid(s,ref,s.name)
        exact_grid(old,ref,old.name)
        stats={'year':y,'valid_old':0,'valid_new':0,'removed_pixel_years':0,
               'unexpected_new_pixels':0,'changed_gt_1e_6':0,
               'changed_gt_0_001':0,'changed_gt_0_01':0,
               'max_abs_change':0.0,'sum_abs_change':0.0,
               'sum_new':0.0,'sum_old_on_common':0.0,'sum_new_on_common':0.0,
               'common_comparison_pixels':0,
               'min_new':np.inf,'max_new':-np.inf}
        def writer(dst):
            for win in win_iter(ref.width,ref.height):
                a={i:s.read(1,window=win) for i,s in sources.items()}
                cm=np.ones_like(a['VCI'],dtype=bool)
                for i in NAMES:cm &= good(a[i],sources[i].nodata)
                r=np.full(cm.shape,NODATA,dtype=np.float32)
                if cm.any():
                    v=np.zeros(int(cm.sum()),dtype=np.float64)
                    for i in NAMES:v+=weights[i]*a[i][cm].astype(np.float64)
                    if not (np.all(np.isfinite(v)) and np.min(v)>=-1e-6 and np.max(v)<=1+1e-6):
                        raise RuntimeError(f'Invalid un-clipped ERPI in {y}; aborting')
                    v=np.clip(v,0,1)
                    r[cm]=v.astype(np.float32)
                    stats['sum_new']+=float(v.sum())
                    stats['min_new']=min(stats['min_new'],float(v.min()))
                    stats['max_new']=max(stats['max_new'],float(v.max()))
                old_a=old.read(1,window=win)
                mo=good(old_a,old.nodata)
                stats['valid_old']+=int(mo.sum());stats['valid_new']+=int(cm.sum())
                stats['removed_pixel_years']+=int(np.count_nonzero(mo & ~cm))
                new_pixels=int(np.count_nonzero(~mo & cm))
                stats['unexpected_new_pixels']+=new_pixels
                if new_pixels:raise RuntimeError(f'Unexpected expansion beyond old final ERPI support in {y}')
                common=mo & cm
                if common.any():
                    oldv=old_a[common].astype(float)
                    newv=r[common].astype(float)
                    d=np.abs(newv-oldv)
                    stats['common_comparison_pixels']+=int(d.size)
                    stats['sum_old_on_common']+=float(oldv.sum())
                    stats['sum_new_on_common']+=float(newv.sum())
                    stats['sum_abs_change']+=float(d.sum())
                    stats['max_abs_change']=max(stats['max_abs_change'],float(d.max()))
                    stats['changed_gt_1e_6']+=int(np.count_nonzero(d>1e-6))
                    stats['changed_gt_0_001']+=int(np.count_nonzero(d>0.001))
                    stats['changed_gt_0_01']+=int(np.count_nonzero(d>0.01))
                dst.write(r,1,window=win)
        atomic_tiff(out/'ERI'/f'ERI_{y}.tif',ref,writer)
        stats['mean_new']=stats['sum_new']/stats['valid_new'] if stats['valid_new'] else np.nan
        stats['mean_abs_change_on_common']=(stats['sum_abs_change']/stats['common_comparison_pixels']
                                            if stats['common_comparison_pixels'] else np.nan)
        stats['old_mean_on_common']=(stats['sum_old_on_common']/stats['common_comparison_pixels']
                                     if stats['common_comparison_pixels'] else np.nan)
        stats['new_mean_on_common']=(stats['sum_new_on_common']/stats['common_comparison_pixels']
                                     if stats['common_comparison_pixels'] else np.nan)
        return stats
    finally:
        old.close()
        for s in sources.values():s.close()


def copy_four_components(orig,out,ref,resume):
    """Make candidate self-contained and compatible with final-raster QA script."""
    for y in YEARS:
        for ind in OTHER:
            src=orig/'indicators'/f'{ind}_{y}.tif'
            dst=out/'indicators'/f'{ind}_{y}.tif'
            if dst.exists():
                # Because the file can be large, check both length and SHA256.
                if dst.stat().st_size != src.stat().st_size or sha256(dst)!=sha256(src):
                    raise RuntimeError(f'Bundled original indicator differs from source: {dst}')
                continue
            if not resume and dst.exists():
                raise FileExistsError(dst)
            tmp=dst.with_name(dst.name+'.partial')
            if tmp.exists():tmp.unlink()
            try:
                shutil.copy2(src,tmp)
                if tmp.stat().st_size != src.stat().st_size or sha256(tmp)!=sha256(src):
                    raise RuntimeError(f'Copy checksum mismatch: {src}')
                os.replace(tmp,dst)
                with rasterio.open(dst) as ds:exact_grid(ds,ref,str(dst))
            except BaseException:
                if tmp.exists():tmp.unlink()
                raise


def run(args):
    root=args.root
    orig=root/ORIG_NAME
    out=args.output or (root/OUTPUT_NAME)
    if out.resolve()==orig.resolve() or orig.resolve() in out.resolve().parents:
        raise ValueError('Output must be a separate directory, not the original ERPI folder or one of its children')
    inputs=source_inputs(root,orig)
    audit_csv=args.recheck_csv or orig/'SOURCE_PROCESSING_AUDIT'/'NTL_EXACT_RECHECK'/'NTL_exact_full_grid_recheck.csv'
    audited=validate_previous_audit(audit_csv)
    with rasterio.open(orig/'ERI'/'ERI_2001.tif') as ref:
        for y in YEARS:
            with rasterio.open(inputs[f'ERPI_old_{y}']) as s:exact_grid(s,ref,s.name)
        config={'start_year':2001,'end_year':2020,'fill_value':FILL,
                'no_data':NODATA,'resampling':'bilinear','number_of_threads':THREADS,
                'old_footprint_preserved':True,'NTL_sample_per_year':NTL_SAMPLES_PER_YEAR,
                'CLCD_low_coverage_policy':'reuse existing LCN; sensitivity not yet performed',
                'original_project':str(orig.resolve()), 'candidate_project':str(out.resolve()),
                'inputs':{name:signature(path) for name,path in inputs.items()},
                'exact_recheck_csv':signature(audit_csv)}
        establish_candidate(root,out,config,args.resume)
        print(f'Old product (READ ONLY): {orig}',flush=True)
        print(f'Corrected candidate: {out}',flush=True)
        ntl_rows=[]
        for y in YEARS:
            dest=out/'intermediate'/'NTL'/f'NTL_{y}.tif'
            if check_complete_raster(dest,ref):
                print(f'[NTL] {y}: existing complete candidate (resume)',flush=True)
                continue
            print(f'[NTL] {y}: masking {int(FILL)} before full-grid bilinear warp',flush=True)
            ntl_rows.append(cleaned_ntl_one(inputs[f'NTL_source_{y}'],inputs[f'NTL_old_{y}'],dest,ref,y))
            write_csv_atomic(pd.DataFrame(ntl_rows),out/'QA'/'NTL_processing_progress.csv')
        q=calculate_quantiles(out)
        print(f'Corrected pooled log1p NTL q01={q["log1p_q01"]:.8g}; q99={q["log1p_q99"]:.8g}',flush=True)
        hci_rows=[]
        for y in YEARS:
            dest=out/'indicators'/f'HCI_{y}.tif'
            if check_complete_raster(dest,ref):
                print(f'[HCI] {y}: existing complete candidate (resume)',flush=True)
                continue
            print(f'[HCI] {y}: recompute with new global quantiles',flush=True)
            hci_rows.append(make_hci_one(y,orig,out,q,ref))
            write_csv_atomic(pd.DataFrame(hci_rows),out/'QA'/'HCI_progress.csv')
        wt=entropy_weights(orig,out,ref)
        print('Recomputed weights: '+', '.join(f'{i}={wt[i]:.8f}' for i in NAMES),flush=True)
        erpi_rows=[]
        for y in YEARS:
            dest=out/'ERI'/f'ERI_{y}.tif'
            if check_complete_raster(dest,ref):
                print(f'[ERI] {y}: existing complete candidate (resume)',flush=True)
                # On resume, rebuild the full QA summary via a separate scan below.
                continue
            print(f'[ERI] {y}: recompute and compare with original',flush=True)
            erpi_rows.append(make_erpi_one(y,orig,out,wt,ref))
            write_csv_atomic(pd.DataFrame(erpi_rows),out/'QA'/'ERPI_rebuild_progress.csv')
        # Full comparison on all 20 products, including products skipped on resume.
        comparison=[]
        for y in YEARS:
            with rasterio.open(out/'ERI'/f'ERI_{y}.tif') as new, \
                 rasterio.open(orig/'ERI'/f'ERI_{y}.tif') as old:
                exact_grid(new,ref,new.name);exact_grid(old,ref,old.name)
                nnew=nold=ncommon=nrm=nadd=nd01=nd0001=0
                snew=sabs=0.0;maxd=0.0
                for win in win_iter(ref.width,ref.height):
                    a=new.read(1,window=win);b=old.read(1,window=win)
                    ma=good(a,new.nodata);mb=good(b,old.nodata)
                    nnew+=int(ma.sum());nold+=int(mb.sum())
                    nrm+=int(np.count_nonzero(mb & ~ma));nadd+=int(np.count_nonzero(ma & ~mb))
                    if ma.any():snew+=float(a[ma].astype(float).sum())
                    c=ma & mb
                    if c.any():
                        d=np.abs(a[c].astype(float)-b[c].astype(float));ncommon+=int(d.size)
                        nd01+=int(np.count_nonzero(d>0.01));nd0001+=int(np.count_nonzero(d>1e-4))
                        maxd=max(maxd,float(d.max()));sabs+=float(d.sum())
                if nadd:raise RuntimeError(f'Unexpected new-support pixels: {y}: {nadd}')
                comparison.append({'year':y,'old_valid':nold,'new_valid':nnew,
                                   'removed_valid':nrm,'unexpected_new_valid':nadd,
                                   'common_pixels':ncommon,'changed_gt_0_0001':nd0001,
                                   'changed_gt_0_01':nd01,'max_abs_change':maxd,
                                   'mean_abs_change_on_common':sabs/ncommon if ncommon else np.nan,
                                   'mean_corrected_erpi':snew/nnew if nnew else np.nan})
        dcomp=pd.DataFrame(comparison)
        write_csv_atomic(dcomp,out/'QA'/'ERPI_old_vs_corrected_fullscan.csv')
        old_w=pd.read_csv(inputs['old_weights'])
        wcomp=pd.DataFrame([{'indicator':i,'old_weight':float(old_w.loc[old_w.indicator==i,'weight'].iloc[0]),
                            'corrected_weight':wt[i]} for i in NAMES])
        wcomp['delta']=wcomp.corrected_weight-wcomp.old_weight
        write_csv_atomic(wcomp,out/'QA'/'fixed_weights_old_vs_corrected.csv')
        print('Bundling unchanged four components with verified SHA256 (may take time)...',flush=True)
        if not args.skip_bundle:
            copy_four_components(orig,out,ref,args.resume)
        else:
            print('WARNING: --skip-bundle; candidate cannot pass standalone reconstruction audit until components are bundled.',flush=True)
        previous_q=json.loads(inputs['old_pooled_ntl_quantiles'].read_text(encoding='utf-8'))
        report=['# ERPI 2001–2020 | corrected PANDA nighttime-light candidate','',
                f'**Candidate:** `{out}`','',
                '**Status:** CANDIDATE ONLY. Full standalone final-raster audit and updated sensitivity are required.','',
                '## Confirmed correction','',
                '- Source PANDA fill -32768 is masked BEFORE bilinear reprojection; original nighttime-light grid/method are otherwise retained.',
                '- Original VCI, LCN, TSI and MSI rasters are unchanged and used to retain the published spatial footprint.',
                '- Corrected NTL pooled q01/q99, all HCI layers, pooled entropy weights and all 20 ERPI rasters were regenerated.',
                '- Cells that become NoData after corrected NTL masking are also removed from ERPI.',
                '- The existing CLCD low-coverage issue is NOT repaired by this correction and remains a separate method/sensitivity question.','',
                '## NTL normalization','',
                f'- Old log1p q01/q99: {previous_q["log1p_q01"]:.10g}, {previous_q["log1p_q99"]:.10g}.',
                f'- Corrected log1p q01/q99: {q["log1p_q01"]:.10g}, {q["log1p_q99"]:.10g}.','',
                '## Full pixel comparison','',
                f'- Total old valid pixel-years: {int(dcomp.old_valid.sum()):,}.',
                f'- Total corrected valid pixel-years: {int(dcomp.new_valid.sum()):,}.',
                f'- Removed pixel-years: {int(dcomp.removed_valid.sum()):,}.',
                f'- Pixels newly valid beyond original support: {int(dcomp.unexpected_new_valid.sum()):,}.',
                f'- Common-support pixel-years with |ΔERPI| > 0.01: {int(dcomp.changed_gt_0_01.sum()):,}.',
                f'- Maximum absolute ERPI change: {dcomp.max_abs_change.max():.8g}.','',
                '## Next validation','',
                'Run audit_ERPI_final_rasters_2001_2020.py with --root pointing to this candidate folder.',
                'Then regenerate the weighting/leave-one-out and manuscript figures from these corrected files.',
                'Do not replace the previous published-candidate files, weights or QA numbers before independent validation.','']
        (out/'QA'/'CORRECTION_CANDIDATE_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
        jdump({'old_project':str(orig),'candidate_project':str(out),
               'years':list(YEARS),'corrected_pooled_quantiles':q,
               'fixed_weights':wt,'standalone_bundled':not args.skip_bundle,
               'source_fill_masked_before_warp':FILL,
               'status':'candidate_requires_independent_audit'},out/'method_config_NTL_corrected.json')
        print('\nCANDIDATE FINISHED. See '+str(out/'QA'/'CORRECTION_CANDIDATE_REPORT.md'),flush=True)


def main():
    ap=argparse.ArgumentParser(description='Non-destructive corrected PANDA NTL ERPI candidate, 2001–2020')
    ap.add_argument('--root',type=Path,default=Path(os.environ.get("ERPI_DATA_ROOT", "./data")).expanduser().resolve())
    ap.add_argument('--output',type=Path,help='Separate candidate folder; defaults to v1 alongside original')
    ap.add_argument('--recheck-csv',type=Path,help='Full-grid exact NTL recheck CSV if moved')
    ap.add_argument('--resume',action='store_true',help='Resume a previously initialized candidate with unchanged input manifest')
    ap.add_argument('--skip-bundle',action='store_true',help='Skip copying unchanged four indicators (not standalone; QA reconstruction then unavailable)')
    args=ap.parse_args()
    run(args)


if __name__=='__main__':
    main()
