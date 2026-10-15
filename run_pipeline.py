#!/usr/bin/env python3
"""Dispatch the preserved ERPI scripts with explicit paths.

Code wrappers set ERPI_DATA_ROOT / ERPI_PRODUCT_ROOT. They do not change science.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPT_MAP = {
  'initial': ('scripts/01_preprocessing/generate_initial_legacy_product.py', 'none'),
  'ntl-recheck': ('scripts/03_validation/recheck_NTL_source_fullgrid.py', 'data'),
  'correct': ('scripts/02_erpi_generation/generate_NTL_corrected_ERPI.py', 'data'),
  'raster-audit': ('scripts/03_validation/audit_final_rasters.py', 'product'),
  'corrected-audit': ('scripts/03_validation/audit_NTL_corrected_candidate.py', 'product'),
  'clcd-audit': ('scripts/03_validation/assess_CLCD_coverage.py', 'data'),
  'sensitivity': ('scripts/03_validation/technical_sensitivity_analysis.py', 'none'),
  'npp': ('scripts/03_validation/NPP_context_comparison.py', 'data'),
  'figures': ('scripts/04_visualization/make_publication_figures.py', 'product'),
}

def main():
    parser=argparse.ArgumentParser(description='ERPI analysis entrypoint; subprocess exit status is propagated')
    parser.add_argument('stage',choices=sorted(SCRIPT_MAP))
    parser.add_argument('--data-root', type=Path, required=True, help='Folder containing CLCD, NDVI, NightLight, NPP and ERA5 files')
    parser.add_argument('--product-root', type=Path, help='Final corrected product directory')
    parser.add_argument('--allow-legacy-initial',action='store_true',help='Explicitly allow initial workflow to write the legacy directory')
    args,rest=parser.parse_known_args()
    if args.stage=='initial' and not args.allow_legacy_initial:
        parser.error('Initial workflow may overwrite legacy outputs; repeat with --allow-legacy-initial only in a safe new data root.')
    data=args.data_root.expanduser().resolve()
    product=(args.product_root or data/'ERI_recalculated_2001_2020_NTLfix_candidate_v1').expanduser().resolve()
    env=os.environ.copy(); env['ERPI_DATA_ROOT']=str(data); env['ERPI_PRODUCT_ROOT']=str(product)
    script,argkind=SCRIPT_MAP[args.stage]
    command=[sys.executable,str(ROOT/script)]
    if argkind=='data': command+=['--root',str(data)]
    if argkind=='product': command+=['--root',str(product)]
    if args.stage=='correct': command+=['--output',str(product)]
    if args.stage=='npp': command+=['--erpi-root',str(product)]
    command+=rest
    print('Executing:', ' '.join(command),flush=True)
    raise SystemExit(subprocess.call(command,cwd=str(ROOT),env=env))

if __name__=='__main__': main()
