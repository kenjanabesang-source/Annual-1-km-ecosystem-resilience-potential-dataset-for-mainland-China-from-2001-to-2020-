#!/usr/bin/env python3
"""Non-writing lightweight source and result inventory (not numerical validation)."""
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--product-root',type=Path)
    a=p.parse_args(); r=a.data_root.expanduser().resolve()
    product=(a.product_root or r/'ERI_recalculated_2001_2020_NTLfix_candidate_v1').expanduser().resolve()
    original=r/'ERI_recalculated_2001_2020'
    groups={
      'CLCD input':[r/'CLCD'/f'CLCD_v01_{y}_albert.tif' for y in range(2001,2021)],
      'NDVI input':[r/'NDVI'/f'{y}_NDVI.tif' for y in range(2001,2021)],
      'NTL input':[r/'NightLight'/f'PANDA_China_{y}.tif' for y in range(2001,2021)],
      'NPP input (optional)':[r/'NPP'/f'NPP_{y}.tif' for y in range(2001,2021)],
      'Climate':[r/'temperature_2m_2000_2025.tif',r/'volumetric_soil_water_layer_1_2000_2025.tif'],
      'Legacy component files':[original/'indicators'/f'{k}_{y}.tif' for y in range(2001,2021) for k in ('VCI','LCN','TSI','MSI')],
      'Corrected ERPI':[product/'ERI'/f'ERI_{y}.tif' for y in range(2001,2021)],
      'Corrected components':[product/'indicators'/f'{k}_{y}.tif' for y in range(2001,2021) for k in ('VCI','LCN','HCI','TSI','MSI')],
      'Corrected weights':[product/'entropy_weights_fixed_2001_2020.csv'],
    }
    print(f'Input data root: {r}\nProduct root: {product}\n')
    for title,paths in groups.items():
        found=sum(p.is_file() for p in paths)
        print(f'{title}: {found}/{len(paths)} at expected paths')
        if found<len(paths): print('  First missing:',next(str(p) for p in paths if not p.is_file()))
    w=product/'entropy_weights_fixed_2001_2020.csv'
    if w.is_file():
        with w.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        try:
            weights={row['indicator']:float(row['weight']) for row in rows}
            print('Stored weights:',weights)
            print('HCI weight ~0.155397:', abs(weights.get('HCI',-1)-0.155397)<5e-4)
        except Exception as exc: print('Weights format inspection failed:',exc)
    print('\nThis does not check pixel values, NTL fill handling, raster hashes, or scientific reproducibility.')

if __name__=='__main__':main()
