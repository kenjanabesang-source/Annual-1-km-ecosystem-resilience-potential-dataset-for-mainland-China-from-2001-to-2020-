# Example run sequence

**Only run after confirming paths and backing up existing products.** This is a command template, not an execution record.

```bash
conda env create -f environment.yml
conda activate erpi-repro
python scripts/preflight_inventory.py --data-root "/data/China"

# Only if legacy intermediate files must be produced from scratch (writes outputs):
# python run_pipeline.py initial --data-root "/data/China" --allow-legacy-initial

python run_pipeline.py ntl-recheck --data-root "/data/China"
python run_pipeline.py correct --data-root "/data/China"
python run_pipeline.py raster-audit --data-root "/data/China"
python run_pipeline.py corrected-audit --data-root "/data/China"
python run_pipeline.py clcd-audit --data-root "/data/China"
python run_pipeline.py sensitivity --data-root "/data/China"
python run_pipeline.py npp --data-root "/data/China"
python run_pipeline.py figures --data-root "/data/China" --only fig2,fig4,fig5,figS1,figS2
```

For a Windows drive:

```powershell
python run_pipeline.py sensitivity --data-root "F:\data\China"
```

Run `python run_pipeline.py --help` for stage choices. Pass `--product-root /path/to/corrected_product` to override the output location. The independent NPP module uses `--erpi-root` internally.
