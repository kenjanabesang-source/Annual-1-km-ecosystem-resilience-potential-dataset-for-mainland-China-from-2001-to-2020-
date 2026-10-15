# ERPI · China · 2001–2020

**Annual 1-km ecosystem resilience potential proxy index — source processing, validation and figure-generation code**

This code accompanies the manuscript *Annual 1-km ecosystem resilience potential dataset for mainland China from 2001 to 2020* (manuscript title subject to final editorial revision). **ERPI is a proxy-based comparative index, not an observation of ecosystem resistance, post-disturbance recovery or tipping points.**

## What is here

- **Original processing workflow** deriving annual vegetation (VCI), land-cover naturalness (LCN), light-pressure (HCI), temperature-baseline proximity (TSI) and soil-moisture-baseline proximity (MSI) layers, fixed pooled entropy weights, and annual composites.
- **NTL correction** masking PANDA-China fill value `-32768` *before* bilinear reprojection and recomputing corrected NTL/HCI, the fixed weights and all 20 ERPI outputs.
- **Technical validation** of raster integrity, source and coverage checks, equal-weight/leave-one-out sensitivity, fixed-support statistics and descriptive trends.
- **Independent NPP context comparison**, and publication-figure generation.

### Status and important limitation

**Pre-release source-code archive.** All nine Python files have been syntax-checked, but the source GeoTIFFs, field data and complete published output archive were **not uploaded here**; an end-to-end reproduction has **not** been run. Numerical values reported in the manuscript cannot yet be certified from this package alone. Before public release, complete the checklist in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository layout

```text
ERPI_GitHub_Final_Release/
├── README.md, LICENSE, CITATION.cff, environment.yml, CHANGELOG.md
├── run_pipeline.py                  # uniform dispatch, not a rewritten science model
├── scripts/
│   ├── 01_preprocessing/            # legacy full calculation, kept for traceability
│   ├── 02_erpi_generation/          # corrected PANDA NTL candidate and composite
│   ├── 03_validation/               # source audits, raster QA, sensitivity, NPP
│   ├── 04_visualization/            # corrected-product figures
│   └── preflight_inventory.py
├── archive/original_sources/        # unmodified original uploads
├── docs/                             # workflow, reproducibility, access, patch diffs
├── examples/                         # practical run order and commands
├── checks/                           # SHA256 manifest and static test report
└── data/README.md                    # no actual rasters are shipped
```

## Installation

Conda (recommended for GDAL/rasterio consistency):

```bash
conda env create -f environment.yml
conda activate erpi-repro
python run_pipeline.py --help
```

`requirements.txt` is provided as an alternative, but no exact numerical environment lockfile exists yet. Capture a tested export once the original data are available.

## Data layout and paths

Pass a local *parent* directory via `--data-root`; the scripts no longer require the author's `F:\data\China` drive. The source files retain their **original expected names**, including:

```text
<DATA_ROOT>/
  CLCD/CLCD_v01_2001_albert.tif ... CLCD_v01_2020_albert.tif
  NDVI/2001_NDVI.tif ... 2020_NDVI.tif
  NightLight/PANDA_China_2001.tif ... PANDA_China_2020.tif
  temperature_2m_2000_2025.tif
  volumetric_soil_water_layer_1_2000_2025.tif
  NPP/NPP_2001.tif ... NPP_2020.tif              # optional NPP comparison
  China_boundary.shp (+ required shapefile sidecars)
  ERI_recalculated_2001_2020/                 # legacy intermediate product
  ERI_recalculated_2001_2020_NTLfix_candidate_v1/  # corrected candidate
```

The boundary file can be resolved by the legacy code by case-insensitive matching. CLCD basename variants may also be accepted; see script. The correct CRS and band-to-year mapping must be verified against your source data.

## Run stages

First inspect local data **without processing rasters**:

```bash
python scripts/preflight_inventory.py --data-root /path/to/China
```

If the legacy product **already exists**, DO NOT rerun the original workflow by default. Use it only if you need to reconstruct the initial intermediates from scratch in a safe isolated directory:

```bash
python run_pipeline.py initial --data-root /path/to/China --allow-legacy-initial
```

The *original* NTL stage is preserved as historical code and does **not** correct the `-32768` source fill issue. Generate the corrected candidate from the legacy non-HCI components and original NTL input:

```bash
python run_pipeline.py ntl-recheck --data-root /path/to/China
python run_pipeline.py correct --data-root /path/to/China
python run_pipeline.py raster-audit --data-root /path/to/China
python run_pipeline.py corrected-audit --data-root /path/to/China
python run_pipeline.py clcd-audit --data-root /path/to/China
python run_pipeline.py sensitivity --data-root /path/to/China
python run_pipeline.py npp --data-root /path/to/China
python run_pipeline.py figures --data-root /path/to/China --only fig2,fig4,fig5,figS1,figS2
```

**Run commands are examples, not confirmed end-to-end execution.** Some operations may need previous QA tables or geospatial boundary vectors. Figure 1 requires a verified cartographic-source declaration; omit it until these conditions are satisfied. Rerunning old validation results in an existing `technical_validation/` directory may reuse an old common-support mask; archive or remove that *derived mask only after backup and confirmation* before recomputing corrected validation.

For Windows PowerShell substitute, for example, `--data-root 'F:\data\China'`. All stages accept `--product-root` if the corrected product is somewhere else. Optional extra flags can be supplied after the listed options and are forwarded to the underlying script.

## Expected data product and validation

The corrected dataset contains 20 annual float32 ERPI GeoTIFFs (`ERI/ERI_<year>.tif`), five annual indicator series, baseline rasters, a fixed weight CSV, strict common mask and QA files. File basenames use **`ERI_`** for historical compatibility, while the manuscript refers to the proxy as **ERPI**; no formulas are silently renamed. Nominal study period: 2001–2020, retrospective reference window 2001–2020, grid: 1-km Albers equal-area CRS.

Validation code outputs `entropy_vs_equal_comparison.csv`, `leave_one_out_comparison.csv`, `trend_summary.csv`, NPP temporal/spatial correlation tables and raster reconstruction reports. These tables should be compared with the manuscript **only after** pointing every script to the corrected product. See [`docs/CODE_PAPER_MAP.md`](docs/CODE_PAPER_MAP.md).

## Data, citation and permissions

**Rasters are not included.** Source files (MODIS NDVI, CLCD, PANDA-China, ERA5-Land and optional MODIS NPP) must be obtained in accordance with their providers' terms. A public corrected output archive URL/DOI is not yet supplied; see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

`CITATION.cff` identifies the initial manuscript authors and is marked pre-release; add the final journal article citation, code release DOI and version after publication. Code is offered under the included MIT license **subject to all code rightsholders' approval before publication**. Source datasets and third-party geographic boundary vectors do not inherit this code license.

## Reproducibility and limitations

- Entropy weights reflect pooled statistical dispersion, **not ecological importance**.
- VCI is pixel-relative; LCN classifies barren cover as unmodified; HCI is darkness rather than a comprehensive human-pressure metric; TSI and MSI quantify deviation from retrospective local baselines, not optima.
- The NTL-corrected code reuses four previously generated components. Do not describe this correction script alone as a full raw-source pipeline.
- NPP is an **independent comparison variable for ecological context**, not observed disturbance resistance or recovery, and shares potential algorithmic ancestry with the NDVI input.
- Users extending the dataset beyond 2020 must decide whether to freeze or re-estimate reference distributions and weights.

**Contact:** corresponding author Xiaoyong Bai (contact details in associated manuscript). Do not report final DOI or manuscript acceptance until they exist.
