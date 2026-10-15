# Data access / files to publish

**Status: deposit pending — no reviewer URL or permanent DOI has been provided.** Do not use fake links.

## Input data (provider downloads, not hosted here)
- MODIS/Terra monthly vegetation indices (MOD13A3.061-derived annual files)
- CLCD v1 annual land cover
- PANDA-China nighttime lights
- ERA5-Land 2-m air temperature and layer-1 soil water, derived annual stacks
- MODIS MOD17A3HGF.061 NPP (independent context comparison only)

Consult the manuscript References for formal dataset citations and provider access instructions. The local annual rasters used here may be derived / transformed versions requiring separate provenance documentation.

## Corrected outputs to archive externally
- `ERI/ERI_2001.tif` ... `ERI/ERI_2020.tif`
- `indicators/{VCI,LCN,HCI,TSI,MSI}_<year>.tif`
- baseline layers and corrected `NTL_pooled_quantiles.json`
- `entropy_weights_fixed_2001_2020.csv`
- strict common-support masks and full technical QA records
- complete metadata/manifest, CRS, file hashes, data dictionary, methods README, license and version information

## Fill before submission
- Review-access URL: **TBD**
- Permanent output DOI: **TBD**
- Code release DOI: **TBD**
- Final data and code reuse permissions: **TBD**
- Exact third-party data versions, annual aggregation procedure and dependencies: **verify**

GitHub is a code host; large GeoTIFFs should be deposited in a suitable public data repository with stable DOI and accessible review links.
