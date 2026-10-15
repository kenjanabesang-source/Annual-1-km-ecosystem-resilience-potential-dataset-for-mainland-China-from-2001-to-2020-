# Reproducibility checks for Scientific Data

## Tested in this package
- Python source parsing / compilation of all portable and archived Python files.
- Expected main routines, command-line switches and version links inspected from source.
- No execution of the full raster workflow (raw source data and GeoTIFF products not present in this chat).

## Must be performed before an official software release

1. Verify all input files, source DOI/version, coordinate system and band-to-year mapping.
2. Capture a tested environment (`conda list --explicit` or similar), OS, GDAL/rasterio versions.
3. Run the original workflow in an isolated **fresh directory**; never overwrite archived research outputs.
4. Independently verify PANDA-China source fill `-32768` and exact full-grid reprojection match.
5. Run corrected candidate and record its full source-input manifest, output hashes, NTL q01/q99, weights and pixel-year support.
6. Check the corrected candidate weighs approximately VCI 0.404919, LCN 0.278581, HCI 0.155397, TSI 0.076801, MSI 0.084303 (compare precise persisted values, not rounded text).
7. Check the strict 20-year common support is 9,055,311 cells; total valid pixel-years 181,800,565; these are reported manuscript values, not checked from this package.
8. Check raster reconstruction maximum abs difference (~2.98e-08 reported in manuscript), no mask mismatch, and CLCD source area coverage sensitivity.
9. **Avoid stale tables**: ensure equal-weight, LOO, trend and NPP calculations read `ERI_recalculated_2001_2020_NTLfix_candidate_v1` and regenerate after the final correction.
10. Compare all manuscript figure numbers and table statistics to newly generated corrected files. Verify Fig.1 geodata license/cartographic provenance.
11. Publish data archive and code release with SHA256, README, license and stable DOI; fill citation metadata.

## Configuration modifications to original uploads
- Minimal portability edits in active scripts are diffed in `docs/PORTABILITY_PATCHES.diff`.
- Original unmodified uploads are retained in `archive/original_sources/` and named accordingly.
- Do not interpret changes to `--root` defaults as scientific algorithm revisions.

## Known dependencies / constraints
- Revised technical/NPP active copies point at corrected product by default, but `archive/` preserves their legacy defaults.
- Original NDVI annual aggregation was performed upstream: the main script does *not* reconstruct monthly NDVI to annual inputs.
- Corrected NTL step does not independently redo CLCD, NDVI, TSI or MSI; reproduce those using the legacy upstream script in an isolated working directory if necessary.
- Figure script checks HCI weight and exact common mask identity, and requires specific QA tables.
- If existing `technical_validation/common_valid_mask_2001_2020.tif` is stale, the technical script will reuse it. Back up and regenerate this derived file when switching datasets.
- Published data availability and licensing remain author obligations.
