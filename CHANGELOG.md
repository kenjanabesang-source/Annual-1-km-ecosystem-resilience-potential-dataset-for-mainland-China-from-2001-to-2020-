# Changelog

These are **processing stages**, not evidence of historically published software versions.

## 0.1.0 — Repository assembly (pre-release)
- Collected nine uploaded source scripts. Preserved exact originals in `archive/original_sources/`.
- Introduced cross-platform paths via environment variables; did not alter formulae.
- Set the corrected NTL candidate as the default product used by validation and NPP scripts.
- Added a dispatcher, preflight inventory, integrity manifest and documentation.

## Provenance: historical scientific workflow (unreleased stages)
- **Initial workflow:** `recalculate_ERI_2001_2020_FINAL.py` generated five indicators and legacy ERI products.
- **Source audit:** the PANDA-China `-32768` undeclared fill was investigated with matched full-grid reprojections.
- **Corrected candidate:** regenerated NTL, HCI, fixed pooled entropy weights and 20 annual ERPI rasters while preserving legacy non-HCI components.
- **Independent audits:** final-raster reconstruction, CLCD native area coverage sensitivity, equal weighting, leave-one-out and NPP comparison.

Do not tag v1.0.0 or call this code validated until a clean end-to-end test and DOI freeze are completed.
