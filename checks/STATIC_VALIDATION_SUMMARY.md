# Static source and GitHub packaging QA

- Date: 2026-10-14 UTC
- Original Python scripts preserved without alteration: **9**
- Portable active script copies: **9**
- Total Python files parsed/compiled: **20**
- Dispatcher `--help` and preflight `--help`: **PASS**
- Original vs. portable changes recorded line-by-line: `docs/PORTABILITY_PATCHES.diff`
- Real raster execution / numerical replication: **NOT RUN** (data missing)
- Code repository DOI / data DOI / source licensing: **PENDING**

## Compile status

| Script | Status |
|---|---|
| archive/original_sources/ERI_NPP_only_validation_2001_2020.py | PASS |
| archive/original_sources/ERI_technical_validation_2001_2020.py | PASS |
| archive/original_sources/ERPI_2001_2020_NTL_corrected_candidate.py | PASS |
| archive/original_sources/assess_CLCD_coverage_impact_NTLfix_2001_2020.py | PASS |
| archive/original_sources/audit_ERPI_NTLfix_candidate_2001_2020.py | PASS |
| archive/original_sources/audit_ERPI_final_rasters_2001_2020.py | PASS |
| archive/original_sources/make_ERPI_publication_figures_v7_1_layoutsafe.py | PASS |
| archive/original_sources/recalculate_ERI_2001_2020_FINAL.py | PASS |
| archive/original_sources/recheck_ERPI_NTL_exact_full_grid.py | PASS |
| run_pipeline.py | PASS |
| scripts/01_preprocessing/generate_initial_legacy_product.py | PASS |
| scripts/02_erpi_generation/generate_NTL_corrected_ERPI.py | PASS |
| scripts/03_validation/NPP_context_comparison.py | PASS |
| scripts/03_validation/assess_CLCD_coverage.py | PASS |
| scripts/03_validation/audit_NTL_corrected_candidate.py | PASS |
| scripts/03_validation/audit_final_rasters.py | PASS |
| scripts/03_validation/recheck_NTL_source_fullgrid.py | PASS |
| scripts/03_validation/technical_sensitivity_analysis.py | PASS |
| scripts/04_visualization/make_publication_figures.py | PASS |
| scripts/preflight_inventory.py | PASS |

## Limitations

Syntax checks confirm source parsability, not that the published results were reproduced. Scientific Data reviewers still need a accessible, complete, DOI-stamped source and dataset archive.
