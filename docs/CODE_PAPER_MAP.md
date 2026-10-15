# Code–paper correspondence

| Manuscript section | Script | Key functions | Output(s) | Stage | Verification notes |
|---|---|---|---|---|---|
| Methods: study grid | generate_initial_legacy_product.py | build_target_grid | method_config.json; common Albers grid | legacy upstream | Grid CRS/shape must be checked from actual GeoTIFFs |
| Methods: NDVI/VCI | generate_initial_legacy_product.py | preprocess_ndvi; compute_baselines; build_indicators | intermediate/NDVI; baselines/NDVI_Q05,NDVI_Q95; indicators/VCI | legacy upstream | Annual NDVI source inputs must exist |
| Methods: CLCD/LCN | generate_initial_legacy_product.py | create_clcd_binary; preprocess_clcd; build_indicators | intermediate/LCN; indicators/LCN | legacy upstream | Source-coverage audit is separate |
| Methods: NTL HCI correction | generate_NTL_corrected_ERPI.py | source fill masking; corrected reprojection; HCI construction | corrected NTL; indicators/HCI | corrected main | Source fill=-32768 must be masked before warp |
| Methods: thermal/moisture proximity | generate_initial_legacy_product.py | preprocess_multiband_annual; temporal_median; temporal_mad; build_indicators | indicators/TSI and MSI; baseline median/MAD | legacy upstream | 26-band year mapping: 2001=band2, 2020=band21 |
| Methods: pooled entropy weights | generate_NTL_corrected_ERPI.py | re-estimate pooled weights | entropy_weights_fixed_2001_2020.csv | corrected main | Check HCI ≈0.155397 |
| Data Records: annual rasters | generate_NTL_corrected_ERPI.py | recompute annual ERPI | ERI/ERI_2001.tif ... ERI_2020.tif | corrected main | Preserves/shrinks legacy valid footprint |
| Technical Validation: source NTL audit | recheck_NTL_source_fullgrid.py | warp_ntl; calc_hci | SOURCE_PROCESSING_AUDIT/NTL_EXACT_RECHECK/* | validation | Requires original source rasters |
| Technical Validation: full raster integrity | audit_final_rasters.py | structural_audit; reconstruction_audit; support_audit | FINAL_ERPI_TECHNICAL_AUDIT/* | validation | Pass corrected --root, not the legacy directory |
| Technical Validation: corrected QA | audit_NTL_corrected_candidate.py | independent corrected candidate QA | NTL correction audit files | validation | Fails on legacy product identity |
| Technical Validation: CLCD coverage | assess_CLCD_coverage.py | native-area coverage sensitivity | CLCD source area coverage QA | validation | Inputs native 30m CLCD required |
| Technical Validation: equal vs entropy | technical_sensitivity_analysis.py | equal_weight_analysis | technical_validation/entropy_vs_equal_comparison.csv | validation | Final corrected product by environment |
| Technical Validation: leave-one-out | technical_sensitivity_analysis.py | leave_one_out_analysis | technical_validation/leave_one_out_comparison.csv | validation | Five exclusions, renormalized weights |
| Technical Validation: common support and trend | technical_sensitivity_analysis.py | build_common_mask; annual_indicator_statistics; trend_and_change_tables | technical_validation/annual_eri_common_mask.csv; trend_summary.csv; change_decomposition_2019_2020.csv | validation | Remove stale derived common mask after backup |
| Technical Validation: NPP comparison | NPP_context_comparison.py | align_npp; temporal_maps; long_term_comparison; annual_comparison | independent_NPP_only_validation/NPP_*csv | validation | NPP not resilience ground truth; corrected --erpi-root |
| Figures 1–5 and S1–S2 | make_publication_figures.py | fig1; fig2; fig3; fig4; fig5; figs1; figs2 | manuscript_figures_FINAL_v7_1_layoutsafe/ | visualization | Requires validated cartographic context for Fig.1 |
