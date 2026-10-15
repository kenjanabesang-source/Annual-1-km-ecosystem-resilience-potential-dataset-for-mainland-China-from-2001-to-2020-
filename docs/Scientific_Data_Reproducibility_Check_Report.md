# Scientific Data — ERPI code reproducibility check

**Assessment date:** 2026-10-14  
**Release state:** *Pre-release code package, not yet data-validated*.

## 1. Source-code recovery and packaging

- Nine relevant source Python files were retrieved from user-supplied materials and preserved in `archive/original_sources/`.
- Nine active, cross-platform copies live under `scripts/`. The exact differences are logged in `docs/PORTABILITY_PATCHES.diff`.
- Path defaults now use `ERPI_DATA_ROOT` and `ERPI_PRODUCT_ROOT`. The technical sensitivity and NPP comparison copies target the NTL-corrected dataset by default.
- An explicit `run_pipeline.py` stage dispatcher and a read-only `scripts/preflight_inventory.py` help verify local paths and run order.
- All 20 Python source files (9 original + 9 active + 2 utility scripts) passed static parsing/compilation. Dispatcher and preflight help commands returned successfully.

**Verification scope:** No original remote-sensing inputs or final GeoTIFF outputs were available for execution here. These checks are **not evidence that raster values, published weights, 20-year support masks or Figures 1–7 were reproduced**.

## 2. Code–manuscript correspondence

`docs/CODE_PAPER_MAP.md` and the accompanying workbook cover 16 links between manuscript sections, code functions and intended outputs. Main functions are found in the submitted source scripts, including upstream VCI/LCN/TSI/MSI, corrected NTL/HCI, pooled entropy weights, ERPI, NPP comparison, sensitivity analysis, and QA.

## 3. Issues to resolve before public submission

| Priority | Finding | Required follow-up |
|---|---|---|
| BLOCKER | No public/reviewer data link or data DOI provided | Deposit the full corrected data archive with access for review |
| BLOCKER | End-to-end numerical reproduction not run here | Run on the original data and match every manuscript metric |
| HIGH | Core original workflow contains historical NTL fill handling | Clearly designate the NTL-corrected candidate as published product |
| HIGH | Corrected workflow reuses four archived component series | Preserve/reproduce legacy upstream intermediate generation and document provenance |
| HIGH | Legacy technical code can reuse preexisting `technical_validation/common_valid_mask_2001_2020.tif` | Back up and regenerate this derived file when changing product version |
| HIGH | Research manuscript calls component/NPP panels Figures 6 and 7, whereas figure suite names them Fig. S1 and Fig. S2 | Harmonize manuscript references, figure files and captions before submission |
| HIGH | Annual NDVI inputs are already aggregated | Publish the upstream annual-composite recipe and exact source lineage, if not already archived |
| HIGH | Release license not approved by all rights holders | Confirm MIT license for code and separate input-data rights |
| MEDIUM | The CFF contains only a pre-release author subset and no DOI | Confirm entire author list, version, publication and repository DOI |
| MEDIUM | Generic conda dependency constraints rather than a tested lockfile | Export environment from the successful full rerun |
| MEDIUM | Figure 1 requires verified boundary/cartographic sources | Verify all vector licenses and provenance |

## 4. Manuscript values that require rerun verification

The manuscript reports the following, used only as **targets for comparison**, not measurements verified in this session:

- Valid pixel-years: **181,800,565**.
- Strict 20-year common support: **9,055,311** cells.
- Fixed weights (rounded): **VCI 0.404919; LCN 0.278581; HCI 0.155397; TSI 0.076801; MSI 0.084303**.
- Independent ERPI reconstruction: **zero mask mismatches** and maximum absolute error **2.98e-08**.

If rerun values differ, reconcile the code, source-data versions, masks and manuscript before public release. Do not silently overwrite prior research outputs.

## 5. Recommended publication sequence

1. Archive unmodified source scripts and original data/version records.
2. Rebuild or verify original upstream component files in an isolated folder.
3. Verify pre-warp `-32768` fill masking and build the corrected candidate.
4. Run independent source and final-raster audits, then rerun sensitivity/NPP outputs from corrected data.
5. Compare corrected outputs, figure labels and all manuscript values.
6. Record platform, package versions, manifests, hashes, and third-party file provenance.
7. Deposit raster data and code in versioned public repositories and add valid DOI/reviewer links.
8. Obtain license/author approvals and publish a tagged release only after all checks pass.

**Current decision: suitable as a documented GitHub *pre-release* repository; not yet verified as a reproducible Scientific Data final publication package.**
