# ERPI workflow and dependencies

```mermaid
flowchart TD
  A[MODIS annual NDVI] --> P[Legacy full upstream preprocessing]
  B[CLCD annual land cover] --> P
  C[ERA5-Land temperature + soil moisture] --> P
  D[PANDA-China night lights] --> P
  P --> X[Legacy VCI / LCN / TSI / MSI + legacy NTL intermediates]
  D --> N[Pre-warp fill -32768 mask + corrected NTL warp]
  X --> F[Corrected HCI + 2001–2020 pooled references]
  N --> F
  F --> W[Re-estimate fixed pooled entropy weights]
  X --> W
  W --> E[20 annual ERPI 1-km rasters + component bundle]
  E --> Q[Final raster audit + NTL candidate independent audit]
  E --> S[Equal weighting / leave-one-out / common-support]
  E --> V[Independent NPP context comparison]
  E --> G[Publication figures]
```

## Source processing
`generate_initial_legacy_product.py` contains `preprocess_ndvi`, `preprocess_clcd`, `preprocess_ntl`, `preprocess_multiband_annual`, `compute_baselines`, `build_indicators`, `compute_entropy_weights`, and `compute_eri`. Historical code does not mask the undeclared NTL fill before bilinear resampling.

## Correction
`generate_NTL_corrected_ERPI.py` uses source `-32768` masking before reprojection, recomputes NTL quantiles, all HCI layers, fixed entropy weights, and annual ERPI maps. It **reuses** the other four archived components and maintains or reduces original valid support.

## Validation
See `docs/CODE_PAPER_MAP.md` for per-result correspondence. Validate against final corrected rasters only; no data files are included here.
