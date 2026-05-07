# ROGII Strategy Reset - 2026-05-07

## Why We Pivot

The last eight-hour search proved that tiny local residual gains are not worth a
submit. Version 7 improved local 5-fold RMSE by only about 0.024 but regressed
public LB from 13.854 to 13.977. The tester is useful for large direction checks,
but it is not calibrated enough for small parameter tuning.

The competition discussion and public notebooks point to a stronger strategy:
this is a geosteering/structural geology problem first, and a residual ML problem
second.

## External Signals Checked

- Kaggle discussion `besides regression, also dwt (time warping)!`: GR-to-typewell
  matching matters, but naive DTW is not enough. Direction can reverse, noise is
  large, and the search must be local/restricted.
- ROGII welcome thread and the official PPT: horizontal-well GR after Prediction
  Start should be interpreted against typewell GR, but TVT can increase, decrease,
  or stay almost constant. The PPT also says the horizontal GR before PS can
  correlate better with after-PS GR than the typewell does.
- Public notebooks:
  - `konbu17/rogii-plane-fit-formation-top-knn`: public LB 11.912, local OOF
    around 12.11. Key formula: TVT ~= -Z + ANCC + per-well anchor.
  - `romantamrazov/rogii-super-baseline-lb-12-602`: formation plane KNN +
    row-level ANCC KNN + beam typewell features + LGB/XGB/CatBoost stack.
  - `pilkwang/12-049-rogii-eda-leakageriskdiscussion`: broad feature/validation
    framework, typewell alignment features, leakage cautions.
  - `ravaghi/wellbore-geology-prediction-lightgbm`: strong LightGBM starter with
    beam typewell features.

## Load-Bearing Insight

Training horizontal files contain formation-top surfaces:

`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`

These are train-only columns, but they are not useless. They let us learn a
spatial geological surface from nearby wells. Strong public code reports that,
within a well:

`TVT ~= -Z + ANCC + b_well`

where `b_well` is estimated from the known prefix:

`b_well = median(TVT_input + Z - predicted_ANCC)`

If predicted ANCC is good, TVT is almost solved. This is a much bigger lever
than predicting a generic residual from row features.

## New Architecture

1. Build a spatial geological map.
   - Per-well centroid KNN.
   - Weighted 2D plane fit for each formation top.
   - Row-level `(X, Y)` KNN for ANCC using all training rows.
   - Self-well exclusion during CV.

2. Convert geometry into strong priors.
   - `tvt_from_plane = -Z + ANCC_plane + b_prefix`
   - `tvt_from_row_knn = -Z + ANCC_row_knn + b_prefix`
   - Use distances/variance between neighbors as uncertainty features.

3. Use constrained geosteering as correction, not as the whole model.
   - Calibrate typewell GR to known-prefix horizontal GR.
   - Use local beam search around last known TVT.
   - Allow stay/up/down moves, not monotone-only DTW.
   - Penalize motion so flat geology remains easy.
   - Feed beam deltas, GR mismatch, and confidence gaps to the residual model.

4. Train residual stack.
   - Target: `TVT - last_known_TVT`.
   - Models: LightGBM seeds + XGBoost; CatBoost optional if Kaggle runtime allows.
   - Include formula priors as stack members, not just features.
   - Smooth/slope-clip only when OOF proves it helps.

5. Validate harder.
   - Keep ordinary GroupKFold for continuity.
   - Add spatial-cluster folds by well centroid to simulate hidden locations.
   - Track fold count wins and per-well catastrophic regressions.
   - Do not submit tiny local gains. Require a meaningful improvement and sane
     behavior under both GroupKFold and spatial folds.

## Bold Experiments Worth Running

1. Surface model search:
   weighted plane vs quadratic trend vs kriging-style radial basis; tune K,
   anisotropy, and row-level neighbor count.

2. Formation choice stack:
   ANCC is the obvious anchor, but train all six formation-derived TVT formulas
   and let ridge/LightGBM learn when each is trustworthy.

3. Particle/Kalman geosteering:
   state is `(TVT, dTVT/dMD)`, transition allows flat/up/down, emission is
   calibrated GR mismatch against typewell. This matches the PPT more directly
   than free-form DTW.

4. Offset-well analog transfer:
   nearest wells by `(X, Y, azimuth, PS TVT, prefix GR signature)` transfer their
   tail delta shape after affine TVT/MD normalization.

5. Two-stage uncertainty gating:
   if surface prior uncertainty is low, trust geometry; if high, let GR/typewell
   beam and residual ML move more aggressively.

## Immediate Next Build

Port the formation-stack baseline into our own scorer/notebook:

1. Add local feature builder with formation-plane KNN + row-level ANCC KNN.
2. Add a 5-fold scorer and spatial-cluster scorer.
3. Reproduce a local OOF near the public `~12.1` range.
4. Build a Kaggle notebook version from our code.
5. Run the required 8-hour guarded search/ablation cycle before any next submit.

No more pure residual hyperparameter search as the primary path.
