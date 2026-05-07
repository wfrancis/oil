# Handover For Claude - ROGII, 2026-05-07

## TL;DR

We are not stuck because Rust is slow or because the residual model needs more
hyperparameter search. We are stuck because our strategy is one abstraction too
high: the strong public work treats this as a spatial geology/formation-surface
problem first, then uses ML for residuals.

Current best submitted public LB:

- v5/v6: `13.854`
- v7: `13.977` after an 8-hour guarded local search; worse, so do not continue
  tiny residual tuning.

Hard constraint from William:

> Do not submit again until at least 8 hours of active CPU/search wall time has
> been recorded after the latest Kaggle submission.

The guard has been reset after v7 and currently blocks submit:

`submit_ready=False accumulated_hours=0.000 required_hours=8.000`

## Latest Submission History

| Version | UTC Time | Public LB | Notes |
| --- | ---: | ---: | --- |
| v3 | 2026-05-06 22:52:04 | 96.973 | Banded DTW + geology + RTS; failed catastrophically. |
| v4 | 2026-05-07 02:10:49 | 15.883 | Last known `TVT_input` constant baseline. |
| v5 | 2026-05-07 03:04:02 | 13.854 | Residual LightGBM, shrink 0.75. |
| v6 | 2026-05-07 03:35:59 | 13.854 | Exact visible train/test lookup + v5 fallback; tied v5. |
| v7 | 2026-05-07 13:21:27 | 13.977 | Best 8-hour local residual config; regressed publicly. |

The v7 lesson matters: local 5-fold improved from about `14.5861` to `14.5624`
but public LB got worse. The local harness catches big moves but is not reliable
for tiny parameter deltas.

## Files Added Or Changed In This Round

Important repo files:

- `rogii/STRATEGY_RESET_2026-05-07.md`
  - Strategy pivot memo. Read this first.
- `rogii/HANDOVER_FOR_CLAUDE.md`
  - This file.
- `rogii/bench/submit_guard.py`
  - Fail-closed submit guard with `status`, `check`, `record`,
    `reset-cycle`.
- `rogii/bench/safe-submit`
  - Guarded wrapper around `kaggle competitions submit`.
- `rogii/bench/run_max_cpu_search.py`
  - 8-hour local residual search harness. Useful as infrastructure, but do not
    use it as the main scientific path anymore.
- `rogii/bench/local_score.py`
  - Residual scorer gained LightGBM params for the search harness.
- `rogii/bench/SUBMIT_GATE.md`
  - Documents the hard submit lock and local scoring gate.
- `rogii/rust/local_score/*`
  - Rust scorer optimized for M1 Pro: Rayon, native target CPU, release LTO,
    faster CSV hot path, selected-column parsing.
- `rogii/notebook/kaggle_cell_v6.py`
  - v6/v7 notebook cell. v7 currently has the worse tuned fallback:
    `SHRINK=0.65`, `colsample_bytree=1.0`, `reg_lambda=0.5`.
- `rogii/notebook/submission.ipynb`
  - Rebuilt from `kaggle_cell_v6.py`.

## Submit Guard Commands

Check:

```bash
python3 rogii/bench/submit_guard.py status
python3 rogii/bench/submit_guard.py check
```

Record completed search time:

```bash
python3 rogii/bench/submit_guard.py record \
  --label some_run \
  --wall-seconds 3600 \
  --note "active local search, no submit"
```

Submit only through:

```bash
./rogii/bench/safe-submit rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k wbfranci/rogii-eagle-ford-dtw-rts-v1 \
  -v <kernel_version> \
  -m "<message>"
```

Do not use raw `kaggle competitions submit` unless the guard passes and you are
intentionally bypassing the wrapper for a known reason.

## What We Learned From The Discussion

Competition discussion thread:

`https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431`

Load-bearing comments:

- It is geosteering.
- Typewell GR encodes geologic location/TVT.
- Horizontal GR should be matched to typewell GR, but not via naive DTW.
- Direction can reverse depending on whether the drill is travelling upward or
  downward through geology.
- Noise is large; matching must be local/restricted.
- The drill may move far in MD/XYZ while staying nearly at the same geological
  position.

Official PPT extracted text:

- Goal is to infer TVT after Prediction Start from horizontal XYZ/GR and
  typewell TVT/GR.
- TVT can increase, decrease, or stay constant.
- Horizontal GR before PS can correlate better with after-PS GR than the
  typewell GR does.
- Nearby offset wells help because geology can be flat or dipping, and drilling
  azimuth matters.

## Strong Public Signals

Pulled locally to `/tmp/rogii_public_kernels/pulled` and converted to Python in
`/tmp/rogii_public_kernels/extracted`.

Most important notebooks:

- `konbu17/rogii-plane-fit-formation-top-knn`
  - Reports public LB `11.912`, local OOF around `12.11`.
  - Core insight:
    `TVT ~= -Z + ANCC + b_well`
  - `b_well = median(TVT_input + Z - predicted_ANCC)` from known prefix.
  - Uses centroid KNN weighted 2D plane fit for formation tops and row-level
    `(X, Y)` KNN for ANCC.
- `romantamrazov/rogii-super-baseline-lb-12-602`
  - Reports LB `12.602`.
  - Similar formation surface features plus beam typewell features and
    LGB/XGB/CatBoost stack.
- `pilkwang/12-049-rogii-eda-leakageriskdiscussion`
  - Huge EDA/modeling framework.
  - Useful for leakage-safe feature definitions, typewell alignment features,
    candidate-path features, and validation thinking.
- `ravaghi/wellbore-geology-prediction-lightgbm`
  - Strong LightGBM baseline with GR/typewell beam features.

## New Main Strategy

Port the formation-surface stack into our own local scorer/notebook:

1. Spatial geological map.
   - Use train-only formation-top columns:
     `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`.
   - Build centroid KNN weighted 2D plane fit for each formation.
   - Build row-level `(X, Y)` KNN for ANCC using all training rows.
   - Exclude the same well during CV.

2. Geometry priors.
   - `tvt_from_plane = -Z + ANCC_plane + b_prefix`
   - `tvt_from_row_knn = -Z + ANCC_row_knn + b_prefix`
   - Use neighbor distance, row-level std, and plane-vs-row disagreement as
     uncertainty features.

3. Geosteering correction.
   - Calibrate typewell GR to known-prefix horizontal GR.
   - Local beam/particle search around last known TVT.
   - Allow up/down/flat moves.
   - Penalize motion heavily enough that flat geology remains the default.
   - Feed beam deltas, GR mismatch, and confidence gaps to residual model.

4. Residual/stacking.
   - Target: `TVT - last_known_TVT`.
   - Use LightGBM seeds + XGBoost; CatBoost optional on Kaggle GPU.
   - Treat formula priors as stack members as well as features.
   - Only use smoothing/slope clipping if OOF proves it helps.

5. Validation.
   - Keep 5-fold GroupKFold for comparability.
   - Add spatial-cluster folds by well centroid to better simulate hidden
     locations.
   - Do not submit tiny local gains. Require a meaningful gain, fold wins, and
     no catastrophic per-well regressions.

## Immediate Next Steps

1. Implement `rogii/bench/formation_stack_score.py` or equivalent module.
2. Reproduce a local OOF near the public `~12.1` range using train data.
3. Build a Kaggle notebook cell from our code, not a raw copy of public code.
4. Run at least 8 guarded hours of local ablations after v7 before any submit.
5. Submit only if the new geology-surface model clearly beats v5/v6 locally and
   passes the spatial-fold sanity check.

## Current Git/Push State

Important: `/Users/william/drilling_oil_gas` is currently not a git repository.
There is no `.git` directory under this workspace. To commit/push, initialize a
repo or move these files into the intended Git checkout/remotes first.

The Kaggle kernel remote is:

`wbfranci/rogii-eagle-ford-dtw-rts-v1`

But that is a Kaggle kernel, not a git remote.
