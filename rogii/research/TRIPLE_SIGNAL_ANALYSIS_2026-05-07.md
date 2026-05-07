# Public-kernel scout: Triple-Signal beats konbu17 — 2026-05-07

Top public notebook on the ROGII competition is no longer konbu17.
A user `shinyanagai123` published `Triple-Signal Beam Search + Dual PF
+ LightGBM` (4 hours ago) at **public LB 11.284**, beating konbu17's
11.912 by 0.6 RMSE.

(For context: top of LB at handover was 11.247. Triple-Signal at
11.284 is essentially at the top.)

Source saved at `rogii/research/public_kernels/triple-signal-beam-
search-dual-pf-lightgbm.ipynb`.

## Architecture

5 signals fed into a single LightGBM regressor (no XGB, no Ridge stack,
no multi-seed):

  1. **TVT Particle Filter (Z-velocity coupled)** — 500 particles
     tracking (TVT, dTVT/dMD). The velocity is informed by `dz/dmd`
     via a per-well learned linear coupling
        `v_tvt = beta * dz_dmd + intercept,  sigma = std(residuals)`
     where (beta, intercept, sigma) come from `np.linalg.lstsq` on the
     visible prefix's (dz, dtvt) per-row deltas.
     This is the load-bearing physics: when the operator angles the
     bit, dz/dmd directly drives dtvt/dmd. Likelihood is the typewell
     GR match at predicted TVT (Gaussian, sigma calibrated per-well
     from the prefix residual std).

  2. **ANCC Particle Filter (S = TVT + Z tracker)** — 500 particles
     tracking the smooth surface S = TVT + Z. Just (pos, rate) state,
     simpler dynamics. Likelihood: typewell GR at TVT = S - Z.
     Used as the PRIMARY pf signal in feature_builder when both are
     available (the Z-velocity PF is the fallback).

  3. **Beam Search (cons + loose)** — same Viterbi we already have.
     Two configs: `(beam=10, move_cost=20, emit_scale=144, radius=3)`
     and `(beam=10, move_cost=8, emit_scale=64, radius=3)`.

  4. **Spatial ANCC plane fit** — same as konbu17's. Per-well centroid
     KNN K=10, weighted 2D plane, std-scaled X/Y kdtree.

  5. **Dense ANCC IDW** — like konbu17's row KNN but DOWNSAMPLED:
     40 evenly-spaced points per well (vs all rows). K=20 IDW. Faster
     and smoother. Builds neighborhood-std as an uncertainty feature.

All 5 signals contribute their prediction + uncertainty + cross-
comparison-with-the-others into a feature matrix. The LGBM gates them.

## Why it works

- **Z-velocity is the strongest physical signal we've ignored.** The
  operator's drilling action directly couples (dz, dtvt) via the
  bit's pitch angle. Their PF learns this coupling on the prefix (per
  well) and uses it for forward propagation. We just emit dz/dmd as
  a static feature; they USE it as the PF transition kernel.

- **Tracking S = TVT + Z is smarter than tracking TVT directly.**
  S is a smooth formation-surface coordinate; TVT changes more
  jaggedly along the lateral. The rate of S is small and stable.
  Convert back to TVT at inference: TVT = S - Z.

- **PF as a SIGNAL not a PREDICTOR.** Our earlier PF prototype
  failed because we tried it as a standalone predictor. Triple-
  Signal feeds the PF mean + std into a GBM that has dozens of
  other signals to weight against. The GBM learns when to trust
  the PF (e.g., when the prefix typewell-GR alignment is good and
  the PF std is small).

- **No OOF measurement.** Their notebook trains on 100% of train
  data with a single fit, no GroupKFold. They iterate against the
  public LB directly. Risky — but the structure of their model is
  rich enough that they're probably not over-fitting much.

## What's in their feature matrix (~50 features)

```
PF block (5):     pf_pred, pf_std, pf_delta, pf_std_trend, pf_std_ratio
Spatial (5):      spatial_tvt, spatial_delta, spatial_ancc, spatial_dist, c_well
Dense (6):        dense_tvt, dense_delta, dense_ancc, dense_dist, c_well_dense, dense_nb_std
Dense reliability (4):  dense_known_rmse, dense_known_bias, dense_known_max_err, dense_known_nb_std
Beam (5):         beam_cons, beam_loose, beam_cons_delta, beam_loose_delta, beam_gap
Cross (6):        pf_vs_spatial, pf_vs_spatial_abs, pf_vs_dense, spatial_vs_dense,
                  pf_vs_beam_cons, dense_vs_beam_cons
Position (10):    md_from_ps, md_from_ps_sq, z_from_ps, x_from_ps, y_from_ps, xy_dist,
                  eval_frac, z, dz_dmd, dx_dmd, dy_dmd
GR (5):           gr, gr_roll21, gr_std21, gr_roll51, gr_std51, gr_cumdev
Context (~9):     last_known_tvt, known_len, eval_len, eval_len_ratio, slope_all,
                  slope_recent, slope_z_recent, known_tvt_range, known_tvt_std,
                  known_gr_mean, known_gr_std, known_tw_rmse,
                  tw_tvt_range, tw_gr_mean, tw_gr_std
TW alignment (3): tw_gr_at_pf, gr_minus_tw_at_pf, gr_tw_off_-60, gr_tw_off_60
Slope-baseline (4): baseline_slope_recent, pf_minus_slope, spatial_minus_slope,
                    dense_minus_slope
```

That's ~55 features total (vs konbu17's ~80, our v9's 166).

## LightGBM hyperparameters (notably gentler than konbu17/v9)

```
n_estimators=3000   (konbu17/v9: 5000)
learning_rate=0.03  (konbu17/v9: 0.06)        — half the LR
num_leaves=64       (konbu17/v9: 89)
min_child_samples=50 (konbu17/v9: 10)         — 5x larger leaves
reg_lambda=5.0      (konbu17/v9: 87.28)       — much less regularization
reg_alpha=0.1       (konbu17/v9: 2.03)
subsample=0.8       (konbu17/v9: 0.645)
colsample_bytree=0.8 (konbu17/v9: 0.821)
```

Their model is BIGGER per leaf (50 min samples × 64 leaves) but with
LESS regularization. That fits because they have FEWER but
HIGHER-QUALITY features (the 2 PFs replace dozens of GR-rolling /
typewell-offset / formation-stack features).

## Implications for our v12

**v12 = v9 (or v11) + Triple-Signal's TVT-PF + ANCC-PF as additional features.**

We already have:
- Spatial ANCC (plane fit) ✓
- Dense ANCC (row KNN with n_q=8000) ✓
- Beam search cons + loose ✓
- MLP-ANCC (3 seeds) ← new in v10/v11
- Aniso-exponential ← new in v11

We're MISSING:
- **TVT Particle Filter (Z-velocity coupled)**
- **ANCC Particle Filter (S = TVT + Z tracker)**

Both are ~200 lines each (already in their notebook). Per-well wall
time is fast (~0.5–2 sec).

The architectural bet: WE HAVE MORE FEATURES THAN THEY DO. Adding
their 2 PFs to v11's stack should give a STRICTLY MORE INFORMATIVE
feature set. The GBM gates everything. Expected v12 OOF: 10.5–11.0.

## Risk

We don't know how much of their LB 11.284 comes from their PFs
specifically vs the simpler GBM hyperparameters they use. Could be:
- 80% from the PFs (best case for us — adding them straight wins 0.5)
- 50% from each (we still gain 0.3)
- 20% from PFs (smaller gain)

Plus: their notebook trains on 100% train with no OOF. Our v9 OOF
already measures 11.41 with stricter validation. If we add their PFs
and use our 5-fold OOF + 3 LGB seeds + Ridge stack on top, it should
both score better AND generalize better on private LB.

## Plan

1. **Wait for v9 Kaggle result** (currently RUNNING). Validates our
   v9 OOF=11.41 vs the actual public LB.
2. **Port their `run_pf_z_velocity` and `run_pf_ancc`** into
   `rogii/src/particle_geosteer.py` (replace our earlier prototype
   with their working version) or as a new module.
3. **Add PF-derived features** to `rogii/src/feature_builder.py`:
   `pf_pred`, `pf_std`, `pf_delta`, `pf_std_trend`, `pf_std_ratio`,
   `pf_vs_spatial`, `pf_vs_dense`, `pf_vs_beam_cons`,
   `tw_gr_at_pf`, `gr_minus_tw_at_pf`, `gr_tw_off_±60`,
   `pf_minus_slope`. ~15 new features per row.
4. **Build kaggle_cell_v12.py** = v11 + PF features. Same LGB×3 +
   XGB + Ridge + EWM on top.
5. **Submit v12** as the high-water mark candidate.

## Files

- `rogii/research/public_kernels/triple-signal-beam-search-dual-pf-lightgbm.ipynb` —
  full notebook source.
- `rogii/research/TRIPLE_SIGNAL_ANALYSIS_2026-05-07.md` — this file.
