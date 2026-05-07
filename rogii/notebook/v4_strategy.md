# v4 Strategy: from RMSE 96 to RMSE <30

## Diagnosis recap

5-well local bench: mean RMSE 63.3 ft, mean |bias| ≈ 53 ft, mean spread ≈ 33 ft. **Bias is POSITIVE in all 5 wells** — systematic deep-drift. RMSE² ≈ bias² + spread², so eliminating bias drops local RMSE to ~33; tightening spread to ~10 ft (top LB regime) gets us under 15. Public LB 96 vs local 63 = harder hidden test (longer eval zones, fewer pinned rows), same failure mode.

## Root cause: hard-coded band-centre slope in `_dtw_forward`

`alignment.py:148-162` sets `slope_num = n_t - 1 - j_seed`, `jc(k) = j_seed + slope_num · k / (L-1)`. **The band centre is forced to march from `(i_anchor, j_anchor)` all the way to `(N_h-1, n_t-1)`** — it assumes the warp traverses the ENTIRE remaining typewell column. A typewell logs ASTNU→BUDA over ~400-600 ft of TVT; a real lateral steers ±5-15 ft within ONE landing zone. The slope is wrong by 1–2 orders of magnitude. Once `j_centre > j_seed + band`, the Itakura monotonicity (`j_lo ← max(j_lo, j_seed)`) prevents recovery. Predictions drift deeper, monotonically. **This is the per-well positive bias.**

## Top 3 fixes ordered by leverage

1. **Fix DTW band-centre slope** — derive j-progression rate from cased-section apparent dip × typewell `dTVT/drow`, not "warp ends at last row". RMSE: 96 → ~50.
2. **Multi-anchor DTW + GR offset calibration** — use ALL finite TVT_input rows as soft pin-points; current pipeline discards 99% of alignment info. RMSE: ~50 → ~32.
3. **GR-only Geology classifier on test typewells** — LGBM on train typewells, predicts Geology on test typewells, restores EGFDL/BUDA floor and Bayesian nudge. RMSE: ~32 → ~22.

## Detailed analysis per fix

### Fix 1: Replace fixed-end band-centre slope with dip-derived slope

- **Hypothesis**: 53-ft bias is a direct mechanical consequence of `slope_num = n_t - 1 - j_seed`. The lateral is sub-horizontal in TVT; the typewell is vertical; the warp shouldn't advance through 600 ft of typewell over the lateral. With cased-section apparent dip ≈ 0.001 ft/ft (already computed by `regional_dip_prior`), the lateral's eval-zone TVT change is `dip · L_lateral ≈ 5-15 ft`, not 400-600 ft.
- **Implementation**: in `_dtw_forward`, override `slope_num` with `clip(round(L · expected_dip_ft_per_row / median_dtvt_per_typewell_row), 0, 2·denom)`. Plumb `expected_dip_ft_per_row` through `dtw_align_gr → predict_well_dtw → inference.predict_well` (where `dip_est` already exists). Default to 0 when dip non-finite. Bump `band_pct` default 0.15 → 0.25 to absorb the slope reduction.
- **Estimated RMSE**: bias near-zero after fix (residual ~2-5 ft from Theil-Sen dip uncertainty). Public 96 → ~50. Local 63 → ~33. Spread unchanged.
- **Risk**: faulted laterals or steep steering need slope > 0; the dip estimator handles this. Fall back to slope=0 if `dip_std > 0.005`.
- **Effort**: **Low** — ~20 LOC, one signature change. Numba re-JITs transparently.

### Fix 2: Multi-anchor DTW + cased-section GR offset calibration

- **Hypothesis**: v3 anchors at ONLY the last finite TVT_input. The cased section has 100s-1000s of finite TVT_input rows where `j_known(i) = searchsorted(t_tvt, h_tvt_in[i])` is GROUND TRUTH. Discarding them: (a) loses warp-shape constraint along the cased portion; (b) forfeits the only joint `(h_GR_at_known_j, t_GR[j])` calibration sample we have for cancelling the GR baseline offset between tools (your concern A).
- **Implementation**: (1) Pre-compute `j_known(i)` for every finite TVT_input row. (2) Run DTW over the full `[0, N_h)` with each pinned row contributing `+λ·(j - j_known(i))²` to the DP cost (λ ≈ 1000; effectively forces warp through). (3) Free byproduct: `gr_offset = median(h_GR[finite_in] - t_GR[j_known(finite_in)])`; subtract from horizontal GR before z-scoring. **This addresses issue A with data, not heuristics.**
- **Estimated RMSE**: independent of Fix 1 — Fix 1 fixes mean slope; Fix 2 fixes shape and GR baseline. Cuts spread from ~33 to ~20 ft (phantom-GR-match jitter near zone boundaries gets pinned out). Combined LB: 50 → ~32. Local: 33 → ~20.
- **Risk**: a single bad TVT_input acts as a hard pin and biases the warp. Guardrail: robust-fit `j_known` vs `i` (Theil-Sen), reject points with residual >5σ before pinning.
- **Effort**: **Medium** — touches the Numba kernel; needs band feasibility logic around pinned rows. 80-120 LOC.

### Fix 3: GR-only Geology classifier on test typewells

- **Hypothesis**: v3 geology priors (floor/ceiling/nudge) short-circuit on test because test typewells have `Geology = null`. Train typewells have Geology + GR + TVT — a fully supervised problem `Geology = f(GR, depth_in_typewell, GR_window_stats)`. 773 typewells × ~1000 rows is a strong training set for LGBM.
- **Implementation**: (geologist agent owns the classifier; see below.) On inference, predict Geology for every test typewell row before calling `fit_formation_gr_model`. After Fix 1+2 reduce DTW bias, the EGFDL/BUDA contact floor remains the highest-leverage geology constraint: laterals never drill more than ~10 ft into the Buda.
- **Estimated RMSE**: Geology floor catches the 5-15% of rows that DTW still drifts past the contact (residual phantom matches). Posterior nudge picks up another 2-4 ft. Combined: ~10 ft RMSE win. 32 → ~22.
- **Risk**: classifier mispredicts contact location → floor clips wrongly. Guardrails: only apply floor if classifier confidence on the contact row ≥0.7; raise nudge log-prob gap threshold from 2 to 3 nats; reject the classifier entirely if EGFDL+BUDA OOF F1 < 0.80.
- **Effort**: **Medium** — ~150 LOC for the classifier + a training notebook; ~30 LOC inference integration in `predict_well`.

## Fixes considered but deprioritized

- **(D) Residual-correction model trained on v3 DTW residuals**: dominant residual signal is the band-slope bug, not a learnable feature. Training on it learns "subtract 53 ft" which doesn't generalize once Fix 1 lands (bias scales with lateral length / typewell extent, not constant). Revisit AFTER Fix 1 if local RMSE > 25.
- **(E) Step-pattern relaxation (add `(2,1)`)**: helps only when the lateral moves through TVT FASTER than typewell density — the rare case. Dominant failure is the opposite. Defer.
- **(A) GR baseline calibration standalone**: subsumed by Fix 2 (free byproduct of multi-anchor calibration).
- **Fault-jump detector**: `geology.py:884` is a stub. Not worth building until base RMSE < 25.

## Concrete instructions for the SWE agent

**`/Users/william/drilling_oil_gas/rogii/src/alignment.py`** (Fix 1):

1. Add `slope_num_override: int = -1` to `_dtw_forward`. When ≥ 0, use it instead of `n_t - 1 - j_seed`.
2. Add `expected_dip_ft_per_row: float | None = None` to `dtw_align_gr`. Convert via `dtvt_per_row = median(abs(diff(t_tvt)))`, `slope = abs(expected_dip_ft_per_row) / dtvt_per_row`, `slope_num_override = clip(round(slope * (L-1)), 0, 2*(L-1))`. Default to 0 if dip non-finite.
3. Bump `band_pct` default 0.15 → 0.25.
4. Add `expected_dip_ft_per_row` to `predict_well_dtw` and forward.

**`/Users/william/drilling_oil_gas/rogii/src/inference.py`** (Fix 1):

5. In `predict_well` step 4: `raw = predict_well_dtw(horizontal_df, typewell_df, expected_dip_ft_per_row=dip_est)`. (`dip_est` already computed at step 3.)

**`alignment.py`** (Fix 2):

6. Refactor `_dtw_forward` to accept `pin_i: int64[:]`, `pin_j: int64[:]`, `pin_w: float64[:]` arrays (Numba-friendly) and add `+pin_w[k] · (j - pin_j[k])²` at every pinned row.
7. In `predict_well_dtw`: compute `j_known(i) = searchsorted(t_tvt, h_tvt_in[i])` for every finite TVT_input row (subsample to ≤ 100 evenly-spaced points if cased section is huge — keeps DP cost bounded). Pass as pin-points with weight 1000.
8. Same step: `gr_offset = median(h_gr[finite_in] - t_gr[j_known(finite_in)])`; subtract from horizontal GR before `_safe_window_zscore`.

**Tests** (`/Users/william/drilling_oil_gas/rogii/tests/`):

- Unit: `_dtw_forward` with `slope_num_override=0` and identical GR at one j produces constant-j path.
- Integration: `predict_well` on 5 local wells; assert mean |bias| < 10 ft after Fix 1, < 5 ft after Fixes 1+2.

## Concrete instructions for the geologist agent

Build `/Users/william/drilling_oil_gas/rogii/src/geology_clf.py` (new file):

- **Training set**: every train typewell row with finite GR + `Geology ∈ FORMATION_ORDER`. Features per row: `gr` (raw), `gr_zscore_in_well` (robust: median, IQR/1.349), `depth_pct = (TVT - TVT.min())/(TVT.max() - TVT.min())`, `gr_window_mean_25`, `gr_window_std_25` (reflect-padded), `gr_gradient_25` (Sav-Gol deriv1, win=25, ord=2), `gr_lag_50`. Target: Geology label.
- **Split**: `GroupKFold(n_splits=5)` grouped by typewell name (NEVER leak rows from same well into val).
- **Model**: LightGBM multiclass, `num_class=6`, `lr=0.05`, `num_leaves=63`, `min_data_in_leaf=200`, `n_estimators=400`, early-stop on val log-loss. Reject model if EGFDL+BUDA OOF F1 < 0.80.
- **Persist** to `<src>/geology_clf.pkl`; bake into kernel via `kaggle_cell.py`.
- **Inference**: `predict_typewell_geology(typewell_df) → typewell_df_with_Geology`. Inject into `inference.predict_well` BEFORE `fit_formation_gr_model`.

## Estimated RMSE after fixes

Cumulative public LB:

| Stage          | Public LB RMSE |
|----------------|----------------|
| v3 baseline    | 96.97          |
| + Fix 1 (slope)| ~50            |
| + Fix 2 (multi-anchor + GR offset) | ~32 |
| + Fix 3 (geology clf)              | ~22 |

Stretch (+ residual model, fault detector): ~15-18.

## Open questions for the integrator

1. **Test typewell TVT populated?** Strategy assumes test typewells have TVT + GR but missing Geology. If TVT is also null, the DTW approach is moot and we'd need synthetic typewells from cross-train clustering.
2. **Median apparent dip distribution train vs hidden test.** If hidden test laterals span >100 ft of TVT (crossing zones), Fix 1's per-well dip estimator handles it, but worth confirming public-test apparent dip matches train.
3. **Numba available at submission time?** Fallback Python loop is ~50× slower; verify `_HAS_NUMBA = True` in the kernel runtime before counting on Fix 1's <1s/well budget.
4. **Public 96 vs local 63 = +33 RMSE gap.** Is this fully harder-hidden-test, or is there an integration bug in `kaggle_cell.py`? Sanity check: run `predict_well` on visible `test/` wells via the assembled kernel vs directly via modules; outputs must match bit-for-bit.
