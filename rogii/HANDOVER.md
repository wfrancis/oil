# ROGII Wellbore Geology Prediction — handover for Codex

**Status as of 2026-05-06:** rank 251 of 304, public LB **15.883 RMSE**. Top of LB is **11.247**. Auto Kaggle Agent baseline is **12.6**. We are below the auto-agent baseline. The gap to top is ~4.6 RMSE — closeable but only with work the previous entrant did not finish.

The competition runs until **2026-08-05**. Three months of iteration time remain.

---

## TL;DR

1. The Eagle Ford lateral stays in-zone; constant-extrapolation (just predict the last known `TVT_input` for the entire eval zone) **scores 11.28 locally on 50 train wells** but **15.88 on hidden test**. That ~4.6 RMSE gap is the work that remains.
2. The previous entrant's DTW-based v3 pipeline (alignment.py + geology.py + inference.py + alignment_v2.py) **scored 96.97 — a disaster**. Don't revive it as the primary predictor. The DTW finds spurious TVT motion that mechanically introduces +30 to +120 ft per-well positive bias.
3. **The asset that survived:** a Geology classifier trained on 773 train typewells with 99.88% OOF accuracy (`src/geology_classifier.py`, model at `data/geology_clf.joblib`). Test typewells lack the Geology column; this restores it.
4. **The thing to build next:** a small slope correction on top of the constant baseline. The 4.6 RMSE gap is consistent with the lateral having a real but small TVT drift during the eval zone (Eagle Ford regional dip ~0.001 ft/ft × eval zone ~3000 rows ≈ 3-30 ft of drift). Capture that drift and you close most of the gap.

---

## Codebase inventory

Working directory: `/Users/william/drilling_oil_gas/rogii/`

```
rogii/
├── HANDOVER.md                       <- this file
├── data/
│   ├── competition/                  <- unzipped competition data
│   │   ├── train/                    <- 773 train wells (h_well + typewell + .png)
│   │   ├── test/                     <- 3 example test wells (full hidden set on Kaggle)
│   │   └── sample_submission.csv
│   └── geology_clf.joblib            <- trained Geology classifier (1.4 MB, 99.88% OOF)
├── notebook/
│   ├── kernel-metadata.json          <- Kaggle kernel metadata; id wbfranci/rogii-eagle-ford-dtw-rts-v1
│   ├── kaggle_cell.py                <- v3 (DTW+RTS) cell content (118KB, base64-embeds 3 modules)
│   ├── kaggle_cell_v4.py             <- v4 (constant baseline) cell content (~3KB)
│   ├── submission.ipynb              <- current notebook ipynb (rebuilt by build_ipynb*.py)
│   ├── assemble.py                   <- builds kaggle_cell.py from src/ modules
│   ├── build_ipynb.py                <- wraps a .py cell into .ipynb
│   ├── build_ipynb_v4.py             <- v4 variant (uses kaggle_cell_v4.py)
│   └── v4_strategy.md                <- PhD ML strategist's doc — read this
└── src/
    ├── alignment.py                  <- v3 DTW (Numba JIT, single anchor, ~580 lines)
    ├── alignment_v2.py               <- multi-anchor + GR offset calibration (734 lines)
    ├── geology.py                    <- Eagle Ford formation priors (922 lines)
    ├── geology_classifier.py         <- LightGBM Geology classifier from train typewells (580 lines)
    └── inference.py                  <- v3 orchestrator (RTS smoother + per-well inference)
```

---

## What was tried, what worked, what failed

### Pipeline iterations

| Version | Approach | Local RMSE | LB RMSE | Notes |
|---|---|---|---|---|
| v1, v2 | (path bugs) | — | — | Empty submission, didn't score |
| v3 | Banded DTW + Eagle Ford geology priors + RTS smoother | 63.28 (5 wells) | **96.97** | Geology priors short-circuited because test typewells have empty Geology column. DTW alone introduced +30-120 ft per-well positive bias |
| v4 | `predict last known TVT_input` constant for entire eval zone | 11.28 (50 wells) | **15.88** | Validated locally; hidden test ~50% more variance than the train sample I picked |

### The 96.97 → 15.88 jump in one iteration

The biggest insight was discovering that **constant extrapolation beats every DTW variant we tried**. The Eagle Ford geosteerer's job is to keep the bit at constant TVT in the pay zone. Dropping all the alignment machinery cut RMSE by 6×.

### The 15.88 → 11.28 (local) → ~7-9 (top LB) gap

The 4.6 RMSE gap between local-50-well and hidden-test is the next entrant's job. Hypothesis: hidden test wells have some combination of:
- Longer eval zones (more drift)
- Larger TVT excursions during the eval zone (operators steering through dip)
- Wells with no `TVT_input` finite values (constant baseline returns 0, catastrophic)

### What the agents built that's still useful

**Geology classifier** (`src/geology_classifier.py`):
- Trained on 773 train typewells, 99.88% OOF accuracy (multiclass, 6 main South Texas Eagle Ford formations)
- Model file: `data/geology_clf.joblib` (~1.4 MB, fits easily in Kaggle output)
- Public API: `train_geology_classifier()`, `predict_geology()`, `fill_missing_geology()`
- Sub-zone → main formation mapping documented in module
- **Test typewells have empty Geology column.** This module fills it. v3 needed this and didn't have it — that's why geology priors short-circuited.

**alignment_v2.py** — multi-anchor DTW + per-well GR offset calibration:
- Uses ALL finite `TVT_input` rows as hard tie-points (v3 used only the last)
- Subtracts per-well GR baseline offset (median diff between horizontal-cased GR and typewell GR at matched TVT)
- The strategist also identified a **load-bearing bug in the band-centre slope** (`alignment.py:_dtw_forward`, `slope_num = n_t - 1 - j_seed`) which forces the warp to march to the END of the typewell. This was patched in `alignment_v2._dtw_forward_ext` to accept `slope_num_override` and is plumbed through `multi_anchor_dtw` with a dip-derived value.
- **Empirically: did not improve over v3 on local validation.** Even with the slope fix, DTW finds GR matches that lead the prediction astray. The DTW concept is not the right tool here; the constant baseline beats it.

**`notebook/v4_strategy.md`** — the PhD ML strategist's prioritized fix list. Useful as background reading; the `band_pct=0.25` recommendation is wrong (band exceeds typewell length, neutralizing the slope constraint), but the diagnosis of the band-slope bug is correct and instructive.

---

## Empirical findings (load-bearing data, do not re-derive)

These were measured on real competition data (`data/competition/train/`):

1. **TVT magnitude**: 11,000+ ft range (NOT 0-100 as the discussion-thread diagram suggests). TVT correlates with TVD-like values, NOT a thin stratigraphic offset.

2. **Test typewells have `Geology = null`** for every row. Train typewells have Geology populated for ~70-100% of rows.

3. **Test horizontal CSVs are missing the formation top columns** (`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`). They only have `MD, X, Y, Z, GR, TVT_input`. Train horizontals have all columns.

4. **Last-known-TVT_input baseline on 50 train wells:**
   - Mean RMSE: **11.28**
   - Median RMSE: 9.36
   - Min: 3.09, Max: 35.05
   - Mean bias: **-0.30** (essentially zero!)
   - Mean spread: 8.08

5. **Median of last 100 / 50 / mean of last 100 TVT_input** all give ~11.5 RMSE. Constant-of-last-known is the winner among these simple variants.

6. **Linear extrapolation** (fit `TVT(MD) = a + b·MD` on cased section, predict eval zone): mean RMSE **74.98**. Don't do this. Cased-section trends don't extrapolate; the lateral changes regime at PS.

7. **DTW v3 pipeline mean local RMSE: 63.28** (5 wells), with bias +53.36. The bias dominates the spread in every well.

8. **DTW alignment_v2 (multi-anchor + GR calibration + slope fix) mean local RMSE: 59.59** (band_pct=0.03), with bias +48.58. Still bad. The slope fix improved one well dramatically (RMSE 21 → 9.5) but made others slightly worse.

9. **The Kaggle Agent autonomous baseline scored 12.6.** Anything below 12.6 is below the autonomous agent.

---

## Top 5 priorities for the next entrant

### Priority 1: Diagnose the 11.28 (local) → 15.88 (LB) gap

This is the single most important move. The local validation said we'd score ~11. We scored 15.88. Why?

Hypotheses, in order of testing:
1. **Hidden test wells have longer eval zones**: download the Kaggle "v4" submission's logs (`kaggle kernels output wbfranci/rogii-eagle-ford-dtw-rts-v1 -p /tmp/v4`), inspect the per-well row counts and eval-zone sizes vs. our 50-well train sample.
2. **Hidden test wells have wells with no finite `TVT_input`**: my code falls back to predicting 0 for those. With TVT in the 11,000+ range, that's a ~11,000 ft error per row. **Even one such well in 200 catastrophically inflates RMSE.** Check this first.
3. **Hidden test has higher TVT drift during the eval zone**: validate by computing `TVT[-1] - TVT_input[last_finite]` for each train well — that's the "true drift" that constant extrapolation misses. If train distribution has tail above ±20 ft, hidden test could be similar.

**Concrete first step:** add a defensive fallback in v4. For wells with no finite `TVT_input`, predict the **median TVT across all train wells** (not zero). That alone might cut LB RMSE by several points if any hidden test well lacks an anchor.

### Priority 2: Slope-corrected constant baseline (v5)

Constant extrapolation captures the in-zone-steering prior; it does NOT capture the regional dip. Eagle Ford regional dip is 3-5°/mile SE = ~0.001 ft TVT / ft MD. Over a 3000-ft eval zone, that's 3 ft of TVT drift. That's likely a meaningful chunk of the 4.6 RMSE gap.

```python
# v5 sketch
def predict_well_v5(horizontal_df):
    md = horizontal_df["MD"].to_numpy()
    tvt_in = horizontal_df["TVT_input"].to_numpy()
    finite = np.isfinite(tvt_in)
    if not finite.any():
        return TRAIN_MEDIAN_TVT  # ~11800
    last_idx = np.flatnonzero(finite)[-1]
    last_tvt = tvt_in[last_idx]
    last_md = md[last_idx]
    # Robust dip from the last 100-300 finite rows (Theil-Sen)
    use_idx = np.flatnonzero(finite)[-300:]
    if len(use_idx) >= 30:
        from scipy.stats import theilslopes
        slope, _, lo, hi = theilslopes(tvt_in[use_idx], md[use_idx], alpha=0.95)
        # Cap slope at physically plausible dip range
        slope = np.clip(slope, -0.005, 0.005)
        # Down-weight if uncertainty is large
        ci_half = 0.5 * (hi - lo)
        if ci_half > 0.002:
            slope *= 0.3  # widely-uncertain dip → conservative correction
    else:
        slope = 0.0
    pred = last_tvt + slope * (md - last_md)
    pred[finite] = tvt_in[finite]
    return pred
```

The risk: Theil-Sen on a noisy cased section can give garbage. Keep the slope **small** and **down-weighted by CI uncertainty**. Compare on local validation before submitting.

### Priority 3: Per-row residual ML

For each train well, simulate the eval-zone gap (mask `TVT_input` from a randomly-chosen PS row to the end) and train LightGBM to predict the residual `TVT - last_known_TVT_input` from features:

```python
features = [
    'md_since_anchor',          # MD - last_finite_MD
    'gr',                       # current GR value
    'gr_z_in_well',             # z-scored against well's own GR
    'gr_window_mean_25',
    'gr_window_std_25',
    'z',                        # current TVD
    'z_minus_anchor_z',         # delta TVD
    'x', 'y',                   # spatial coordinates
    'cased_section_dip',        # Theil-Sen slope from cased section
    'cased_section_mean_tvt',
    'cased_section_std_tvt',
    'eval_zone_length',         # rows past anchor
    'gr_minus_typewell_at_pred_tvt',  # residual from typewell
    # If geology classifier is plugged in:
    'predicted_formation_argmax',  # one-hot or ordinal
    'predicted_formation_egfdl_prob',
]
```

GroupKFold by WELLNAME. Target: per-row TVT residual. With 773 wells × ~5000 rows = ~4M training samples (subsample to ~200K), LightGBM trains in minutes.

This residual model layered on top of v5 has the potential to cut RMSE meaningfully — it captures patterns like "wells where the GR drops sharply tend to drift up in TVT" that hand-engineered features miss.

### Priority 4: Use the Geology classifier output

`fill_missing_geology(typewell_df, model_path='/kaggle/working/geology_clf.joblib')` — drop-in replacement for the typewell input, gives test typewells a Geology column.

Use cases:
- Per-formation TVT priors: train wells in `EGFDL` have a typical TVT range. Test laterals classified into EGFDL should be near that range — soft constraint.
- Fault/anchor detection: sharp Geology label transitions in the typewell signal formation tops. Map the lateral's GR signature against those tops.

Caveat: this is the geology-priors path that v3 attempted and didn't beat constant. Use it as a **soft constraint**, not a hard one.

### Priority 5: Spatial models from (X, Y)

Train wells have known X, Y. Test wells have known X, Y. **Nearest-neighbour interpolation in (X, Y) space** of the per-well TVT signature gives a strong regional prior.

For each test well:
1. Find the K=5 nearest train wells by (X, Y) Euclidean distance
2. Compute the average `TVT_input(MD)` curve from those neighbours
3. Use it as a prior for the eval zone

This captures regional structural information (faults, dip, formation thinning) that the typewell alone cannot.

---

## Reproduction steps

**Setup:**
```bash
cd /Users/william/drilling_oil_gas/rogii
# Already in place:
# - data/competition/  (unzipped competition data)
# - data/geology_clf.joblib (trained classifier)
# - src/*.py (5 modules)
# - notebook/*.py + .ipynb (build/push scripts)
```

**Validate any new pipeline locally:**
```bash
python3 << 'EOF'
import sys, glob
sys.path.insert(0, "src")
import polars as pl
import numpy as np

# Drop in your predict_well_v5(h_df) here

train_files = sorted(glob.glob("data/competition/train/*__horizontal_well.csv"))[:50]
rmses, biases = [], []
for h_path in train_files:
    h_df = pl.read_csv(h_path)
    tvt_input = h_df["TVT_input"].to_numpy()
    tvt_true = h_df["TVT"].to_numpy()
    eval_mask = np.isnan(tvt_input)
    if eval_mask.sum() == 0:
        continue
    pred = predict_well_v5(h_df)  # YOUR FUNCTION
    err = pred[eval_mask] - tvt_true[eval_mask]
    rmses.append(float(np.sqrt(np.mean(err**2))))
    biases.append(float(np.mean(err)))
print(f"Mean RMSE: {np.mean(rmses):.2f}, mean bias: {np.mean(biases):+.2f}")
EOF
```

Target: beat 11.28 mean RMSE. Aim for under 10.

**Build the Kaggle cell:**
```bash
# Edit notebook/kaggle_cell_v4.py with your new logic
python3 notebook/build_ipynb_v4.py
cd notebook && kaggle kernels push -p .
```

The kernel id is `wbfranci/rogii-eagle-ford-dtw-rts-v1`. The notebook re-runs on push and you can monitor logs via:
```bash
kaggle kernels output wbfranci/rogii-eagle-ford-dtw-rts-v1 -p /tmp/run
cat /tmp/run/rogii-eagle-ford-dtw-rts-v1.log | python3 -m json.tool | less
```

**Submit:**
- Click "Submit to Competition" in the Kaggle UI on the notebook output page (currently the only working path — the API submit isn't wired here).
- 5 submissions per day. 2 final-submission slots.

**Path discovery on Kaggle:**
The competition data lives at `/kaggle/input/competitions/rogii-wellbore-geology-prediction/test/` (TWO levels deep). The `kaggle_cell_v4.py` already auto-discovers this — preserve that walk in any new cell.

---

## Hard constraints to respect

1. **Code Competition**: submission is a Notebook re-run on hidden test set. CSV submission via `kaggle competitions submit` is NOT accepted. Submit via Kaggle UI on the notebook page.
2. **No internet at scoring time**: any external data must be packaged as a Kaggle Dataset and added as input to the notebook.
3. **9-hour CPU/GPU budget**: more than enough for our scale. Don't over-engineer for speed.
4. **Notebook output `submission.csv`**: must be at `/kaggle/working/submission.csv` with columns `id, tvt`.
5. **id format**: `{WELLNAME}_{row_index}` for every row where `TVT_input` is NaN. Do not emit rows for finite-`TVT_input` rows.
6. **Path conventions:** `/kaggle/working/` for outputs, `/kaggle/input/competitions/<slug>/` for competition data. The Geology classifier joblib should be placed in `/kaggle/working/` at runtime if you bundle it into the notebook (or ship it as a dataset input — see below).

---

## Shipping the Geology classifier joblib to Kaggle

The classifier joblib is 1.4 MB. Two options:

1. **Bundle in notebook as base64 string**: write a setup cell that decodes and writes to `/kaggle/working/geology_clf.joblib`. Same pattern as v3's `kaggle_cell.py` does for the .py modules. Add `assemble_v5.py` that base64-encodes the joblib alongside the source modules.

2. **Upload as a Kaggle Dataset**: `kaggle datasets create -p <dir>`, then add it as a secondary input on the notebook. Cleaner separation; reusable across notebook versions. The kernel-metadata.json `dataset_sources` field accepts dataset slugs.

Option 1 is simpler for one-off; option 2 is right if you'll iterate on the classifier separately.

---

## Open hypotheses to test (anyone)

1. **Sub-zone vs main-six classification**: classifier is trained at main-six granularity. The 22 sub-zone labels (LBHL, UEGFD TGT, etc.) are operator-specific landing zones. Maybe predicting at sub-zone resolution gives the model finer geological signal.

2. **The cross-section .png files in train/**: hand-drawn geosteering interpretations. Visual encoding of TVT labels. A small ConvNeXt-tiny trained to predict layer dip from these PNGs could give a per-well "structural prior". Not done.

3. **Multi-typewell averaging**: each well has one typewell, but nearby train wells' typewells are also informative. Cluster typewells by (X, Y) and average within cluster.

4. **Public/private split shake-up risk**: 26%/74% split. Our public score is 15.88; private will likely be in the 13-18 RMSE band. Expect 5-15 places of churn at this rank level. Don't over-tune to public.

5. **External Eagle Ford LAS data** (USGS, Texas RRC, BEG): rules allow it. Could be pretraining for a Geology classifier with even higher accuracy or for the cross-section vision encoder.

---

## Final note on what NOT to do

- **Don't revive the DTW pipeline as the primary predictor.** It's load-bearing wrong for in-zone laterals. Keep alignment_v2 around as a feature extractor (e.g., DTW-warped GR similarity as an ML feature), not as the prediction.
- **Don't trust local validation alone.** Local 11.28 → LB 15.88 is a 50% variance increase. Always cross-check against the public LB before declaring success.
- **Don't waste agent budget on the geology priors module** until the constant-baseline-with-slope-correction is competitive. Geology priors helped exactly zero on v3.
- **Don't tune to public LB scores.** With 26% public test, public-LB optimization is genuinely random above the noise floor.

---

## Contact / context

- Kaggle account: **wbfranci** (William)
- Kernel: `wbfranci/rogii-eagle-ford-dtw-rts-v1` (current version: 4)
- Competition close: **2026-08-05**
- Top LB at handover: 11.247 (lucataco)
- Our position at handover: **rank 251, 15.883 RMSE**

Operator brief: William is the architect, not the coder. AI agents implement under his direction. He thinks in HFT-mechanical-sympathy terms (platform speed, iteration throughput, benchmark-driven engineering). Match that level. The previous entrant deployed three Opus agents in parallel for v3+v4 — that pattern works for parallel module development but the coordination overhead matters; a single agent with the right brief is often faster.

Memory of strategic context lives in `~/.claude/projects/-Users-william-drilling-oil-gas/memory/`.
