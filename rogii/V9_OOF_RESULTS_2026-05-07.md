# v9 OOF Results — 2026-05-07

Full 5-fold GroupKFold OOF over 773 train wells / 3.78M training rows /
166 features (130 KNN+plane+formula features + 34 MLP-derived features +
2 admin). 1 LGB seed (42), beam features OFF (would add ~0.1 if on),
no XGB, no ridge stack (those add 0.2–0.4 to v10/v11).

```
fold rmses    [12.5686, 10.5477, 10.7895, 11.8961, 11.0717]
overall RMSE  11.4059
per-well median   7.83 ft
per-well mean     9.28 ft
per-well p90     15.99 ft
per-well max     56.10 ft
total wall time  4370 s = 73 min
```

## Versus v8 (KNN-only baseline)

| metric | v8 | v9 | Δ |
| --- | ---: | ---: | ---: |
| overall RMSE | 12.03 | 11.41 | **-0.62** |
| median well | 8.16 | 7.83 | -0.33 |
| mean well | 9.71 | 9.28 | -0.43 |
| p90 well | 16.64 | 15.99 | -0.65 |
| max well | 56.13 | 56.10 | -0.03 (essentially unchanged) |

The MLP-ANCC features improve typical-case accuracy (median, mean, p90)
but DO NOT reduce max-well-RMSE meaningfully. The GBM already absorbed
most of the imputer-level catastrophic outliers (MLP-imputer level
max=165.66 → after-GBM max=56.10), so adding MLP features to v9 doesn't
reduce the v8 GBM's residual tail.

## Anchor-shrinkage sweep (rogii/bench/anchor_shrinkage_results.json)

```
method                    overall   max well    delta vs base
base (no post-process)    11.41     56.10       -
constant baseline (α=0)   15.91     70.64       +4.50 / +14.5
α=0.5                     12.66     63.16       +1.25 / +7.1
α=0.7                     11.86     60.28       +0.45 / +4.2
α=0.85 (v10 default)      11.53     ~58         +0.12 / +1.5
α=0.9                     11.45     57.47       +0.04 / +1.4
α=1.0 (no shrinkage)      11.41     56.10       0    / 0

band=15  hard cap         12.09     60.94       +0.68 / +4.8
band=20                   11.72     58.98       +0.31 / +2.9
band=25                   11.53     57.45       +0.12 / +1.4
band=30                   11.44     56.53       +0.03 / +0.4
band=40 hard cap          11.37     56.10       -0.04 / 0      <-- WIN
band=60                   11.40     56.10       -0.01 / 0
```

**Multiplicative shrinkage (α<1) is uniformly bad** — it hurts overall
RMSE *and* doesn't reduce max-well-RMSE. This refutes my v10 design
choice of α=0.85.

**Hard cap at ±40 ft is the only winning intervention.** -0.04 overall
RMSE for free, no max-well change. Population p99 of
`eval_offset_from_anchor` is 37.7 ft, so band=40 chops only the
extreme tail.

Calibrated v10 / v11 to `SHRINK_ALPHA=1.0` and `HARD_CAP_BAND=40.0`.

## What this teaches

1. **The catastrophic-tail wells are a modeling problem at the
   feature/architecture level**, not a post-processing one. Anchor
   shrinkage was the wrong lever.

2. **The GBM already does most of what shrinkage was supposed to do**
   — the formula features include `last_known_TVT` directly, and the
   target is delta-anchored. The GBM will predict 0 (=anchor) when
   uncertain. Adding multiplicative shrinkage on top double-counts.

3. **Hard cap at p99 is a free safety net.** Costs essentially nothing
   on overall RMSE, prevents catastrophic predictions on hidden test
   wells that might be even more drift-prone than the OOF.

4. **The remaining max-well-RMSE 56 ft is intrinsic to v9.** To go
   below that, we need targeted DRIFT correction — likely v11's
   aniso layer (different inductive bias) or a per-row uncertainty
   gate (more complex).

## Updated submission lineup

| Submission | OOF measured / projected | Max-well | Description |
| --- | ---: | ---: | --- |
| v8 (KNN+plane+GBM, 3 seeds, beam, XGB, ridge, EWM) | ~11.85 LB-projected | ~56 | konbu17 baseline |
| v9 (+ MLP, 5 LGB + 3 MLP seeds, beam, XGB, ridge, EWM, band=40) | 11.41 → ~11.20 OOF | 56 | adds neural ANCC layer |
| v10 (= v9 — same config, different name kept for clarity) | ~11.20 | 56 | same as v9 with calibrated cap |
| v11 (+ aniso-exponential, all the above) | ~10.80 | TBD | adds anisotropic-kriging spatial layer |

Strategy: submit v8, v9/v10, and v11 across the available daily slots.
Pick the final 2 selections on the Pareto front of (overall LB,
max-well-RMSE-from-OOF). The likely picks: v11 (sharp) + v9 (insurance
if v11 has unexpected max-well-RMSE blow-ups on hidden test).

## Files

- `/tmp/v9_oof.csv` (3.78M rows; row-level OOF predictions for v9)
- `rogii/bench/anchor_shrinkage_results.json` (sweep numbers)
- `rogii/bench/anchor_shrinkage_score.py` (the sweep script)
- `rogii/src/anchor_shrinkage.py` (the shrinkage primitives)
- `rogii/notebook/{kaggle_cell_v10.py,kaggle_cell_v11.py}` (calibrated)
