# Private-LB Shake-Up Strategy — 2026-05-07

Triggered by observing another Kaggle competition's final leaderboard:

  Public-LB top-12 was completely reshuffled on private. Δ ranks: +996,
  +190, +740, +959, +497, +912, +912, +1016, +935, +1013, +967, +644.
  The actual top-12 of the *private* board emerged from rank 700–1000+
  on public. People who fine-tuned to public got punished.

ROGII has the same structural risk: **public LB = 26% of test, private =
74%.** Tightly clustered scores (top is 11.247, ours v8 OOF is 12.03,
v6 LB is 13.854) plus a small public sample = high variance in the
public→private mapping.

## Implication for our optimization target

The competition metric is overall RMSE. But because:

  1. Test rows are clustered by well (~3000–7000 rows per well)
  2. A single catastrophic well's squared error dominates the sum
  3. Private 74% may contain 1–2 wells with neighbour-sparsity patterns
     not seen in public 26%

…the load-bearing metric for **private-LB stability** is
**max-well-RMSE**, not overall RMSE.

Math: a single 300-ft-RMSE well at 4000 rows contributes 3.6×10⁸ to the
squared-error sum. Fifty normal wells at 12-ft RMSE contribute 2.9×10⁷.
**One bad well doubles overall RMSE.** This is exactly why konbu17's
public 11.912 is risky on private — their max-well-RMSE is 300.85.

## What this changes about v8/v9/v10

| Model | OOF | Max well | Private-LB risk |
| --- | ---: | ---: | --- |
| konbu17 (KNN only) | 12.11 | 300.85 | high — one outlier and your 11.9 becomes 14+ |
| v8 (KNN + LGB) | 12.03 | 56.13 | medium — GBM caught the worst KNN cases |
| v9 (KNN + MLP + LGB) | TBD | likely <40 | low — MLP collapses the catastrophic tail |
| v10 stacker (v8 + v9 + …) | TBD | likely <30 | lowest — diversity averages noise |

The MLP's main contribution to v9 isn't lower mean RMSE (it's actually
+2 ft worse on the median: 14.67 vs 12.30 KNN). It's the **catastrophic-
tail collapse**: max-well-RMSE 300 → 166, wells>60ft 46 → 11. **That is
exactly the robustness signal that wins private LBs.**

## Practical reorientation

1. **Track max-well-RMSE as a co-metric** alongside overall RMSE in
   every benchmark from this point forward. A model with overall RMSE
   12.5 and max-well 30 is *better* than overall 12.0 and max-well 100
   for private-LB purposes.

2. **Multi-seed averaging is now top priority.** Cheapest variance
   reduction. v9's Kaggle cell already runs 3 LGB seeds; we should also
   ensemble multiple MLP seeds. (The neural-ANCC agent reported that 3
   MLP seeds gave only marginal gains — that needs re-checking under
   the max-well-RMSE metric, not the median.)

3. **Diverse model stacking** is now Priority 1 (the ongoing stacker
   agent). v6 (constant lookup) + v8 (KNN-LGB) + v9 (MLP-stacked-LGB)
   averaged with positive Ridge weights is canonical private-LB
   stabilization.

4. **Submit slot discipline.** 5 submits/day, 2 final selections.
   Probe substantively *different* architectures, not minor variants.
   Plan: v8 → v9 → stacker. Three submits, three different models, three
   different bias profiles. Pick the two with best OOF max-well-RMSE
   for final selection — even if one of them has slightly worse mean.

5. **Don't trust public LB above the noise floor.** With 26% of test,
   public RMSE has a non-trivial standard error. A 0.2 difference
   between two models on public LB is within noise. Trust local
   GroupKFold OOF.

6. **Be skeptical of the LB top.** Top of LB is 11.247 (lucataco). If
   they over-fit to public, they may fall to 13+. Our v9 at projected
   OOF ~11–12 with low max-well-RMSE could place much higher on private
   than on public — that's the upside.

## What we are *not* doing

- Tuning v9 LGB hyperparameters past the OOF-validated konbu17 base.
- Adding ad-hoc features that improve overall RMSE but not max-well-
  RMSE.
- Submitting incremental variants. Save the submit slots for genuinely
  diverse architectures.

## Cumulative architectural diversity portfolio

To minimise correlated errors in the ensemble we want models with
different inductive biases:

  - **Local + LGB**: konbu17/v8 (sharp on dense neighbours)
  - **Global + LGB**: v9 (smooth on sparse neighbours via MLP-ANCC)
  - **Constant baseline**: v6 (last_known_TVT_input; floor)
  - *Optional next:* anisotropic-kriging spatial layer (different
    error profile from KNN and MLP alike)
  - *Optional next:* sequence-aware model (transformer or PF) for the
    well-trajectory inductive bias the GBM lacks

The stacker can absorb 4–6 of these with non-negative Ridge weights;
each one adds insurance against a particular failure mode.
