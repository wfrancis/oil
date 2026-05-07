# Submit Gate

## Hard Submission Lock

After the v6 attempt tied v5 at `13.854`, the next submission has a hard
minimum compute requirement:

> Do not submit again until at least 8 hours of active CPU/search wall time have
> been recorded after the latest Kaggle submission.

Use the guarded wrapper for future submissions:

```bash
python3 rogii/bench/submit_guard.py status
./rogii/bench/safe-submit -c rogii-wellbore-geology-prediction ...
```

Raw `kaggle competitions submit ...` is forbidden until
`submit_guard.py check` passes.

The local tester should answer one question before we spend a Kaggle submit:

> Did this change beat the current submitted strategy under a leak-free train-well holdout?

Use the 5-fold row-weighted RMSE. Do not use the old first-50-wells sample; that
sample made v4 look like `11.28` locally and was too lucky.

## Calibration From The v5 Submit

| Strategy | Local 5-fold row-weighted RMSE | Public LB RMSE |
| --- | ---: | ---: |
| v4 constant | 15.9099 | 15.883 |
| v5 residual LightGBM, shrink 0.75 | 14.5861 | 13.854 |

The tester called the direction correctly:

| Delta | Local | Public LB |
| --- | ---: | ---: |
| v5 - v4 | -1.3238 | -2.029 |

So the current local CV is useful as a submit gate, but conservative for v5.

## Fast Baseline

Fast row-weighted control, enough to catch scorer drift:

```bash
./rogii/bench/score-rust run-baseline \
  --strategy constant \
  --n-folds 1 \
  --fold 0 \
  --json
```

This scores all `773` train wells once. On the local M1 Pro native build it is
about `0.20-0.25s` after warmup.

Fold-level control, when you need the same per-fold table as the residual gate:

```bash
mkdir -p /tmp/rogii_gate/constant
for f in 0 1 2 3 4; do
  ./rogii/bench/score-rust run-baseline \
    --strategy constant \
    --n-folds 5 \
    --fold "$f" \
    --json > "/tmp/rogii_gate/constant/fold_${f}.json"
done

python3 rogii/bench/summarize_scores.py /tmp/rogii_gate/constant/*.json
```

The v4 constant control should stay near `15.91` row-weighted RMSE. The
fold-level loop is about `0.23s` of scorer time after warmup. If the RMSE moves
materially, the fold definition or scoring harness changed.

## v5-Style Residual Gate

```bash
mkdir -p /tmp/rogii_gate/residual
for f in 0 1 2 3 4; do printf '%s\n' "$f"; done | xargs -P 4 -I{} sh -c '
  python3 rogii/bench/local_score.py run-residual \
    --n-folds 5 \
    --fold "{}" \
    --train-rows 1200000 \
    --shrink 0.75 \
    --threads 2 \
    --json > "/tmp/rogii_gate/residual/fold_{}.json"
'

python3 rogii/bench/summarize_scores.py /tmp/rogii_gate/residual/*.json
```

Submit only when the candidate clears the current best local gate by a real
margin. A good default rule is:

- row-weighted RMSE improves by at least `0.35`
- at least `3/5` folds improve
- no fold regresses by more than `0.50` unless the row-weighted gain is large

For non-LightGBM candidates, use `local_score.py export`, run the candidate on
the exported `test/` folder, then score the resulting `id,tvt` CSV with
`local_score.py score-csv`.
