"""Lightweight pilot for multi-seed LGB variance-reduction.

Uses v8 features (no MLP) on a 200-well subset to give a fast empirical
read on multi-seed LGB ensemble variance reduction. Total runtime when
the system is uncontended: ~10-15 min.

Calibrated: when the runner-time matters more than the signal strength,
use this; for production v9 numbers, run multiseed_score.py.

Usage
-----
  python3 bench/multiseed_pilot.py            # 200-well, 5 LGB seeds
  python3 bench/multiseed_pilot.py --limit 100 --seeds 42 7 123
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from feature_builder import (  # noqa: E402
    FormationPlaneKNN, RowKNN, build_dataset,
)


LGB_PARAMS_BASE = dict(
    boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5, n_estimators=2000,
    n_jobs=-1, reg_alpha=2.03, reg_lambda=87.28,
    subsample=0.645, subsample_freq=1, colsample_bytree=0.821,
    objective="regression", metric="rmse", verbose=-1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 7, 123, 999, 31337])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cache", type=str,
                    default="/tmp/v8_pilot_features.pkl")
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "bench" / "multiseed_pilot_results.json"))
    args = ap.parse_args()

    overall_t = time.perf_counter()

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.force_rebuild:
        print(f">> Loading cached features from {cache_path}", flush=True)
        train_df = pd.read_pickle(cache_path)
    else:
        train_dir = ROOT / "data" / "competition" / "train"
        all_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(all_paths), size=min(args.limit, len(all_paths)), replace=False))
        paths = [all_paths[i] for i in idx]
        print(f">> Building v8 features over {len(paths)} train wells", flush=True)

        t0 = time.perf_counter()
        plane = FormationPlaneKNN.fit(paths)
        print(f"   plane: {time.perf_counter() - t0:.1f}s", flush=True)
        t0 = time.perf_counter()
        row = RowKNN.fit(paths)
        print(f"   row KNN: {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        train_df = build_dataset(
            paths, plane, row, is_train=True, mlp_imputer=None,
            primary_formation="ANCC", enable_beam=False,
            label="train", progress_every=50,
        )
        print(f"   features: {train_df.shape}, {time.perf_counter() - t0:.1f}s",
              flush=True)
        train_df.to_pickle(cache_path)
        print(f"   cached to {cache_path}", flush=True)

    feature_cols = [c for c in train_df.columns
                    if c not in {"well", "prediction_id", "target"}]
    print(f"   #features: {len(feature_cols)}", flush=True)

    seed_oofs = {}
    seed_metrics = {}
    seed_fold_rmses = {}
    target_arr = train_df["target"].to_numpy()
    last_known = train_df["last_known_tvt"].to_numpy()
    well_arr = train_df["well"].to_numpy()

    for s in args.seeds:
        gkf = GroupKFold(n_splits=args.n_folds, shuffle=True, random_state=s)
        splits = list(gkf.split(train_df, train_df["target"],
                                 groups=train_df["well"]))
        params = dict(LGB_PARAMS_BASE)
        params["random_state"] = s
        params["bagging_seed"] = s
        params["feature_fraction_seed"] = s

        oof = np.zeros(len(train_df), dtype=np.float32)
        fold_rmses = []
        t_seed = time.perf_counter()
        for fold, (tr, va) in enumerate(splits):
            x_tr = train_df.iloc[tr][feature_cols]
            y_tr = train_df.iloc[tr]["target"]
            x_va = train_df.iloc[va][feature_cols]
            y_va = train_df.iloc[va]["target"]
            dtr = lgb.Dataset(x_tr, label=y_tr)
            dva = lgb.Dataset(x_va, label=y_va, reference=dtr)
            m = lgb.train(
                params, dtr, valid_sets=[dva],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            oof[va] = m.predict(
                x_va, num_iteration=m.best_iteration
            ).astype(np.float32)
            r = float(np.sqrt(np.mean((oof[va] - y_va.values) ** 2)))
            fold_rmses.append(r)

        seed_oofs[s] = oof
        seed_fold_rmses[s] = fold_rmses

        # metrics
        overall_rmse = float(np.sqrt(np.mean((oof - target_arr) ** 2)))
        truth = last_known + target_arr
        pred = last_known + oof
        e = pred - truth
        df = pd.DataFrame({"well": well_arr, "err": e})
        well_rmse = df.groupby("well")["err"].apply(
            lambda s: float(np.sqrt(np.mean(s.values ** 2)))
        )
        seed_metrics[s] = dict(
            overall_rmse=overall_rmse,
            well_median=float(well_rmse.median()),
            well_mean=float(well_rmse.mean()),
            well_p90=float(well_rmse.quantile(0.9)),
            well_p99=float(well_rmse.quantile(0.99)),
            well_max=float(well_rmse.max()),
        )
        print(f"   seed={s}: overall={overall_rmse:.4f}  "
              f"well_max={seed_metrics[s]['well_max']:.2f}  "
              f"well_p90={seed_metrics[s]['well_p90']:.2f}  "
              f"({time.perf_counter() - t_seed:.1f}s)", flush=True)

    # Average across seeds
    avg_oof = np.mean(np.stack(list(seed_oofs.values()), axis=0), axis=0)
    overall_avg = float(np.sqrt(np.mean((avg_oof - target_arr) ** 2)))
    truth = last_known + target_arr
    pred = last_known + avg_oof
    e = pred - truth
    df = pd.DataFrame({"well": well_arr, "err": e})
    well_rmse = df.groupby("well")["err"].apply(
        lambda s: float(np.sqrt(np.mean(s.values ** 2)))
    )
    avg_metrics = dict(
        overall_rmse=overall_avg,
        well_median=float(well_rmse.median()),
        well_mean=float(well_rmse.mean()),
        well_p90=float(well_rmse.quantile(0.9)),
        well_p99=float(well_rmse.quantile(0.99)),
        well_max=float(well_rmse.max()),
    )
    per_seed_mean_overall = float(np.mean([m["overall_rmse"]
                                           for m in seed_metrics.values()]))
    per_seed_mean_max = float(np.mean([m["well_max"]
                                       for m in seed_metrics.values()]))
    per_seed_mean_p90 = float(np.mean([m["well_p90"]
                                       for m in seed_metrics.values()]))

    print()
    print(f">> per-seed mean: overall={per_seed_mean_overall:.4f}  "
          f"well_max={per_seed_mean_max:.2f}  "
          f"well_p90={per_seed_mean_p90:.2f}", flush=True)
    print(f">> averaged OOF: overall={avg_metrics['overall_rmse']:.4f}  "
          f"well_max={avg_metrics['well_max']:.2f}  "
          f"well_p90={avg_metrics['well_p90']:.2f}", flush=True)
    print(f">> Δ (avg - per-seed-mean): "
          f"Δoverall={avg_metrics['overall_rmse'] - per_seed_mean_overall:+.4f}  "
          f"Δwell_max={avg_metrics['well_max'] - per_seed_mean_max:+.2f}  "
          f"Δwell_p90={avg_metrics['well_p90'] - per_seed_mean_p90:+.2f}",
          flush=True)

    out = {
        "config": {
            "n_wells": int(train_df["well"].nunique()),
            "n_rows": int(len(train_df)),
            "n_features": int(len(feature_cols)),
            "feature_set": "v8 (no MLP)",
            "lgb_seeds": list(args.seeds),
            "n_folds": args.n_folds,
        },
        "per_seed": {str(s): {
            "metrics": seed_metrics[s],
            "fold_rmses": seed_fold_rmses[s],
        } for s in args.seeds},
        "avg_metrics": avg_metrics,
        "per_seed_mean": dict(
            overall_rmse=per_seed_mean_overall,
            well_max=per_seed_mean_max,
            well_p90=per_seed_mean_p90,
        ),
        "delta_avg_minus_perseed_mean": dict(
            overall_rmse=avg_metrics["overall_rmse"] - per_seed_mean_overall,
            well_max=avg_metrics["well_max"] - per_seed_mean_max,
            well_p90=avg_metrics["well_p90"] - per_seed_mean_p90,
        ),
        "wall_seconds": time.perf_counter() - overall_t,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f">> saved {args.out}")
    print(f"   total wall: {time.perf_counter() - overall_t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
