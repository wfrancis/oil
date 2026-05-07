"""Quick LightGBM baseline on the SAME 100-well feature parquet, so we can
compare apples-to-apples against the Sequence Transformer prototype.

This is NOT the v8/v9 GBM with full hyperparameter tuning — it's a basic
GBM using the standard v9 LGB params, run on the small subset to give us
an apples-to-apples reference number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feats = pl.read_parquet(args.features)
    print(f"loaded features: {feats.shape}", flush=True)
    feature_cols = [
        c for c in feats.columns
        if c not in {"well", "prediction_id", "target", "row_idx",
                     "last_known_tvt", "known_len", "hidden_len"}
    ]
    print(f"#features: {len(feature_cols)}", flush=True)

    X = feats.select(feature_cols).to_numpy()
    y = feats.get_column("target").to_numpy()
    groups = feats.get_column("well").to_numpy()

    # NaN/Inf handling
    X = np.where(np.isfinite(X), X, 0.0)

    gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(gkf.split(X, y, groups=groups))

    LGB_PARAMS = dict(
        boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
        min_child_samples=10, min_child_weight=0.5, n_estimators=2000,
        n_jobs=-1, reg_alpha=2.03, reg_lambda=87.28,
        subsample=0.645, subsample_freq=1, colsample_bytree=0.821,
        objective="regression", metric="rmse", verbose=-1, random_state=42,
    )

    oof = np.zeros(len(X), dtype=np.float64)
    fold_rmses = []
    t_overall = time.perf_counter()
    for fold, (tr, va) in enumerate(splits):
        t0 = time.perf_counter()
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
        m = lgb.train(
            LGB_PARAMS, dtr,
            valid_sets=[dva], num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(125, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        rmse = float(np.sqrt(np.mean((oof[va] - y[va]) ** 2)))
        fold_rmses.append(rmse)
        print(
            f"   fold {fold+1}: rmse={rmse:.4f}  best_iter={m.best_iteration}  "
            f"{time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    overall = float(np.sqrt(np.mean((oof - y) ** 2)))
    # Per-well
    df = pl.DataFrame({"well": groups, "err": oof - y})
    well_rmse = (
        df.group_by("well")
        .agg(pl.col("err").pow(2).mean().sqrt().alias("rmse"))
        .get_column("rmse")
        .to_numpy()
    )
    summary = {
        "n_features": len(feature_cols),
        "n_rows": len(X),
        "fold_rmses": fold_rmses,
        "overall_rmse": overall,
        "well_rmse_median": float(np.median(well_rmse)),
        "well_rmse_mean": float(np.mean(well_rmse)),
        "well_rmse_p90": float(np.quantile(well_rmse, 0.9)),
        "well_rmse_max": float(np.max(well_rmse)),
        "wall_time_s": time.perf_counter() - t_overall,
    }
    print("\n==== LightGBM baseline OOF (same 100-well subset) ====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    Path("/tmp/seq_gbm_oof.json").write_text(json.dumps(summary, indent=2))
    print("saved /tmp/seq_gbm_oof.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
