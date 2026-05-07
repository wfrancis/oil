"""Run the full v9 OOF on M1 Pro and dump per-row OOF predictions for the
stacker. This is a one-shot script (not part of the regular bench/) — it
exists so we can run it under nohup without inline heredoc pitfalls.

Outputs:
  /tmp/v9_oof.csv         per-row OOF predictions for all train wells
  /tmp/v9_well_rmse.csv   per-well RMSE summary
  /tmp/v9_full_oof.log    is the stdout/stderr file (set externally)
"""

from __future__ import annotations

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

from feature_builder import (
    FORMATIONS, FormationPlaneKNN, RowKNN, MLPAnccImputer, build_dataset,
)


def main() -> int:
    t_overall = time.perf_counter()
    train_dir = ROOT / "data" / "competition" / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    print(f">> Full v9 OOF over {len(paths)} train wells", flush=True)

    t0 = time.perf_counter()
    plane = FormationPlaneKNN.fit(paths)
    print(f"   plane fit: {len(plane.df)} wells in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    row = RowKNN.fit(paths)
    print(f"   row KNN fit: {len(row.targets):,} rows in {time.perf_counter() - t0:.1f}s", flush=True)

    print(">> Training MLP on full data (single fit, no fold) ...", flush=True)
    t0 = time.perf_counter()
    mlp = MLPAnccImputer.fit(
        paths, formations=FORMATIONS,
        num_freqs=8, hidden=256, epochs=12, rows_per_epoch=500_000,
        seed=42, verbose=True,
    )
    print(f"   MLP fit: {time.perf_counter() - t0:.1f}s", flush=True)

    print(">> Building train features (with MLP, NO beam for speed) ...", flush=True)
    t0 = time.perf_counter()
    train_df = build_dataset(
        paths, plane, row, is_train=True, mlp_imputer=mlp,
        primary_formation="ANCC", enable_beam=False, label="train",
    )
    print(f"   train shape: {train_df.shape}, {time.perf_counter() - t0:.1f}s", flush=True)

    if train_df.empty:
        print("FATAL: empty train_df", flush=True)
        return 1

    feature_cols = [c for c in train_df.columns if c not in {"well", "prediction_id", "target"}]
    print(f"   #features: {len(feature_cols)}", flush=True)
    mlp_cols = [c for c in feature_cols if c.startswith("mlp_")]
    print(f"   #MLP features: {len(mlp_cols)}", flush=True)

    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(gkf.split(train_df, train_df["target"], groups=train_df["well"]))

    LGB_PARAMS = dict(
        boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
        min_child_samples=10, min_child_weight=0.5, n_estimators=3000,
        n_jobs=-1, reg_alpha=2.03, reg_lambda=87.28,
        subsample=0.645, subsample_freq=1, colsample_bytree=0.821,
        objective="regression", metric="rmse", verbose=-1, random_state=42,
    )

    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_rmses = []
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(train_df.iloc[tr][feature_cols], label=train_df.iloc[tr]["target"])
        dva = lgb.Dataset(
            train_df.iloc[va][feature_cols],
            label=train_df.iloc[va]["target"],
            reference=dtr,
        )
        m = lgb.train(
            LGB_PARAMS, dtr, valid_sets=[dva],
            num_boost_round=3000,
            callbacks=[lgb.early_stopping(125, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        oof[va] = m.predict(train_df.iloc[va][feature_cols],
                            num_iteration=m.best_iteration).astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof[va] - train_df.iloc[va]["target"].values) ** 2)))
        fold_rmses.append(rmse)
        print(f"   fold {fold}: rmse={rmse:.4f}  best_iter={m.best_iteration}", flush=True)

    overall = float(np.sqrt(np.mean((oof - train_df["target"].values) ** 2)))
    print(f"\n>> v9 OOF RMSE = {overall:.4f}", flush=True)
    print(f"   fold rmses: {[round(r, 4) for r in fold_rmses]}", flush=True)

    truth = train_df["last_known_tvt"].values + train_df["target"].values
    pred = train_df["last_known_tvt"].values + oof
    err = pred - truth
    df_eval = pd.DataFrame({"well": train_df["well"], "err": err})
    well_rmse = df_eval.groupby("well").apply(
        lambda g: float(np.sqrt(np.mean(g["err"] ** 2)))
    )
    print(
        "   per-well: median={:.2f}  mean={:.2f}  p90={:.2f}  max={:.2f}".format(
            well_rmse.median(), well_rmse.mean(),
            well_rmse.quantile(0.9), well_rmse.max()
        ),
        flush=True,
    )

    oof_df = pd.DataFrame({
        "prediction_id": train_df["prediction_id"],
        "well": train_df["well"],
        "row_idx": train_df["row_idx"].astype(np.int32),
        "target": train_df["target"].values,
        "oof_pred_v9": oof.astype(np.float64),
        "last_known_tvt": train_df["last_known_tvt"].astype(np.float64),
    })
    oof_df.to_csv("/tmp/v9_oof.csv", index=False)
    print("   saved /tmp/v9_oof.csv", flush=True)
    well_rmse.to_csv("/tmp/v9_well_rmse.csv", header=["rmse"])
    print("   saved /tmp/v9_well_rmse.csv", flush=True)
    print(f"   total wall time: {time.perf_counter() - t_overall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
