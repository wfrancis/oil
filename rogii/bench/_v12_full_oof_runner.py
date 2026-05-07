"""v12 LOCAL OOF runner with PROPER per-fold retraining.

v11 OOF was LEAKED (aniso fit on all train wells, not per-fold). v12
fixes this by retraining the leak-prone imputers (MLP, aniso) PER FOLD
on the train-fold wells only. The cheap imputers (FormationPlaneKNN,
RowKNN) get self-well exclusion at query time so they don't leak.

PFs are inherently leak-safe: each well's PF only uses that well's own
typewell + horizontal data. No cross-well training. So PFs are run
ONCE for all wells and reused across folds.

Architecture (6 spatial signals + LGB):
  1. plane fit KNN K=10        (self-well exclude at query)
  2. row KNN K=20 n_q=8000     (self-well exclude at query)
  3. MLP+PE-L8 multi 3-seed    (RETRAINED per fold)
  4. aniso-exp kriging         (RETRAINED per fold)
  5. TVT-PF (Z-vel coupled)    (per-well, no fold dep)
  6. ANCC-PF (S=TVT+Z tracker) (per-well, no fold dep)

Outputs:
  /tmp/v12_oof.csv         row-level OOF
  /tmp/v12_well_rmse.csv   per-well summary
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from feature_builder import (
    FORMATIONS, FormationPlaneKNN, RowKNN, MLPAnccImputer,
    AnisoFormationImputer, build_dataset,
)
from triple_signal_pf import run_pfs_for_wells


def _load_pl(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        str(path),
        infer_schema_length=2000,
        null_values=["", "NA", "NaN", "nan", "null"],
        truncate_ragged_lines=True,
    )


def main() -> int:
    t_overall = time.perf_counter()
    train_dir = ROOT / "data" / "competition" / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    print(f">> v12 OOF over {len(paths)} train wells (per-fold MLP+aniso retrain)", flush=True)

    # Cheap imputers fit ONCE on all train (they self-exclude at query time)
    t0 = time.perf_counter()
    plane_full = FormationPlaneKNN.fit(paths)
    row_full = RowKNN.fit(paths)
    print(f"   plane+rowKNN: {time.perf_counter() - t0:.1f}s", flush=True)

    # PFs: run once for all wells (no cross-well dependency, so safe)
    print(">> Running TVT-PF + ANCC-PF for all 773 wells (parallel) ...", flush=True)
    t0 = time.perf_counter()
    well_dfs = {}
    typewell_dfs = {}
    for p in paths:
        wid = p.stem.replace("__horizontal_well", "")
        well_dfs[wid] = _load_pl(p)
        tw_path = train_dir / f"{wid}__typewell.csv"
        if tw_path.exists():
            typewell_dfs[wid] = _load_pl(tw_path)
    pf_results = run_pfs_for_wells(
        well_dfs, typewell_dfs,
        n_workers=-1, n_particles=500, seed=42,
    )
    print(f"   PFs done: {len(pf_results)} wells in {time.perf_counter() - t0:.1f}s", flush=True)
    del well_dfs, typewell_dfs

    # GroupKFold split
    well_ids = [p.stem.replace("__horizontal_well", "") for p in paths]
    n_paths = len(paths)
    dummy_y = np.zeros(n_paths)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(gkf.split(np.arange(n_paths), dummy_y, groups=well_ids))

    # Build features per fold (with PER-FOLD MLP and aniso)
    LGB_PARAMS = dict(
        boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
        min_child_samples=10, min_child_weight=0.5, n_estimators=3000,
        n_jobs=-1, reg_alpha=2.03, reg_lambda=87.28,
        subsample=0.645, subsample_freq=1, colsample_bytree=0.821,
        objective="regression", metric="rmse", verbose=-1, random_state=42,
    )

    fold_dfs_oof = []
    fold_rmses = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        t_fold = time.perf_counter()
        tr_paths = [paths[i] for i in tr_idx]
        va_paths = [paths[i] for i in va_idx]
        va_wells = set(well_ids[i] for i in va_idx)

        # Per-fold retrain of leak-prone imputers
        mlp_fold = MLPAnccImputer.fit(
            tr_paths, formations=FORMATIONS,
            num_freqs=8, hidden=256, epochs=12, rows_per_epoch=500_000,
            seeds=[42, 7, 123], verbose=False,
        )
        aniso_fold = AnisoFormationImputer.fit(
            tr_paths, formations=FORMATIONS,
            kernel="exponential", range_scale=1.0, k=20,
        )

        # Build features for THIS fold's val wells (use full train data
        # for plane/row + per-fold MLP/aniso)
        # IMPORTANT: pass ALL paths to build_dataset so plane/row impute
        # works -- they self-exclude held-out wells via wid arg internally.
        # Wait: actually the plane/row use train_paths to BUILD the index
        # which DOES include held-out wells. They self-exclude at QUERY
        # time via the well-id mask. That's leak-safe.
        # For MLP and aniso we built per-fold on tr_paths only -> no leak.
        fold_va_df = build_dataset(
            va_paths, plane_full, row_full,
            is_train=True,
            mlp_imputer=mlp_fold, aniso_imputer=aniso_fold,
            pf_results=pf_results,
            primary_formation="ANCC", enable_beam=False,
            label=f"fold{fold}_val",
        )

        if fold_va_df.empty:
            print(f"   fold {fold}: empty val, skip", flush=True)
            continue

        # Train fold features
        fold_tr_df = build_dataset(
            tr_paths, plane_full, row_full,
            is_train=True,
            mlp_imputer=mlp_fold, aniso_imputer=aniso_fold,
            pf_results=pf_results,
            primary_formation="ANCC", enable_beam=False,
            label=f"fold{fold}_tr",
        )

        feat_cols = [c for c in fold_tr_df.columns
                     if c not in {"well", "prediction_id", "target"}]

        dtr = lgb.Dataset(fold_tr_df[feat_cols], label=fold_tr_df["target"])
        dva = lgb.Dataset(fold_va_df[feat_cols], label=fold_va_df["target"], reference=dtr)
        m = lgb.train(
            LGB_PARAMS, dtr, valid_sets=[dva],
            num_boost_round=3000,
            callbacks=[lgb.early_stopping(125, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        oof_pred = m.predict(fold_va_df[feat_cols], num_iteration=m.best_iteration).astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof_pred - fold_va_df["target"].values) ** 2)))
        fold_rmses.append(rmse)
        print(f"   fold {fold}: rmse={rmse:.4f}  best_iter={m.best_iteration}  "
              f"feats={len(feat_cols)}  fold_time={time.perf_counter() - t_fold:.0f}s", flush=True)

        # Save OOF for this fold's val wells
        out_block = pd.DataFrame({
            "prediction_id": fold_va_df["prediction_id"],
            "well": fold_va_df["well"],
            "row_idx": fold_va_df["row_idx"].astype(np.int32),
            "target": fold_va_df["target"].values,
            "oof_pred_v12": oof_pred.astype(np.float64),
            "last_known_tvt": fold_va_df["last_known_tvt"].astype(np.float64),
        })
        fold_dfs_oof.append(out_block)
        del fold_tr_df, fold_va_df, m

    if not fold_dfs_oof:
        print("FATAL: no folds produced OOF", flush=True)
        return 1

    oof_df = pd.concat(fold_dfs_oof, ignore_index=True)
    err = oof_df["oof_pred_v12"].values - oof_df["target"].values
    overall = float(np.sqrt(np.mean(err * err)))
    print(f"\n>> v12 OOF RMSE = {overall:.4f}", flush=True)
    print(f"   fold rmses: {[round(r, 4) for r in fold_rmses]}", flush=True)

    well_rmse = oof_df.assign(err=err).groupby("well").apply(
        lambda g: float(np.sqrt(np.mean(g["err"] ** 2)))
    )
    print(
        "   per-well: median={:.2f}  mean={:.2f}  p90={:.2f}  max={:.2f}".format(
            well_rmse.median(), well_rmse.mean(),
            well_rmse.quantile(0.9), well_rmse.max()
        ),
        flush=True,
    )

    print(f"\n   vs v9 OOF (11.41):  delta = {overall - 11.4059:+.4f}", flush=True)

    oof_df.to_csv("/tmp/v12_oof.csv", index=False)
    well_rmse.to_csv("/tmp/v12_well_rmse.csv", header=["rmse"])
    print(f"   saved /tmp/v12_oof.csv and /tmp/v12_well_rmse.csv", flush=True)
    print(f"   total wall time: {time.perf_counter() - t_overall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
