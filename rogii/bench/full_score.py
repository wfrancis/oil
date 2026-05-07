"""Full pipeline validator: features + LGB + (optional XGB) + Ridge stack.

Drops in the konbu17-style feature builder, runs 5-fold GroupKFold OOF on a
subset of train wells, and reports OOF RMSE.

Three modes:
  smoke       - 1-seed LightGBM, beam disabled, small well sample
  oof         - 5-fold LightGBM, full GBM
  oof-stack   - LGB×3 + XGB + Ridge stack (mirrors konbu17)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_builder import (
    FORMATIONS,
    FormationPlaneKNN,
    RowKNN,
    build_dataset,
)

DEFAULT_TRAIN_DIR = ROOT / "data" / "competition" / "train"

logger = logging.getLogger("rogii.full_score")


def _select_paths(train_dir: Path, *, limit: int, seed: int) -> list[Path]:
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if limit > 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(paths), min(limit, len(paths)), replace=False)
        idx.sort()
        paths = [paths[i] for i in idx]
    return paths


def _per_well_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    out = (
        scored.assign(
            err=scored["pred_tvt"] - scored["true_tvt"],
        )
        .groupby("well", sort=True)
        .agg(rows=("err", "size"),
             rmse=("err", lambda s: float(np.sqrt(np.mean(s ** 2)))),
             bias=("err", "mean"))
        .reset_index()
    )
    return out


def _train_lgb_seed(train_df: pd.DataFrame, feature_cols: list[str], splits, seed: int,
                    n_estimators: int = 5000, learning_rate: float = 0.06,
                    early_stopping: int = 125) -> dict:
    import lightgbm as lgb

    params = dict(
        boosting_type="gbdt",
        learning_rate=learning_rate,
        num_leaves=89,
        min_child_samples=10,
        min_child_weight=0.5,
        n_estimators=n_estimators,
        n_jobs=-1,
        reg_alpha=2.03,
        reg_lambda=87.28,
        subsample=0.645,
        subsample_freq=1,
        colsample_bytree=0.821,
        objective="regression",
        metric="rmse",
        verbose=-1,
        random_state=seed,
    )
    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_rmse = []
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(train_df.iloc[tr][feature_cols], label=train_df.iloc[tr]["target"])
        dva = lgb.Dataset(train_df.iloc[va][feature_cols], label=train_df.iloc[va]["target"],
                          reference=dtr)
        m = lgb.train(
            params, dtr, valid_sets=[dva],
            num_boost_round=params["n_estimators"],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        oof[va] = m.predict(train_df.iloc[va][feature_cols],
                             num_iteration=m.best_iteration).astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof[va] - train_df.iloc[va]["target"].values) ** 2)))
        fold_rmse.append(rmse)
        print(f"   LGB seed={seed} fold {fold}: rmse={rmse:.4f}  best_iter={m.best_iteration}", flush=True)
    overall = float(np.sqrt(np.mean((oof - train_df["target"].values) ** 2)))
    print(f"   LGB seed={seed}: OOF rmse = {overall:.4f}", flush=True)
    return {"oof": oof, "rmse": overall, "fold_rmse": fold_rmse}


def cmd_smoke(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    paths = _select_paths(train_dir, limit=args.limit, seed=args.seed)
    print(f"using {len(paths)} train wells", flush=True)

    print(">> build PLANE-FIT formation imputer", flush=True)
    t0 = time.perf_counter()
    formation_imputer = FormationPlaneKNN.fit(paths)
    print(f"   {len(formation_imputer.df)} wells, {time.perf_counter() - t0:.1f}s", flush=True)

    print(">> build row-level KNN imputer", flush=True)
    t0 = time.perf_counter()
    row_imputer = RowKNN.fit(paths)
    print(f"   {len(row_imputer.targets):,} rows, {time.perf_counter() - t0:.1f}s", flush=True)

    print(">> build train features", flush=True)
    t0 = time.perf_counter()
    train_df = build_dataset(
        paths, formation_imputer, row_imputer,
        is_train=True,
        primary_formation=args.primary,
        enable_beam=bool(args.enable_beam),
        label="train",
    )
    print(f"   train shape: {train_df.shape}, {time.perf_counter() - t0:.1f}s", flush=True)

    if train_df.empty:
        raise SystemExit("Empty training feature set")

    feature_cols = [c for c in train_df.columns
                    if c not in {"well", "prediction_id", "target"}]
    print(f"   #features: {len(feature_cols)}", flush=True)

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(gkf.split(train_df, train_df["target"], groups=train_df["well"]))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = {}
    for seed in seeds:
        results[f"lgb_{seed}"] = _train_lgb_seed(
            train_df, feature_cols, splits, seed=seed,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            early_stopping=args.early_stopping,
        )

    if len(seeds) > 1:
        oof_avg = np.mean([r["oof"] for r in results.values()], axis=0)
        rmse_avg = float(np.sqrt(np.mean((oof_avg - train_df["target"].values) ** 2)))
        print(f"\n   simple avg OOF rmse = {rmse_avg:.4f}", flush=True)
    else:
        oof_avg = results[f"lgb_{seeds[0]}"]["oof"]
        rmse_avg = results[f"lgb_{seeds[0]}"]["rmse"]

    final_pred = train_df["last_known_tvt"].to_numpy(dtype=np.float64) + oof_avg.astype(np.float64)
    truth = train_df["last_known_tvt"].to_numpy(dtype=np.float64) + train_df["target"].astype(np.float64)
    scored = pd.DataFrame({
        "well": train_df["well"].values,
        "true_tvt": truth,
        "pred_tvt": final_pred,
    })
    per_well = _per_well_metrics(scored)
    print(f"\n   per-well stats: median rmse={per_well['rmse'].median():.3f}  "
          f"mean rmse={per_well['rmse'].mean():.3f}  "
          f"max rmse={per_well['rmse'].max():.3f}  "
          f"p90 rmse={per_well['rmse'].quantile(0.9):.3f}", flush=True)

    if args.show_worst > 0:
        worst = per_well.sort_values("rmse", ascending=False).head(args.show_worst)
        print(f"\n   worst {args.show_worst} wells:", flush=True)
        for _, r in worst.iterrows():
            print(f"     {r['well']}  rmse={r['rmse']:.3f}  bias={r['bias']:+.3f}  rows={int(r['rows'])}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke", help="Quick LightGBM-only OOF on a subset.")
    smoke.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    smoke.add_argument("--limit", type=int, default=200)
    smoke.add_argument("--n-folds", type=int, default=5)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--seeds", default="42",
                       help="Comma-sep LGB seeds, e.g. '42,7,123'")
    smoke.add_argument("--primary", default="EGFDL", choices=list(FORMATIONS))
    smoke.add_argument("--enable-beam", type=int, default=0,
                       help="1 to enable Viterbi beam-search GR features (slow).")
    smoke.add_argument("--n-estimators", type=int, default=5000)
    smoke.add_argument("--learning-rate", type=float, default=0.06)
    smoke.add_argument("--early-stopping", type=int, default=125)
    smoke.add_argument("--show-worst", type=int, default=10)
    smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
