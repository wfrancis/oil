"""LightGBM objective comparison on a 300-well subset.

Runs 5-fold GroupKFold OOF for the same v8 features (KNN + plane only, NO
MLP, NO beam) but varies the LightGBM `objective`. The goal is to see if a
robust loss (MAE / huber / quantile / tweedie) beats the MSE default on the
catastrophic-tail wells (max-well RMSE ~ 56 ft on full data).

Hyperparameters mirror the v8/v9 LGB_PARAMS in `_v9_full_oof_runner.py`.

Outputs:
    bench/lgb_objectives_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from feature_builder import FormationPlaneKNN, RowKNN, build_dataset  # noqa: E402


# Subset size for runtime budget. The dominant cost is the per-well
# RowKNN.impute (cKDTree.query with n_q=12_000 returning 6k+ neighbors per
# row), which on M1 Pro is ~2.5 s/well for the v8/v9 feature stack. The
# user asked for under 30 minutes total; with ~5 objectives x 5 folds of
# LGB training on top, 150 wells is the safe budget.
N_WELLS_SUBSET = 150
SUBSET_SEED = 42

# All five objectives we want to compare. Each entry overrides keys in
# LGB_PARAMS_BASE. We pick metric="rmse" everywhere so early stopping is
# evaluated on the COMPETITION metric, not the training loss.
OBJECTIVES = [
    {
        "name": "regression",
        "params": {"objective": "regression", "metric": "rmse"},
    },
    {
        "name": "regression_l1",
        "params": {"objective": "regression_l1", "metric": "rmse"},
    },
    {
        "name": "huber",
        # alpha for huber in LightGBM is the threshold parameter. The user
        # asked for alpha=0.95 — we pass it through directly.
        "params": {"objective": "huber", "alpha": 0.95, "metric": "rmse"},
    },
    {
        "name": "quantile_0.5",
        "params": {"objective": "quantile", "alpha": 0.5, "metric": "rmse"},
    },
    {
        "name": "tweedie_1.5",
        # Tweedie requires non-negative targets. The base target is
        # (TVT - last_known_tvt) and is signed, so we will shift+train+unshift
        # at use-site (see `train_objective`). Power=1.5 = the user spec.
        "params": {
            "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "rmse",
        },
    },
]

# Same as v8/v9 LGB_PARAMS, just without `objective` and `metric` (set
# per-objective above).
LGB_PARAMS_BASE = dict(
    boosting_type="gbdt",
    learning_rate=0.06,
    num_leaves=89,
    min_child_samples=10,
    min_child_weight=0.5,
    n_estimators=3000,
    n_jobs=-1,
    reg_alpha=2.03,
    reg_lambda=87.28,
    subsample=0.645,
    subsample_freq=1,
    colsample_bytree=0.821,
    verbose=-1,
    random_state=42,
)


def well_rmse_stats(
    truth: np.ndarray, pred: np.ndarray, wells: np.ndarray
) -> dict:
    """Per-well RMSE stats. Uses polars for the groupby.

    `truth` and `pred` are TVT space (i.e. `last_known_tvt + target/oof_pred`),
    NOT the residuals — that matches `_v9_full_oof_runner.py` line 107-117.
    """
    err = pred - truth
    df = pl.DataFrame({"well": wells, "err": err})
    rmse_per_well = (
        df.group_by("well")
        .agg(rmse=pl.col("err").pow(2).mean().sqrt())
        .get_column("rmse")
        .to_numpy()
    )
    return {
        "median_well_rmse": float(np.median(rmse_per_well)),
        "mean_well_rmse": float(np.mean(rmse_per_well)),
        "p90_well_rmse": float(np.quantile(rmse_per_well, 0.9)),
        "max_well_rmse": float(np.max(rmse_per_well)),
        "n_wells": int(len(rmse_per_well)),
    }


def train_objective(
    name: str,
    params: dict,
    train_df,
    feature_cols: list[str],
    splits: list,
) -> dict:
    """Run 5-fold OOF for one objective. Returns numeric summary."""
    print(f"\n>> objective={name}", flush=True)
    full = {**LGB_PARAMS_BASE, **params}

    # Tweedie cannot consume negative targets, so we shift y by a global
    # offset (max |target| over the whole train set), train, and subtract on
    # predict. This keeps the comparison fair: the shift is invertible.
    y = train_df["target"].to_numpy()
    is_tweedie = name.startswith("tweedie")
    if is_tweedie:
        offset = float(np.abs(y).max() + 1.0)
        y_train_full = y + offset
        if (y_train_full <= 0).any():
            raise RuntimeError("tweedie shift failed; negative remains")
    else:
        offset = 0.0
        y_train_full = y

    X = train_df.select(feature_cols).to_numpy()
    truth_tvt = (
        train_df["last_known_tvt"].to_numpy() + train_df["target"].to_numpy()
    )
    wells = train_df["well"].to_numpy()

    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_rmses = []
    fold_walltimes = []
    fold_best_iters = []

    for fold, (tr, va) in enumerate(splits):
        t0 = time.perf_counter()
        dtr = lgb.Dataset(X[tr], label=y_train_full[tr])
        dva = lgb.Dataset(X[va], label=y_train_full[va], reference=dtr)
        m = lgb.train(
            full,
            dtr,
            valid_sets=[dva],
            num_boost_round=3000,
            callbacks=[
                lgb.early_stopping(125, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred_va = m.predict(X[va], num_iteration=m.best_iteration)
        if is_tweedie:
            pred_va = pred_va - offset
        oof[va] = pred_va.astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof[va] - y[va]) ** 2)))
        wt = time.perf_counter() - t0
        fold_rmses.append(rmse)
        fold_walltimes.append(wt)
        fold_best_iters.append(int(m.best_iteration))
        print(
            f"   fold {fold}: rmse={rmse:.4f}  best_iter={m.best_iteration}  "
            f"walltime={wt:.1f}s",
            flush=True,
        )

    overall_rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    pred_tvt = train_df["last_known_tvt"].to_numpy() + oof
    well_stats = well_rmse_stats(truth_tvt, pred_tvt, wells)

    summary = {
        "objective": name,
        "params_extra": params,
        "overall_rmse": overall_rmse,
        "fold_rmses": fold_rmses,
        "fold_walltime_sec": fold_walltimes,
        "fold_best_iters": fold_best_iters,
        "mean_walltime_per_fold_sec": float(np.mean(fold_walltimes)),
        **well_stats,
    }
    print(
        f"   overall_rmse={overall_rmse:.4f}  "
        f"max_well_rmse={well_stats['max_well_rmse']:.2f}  "
        f"p90_well_rmse={well_stats['p90_well_rmse']:.2f}  "
        f"mean_walltime/fold={summary['mean_walltime_per_fold_sec']:.1f}s",
        flush=True,
    )
    return summary


def main() -> int:
    t_overall = time.perf_counter()
    train_dir = ROOT / "data" / "competition" / "train"
    paths_all = sorted(train_dir.glob("*__horizontal_well.csv"))

    # Deterministic 300-well subsample.
    rng = np.random.default_rng(SUBSET_SEED)
    idx = rng.choice(len(paths_all), size=N_WELLS_SUBSET, replace=False)
    paths = [paths_all[i] for i in sorted(idx)]
    print(
        f">> Subset: {len(paths)}/{len(paths_all)} train wells, seed={SUBSET_SEED}",
        flush=True,
    )

    t0 = time.perf_counter()
    plane = FormationPlaneKNN.fit(paths)
    print(
        f"   plane fit: {len(plane.df)} wells in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    row = RowKNN.fit(paths)
    print(
        f"   row KNN fit: {len(row.targets):,} rows in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    print(
        ">> Building train features (KNN + plane, no MLP, no beam) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    train_pdf = build_dataset(
        paths, plane, row, is_train=True, mlp_imputer=None,
        primary_formation="ANCC", enable_beam=False, label="train",
        progress_every=25,
    )
    print(
        f"   train shape: {train_pdf.shape}, {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    if train_pdf.empty:
        print("FATAL: empty train_df", flush=True)
        return 1

    # Convert to polars for downstream groupby + slicing speed.
    train_df = pl.from_pandas(train_pdf)
    feature_cols = [
        c for c in train_df.columns if c not in {"well", "prediction_id", "target"}
    ]
    print(f"   #features: {len(feature_cols)}", flush=True)

    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(
        gkf.split(
            np.arange(len(train_df)),
            train_df["target"].to_numpy(),
            groups=train_df["well"].to_numpy(),
        )
    )

    results = []
    for cfg in OBJECTIVES:
        try:
            summary = train_objective(
                cfg["name"], cfg["params"], train_df, feature_cols, splits
            )
        except Exception as e:
            summary = {
                "objective": cfg["name"],
                "params_extra": cfg["params"],
                "error": repr(e),
            }
            print(f"   FAILED: {e!r}", flush=True)
        results.append(summary)

    out = {
        "n_wells_subset": N_WELLS_SUBSET,
        "subset_seed": SUBSET_SEED,
        "n_rows": int(len(train_df)),
        "n_features": len(feature_cols),
        "kfold": "GroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "lgb_params_base": LGB_PARAMS_BASE,
        "results": results,
        "total_walltime_sec": time.perf_counter() - t_overall,
    }
    out_path = ROOT / "bench" / "lgb_objectives_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n>> Wrote {out_path}", flush=True)
    print(
        f">> Total wall time: {time.perf_counter() - t_overall:.1f}s",
        flush=True,
    )

    print("\n>> SUMMARY")
    print(
        f"{'objective':<16} {'rmse':>8} {'med_well':>9} "
        f"{'p90_well':>9} {'max_well':>9} {'wt/fold':>9}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['objective']:<16} FAILED  {r['error']}")
            continue
        print(
            f"{r['objective']:<16} "
            f"{r['overall_rmse']:>8.4f} "
            f"{r['median_well_rmse']:>9.2f} "
            f"{r['p90_well_rmse']:>9.2f} "
            f"{r['max_well_rmse']:>9.2f} "
            f"{r['mean_walltime_per_fold_sec']:>9.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
