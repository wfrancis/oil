"""Multi-seed LGB variance-reduction experiment for v9.

Approach
--------
The v9 OOF cell is dominated by feature-building (MLP fit + per-well KNN +
beam scoring), not by LGB training. So we build features ONCE (cache to a
pickled DataFrame), then re-run the GBM stage with multiple LGB random_state
values and average the OOF predictions across seeds. This isolates the
variance-reduction effect of seed averaging from feature noise.

Optional Stage 2 (MLP multi-seed) re-fits the MLP imputer with multiple
seeds, averages their predictions to produce more robust ANCC features,
re-builds the train DataFrame, and re-runs the LGB stage. Stage 2 is
expensive (each MLP fit costs ~60s and each feature build ~15 min on the
M1 Pro) so it is gated behind --stage2 and is skipped unless explicitly
requested.

Outputs
-------
  /tmp/v9_multiseed_oof_<seed>.csv     per-row OOF for each LGB seed
  /tmp/v9_multiseed_avg_oof.csv        per-row averaged OOF across seeds
  /tmp/v9_multiseed_features.pkl       cached train_df (so re-runs are cheap)
  /Users/william/drilling_oil_gas/rogii/bench/multiseed_results.json
                                        per-seed and averaged metrics

Run
---
  python bench/multiseed_score.py             # stage 1 only (5 LGB seeds)
  python bench/multiseed_score.py --stage2    # also re-fit MLP multi-seed
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
    FORMATIONS, FormationPlaneKNN, RowKNN, MLPAnccImputer, build_dataset,
)


LGB_SEEDS = [42, 7, 123, 999, 31337]
MLP_SEEDS = [42, 7, 123]
FEATURES_CACHE = Path("/tmp/v9_multiseed_features.pkl")
RESULTS_JSON = ROOT / "bench" / "multiseed_results.json"

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
    objective="regression",
    metric="rmse",
    verbose=-1,
)


# ---------------------------------------------------------------------------
# Feature building (cached)
# ---------------------------------------------------------------------------

def _train_paths() -> list[Path]:
    return sorted((ROOT / "data" / "competition" / "train").glob(
        "*__horizontal_well.csv"))


def build_features(
    *,
    mlp_seeds: list[int] | None = None,
    cache_path: Path = FEATURES_CACHE,
    force_rebuild: bool = False,
    no_mlp: bool = False,
) -> pd.DataFrame:
    """Build (or load cached) train DataFrame with v9 features.

    If ``mlp_seeds`` is None, fits a single MLP with seed=42 (matches the
    inline kernel). Otherwise fits N MLPs and averages their predictions
    inside a wrapper-MLP so all downstream features see a consensus surface.
    If ``no_mlp`` is True, skips MLP entirely (v8 features only) — used as
    a fast fallback when the v9 OOF runner is too slow.
    """
    if cache_path.exists() and not force_rebuild:
        print(f">> Loading cached features from {cache_path}", flush=True)
        return pd.read_pickle(cache_path)

    paths = _train_paths()
    label = "v8 (no MLP)" if no_mlp else "v9"
    print(f">> Building {label} features over {len(paths)} train wells",
          flush=True)
    t0 = time.perf_counter()
    plane = FormationPlaneKNN.fit(paths)
    print(f"   plane: {len(plane.df)} wells in {time.perf_counter() - t0:.1f}s",
          flush=True)
    t0 = time.perf_counter()
    row = RowKNN.fit(paths)
    print(f"   row KNN: {len(row.targets):,} rows in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    if no_mlp:
        mlp = None
    elif mlp_seeds is None or len(mlp_seeds) == 1:
        seed = 42 if mlp_seeds is None else mlp_seeds[0]
        t0 = time.perf_counter()
        mlp = MLPAnccImputer.fit(
            paths, formations=FORMATIONS,
            num_freqs=8, hidden=256, epochs=12, rows_per_epoch=500_000,
            seed=seed, verbose=False,
        )
        print(f"   MLP fit (seed={seed}): {time.perf_counter() - t0:.1f}s",
              flush=True)
    else:
        # Fit multiple MLPs, wrap them so .predict_xy averages across seeds.
        nets = []
        for s in mlp_seeds:
            t0 = time.perf_counter()
            sub = MLPAnccImputer.fit(
                paths, formations=FORMATIONS,
                num_freqs=8, hidden=256, epochs=12, rows_per_epoch=500_000,
                seed=s, verbose=False,
            )
            print(f"   MLP fit (seed={s}): "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
            nets.append(sub.net)
        mlp = _AveragedMLPImputer(nets, formations=FORMATIONS)

    t0 = time.perf_counter()
    train_df = build_dataset(
        paths, plane, row, is_train=True, mlp_imputer=mlp,
        primary_formation="ANCC", enable_beam=False, label="train",
    )
    print(f"   features: {train_df.shape}, "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    train_df.to_pickle(cache_path)
    print(f"   cached to {cache_path}", flush=True)
    return train_df


class _AveragedMLPImputer:
    """Drop-in replacement for MLPAnccImputer that averages multiple AnccNets.

    feature_builder calls ``mlp_imputer.impute(xy)`` which delegates to
    ``self.net.predict(xy)``. We expose .impute() returning the mean of all
    nested nets' .predict() outputs.
    """

    def __init__(self, nets, formations=FORMATIONS):
        self.nets = nets
        self.formations = formations
        # For symmetry with MLPAnccImputer, expose a wrapper net with .predict
        outer = self

        class _AvgNet:
            def predict(self_inner, xy):
                preds = [n.predict(xy) for n in outer.nets]
                return np.mean(preds, axis=0).astype(np.float32)

        self.net = _AvgNet()

    def impute(self, xy_q: np.ndarray) -> np.ndarray:
        return self.net.predict(xy_q)


# ---------------------------------------------------------------------------
# LGB stage
# ---------------------------------------------------------------------------

def run_lgb_oof(
    train_df: pd.DataFrame,
    *,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, list[float], list[int]]:
    """Run a single-seed 5-fold GroupKFold OOF and return per-row predictions."""
    feature_cols = [c for c in train_df.columns
                    if c not in {"well", "prediction_id", "target"}]
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(gkf.split(train_df, train_df["target"],
                             groups=train_df["well"]))

    params = dict(LGB_PARAMS_BASE)
    params["random_state"] = seed
    # bagging_seed is honoured per-tree by lightgbm; vary it too
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed
    # data_random_seed used in `data_random_seed`-style splits
    params["data_random_seed"] = seed

    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_rmses: list[float] = []
    best_iters: list[int] = []
    target_arr = train_df["target"].to_numpy()

    for fold, (tr, va) in enumerate(splits):
        x_tr = train_df.iloc[tr][feature_cols]
        y_tr = train_df.iloc[tr]["target"]
        x_va = train_df.iloc[va][feature_cols]
        y_va = train_df.iloc[va]["target"]
        dtr = lgb.Dataset(x_tr, label=y_tr)
        dva = lgb.Dataset(x_va, label=y_va, reference=dtr)
        m = lgb.train(
            params, dtr, valid_sets=[dva],
            num_boost_round=3000,
            callbacks=[lgb.early_stopping(125, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        oof[va] = m.predict(
            x_va, num_iteration=m.best_iteration
        ).astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof[va] - y_va.values) ** 2)))
        fold_rmses.append(rmse)
        best_iters.append(int(m.best_iteration or 0))
        print(f"   seed={seed} fold {fold}: rmse={rmse:.4f}  "
              f"best_iter={m.best_iteration}", flush=True)

    return oof, fold_rmses, best_iters


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(oof: np.ndarray, train_df: pd.DataFrame) -> dict:
    target = train_df["target"].to_numpy()
    err_target = oof - target
    overall_rmse = float(np.sqrt(np.mean(err_target ** 2)))

    truth = train_df["last_known_tvt"].to_numpy() + target
    pred = train_df["last_known_tvt"].to_numpy() + oof
    e = pred - truth
    df = pd.DataFrame({"well": train_df["well"].to_numpy(), "err": e})
    well_rmse = df.groupby("well")["err"].apply(
        lambda s: float(np.sqrt(np.mean(s.values ** 2)))
    )
    return dict(
        overall_rmse=overall_rmse,
        well_median=float(well_rmse.median()),
        well_mean=float(well_rmse.mean()),
        well_p75=float(well_rmse.quantile(0.75)),
        well_p90=float(well_rmse.quantile(0.9)),
        well_p95=float(well_rmse.quantile(0.95)),
        well_p99=float(well_rmse.quantile(0.99)),
        well_max=float(well_rmse.max()),
        worst_well=str(well_rmse.idxmax()),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", action="store_true",
                    help="Also run MLP multi-seed feature variant.")
    ap.add_argument("--seeds", type=int, nargs="+", default=LGB_SEEDS,
                    help="LGB seeds to use.")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="Rebuild feature cache even if it exists.")
    ap.add_argument("--mlp-seeds", type=int, nargs="+", default=MLP_SEEDS,
                    help="MLP seeds for stage 2.")
    ap.add_argument("--no-mlp", action="store_true",
                    help="Use v8 features only (no MLP) — fast fallback.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    overall_t = time.perf_counter()

    # ---- Stage 1 ---------------------------------------------------------
    print("=" * 70)
    if args.no_mlp:
        print("Stage 1: LGB multi-seed (v8 features, NO MLP)")
        cache = Path("/tmp/v8_multiseed_features.pkl")
    else:
        print("Stage 1: LGB multi-seed (single MLP=42)")
        cache = FEATURES_CACHE
    print("=" * 70, flush=True)
    train_df = build_features(mlp_seeds=None,
                              cache_path=cache,
                              force_rebuild=args.force_rebuild,
                              no_mlp=args.no_mlp)
    print(f"   features columns: {train_df.shape[1]}, "
          f"rows: {train_df.shape[0]:,}", flush=True)

    seed_oofs: dict[int, np.ndarray] = {}
    seed_metrics: dict[int, dict] = {}
    seed_fold_rmses: dict[int, list[float]] = {}
    seed_best_iters: dict[int, list[int]] = {}

    for s in args.seeds:
        t0 = time.perf_counter()
        print(f"\n>> seed={s}", flush=True)
        oof, fold_rmses, best_iters = run_lgb_oof(train_df, seed=s)
        seed_oofs[s] = oof
        seed_fold_rmses[s] = fold_rmses
        seed_best_iters[s] = best_iters
        m = metrics(oof, train_df)
        seed_metrics[s] = m
        print(f"   seed={s}: overall={m['overall_rmse']:.4f}  "
              f"well_max={m['well_max']:.2f}  "
              f"well_p90={m['well_p90']:.2f}  "
              f"well_p99={m['well_p99']:.2f}  "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

        df_out = pd.DataFrame({
            "prediction_id": train_df["prediction_id"].to_numpy(),
            "well": train_df["well"].to_numpy(),
            "target": train_df["target"].to_numpy(),
            "oof": oof.astype(np.float64),
            "last_known_tvt": train_df["last_known_tvt"].to_numpy(),
        })
        df_out.to_csv(f"/tmp/v9_multiseed_oof_{s}.csv", index=False)
        print(f"   saved /tmp/v9_multiseed_oof_{s}.csv", flush=True)

    # Averaged OOF
    avg_oof = np.mean(np.stack(list(seed_oofs.values()), axis=0), axis=0)
    avg_metrics = metrics(avg_oof, train_df)

    # Save averaged OOF
    df_avg = pd.DataFrame({
        "prediction_id": train_df["prediction_id"].to_numpy(),
        "well": train_df["well"].to_numpy(),
        "target": train_df["target"].to_numpy(),
        "oof": avg_oof.astype(np.float64),
        "last_known_tvt": train_df["last_known_tvt"].to_numpy(),
    })
    df_avg.to_csv("/tmp/v9_multiseed_avg_oof.csv", index=False)
    print("\n>> averaged across seeds:")
    print(f"   overall={avg_metrics['overall_rmse']:.4f}  "
          f"well_max={avg_metrics['well_max']:.2f}  "
          f"well_p90={avg_metrics['well_p90']:.2f}  "
          f"well_p99={avg_metrics['well_p99']:.2f}  "
          f"(worst={avg_metrics['worst_well']})", flush=True)

    # Per-seed mean
    per_seed_overall = float(np.mean([m["overall_rmse"]
                                       for m in seed_metrics.values()]))
    per_seed_max = float(np.mean([m["well_max"]
                                   for m in seed_metrics.values()]))
    per_seed_p90 = float(np.mean([m["well_p90"]
                                   for m in seed_metrics.values()]))
    delta_overall = avg_metrics["overall_rmse"] - per_seed_overall
    delta_max = avg_metrics["well_max"] - per_seed_max
    delta_p90 = avg_metrics["well_p90"] - per_seed_p90

    print(f"\n>> Per-seed mean overall = {per_seed_overall:.4f}")
    print(f"   Per-seed mean well_max = {per_seed_max:.2f}")
    print(f"   Per-seed mean well_p90 = {per_seed_p90:.2f}")
    print(f"   Δ overall = {delta_overall:+.4f}  "
          f"Δ well_max = {delta_max:+.2f}  "
          f"Δ well_p90 = {delta_p90:+.2f}", flush=True)

    results = {
        "stage1": {
            "lgb_seeds": list(args.seeds),
            "per_seed_metrics": {str(k): v for k, v in seed_metrics.items()},
            "per_seed_fold_rmses": {str(k): v for k, v in seed_fold_rmses.items()},
            "per_seed_best_iters": {str(k): v for k, v in seed_best_iters.items()},
            "avg_metrics": avg_metrics,
            "per_seed_mean": dict(
                overall_rmse=per_seed_overall,
                well_max=per_seed_max,
                well_p90=per_seed_p90,
            ),
            "delta_avg_minus_perseed_mean": dict(
                overall_rmse=delta_overall,
                well_max=delta_max,
                well_p90=delta_p90,
            ),
            "n_features": int(train_df.shape[1]) - 3,
            "n_rows": int(train_df.shape[0]),
        },
    }

    # ---- Stage 2 (optional) ---------------------------------------------
    if args.stage2:
        print("\n" + "=" * 70)
        print(f"Stage 2: MLP multi-seed (seeds={args.mlp_seeds})")
        print("=" * 70, flush=True)
        cache2 = Path("/tmp/v9_multiseed_features_mlp_avg.pkl")
        train_df2 = build_features(
            mlp_seeds=args.mlp_seeds,
            cache_path=cache2,
            force_rebuild=args.force_rebuild,
        )
        # Run a single LGB seed (42) to compare apples-to-apples vs single-MLP.
        print(">> stage2: 1-seed LGB (42) on multi-MLP features", flush=True)
        oof2, fold2, biters2 = run_lgb_oof(train_df2, seed=42)
        m2 = metrics(oof2, train_df2)
        print(f"   stage2 single-LGB on multi-MLP: "
              f"overall={m2['overall_rmse']:.4f}  "
              f"well_max={m2['well_max']:.2f}  "
              f"well_p90={m2['well_p90']:.2f}", flush=True)

        # Also run the full multi-LGB-seed on multi-MLP features for the
        # complete quadrant.
        seed_oofs2: dict[int, np.ndarray] = {}
        seed_metrics2: dict[int, dict] = {}
        for s in args.seeds:
            print(f">> stage2 seed={s}", flush=True)
            oof_s, _, _ = run_lgb_oof(train_df2, seed=s)
            seed_oofs2[s] = oof_s
            seed_metrics2[s] = metrics(oof_s, train_df2)
        avg2 = np.mean(np.stack(list(seed_oofs2.values()), axis=0), axis=0)
        avg_m2 = metrics(avg2, train_df2)

        results["stage2"] = {
            "mlp_seeds": list(args.mlp_seeds),
            "single_lgb_metrics": m2,
            "single_lgb_fold_rmses": fold2,
            "per_seed_metrics": {str(k): v for k, v in seed_metrics2.items()},
            "avg_metrics": avg_m2,
        }

    # ---- Save & summarise ------------------------------------------------
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n>> saved {RESULTS_JSON}")
    print(f"   total wall: {time.perf_counter() - overall_t:.1f}s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
