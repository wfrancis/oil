"""Run the meta-stacker on whatever OOF predictions are currently on disk.

Inputs (in priority order):
  /tmp/v9_oof.csv          v9 (no-beam) full 5-fold OOF — required for the
                           non-trivial part of this script. If absent, we
                           fall back to a constant-only "stack" (which is
                           degenerate and useful only as a sanity check).
  /tmp/v9_well_rmse.csv    per-well RMSE summary written by the v9 runner.

The constant-baseline predictor is target=0 (i.e. the model predicts
`last_known_TVT_input` for every row). v6 is implicitly this.

Usage:
  python bench/stack_score.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from stacker import stack_oof  # noqa: E402


V9_OOF = Path("/tmp/v9_oof.csv")


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def main() -> int:
    if not V9_OOF.exists():
        print(f"FATAL: {V9_OOF} does not exist — v9 OOF run not finished",
              flush=True)
        return 1

    t0 = time.perf_counter()
    df = pl.read_csv(V9_OOF, schema_overrides={
        "well": pl.String, "prediction_id": pl.String,
        "row_idx": pl.Int32, "target": pl.Float64,
        "oof_pred_v9": pl.Float64, "last_known_tvt": pl.Float64,
    })
    print(f">> loaded v9 OOF: {df.shape}, {time.perf_counter() - t0:.1f}s",
          flush=True)

    target = df["target"].to_numpy()
    groups = df["well"].to_numpy()
    oof_v9 = df["oof_pred_v9"].to_numpy()
    n = len(target)

    # constant baseline = predict last_known_tvt -> in the residual target
    # space that's exactly 0
    oof_const = np.zeros(n, dtype=np.float64)

    predictions = {
        "v9":       oof_v9,
        "constant": oof_const,
    }

    # ---- single-predictor RMSEs (sanity)
    print("\n>> single-predictor OOF RMSE:", flush=True)
    for name, p in predictions.items():
        rmse = float(np.sqrt(np.mean((p - target) ** 2)))
        print(f"   {name:10s}  {rmse:7.4f}", flush=True)

    # ---- stacker
    t0 = time.perf_counter()
    out = stack_oof(predictions, target=target, groups=groups,
                    n_splits=5, seed=42, alpha=1.0)
    print(f"\n>> stacker fit: {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"\n   simple-mean OOF RMSE     = {out['simple_mean_rmse']:.4f}",
          flush=True)
    print(f"   ridge-stack OOF RMSE     = {out['ridge_oof_rmse']:.4f}",
          flush=True)
    print("\n   per-fold ridge weights:", flush=True)
    for fi, w in enumerate(out["ridge_weights"]):
        as_str = "  ".join(
            f"{n}={v:7.4f}" for n, v in zip(out["names"], w)
        )
        print(f"     fold {fi}:  {as_str}", flush=True)
    print("\n   mean ridge weights:", flush=True)
    for n, v in zip(out["names"], out["mean_ridge_weights"]):
        print(f"     {n:10s}  {v:7.4f}", flush=True)

    # ---- per-well comparison
    print("\n>> per-well comparison: ridge vs each base predictor", flush=True)
    print(f"   ridge max-well-RMSE                      = "
          f"{out['per_well_max_rmse_ridge']:.3f}", flush=True)
    for n in out["names"]:
        frac = out["per_well_better"][n]
        max_n = out["per_well_max_rmse"][n]
        print(f"   ridge beats {n:10s} on  {_fmt_pct(frac)}  of wells "
              f"  (its max-well-RMSE = {max_n:.3f})", flush=True)

    # ---- per-fold table
    print("\n>> per-fold OOF RMSE:", flush=True)
    rows = ["    fold       ridge      mean    " +
            "    ".join(f"{n:>8s}" for n in out["names"])]
    for fi in range(5):
        row = [f"      {fi}    "]
        row.append(f"{out['per_fold_rmse']['ridge'][fi]:8.4f}")
        row.append(f"   {out['per_fold_rmse']['simple_mean'][fi]:7.4f}")
        for n in out["names"]:
            row.append(f"   {out['per_fold_rmse'][n][fi]:7.4f}")
        rows.append("  ".join(row))
    print("\n".join(rows), flush=True)

    # ---- winner
    print(f"\n>> WINNER: {out['best']}  (RMSE {out['best_rmse']:.4f})",
          flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
