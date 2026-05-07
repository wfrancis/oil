"""Sweep shrinkage parameters on v9 OOF and report whether shrinkage
reduces max-well-RMSE without blowing up overall RMSE.

The fundamental trade:
  - alpha=1.0 (no shrinkage) preserves all model signal but lets
    catastrophic predictions through (max well RMSE 165 in current v9).
  - alpha=0.0 (full shrinkage to anchor) gives constant baseline,
    losing all the geosteering signal but capping max well RMSE at the
    constant-baseline level (~10-15 ft per the diagnostic).
  - Somewhere in between is the private-LB-optimal point.

We sweep alpha in {0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} and band in
{20, 30, 40, 50} ft, reporting overall + max-well + p90-well RMSE.

Usage:
    python3 rogii/bench/anchor_shrinkage_score.py \
      --oof /tmp/v9_oof.csv \
      --out /Users/william/drilling_oil_gas/rogii/bench/anchor_shrinkage_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_shrinkage import (
    constant_shrinkage,
    hard_cap,
    evaluate_shrinkage,
)


def sweep(oof_path: Path, out_path: Path) -> int:
    print(f">> Loading {oof_path}", flush=True)
    df = pl.read_csv(str(oof_path))
    needed = {"target", "oof_pred_v9", "well"}
    missing = needed - set(df.columns)
    if missing:
        print(f"FATAL: oof CSV is missing columns: {sorted(missing)}", flush=True)
        return 1

    target = df["target"].to_numpy().astype(np.float64)
    pred = df["oof_pred_v9"].to_numpy().astype(np.float64)
    well = df["well"].to_numpy()

    print(f"   rows: {target.size}, wells: {len(set(well))}", flush=True)

    # Baseline: no shrinkage
    base = evaluate_shrinkage(pred, target, well)
    print(f"\nbase     overall={base.overall_rmse:.4f}  median={base.median_well_rmse:.3f}"
          f"  p90={base.p90_well_rmse:.3f}  max={base.max_well_rmse:.3f}",
          flush=True)

    # Constant baseline: alpha=0
    const = evaluate_shrinkage(np.zeros_like(pred), target, well)
    print(f"\nconst    overall={const.overall_rmse:.4f}  median={const.median_well_rmse:.3f}"
          f"  p90={const.p90_well_rmse:.3f}  max={const.max_well_rmse:.3f}",
          flush=True)

    results = {
        "base": base.__dict__,
        "constant_baseline": const.__dict__,
        "constant_shrinkage": {},
        "hard_cap": {},
    }

    print("\n=== constant shrinkage (alpha sweep) ===", flush=True)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        shrunk = constant_shrinkage(pred, alpha=alpha)
        r = evaluate_shrinkage(shrunk, target, well)
        results["constant_shrinkage"][f"alpha={alpha}"] = r.__dict__
        print(f"alpha={alpha:.1f}  overall={r.overall_rmse:.4f}  "
              f"median={r.median_well_rmse:.3f}  p90={r.p90_well_rmse:.3f}  "
              f"max={r.max_well_rmse:.3f}", flush=True)

    print("\n=== hard cap (band sweep, ft) ===", flush=True)
    for band in [15.0, 20.0, 25.0, 30.0, 40.0, 60.0]:
        capped = hard_cap(pred, band=band)
        r = evaluate_shrinkage(capped, target, well)
        results["hard_cap"][f"band={band}"] = r.__dict__
        print(f"band={band:5.1f}  overall={r.overall_rmse:.4f}  "
              f"median={r.median_well_rmse:.3f}  p90={r.p90_well_rmse:.3f}  "
              f"max={r.max_well_rmse:.3f}", flush=True)

    # Pareto front: lowest overall RMSE among configs with max-well-RMSE < 50
    print("\n=== Pareto pick (max_well_rmse < 50, lowest overall) ===", flush=True)
    candidates = []
    for cat in ("constant_shrinkage", "hard_cap"):
        for k, v in results[cat].items():
            if v["max_well_rmse"] < 50.0:
                candidates.append((v["overall_rmse"], v["max_well_rmse"], cat, k))
    candidates.sort()
    for ov, mx, c, k in candidates[:5]:
        print(f"  {c}:{k}  overall={ov:.4f}  max={mx:.3f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True, help="Path to OOF CSV with columns target, oof_pred_v9, well")
    parser.add_argument("--out", required=True, help="Path to write results JSON")
    args = parser.parse_args(argv)
    return sweep(Path(args.oof), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
