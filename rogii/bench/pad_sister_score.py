"""Local scorer for the pad-sister retrieval predictor.

Validates `rogii.src.pad_sister.PadSisterIndex` against held-out train wells
in the same GroupKFold setup we use for v8 (seed=42, shuffle=True, 5 folds).

Self-well exclusion is enforced — the held-out well's centroid IS in the
sister index but its sister-ranking would obviously include itself. Pass
``exclude_wid=well_id`` to drop it.

Reports:
  * Overall RMSE / MAE / bias
  * Per-well distribution (median, p90, max) — focus on the catastrophic
    outliers v8 misses (max well RMSE 56.13 ft).
  * Sister coverage diagnostics: how many wells got >= 1 sister, what
    fraction of rows had >= 1 sister covering them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pad_sister import PadSisterIndex, _read_horizontal

DEFAULT_TRAIN_DIR = ROOT / "data" / "competition" / "train"

logger = logging.getLogger("rogii.pad_sister_score")


def _stable_score(well: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{well}".encode(), digest_size=8).digest(),
        "big",
    )


def _select_wells(train_dir: Path, *, limit: int, seed: int) -> list[str]:
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    wells = [p.name.replace("__horizontal_well.csv", "") for p in paths]
    if limit > 0:
        wells = sorted(wells, key=lambda w: _stable_score(w, seed))[:limit]
    return wells


def _truth_for_well(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "TVT_input" not in df.columns or "TVT" not in df.columns:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    tvt_in = df["TVT_input"].to_numpy().astype(np.float64)
    tvt = df["TVT"].to_numpy().astype(np.float64)
    mask = ~np.isfinite(tvt_in) & np.isfinite(tvt)
    idx = np.flatnonzero(mask)
    return idx.astype(np.int64), tvt[idx].astype(np.float64)


def cmd_hold(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)

    print(">> Building pad-sister index ...", flush=True)
    t0 = time.perf_counter()
    idx = PadSisterIndex.fit(train_dir)
    print(f"   {len(idx.wid_order)} wells, fit_s={time.perf_counter() - t0:.1f}", flush=True)

    held = _select_wells(train_dir, limit=args.n, seed=args.seed)

    overall_rows: list[dict] = []
    n_with_sisters = 0
    n_total_sisters = 0

    t_pred = time.perf_counter()
    for wid in held:
        path = train_dir / f"{wid}__horizontal_well.csv"
        df = _read_horizontal(path)
        eval_idx, truth = _truth_for_well(df)
        if eval_idx.size == 0:
            continue
        out = idx.predict_well(df, query_wid=wid, k=args.k, exclude_wid=wid)
        sister_count = len(out["sister_wids"])
        n_with_sisters += int(sister_count > 0)
        n_total_sisters += sister_count
        if sister_count == 0:
            continue
        tvt_pred = out["tvt"]
        # Some rows may be NaN (no sister covered them) — replace with
        # last_known_TVT_input as a fallback so we can still score.
        bad = ~np.isfinite(tvt_pred)
        if bad.any():
            tvt_in = df["TVT_input"].to_numpy().astype(np.float64)
            finite_in = np.isfinite(tvt_in)
            if finite_in.any():
                last_anchor = float(tvt_in[np.flatnonzero(finite_in)[-1]])
                tvt_pred = np.where(bad, last_anchor, tvt_pred)
        err = tvt_pred[eval_idx] - truth
        overall_rows.append({
            "well": wid,
            "err": err,
            "sisters": sister_count,
            "score_top": float(out["sister_scores"][0]) if out["sister_scores"] else float("nan"),
            "coverage_frac": float(out["coverage"][eval_idx].mean()),
        })
    pred_s = time.perf_counter() - t_pred

    if not overall_rows:
        print("No predictions made.")
        return

    err = np.concatenate([r["err"] for r in overall_rows])
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    p90_ae = float(np.percentile(np.abs(err), 90))
    well_rmse = np.array([
        float(np.sqrt(np.mean(r["err"] ** 2))) for r in overall_rows
    ])

    print(f"\n=== Pad-sister overall (k={args.k}) ===")
    print(json.dumps({
        "rows": int(err.size),
        "wells": int(len(overall_rows)),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "p90_ae": p90_ae,
        "median_well_rmse": float(np.median(well_rmse)),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "max_well_rmse": float(np.max(well_rmse)),
        "wells_with_sisters": int(n_with_sisters),
        "mean_sisters_per_well": n_total_sisters / max(1, len(held)),
        "pred_s": pred_s,
        "pred_s_per_well": pred_s / max(1, len(held)),
    }, indent=2, sort_keys=True))

    if args.show_worst > 0:
        print(f"\nWorst {args.show_worst} wells:")
        for r in sorted(overall_rows, key=lambda r: -np.sqrt(np.mean(r["err"] ** 2)))[: args.show_worst]:
            wrmse = float(np.sqrt(np.mean(r["err"] ** 2)))
            print(
                f"  {r['well']}  rmse={wrmse:.3f}  bias={np.mean(r['err']):+.3f}  "
                f"sisters={r['sisters']}  top_score={r['score_top']:.3f}  "
                f"coverage={r['coverage_frac']:.2f}  rows={r['err'].size}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hold = sub.add_parser("hold", help="Hold N wells; quick sanity score.")
    hold.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    hold.add_argument("--n", type=int, default=50)
    hold.add_argument("--seed", type=int, default=42)
    hold.add_argument("--k", type=int, default=5)
    hold.add_argument("--show-worst", type=int, default=10)
    hold.set_defaults(func=cmd_hold)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
