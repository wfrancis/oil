"""Local validator for the formation-surface predictor.

Validation protocol (matches Kaggle conditions):
  * Use the existing TVT_input mask in each train well as the eval mask
    (rows where TVT_input is NaN are scored against true TVT).
  * For the held-out fold, exclude those wells from the predictor's
    training data — both row-level KNN and the per-well centroid table.

Outputs OOF metrics: rmse, mae, bias, p90_ae, mean_well_rmse, plus the
worst-well table for diagnosis.

Run modes:
    fold            — single GroupKFold fold, fast smoke
    cv              — full 5-fold GroupKFold OOF
    hold-n          — hold last N wells for a quick sanity check
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

from formation_stack import (
    FORMATION_COLS,
    FormationStackPredictor,
    load_train_horizontals,
)

DEFAULT_TRAIN_DIR = ROOT / "data" / "competition" / "train"

logger = logging.getLogger("rogii.formation_score")


def _stable_score(well: str, seed: int) -> int:
    payload = f"{seed}:{well}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _kfold_assignments(wells: list[str], n_folds: int, seed: int) -> dict[str, int]:
    ordered = sorted(wells, key=lambda w: _stable_score(w, seed))
    return {w: i % n_folds for i, w in enumerate(ordered)}


def _load_with_full_horizontals(train_dir: Path) -> tuple[
    dict[str, pl.DataFrame], dict[str, pl.DataFrame]
]:
    """Load both:
      * train_wells_for_fit — only rows with all formations + TVT finite
      * full_wells          — all rows (for row indexing into TVT_input/TVT)
    Returns (train_for_fit, full).
    """
    train_for_fit = load_train_horizontals(train_dir, formations=FORMATION_COLS)
    full: dict[str, pl.DataFrame] = {}
    for path in sorted(train_dir.glob("*__horizontal_well.csv")):
        wid = path.name.replace("__horizontal_well.csv", "")
        full[wid] = pl.read_csv(
            path,
            infer_schema_length=2000,
            null_values=["", "NA", "NaN", "nan", "null"],
            truncate_ragged_lines=True,
        )
    return train_for_fit, full


def _truth_for_well(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (eval_idx, truth) — rows where TVT_input is NaN and TVT is finite."""
    if "TVT_input" not in df.columns or "TVT" not in df.columns:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    tvt_in = df["TVT_input"].to_numpy().astype(np.float64, copy=False)
    tvt = df["TVT"].to_numpy().astype(np.float64, copy=False)
    mask = ~np.isfinite(tvt_in) & np.isfinite(tvt)
    idx = np.flatnonzero(mask)
    return idx.astype(np.int64), tvt[idx].astype(np.float64)


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"rmse": float("nan"), "wells": 0, "rows": 0}
    err = np.concatenate([r["err"] for r in rows])
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    p90_ae = float(np.percentile(np.abs(err), 90))
    well_rmse = np.array([
        float(np.sqrt(np.mean(r["err"] ** 2))) for r in rows
    ])
    return {
        "rows": int(err.size),
        "wells": len(rows),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "p90_ae": p90_ae,
        "median_well_rmse": float(np.median(well_rmse)),
        "mean_well_rmse": float(np.mean(well_rmse)),
        "max_well_rmse": float(np.max(well_rmse)),
    }


def cmd_cv(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    train_for_fit, full = _load_with_full_horizontals(train_dir)
    wells = sorted(full)
    if args.limit > 0:
        wells = sorted(wells, key=lambda w: _stable_score(w, args.seed))[: args.limit]

    fold_assign = _kfold_assignments(wells, args.n_folds, args.seed)

    overall_rows: list[dict] = []
    median_tvt = float(np.median(np.concatenate([
        df["TVT"].to_numpy().astype(np.float64) for df in train_for_fit.values()
    ])))

    for fold in range(args.n_folds):
        fold_test = [w for w in wells if fold_assign[w] == fold]
        fold_train = {w: df for w, df in train_for_fit.items() if fold_assign.get(w, -1) != fold}
        if not fold_train or not fold_test:
            continue
        t0 = time.perf_counter()
        pred_obj = FormationStackPredictor(
            train_wells=fold_train,
            formations=FORMATION_COLS,
            k_row=args.k_row,
            k_plane=args.k_plane,
            b_method=args.b_method,
            primary_formation=args.primary,
        ).fit()
        fit_s = time.perf_counter() - t0

        fold_rows: list[dict] = []
        t0 = time.perf_counter()
        for wid in fold_test:
            df = full[wid]
            eval_idx, truth = _truth_for_well(df)
            if eval_idx.size == 0:
                continue
            tvt_pred = pred_obj.predict_well(
                df, well_id=wid,
                train_median_tvt=median_tvt,
                strategy=args.strategy,
            )
            err = tvt_pred[eval_idx] - truth
            fold_rows.append({"well": wid, "err": err})
            overall_rows.append({"well": wid, "err": err})
        pred_s = time.perf_counter() - t0

        m = _summarize(fold_rows)
        print(
            f"fold={fold}  rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  "
            f"bias={m['bias']:+.3f}  p90={m['p90_ae']:.3f}  "
            f"wells={m['wells']}  rows={m['rows']}  "
            f"fit_s={fit_s:.1f}  pred_s={pred_s:.1f}"
        )

    overall = _summarize(overall_rows)
    print()
    print("=== Overall OOF ===")
    print(json.dumps(overall, indent=2, sort_keys=True))


def cmd_hold(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    train_for_fit, full = _load_with_full_horizontals(train_dir)
    all_wells = sorted(full)
    held = sorted(all_wells, key=lambda w: _stable_score(w, args.seed))[: args.n]

    fold_train = {w: df for w, df in train_for_fit.items() if w not in set(held)}
    median_tvt = float(np.median(np.concatenate([
        df["TVT"].to_numpy().astype(np.float64) for df in train_for_fit.values()
    ])))

    t0 = time.perf_counter()
    pred_obj = FormationStackPredictor(
        train_wells=fold_train,
        formations=FORMATION_COLS,
        k_row=args.k_row,
        k_plane=args.k_plane,
        b_method=args.b_method,
        primary_formation=args.primary,
    ).fit()
    fit_s = time.perf_counter() - t0

    rows: list[dict] = []
    pred_s_total = 0.0
    for wid in held:
        df = full[wid]
        eval_idx, truth = _truth_for_well(df)
        if eval_idx.size == 0:
            continue
        t0 = time.perf_counter()
        tvt_pred = pred_obj.predict_well(
            df, well_id=wid,
            train_median_tvt=median_tvt,
            strategy=args.strategy,
        )
        pred_s_total += time.perf_counter() - t0
        err = tvt_pred[eval_idx] - truth
        rows.append({"well": wid, "err": err})

    m = _summarize(rows)
    print(
        f"holdout  rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  bias={m['bias']:+.3f}  "
        f"p90={m['p90_ae']:.3f}  wells={m['wells']}  rows={m['rows']}  "
        f"fit_s={fit_s:.1f}  pred_s={pred_s_total:.1f}"
    )
    print()
    if args.show_worst > 0 and rows:
        worst = sorted(rows, key=lambda r: -np.sqrt(np.mean(r["err"] ** 2)))[: args.show_worst]
        print(f"worst {args.show_worst} wells:")
        for r in worst:
            print(
                f"  {r['well']}  rmse={np.sqrt(np.mean(r['err']**2)):.3f}  "
                f"bias={np.mean(r['err']):+.3f}  rows={r['err'].size}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cv = sub.add_parser("cv", help="Full GroupKFold OOF.")
    cv.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    cv.add_argument("--n-folds", type=int, default=5)
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--limit", type=int, default=0, help="Subsample wells (0 = all)")
    cv.add_argument("--k-row", type=int, default=20)
    cv.add_argument("--k-plane", type=int, default=10)
    cv.add_argument("--b-method", default="median", choices=["median", "huber", "trimmed_mean"])
    cv.add_argument("--primary", default="EGFDL", choices=list(FORMATION_COLS))
    cv.add_argument("--strategy", default="row_only",
                    choices=["row_only", "plane_only", "row_avg_plane", "formation_ensemble"])
    cv.set_defaults(func=cmd_cv)

    hold = sub.add_parser("hold", help="Hold last N wells; quick sanity check.")
    hold.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    hold.add_argument("--n", type=int, default=20)
    hold.add_argument("--seed", type=int, default=42)
    hold.add_argument("--k-row", type=int, default=20)
    hold.add_argument("--k-plane", type=int, default=10)
    hold.add_argument("--b-method", default="median", choices=["median", "huber", "trimmed_mean"])
    hold.add_argument("--primary", default="EGFDL", choices=list(FORMATION_COLS))
    hold.add_argument("--strategy", default="row_only",
                      choices=["row_only", "plane_only", "row_avg_plane", "formation_ensemble"])
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
