"""GroupKFold(5, shuffle, seed=42) OOF benchmark of neural ANCC vs KNN.

For each variant we measure:
  - ANCC-prediction RMSE on held-out wells (per-fold + overall pooled)
  - TVT-prediction RMSE using the closed-form anchor:
        tvt_pred = -Z + ANCC_pred + b_well_prefix_median
    where b_well_prefix_median is the median of (TVT_input + Z - ANCC_pred)
    over the visible prefix rows of each held-out well. This anchors the
    surface to each well's vertical bias using its own prefix.
  - Per-well RMSE distribution (median, p90, max).

Variants compared:
  knn               : row-level KNN (K=20, IDW p=1) — konbu17 baseline
  mlp_no_pe         : MLP without positional encoding
  mlp_pe_l8         : MLP + sinusoidal PE, L=8
  mlp_pe_l16        : MLP + sinusoidal PE, L=16
  mlp_pe_l8_multi   : Multi-output MLP (all 6 formations) + PE L=8

Usage
-----
    python3 bench/neural_ancc_score.py [--limit N] [--epochs E]

Outputs JSON to bench/neural_ancc_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_ancc import (  # noqa: E402
    FORMATIONS,
    AnccNet,
    TrainConfig,
    load_train_rows,
)


# ---------------------------------------------------------------------------
# KNN baseline (konbu17 row-level KNN, ANCC only)
# ---------------------------------------------------------------------------

def knn_predict_ancc(
    xy_train: np.ndarray,
    ancc_train: np.ndarray,
    xy_q: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Row-level KNN with IDW p=1 — same recipe as feature_builder.RowKNN
    but ANCC-only. Self-well exclusion is implicit because the train set
    here already excludes the held-out fold's wells.
    """
    scale = xy_train.std(axis=0)
    scale = np.where(scale < 1e-3, 1.0, scale)
    tree = cKDTree(xy_train / scale)
    q = xy_q / scale
    dist, idx = tree.query(q, k=k, workers=-1)
    valid = np.isfinite(dist)
    w = np.where(valid, 1.0 / (dist + 1e-3), 0.0)
    sw = w.sum(axis=1, keepdims=True)
    safe = np.where(sw < 1e-9, 1.0, sw)
    a_n = ancc_train[idx]                         # (M, K)
    pred = (a_n * w).sum(axis=1, keepdims=True) / safe
    bad = sw < 1e-9
    if bad.any():
        pred[bad.squeeze(-1)] = ancc_train.mean()
    return pred.squeeze(-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-well b_prefix and TVT scoring helpers
# ---------------------------------------------------------------------------

def load_well_arrays(train_dir: Path, wells: list[str]) -> dict[str, dict]:
    """Load per-well (X, Y, Z, TVT, TVT_input, ANCC) arrays for the wells we
    care about. Used to compute b_prefix and per-well RMSE on held-out folds.
    """
    out: dict[str, dict] = {}
    for w in wells:
        p = train_dir / f"{w}__horizontal_well.csv"
        try:
            df = pl.read_csv(p, infer_schema_length=10_000)
        except Exception:
            continue
        for c in ["ANCC"]:
            if df[c].dtype == pl.Utf8:
                df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        x = df["X"].to_numpy()
        y = df["Y"].to_numpy()
        z = df["Z"].to_numpy()
        tvt = df["TVT"].to_numpy()
        tvt_in = df["TVT_input"].to_numpy()
        ancc = df["ANCC"].to_numpy()
        out[w] = {"X": x, "Y": y, "Z": z, "TVT": tvt, "TVT_input": tvt_in, "ANCC": ancc}
    return out


def per_well_metrics(
    well_arrays: dict[str, dict],
    pred_ancc_by_well: dict[str, np.ndarray],
    only_hidden: bool = True,
) -> dict:
    """For each well, compute b_prefix (median of TVT_input + Z - ANCC_pred over
    prefix rows), then plug into TVT closed-form on the hidden segment, and
    compute per-well TVT RMSE + ANCC RMSE.
    """
    rows = []
    for w, arrs in well_arrays.items():
        if w not in pred_ancc_by_well:
            continue
        pred_ancc = pred_ancc_by_well[w]
        z = arrs["Z"]
        tvt = arrs["TVT"]
        tvt_in = arrs["TVT_input"]
        ancc_true = arrs["ANCC"]
        prefix_mask = np.isfinite(tvt_in)
        if not prefix_mask.any():
            continue
        b_prefix = float(
            np.median(
                tvt_in[prefix_mask] + z[prefix_mask] - pred_ancc[prefix_mask]
            )
        )
        if only_hidden:
            scoring_mask = ~prefix_mask
        else:
            scoring_mask = np.ones_like(prefix_mask)
        if not scoring_mask.any():
            continue
        tvt_pred = -z + pred_ancc + b_prefix
        # TVT RMSE
        finite = scoring_mask & np.isfinite(tvt) & np.isfinite(tvt_pred)
        if finite.any():
            tvt_err = tvt_pred[finite] - tvt[finite]
            tvt_rmse = float(np.sqrt(np.mean(tvt_err ** 2)))
        else:
            tvt_rmse = float("nan")
        # ANCC RMSE on hidden segment
        finite_a = scoring_mask & np.isfinite(ancc_true) & np.isfinite(pred_ancc)
        if finite_a.any():
            ancc_err = pred_ancc[finite_a] - ancc_true[finite_a]
            ancc_rmse = float(np.sqrt(np.mean(ancc_err ** 2)))
        else:
            ancc_rmse = float("nan")
        rows.append({
            "well": w,
            "rows": int(finite.sum()),
            "ancc_rmse": ancc_rmse,
            "tvt_rmse": tvt_rmse,
            "b_prefix": b_prefix,
        })
    if not rows:
        return {"per_well": [], "summary": {}}
    rmses_t = np.array([r["tvt_rmse"] for r in rows])
    rmses_a = np.array([r["ancc_rmse"] for r in rows])
    finite_t = rmses_t[np.isfinite(rmses_t)]
    finite_a = rmses_a[np.isfinite(rmses_a)]
    summary = {
        "n_wells": len(rows),
        "tvt": {
            "median": float(np.median(finite_t)) if len(finite_t) else float("nan"),
            "p90": float(np.quantile(finite_t, 0.9)) if len(finite_t) else float("nan"),
            "max": float(np.max(finite_t)) if len(finite_t) else float("nan"),
            "mean": float(np.mean(finite_t)) if len(finite_t) else float("nan"),
        },
        "ancc": {
            "median": float(np.median(finite_a)) if len(finite_a) else float("nan"),
            "p90": float(np.quantile(finite_a, 0.9)) if len(finite_a) else float("nan"),
            "max": float(np.max(finite_a)) if len(finite_a) else float("nan"),
            "mean": float(np.mean(finite_a)) if len(finite_a) else float("nan"),
        },
    }
    return {"per_well": rows, "summary": summary}


# ---------------------------------------------------------------------------
# Variant config helper
# ---------------------------------------------------------------------------

def make_cfg(num_freqs: int, out_dim: int, epochs: int, rows_per_epoch: int,
             seed: int = 42) -> TrainConfig:
    return TrainConfig(
        num_freqs=num_freqs,
        hidden=256,
        out_dim=out_dim,
        rows_per_epoch=rows_per_epoch,
        batch_size=4096,
        epochs=epochs,
        lr=1e-3,
        weight_decay=0.0,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(ROOT / "data" / "competition" / "train"))
    parser.add_argument("--out-json", default=str(ROOT / "bench" / "neural_ancc_results.json"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--rows-per-epoch", type=int, default=500_000)
    parser.add_argument("--limit-wells", type=int, default=0,
                        help="If >0, restrict to first N wells (debug only).")
    parser.add_argument("--variants", default="knn,mlp_no_pe,mlp_pe_l8,mlp_pe_l16,mlp_pe_l8_multi")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    print("== loading training rows ==", flush=True)
    t0 = time.perf_counter()
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if args.limit_wells:
        paths = paths[: args.limit_wells]
    xy_all, targets_all, wids_all = load_train_rows(train_dir, paths=paths)
    print(f"   {len(xy_all):,} rows, {len(set(wids_all.tolist()))} wells, "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    f_idx_ancc = FORMATIONS.index("ANCC")

    # Build well->rowindices and well->{TVT, TVT_input, Z, ANCC} for full-well
    # scoring (we need per-well b_prefix, hidden mask, and ANCC truth on the
    # full-well grid, including hidden rows that may have been filtered by
    # drop_nulls during loading).
    print("== loading per-well full arrays for scoring ==", flush=True)
    t0 = time.perf_counter()
    unique_wells = sorted(set(wids_all.tolist()))
    well_arrays = load_well_arrays(train_dir, unique_wells)
    print(f"   {len(well_arrays)} well files loaded, "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    # Build per-well full (X, Y, Z, ANCC, TVT, TVT_input) for prediction
    # — we need to predict ANCC at EVERY row of held-out wells (prefix +
    # hidden) so we can compute b_prefix from prefix and TVT/ANCC RMSE on the
    # hidden segment. This is independent of the (xy_all, targets_all, wids_all)
    # arrays, which were filtered by drop_nulls.
    print("== building full-well xy index ==", flush=True)
    well_full_xy: dict[str, np.ndarray] = {}
    for w, a in well_arrays.items():
        well_full_xy[w] = np.column_stack([a["X"], a["Y"]]).astype(np.float64)

    # GroupKFold by well over the row-level (X, Y, ANCC) corpus
    gkf = GroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(gkf.split(xy_all, targets_all[:, f_idx_ancc], groups=wids_all))

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results: dict = {
        "config": vars(args),
        "n_train_rows": int(len(xy_all)),
        "n_wells": len(unique_wells),
        "variants": {},
    }
    for v in variants:
        results["variants"][v] = {
            "fold_ancc_rmse": [],
            "fold_tvt_summary": [],
            "per_well": [],
            "fit_time_s": [],
            "pred_time_s": [],
        }

    for fold_idx, (tr, va) in enumerate(splits):
        val_well_set = set(np.unique(wids_all[va]).tolist())
        print(f"\n=== fold {fold_idx} : {len(val_well_set)} val wells, "
              f"{len(va):,} val rows, {len(tr):,} train rows ===", flush=True)

        # Predict at FULL well rows for each held-out well (prefix + hidden)
        # so we can compute TVT closed-form on the hidden mask and b_prefix
        # on the visible mask.
        for v in variants:
            print(f"  -- variant {v} --", flush=True)
            if v == "knn":
                # KNN at full-well xy of each held-out well
                xy_train = xy_all[tr]
                ancc_train = targets_all[tr, f_idx_ancc]
                # concat per-well full xy, then de-concat
                xy_q_list = []
                w_lengths = []
                w_order = sorted(val_well_set)
                for w in w_order:
                    xy_q_list.append(well_full_xy[w])
                    w_lengths.append(len(well_full_xy[w]))
                xy_q = np.concatenate(xy_q_list)
                t0 = time.perf_counter()
                pred_q = knn_predict_ancc(xy_train, ancc_train, xy_q, k=20)
                ftime = time.perf_counter() - t0
                # ANCC pooled rmse: only over rows where ANCC truth is finite
                truth_q = []
                for w in w_order:
                    truth_q.append(well_arrays[w]["ANCC"])
                truth_q = np.concatenate(truth_q)
                ok = np.isfinite(truth_q) & np.isfinite(pred_q)
                rmse_pool = float(np.sqrt(np.mean((pred_q[ok] - truth_q[ok]) ** 2)))
                pred_by_well: dict[str, np.ndarray] = {}
                start = 0
                for w, n in zip(w_order, w_lengths):
                    pred_by_well[w] = pred_q[start:start + n]
                    start += n
                fold_pw = per_well_metrics(
                    {w: well_arrays[w] for w in w_order},
                    pred_by_well, only_hidden=True,
                )
                results["variants"][v]["fold_ancc_rmse"].append(rmse_pool)
                results["variants"][v]["fold_tvt_summary"].append(fold_pw["summary"])
                results["variants"][v]["per_well"].extend(fold_pw["per_well"])
                results["variants"][v]["fit_time_s"].append(0.0)
                results["variants"][v]["pred_time_s"].append(ftime)
                summary = fold_pw["summary"]
                print(f"    knn  ANCC pool rmse={rmse_pool:.3f}  "
                      f"TVT median={summary['tvt']['median']:.3f}  "
                      f"p90={summary['tvt']['p90']:.3f}  "
                      f"max={summary['tvt']['max']:.3f}  "
                      f"({ftime:.1f}s)", flush=True)
                continue

            # MLP variants
            multi = v.endswith("_multi")
            num_freqs = (
                0 if v == "mlp_no_pe"
                else (16 if "l16" in v else 8)
            )
            out_dim = len(FORMATIONS) if multi else 1
            cfg = make_cfg(num_freqs=num_freqs, out_dim=out_dim, epochs=args.epochs,
                           rows_per_epoch=args.rows_per_epoch, seed=42 + fold_idx)

            t0 = time.perf_counter()
            xy_train = xy_all[tr]
            t_train = (
                targets_all[tr] if multi else targets_all[tr, f_idx_ancc:f_idx_ancc + 1]
            )
            net = AnccNet(cfg)
            hist = net.fit(xy_train, t_train, verbose=args.verbose)
            ftime = time.perf_counter() - t0

            # Predict at full-well xy of each held-out well
            t0 = time.perf_counter()
            w_order = sorted(val_well_set)
            xy_q_list = [well_full_xy[w] for w in w_order]
            w_lengths = [len(x) for x in xy_q_list]
            xy_q = np.concatenate(xy_q_list)
            pred_q_full = net.predict(xy_q)
            pred_q = pred_q_full[:, f_idx_ancc] if multi else pred_q_full[:, 0]
            ptime = time.perf_counter() - t0

            truth_q = np.concatenate([well_arrays[w]["ANCC"] for w in w_order])
            ok = np.isfinite(truth_q) & np.isfinite(pred_q)
            rmse_pool = float(np.sqrt(np.mean((pred_q[ok] - truth_q[ok]) ** 2)))
            pred_by_well = {}
            start = 0
            for w, n in zip(w_order, w_lengths):
                pred_by_well[w] = pred_q[start:start + n]
                start += n
            fold_pw = per_well_metrics(
                {w: well_arrays[w] for w in w_order},
                pred_by_well, only_hidden=True,
            )
            results["variants"][v]["fold_ancc_rmse"].append(rmse_pool)
            results["variants"][v]["fold_tvt_summary"].append(fold_pw["summary"])
            results["variants"][v]["per_well"].extend(fold_pw["per_well"])
            results["variants"][v]["fit_time_s"].append(ftime)
            results["variants"][v]["pred_time_s"].append(ptime)
            summary = fold_pw["summary"]
            print(f"    {v:<22s} ANCC pool rmse={rmse_pool:.3f}  "
                  f"TVT median={summary['tvt']['median']:.3f}  "
                  f"p90={summary['tvt']['p90']:.3f}  "
                  f"max={summary['tvt']['max']:.3f}  "
                  f"(fit {ftime:.1f}s, pred {ptime:.1f}s)", flush=True)

    # Aggregate overall summaries
    print("\n=== OVERALL ===", flush=True)
    overall = {}
    for v in variants:
        per_well = results["variants"][v]["per_well"]
        if not per_well:
            continue
        rmses_t = np.array([r["tvt_rmse"] for r in per_well])
        rmses_a = np.array([r["ancc_rmse"] for r in per_well])
        finite_t = rmses_t[np.isfinite(rmses_t)]
        finite_a = rmses_a[np.isfinite(rmses_a)]
        # Rows-weighted pooled RMSE
        pooled_ancc = float(np.mean(results["variants"][v]["fold_ancc_rmse"]))
        # Per-fold avg of summaries
        ancc = {
            "median": float(np.median(finite_a)),
            "p90": float(np.quantile(finite_a, 0.9)),
            "max": float(np.max(finite_a)),
            "mean": float(np.mean(finite_a)),
        }
        tvt = {
            "median": float(np.median(finite_t)),
            "p90": float(np.quantile(finite_t, 0.9)),
            "max": float(np.max(finite_t)),
            "mean": float(np.mean(finite_t)),
        }
        overall[v] = {
            "ancc_pooled_rmse_avg_over_folds": pooled_ancc,
            "ancc_per_well_summary": ancc,
            "tvt_per_well_summary": tvt,
            "n_wells": int(len(per_well)),
            "fit_time_s_total": float(np.sum(results["variants"][v]["fit_time_s"])),
            "pred_time_s_total": float(np.sum(results["variants"][v]["pred_time_s"])),
        }
        print(f"  {v:<22s}  ANCC pool {pooled_ancc:.3f}  "
              f"TVT med {tvt['median']:.3f}  p90 {tvt['p90']:.3f}  max {tvt['max']:.3f}  "
              f"(fit {overall[v]['fit_time_s_total']:.1f}s)", flush=True)
    results["overall"] = overall

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"\n>> wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
