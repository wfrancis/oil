"""GroupKFold(5, shuffle, seed=42) OOF benchmark of anisotropic kriging vs KNN.

Mirrors bench/neural_ancc_score.py protocol:
  - 5-fold GroupKFold over the same 765 train wells used by neural_ancc_score.
  - Per-fold pooled ANCC RMSE on held-out wells (full-well prediction grid).
  - Per-well closed-form TVT_pred = -Z + ANCC_pred + b_prefix, where
      b_prefix = median(TVT_input + Z - ANCC_pred) over visible-prefix rows.
  - TVT and ANCC RMSE summary (median, p90, max, mean) across all val wells.

Variants compared (only these — no tuning saga):
  aniso_gaussian       : K=20, kernel='gaussian',   range_scale=1.0
  aniso_exponential    : K=20, kernel='exponential', range_scale=1.0
  aniso_gaussian_K30   : K=30, kernel='gaussian',   range_scale=1.0
  aniso_gaussian_rs0.5 : K=20, kernel='gaussian',   range_scale=0.5

The objective: beat the konbu17-style row KNN baseline (ANCC pool 30.74,
TVT med 12.30) by >=0.3 RMSE. If yes, this becomes v10's row imputer.

Usage:
    python3 bench/aniso_score.py [--limit-wells N]
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

from aniso_kriging import AnisoFormationKNN, estimate_anisotropy_from_field  # noqa: E402
from neural_ancc import FORMATIONS, load_train_rows  # noqa: E402


# ---------------------------------------------------------------------------
# Per-well b_prefix and TVT scoring helpers (copy of neural_ancc_score logic)
# ---------------------------------------------------------------------------

def load_well_arrays(train_dir: Path, wells: list[str]) -> dict[str, dict]:
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
        finite = scoring_mask & np.isfinite(tvt) & np.isfinite(tvt_pred)
        if finite.any():
            tvt_err = tvt_pred[finite] - tvt[finite]
            tvt_rmse = float(np.sqrt(np.mean(tvt_err ** 2)))
        else:
            tvt_rmse = float("nan")
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
# Variant config table
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {
    "aniso_gaussian":       {"k": 20, "kernel": "gaussian",    "range_scale": 1.0},
    "aniso_exponential":    {"k": 20, "kernel": "exponential", "range_scale": 1.0},
    "aniso_gaussian_K30":   {"k": 30, "kernel": "gaussian",    "range_scale": 1.0},
    "aniso_gaussian_rs0.5": {"k": 20, "kernel": "gaussian",    "range_scale": 0.5},
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=str(ROOT / "data" / "competition" / "train"))
    parser.add_argument("--out-json", default=str(ROOT / "bench" / "aniso_results.json"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-wells", type=int, default=0,
                        help="If >0, restrict to first N wells (debug only).")
    parser.add_argument("--variants", default=",".join(VARIANTS.keys()))
    parser.add_argument("--batch-size", type=int, default=50_000,
                        help="Query batch size for aniso (memory ~ B*K*K*16B).")
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

    print("== loading per-well full arrays for scoring ==", flush=True)
    t0 = time.perf_counter()
    unique_wells = sorted(set(wids_all.tolist()))
    well_arrays = load_well_arrays(train_dir, unique_wells)
    print(f"   {len(well_arrays)} well files loaded, "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    print("== building full-well xy index ==", flush=True)
    t0 = time.perf_counter()
    well_full_xy: dict[str, np.ndarray] = {}
    for w, a in well_arrays.items():
        well_full_xy[w] = np.column_stack([a["X"], a["Y"]]).astype(np.float64)
    print(f"   built in {time.perf_counter()-t0:.1f}s", flush=True)

    # GroupKFold
    print("== computing GroupKFold splits ==", flush=True)
    t0 = time.perf_counter()
    gkf = GroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(gkf.split(xy_all, targets_all[:, f_idx_ancc], groups=wids_all))
    print(f"   {len(splits)} splits in {time.perf_counter()-t0:.1f}s", flush=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant {v!r}; valid={list(VARIANTS)}")

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
            "config": VARIANTS[v],
        }

    # Pre-encode all wids to ints once (faster than per-fold list/set ops on
    # an N=5M object array of strings).
    print("== integer-encoding well IDs ==", flush=True)
    t0 = time.perf_counter()
    wids_all_int = np.zeros(len(wids_all), dtype=np.int32)
    well_to_int = {w: i for i, w in enumerate(unique_wells)}
    for i, w in enumerate(wids_all):
        wids_all_int[i] = well_to_int[w]
    print(f"   {time.perf_counter()-t0:.1f}s", flush=True)

    for fold_idx, (tr, va) in enumerate(splits):
        val_well_set = np.unique(wids_all_int[va])
        print(f"\n=== fold {fold_idx} : {len(val_well_set)} val wells, "
              f"{len(va):,} val rows, {len(tr):,} train rows ===", flush=True)

        # Build aniso index ONCE per fold (anisotropy from train ANCC field)
        # then re-use across all variants. range_scale and kernel can be
        # changed per variant cheaply since we'd need to re-fit the kdtree
        # only when range_scale changes (because L scales). For simplicity
        # we re-fit per variant — fit time is small (~5s) compared to query.
        xy_train = xy_all[tr].astype(np.float64)
        ancc_train = targets_all[tr, f_idx_ancc].astype(np.float64)
        wids_train_int = wids_all_int[tr]

        # Estimate anisotropy once on train fold (shared across variants)
        t0 = time.perf_counter()
        R, sigma = estimate_anisotropy_from_field(xy_train, ancc_train)
        print(f"   aniso est R={R.flatten().round(3).tolist()} sigma={sigma.round(3).tolist()} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

        # Pre-collect val xy_q
        w_order = sorted(val_well_set)
        xy_q_list = [well_full_xy[w] for w in w_order]
        w_lengths = [len(x) for x in xy_q_list]
        xy_q = np.concatenate(xy_q_list)
        truth_q = np.concatenate([well_arrays[w]["ANCC"] for w in w_order])

        for v in variants:
            cfg = VARIANTS[v]
            print(f"  -- variant {v} (k={cfg['k']} kernel={cfg['kernel']} rs={cfg['range_scale']}) --", flush=True)
            t0 = time.perf_counter()
            knn = AnisoFormationKNN.fit(
                xy_train, ancc_train, wids_train_int, [],
                anisotropy=(R, sigma),
                range_scale=cfg["range_scale"],
            )
            ftime = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_q, _, _ = knn.query(
                xy_q, k=cfg["k"], kernel=cfg["kernel"],
                batch_size=args.batch_size,
            )
            ptime = time.perf_counter() - t0

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
            results["variants"][v]["fit_time_s"].append(ftime)
            results["variants"][v]["pred_time_s"].append(ptime)
            summary = fold_pw["summary"]
            print(f"    {v:<22s} ANCC pool rmse={rmse_pool:.3f}  "
                  f"TVT median={summary['tvt']['median']:.3f}  "
                  f"p90={summary['tvt']['p90']:.3f}  "
                  f"max={summary['tvt']['max']:.3f}  "
                  f"(fit {ftime:.1f}s, pred {ptime:.1f}s)", flush=True)

            # Free memory
            del knn

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
        pooled_ancc = float(np.mean(results["variants"][v]["fold_ancc_rmse"]))
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
            "config": results["variants"][v]["config"],
        }
        print(f"  {v:<22s}  ANCC pool {pooled_ancc:.3f}  "
              f"TVT med {tvt['median']:.3f}  p90 {tvt['p90']:.3f}  max {tvt['max']:.3f}  "
              f"(fit {overall[v]['fit_time_s_total']:.1f}s, pred {overall[v]['pred_time_s_total']:.1f}s)",
              flush=True)
    results["overall"] = overall

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"\n>> wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
