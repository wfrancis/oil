"""Formation top analysis for ROGII competition.

Computes within-well anchor stability, spatial smoothness, coverage,
cross-formation correlation, and a primary-anchor recommendation
across all 773 train horizontal wells.

Pure Polars, no pandas. mmap CSV reads.
"""

from __future__ import annotations

import glob
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import polars as pl

TRAIN_DIR = "/Users/william/drilling_oil_gas/rogii/data/competition/train"
OUT_PATH = "/Users/william/drilling_oil_gas/rogii/bench/formation_stats.json"

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
FT_PER_MILE = 5280.0


def well_id(path: str) -> str:
    return os.path.basename(path).split("__", 1)[0]


def per_well_stats(path: str) -> dict[str, Any]:
    """Return per-well stats: id, centroid, F median, F coverage, F std of (TVT - (-Z + F))."""
    # mmap=True via low_memory=False + scan_csv collect; polars CSV reader uses mmap by default on local files.
    df = pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["", "NA", "nan", "NaN"],
        try_parse_dates=False,
    )
    # Some wells have mixed-type formation columns (string sentinels) — coerce to Float64.
    coerce_cols = ["MD", "X", "Y", "Z", "TVT", "GR", "TVT_input", *FORMATIONS]
    casts = []
    for c in coerce_cols:
        if c in df.columns and df.schema[c] != pl.Float64:
            casts.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
    if casts:
        df = df.with_columns(casts)
    n_rows = df.height
    out: dict[str, Any] = {
        "well": well_id(path),
        "n_rows": n_rows,
        "x_mean": float(df["X"].mean()) if "X" in df.columns else None,
        "y_mean": float(df["Y"].mean()) if "Y" in df.columns else None,
    }

    # residuals per formation, with TVT_input mask if available — but spec says TVT (full). Use TVT.
    # Compute residual r_F = TVT - (-Z + F) = TVT + Z - F per row, then drop nulls.
    # Coverage
    for F in FORMATIONS:
        if F not in df.columns:
            out[f"{F}_cov"] = 0.0
            out[f"{F}_std"] = None
            out[f"{F}_median"] = None
            out[f"{F}_n_finite"] = 0
            continue
        col = df[F]
        finite = col.is_finite() & col.is_not_null()
        n_fin = int(finite.sum())
        out[f"{F}_n_finite"] = n_fin
        out[f"{F}_cov"] = float(n_fin / n_rows) if n_rows > 0 else 0.0
        if n_fin >= 5:
            sub = df.filter(finite).select(["TVT", "Z", F])
            r = sub["TVT"] + sub["Z"] - sub[F]  # TVT - (-Z + F)
            out[f"{F}_std"] = float(r.std()) if r.std() is not None else None
            out[f"{F}_median"] = float(sub[F].median()) if sub[F].median() is not None else None
        else:
            out[f"{F}_std"] = None
            out[f"{F}_median"] = None

    # Cross-formation residual correlation matrix per well: correlation of (TVT - (-Z + F_i)) with (TVT - (-Z + F_j))
    # Build a residual frame and compute pairwise corr.
    res_cols = {}
    for F in FORMATIONS:
        if F in df.columns:
            r = (df["TVT"] + df["Z"] - df[F]).rename(f"r_{F}")
            res_cols[F] = r
    if res_cols:
        rdf = pl.DataFrame({k: v for k, v in res_cols.items()})
        # drop rows where any are null/non-finite
        mask = pl.lit(True)
        for k in res_cols.keys():
            mask = mask & rdf[k].is_finite() & rdf[k].is_not_null()
        rdf = rdf.filter(mask) if rdf.height > 0 else rdf
        if rdf.height >= 10:
            arr = rdf.to_numpy()
            with np.errstate(invalid="ignore"):
                cm = np.corrcoef(arr, rowvar=False)
            keys = list(res_cols.keys())
            corr = {}
            for i, ki in enumerate(keys):
                for j, kj in enumerate(keys):
                    if j > i:
                        v = cm[i, j]
                        if np.isfinite(v):
                            corr[f"{ki}__{kj}"] = float(v)
            out["pair_corr"] = corr
        else:
            out["pair_corr"] = {}
    else:
        out["pair_corr"] = {}
    return out


def main() -> None:
    paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*__horizontal_well.csv")))
    print(f"found {len(paths)} horizontal well CSVs")

    results: list[dict[str, Any]] = []
    # 10-core M1 Pro -> 8 workers leaves headroom
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(per_well_stats, p): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"FAIL {futs[fut]}: {e}")
            done += 1
            if done % 100 == 0:
                print(f"  processed {done}/{len(paths)}")

    # === Q1: within-well std of TVT - (-Z + F) ===
    table_q1: dict[str, dict[str, float]] = {}
    for F in FORMATIONS:
        stds = np.array(
            [r[f"{F}_std"] for r in results if r.get(f"{F}_std") is not None],
            dtype=float,
        )
        if stds.size == 0:
            table_q1[F] = {"n": 0}
            continue
        table_q1[F] = {
            "n_wells": int(stds.size),
            "median_std_ft": float(np.median(stds)),
            "p95_std_ft": float(np.percentile(stds, 95)),
            "p99_std_ft": float(np.percentile(stds, 99)),
            "max_std_ft": float(np.max(stds)),
            "mean_std_ft": float(np.mean(stds)),
        }

    # === Q2: spatial smoothness (variogram-like) ===
    # Pairs binned by 2D distance in miles
    bins_mi = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    # Build per-well centroid + median F
    centroids = []
    for r in results:
        if r["x_mean"] is None or r["y_mean"] is None:
            continue
        centroids.append(r)

    # Vectorize pairwise distances using numpy
    xs = np.array([r["x_mean"] for r in centroids], dtype=float)
    ys = np.array([r["y_mean"] for r in centroids], dtype=float)
    n = xs.size
    print(f"variogram on {n} wells -> {n*(n-1)//2:,} pairs")
    # pairwise dist in feet, convert to miles
    # Use blocked pairwise to avoid OOM (n ~773 -> 0.6 MB matrix, fine)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_mi = np.sqrt(dx * dx + dy * dy) / FT_PER_MILE
    iu = np.triu_indices(n, k=1)
    d_pairs = dist_mi[iu]

    table_q2: dict[str, dict[str, dict[str, float]]] = {}
    for F in FORMATIONS:
        med = np.array(
            [r.get(f"{F}_median", np.nan) for r in centroids],
            dtype=float,
        )
        valid_i = np.isfinite(med)
        med_pairs = np.abs(med[iu[0]] - med[iu[1]])
        valid_pairs = valid_i[iu[0]] & valid_i[iu[1]]
        bin_stats: dict[str, dict[str, float]] = {}
        for lo, hi in bins_mi:
            sel = (d_pairs >= lo) & (d_pairs < hi) & valid_pairs
            vals = med_pairs[sel]
            label = f"{lo:g}-{hi:g}mi"
            if vals.size > 0:
                bin_stats[label] = {
                    "n_pairs": int(vals.size),
                    "median_abs_dF_ft": float(np.median(vals)),
                    "mean_abs_dF_ft": float(np.mean(vals)),
                    "p95_abs_dF_ft": float(np.percentile(vals, 95)),
                }
            else:
                bin_stats[label] = {"n_pairs": 0}
        table_q2[F] = bin_stats

    # === Q3: coverage ===
    table_q3: dict[str, Any] = {}
    n_wells = len(results)
    n_total_rows = int(sum(r["n_rows"] for r in results))
    for F in FORMATIONS:
        n_fin_rows = int(sum(r.get(f"{F}_n_finite", 0) for r in results))
        n_wells_with_F = int(sum(1 for r in results if r.get(f"{F}_n_finite", 0) > 0))
        n_wells_full = int(
            sum(1 for r in results if r["n_rows"] > 0 and r.get(f"{F}_n_finite", 0) == r["n_rows"])
        )
        table_q3[F] = {
            "frac_rows_finite": float(n_fin_rows / n_total_rows) if n_total_rows else 0.0,
            "n_rows_finite": n_fin_rows,
            "n_wells_with_any_F": n_wells_with_F,
            "n_wells_with_F_complete": n_wells_full,
        }
    table_q3["_meta"] = {
        "n_wells_total": n_wells,
        "n_total_rows": n_total_rows,
    }

    # missing pattern: wells with at least one formation entirely absent
    missing_patterns: dict[str, int] = {}
    for r in results:
        miss = tuple(F for F in FORMATIONS if r.get(f"{F}_n_finite", 0) == 0)
        key = ",".join(miss) if miss else "<none missing>"
        missing_patterns[key] = missing_patterns.get(key, 0) + 1
    table_q3["missing_patterns"] = missing_patterns

    # === Q4: cross-formation residual correlations ===
    pair_lists: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.get("pair_corr", {}).items():
            if math.isfinite(v):
                pair_lists.setdefault(k, []).append(v)
    table_q4: dict[str, dict[str, float]] = {}
    for k, lst in pair_lists.items():
        a = np.array(lst, dtype=float)
        table_q4[k] = {
            "n_wells": int(a.size),
            "median_corr": float(np.median(a)),
            "mean_corr": float(np.mean(a)),
            "p05_corr": float(np.percentile(a, 5)),
            "p95_corr": float(np.percentile(a, 95)),
        }

    # === ensemble RMSE simulation: assume per-well bias is removed (well-specific b_F),
    # so residual = TVT - (-Z + F + b_F_well). Within-well std == intra-well std of (TVT + Z - F).
    # If we ensemble K formations with average correlation rho and equal weighting,
    # the variance of the ensemble is (1/K + (K-1)/K * rho_avg) * sigma^2 where sigma^2 is the avg per-formation variance.
    # We take per-well sigma_F and rho_F,F' and compute predicted ensemble RMSE (population over wells).
    sigma_per_F: dict[str, float] = {}
    for F in FORMATIONS:
        s = table_q1[F].get("median_std_ft")
        if s is not None:
            sigma_per_F[F] = float(s)

    # Average residual correlation across wells (already in table_q4 median_corr)
    avg_rho: dict[tuple[str, str], float] = {}
    for k, v in table_q4.items():
        a, b = k.split("__")
        avg_rho[(a, b)] = v["median_corr"]
        avg_rho[(b, a)] = v["median_corr"]
    for F in FORMATIONS:
        avg_rho[(F, F)] = 1.0

    def ensemble_var(formations: list[str]) -> float:
        K = len(formations)
        if K == 0:
            return float("inf")
        # predicted std: weights all 1/K. Var(sum w_i x_i) = sum w_i^2 sigma_i^2 + sum_{i!=j} w_i w_j rho_ij sigma_i sigma_j
        s = 0.0
        for i, Fi in enumerate(formations):
            for j, Fj in enumerate(formations):
                rho = avg_rho.get((Fi, Fj), 0.0) if i != j else 1.0
                s += (1.0 / K) * (1.0 / K) * rho * sigma_per_F[Fi] * sigma_per_F[Fj]
        return s

    ranked_singles = sorted(
        ((F, sigma_per_F[F]) for F in sigma_per_F), key=lambda t: t[1]
    )
    # try the best K=2,3,4 ensembles by exhaustive search over the formations we have
    from itertools import combinations

    have = list(sigma_per_F.keys())
    ensemble_results = []
    for K in (1, 2, 3, 4, 5, 6):
        best = None
        for combo in combinations(have, K):
            v = ensemble_var(list(combo))
            if best is None or v < best[1]:
                best = (combo, v)
        if best:
            ensemble_results.append({
                "K": K,
                "formations": list(best[0]),
                "predicted_std_ft": float(math.sqrt(best[1])),
            })

    table_q5 = {
        "ranked_singles_by_median_std_ft": [{"F": F, "median_std_ft": s} for F, s in ranked_singles],
        "best_ensemble_per_K": ensemble_results,
    }

    out = {
        "q1_within_well_std": table_q1,
        "q2_spatial_smoothness": table_q2,
        "q3_coverage": table_q3,
        "q4_cross_formation_corr": table_q4,
        "q5_ensemble_recommendation": table_q5,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {OUT_PATH}")

    # Print a summary
    print("\n=== Q1 within-well std of TVT - (-Z + F) ===")
    for F in FORMATIONS:
        s = table_q1[F]
        if "median_std_ft" not in s:
            print(f"  {F}: no data")
            continue
        print(
            f"  {F}: median={s['median_std_ft']:.4f}  p95={s['p95_std_ft']:.4f}  max={s['max_std_ft']:.4f}  n_wells={s['n_wells']}"
        )

    print("\n=== Q2 spatial smoothness (median |dF| in ft, by distance bin) ===")
    print(f"  {'F':>6} | " + " | ".join(f"{lo:g}-{hi:g}mi" for lo, hi in bins_mi))
    for F in FORMATIONS:
        b = table_q2[F]
        row = [F]
        for lo, hi in bins_mi:
            label = f"{lo:g}-{hi:g}mi"
            v = b.get(label, {})
            if "median_abs_dF_ft" in v:
                row.append(f"{v['median_abs_dF_ft']:.1f}({v['n_pairs']})")
            else:
                row.append("-")
        print("  " + " | ".join(f"{x:>10}" for x in row))

    print("\n=== Q3 coverage ===")
    for F in FORMATIONS:
        c = table_q3[F]
        print(
            f"  {F}: rows={c['frac_rows_finite']*100:.2f}%  wells_with_any={c['n_wells_with_any_F']}/{n_wells}  wells_complete={c['n_wells_with_F_complete']}"
        )
    print("  missing patterns (top 6):")
    for k, v in sorted(table_q3["missing_patterns"].items(), key=lambda t: -t[1])[:6]:
        print(f"    [{k}]: {v} wells")

    print("\n=== Q4 median per-well correlation of residuals ===")
    for k in sorted(table_q4.keys()):
        v = table_q4[k]
        print(f"  {k}: median_rho={v['median_corr']:.3f}  p05={v['p05_corr']:.3f}  p95={v['p95_corr']:.3f}  n={v['n_wells']}")

    print("\n=== Q5 ensemble recommendation ===")
    print("  best ensemble per K (predicted within-well std after subtracting per-well bias):")
    for r in ensemble_results:
        print(f"    K={r['K']}: {r['formations']}  predicted_std={r['predicted_std_ft']:.4f} ft")


if __name__ == "__main__":
    main()
