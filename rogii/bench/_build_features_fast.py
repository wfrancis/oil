"""Sequential feature builder with reduced n_q for the prototype.

The default RowKNN.impute uses n_q=12000 which is huge. For a 100-well
prototype with ~600k tree points we don't need that many candidates;
n_q=2000 gives ~indistinguishable accuracy with 6x faster queries.

This script also patches RowKNN.impute to use workers=1 (single threaded)
to avoid stomping on parallel CPU jobs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_builder import (   # noqa: E402
    FormationPlaneKNN,
    RowKNN,
    build_hidden_features,
    ROW_K_DEFAULT,
)


def patch_row_knn(n_q: int):
    """Replace RowKNN.impute with a faster, single-threaded variant."""
    def _impute(self, xy_q, self_wid=None, k=ROW_K_DEFAULT, n_q=n_q):
        q = xy_q / self.scale
        n_q_eff = min(n_q, len(self.targets))
        dist, idx = self.tree.query(q, k=n_q_eff, workers=1)
        if self_wid is not None:
            mask_self = self.wids[idx] == self_wid
            dist = np.where(mask_self, np.inf, dist)
        order = np.argpartition(dist, kth=min(k - 1, n_q_eff - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)
        w = np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0)
        sw = w.sum(axis=1)
        no_n = sw < 1e-9
        safe = np.where(no_n, 1.0, sw)
        f_n = self.targets[idx_k]
        preds = (f_n * w[:, :, None]).sum(axis=1) / safe[:, None]
        if no_n.any():
            global_mean = self.targets.mean(axis=0)
            preds[no_n] = global_mean
        diff_sq = (f_n - preds[:, None, :]) ** 2
        var = (diff_sq * w[:, :, None]).sum(axis=1) / safe[:, None]
        stds = np.sqrt(np.maximum(var, 0.0))
        d_finite = np.where(valid_k, d_k, np.inf)
        min_dist = d_finite.min(axis=1)
        return (preds.astype(np.float32),
                stds.astype(np.float32),
                min_dist.astype(np.float32))

    RowKNN.impute = _impute


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-n-q", type=int, default=2000)
    ap.add_argument("--imputer-paths", default="all",
                    help="'all' to use all 773 wells for imputers; "
                         "'subset' to use only the chosen subset.")
    args = ap.parse_args()

    train_dir = ROOT / "data" / "competition" / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    sub = [paths[i] for i in idx[: args.n_wells]]
    print(f">> {len(sub)} of {len(paths)} wells (seed={args.seed})", flush=True)

    imputer_paths = paths if args.imputer_paths == "all" else sub
    t0 = time.perf_counter()
    plane = FormationPlaneKNN.fit(imputer_paths)
    print(f"plane fit ({len(imputer_paths)} wells): "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    row = RowKNN.fit(imputer_paths)
    print(f"row KNN fit ({len(imputer_paths)} wells, "
          f"{len(row.targets):,} rows): {time.perf_counter() - t0:.1f}s",
          flush=True)

    patch_row_knn(args.row_n_q)
    print(f">> patched RowKNN.impute (n_q={args.row_n_q}, workers=1)", flush=True)

    parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    print(f">> building features for {len(sub)} wells (sequential)", flush=True)
    for i, p in enumerate(sub):
        wid = p.stem.replace("__horizontal_well", "")
        try:
            h = pd.read_csv(p)
            t = pd.read_csv(p.parent / f"{wid}__typewell.csv")
        except Exception:
            continue
        if "TVT" not in h.columns:
            continue
        feats = build_hidden_features(
            h, t, wid,
            is_train=True,
            formation_imputer=plane,
            row_imputer=row,
            mlp_imputer=None,
            primary_formation="ANCC",
            enable_beam=False,
        )
        if feats is not None:
            parts.append(feats)
        if (i + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(sub) - (i + 1))
            print(
                f"   {i+1}/{len(sub)}  {rate:.1f}s/well  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                flush=True,
            )

    print(f">> assembled {len(parts)} parts in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    big = pd.concat(parts, ignore_index=True)
    print(f">> merged shape={big.shape}", flush=True)
    pl.from_pandas(big).write_parquet(args.out)
    print(f">> saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
