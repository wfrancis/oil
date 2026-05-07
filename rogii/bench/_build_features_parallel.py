"""Parallel feature builder for the sequence-transformer prototype.

Pre-fits global imputers, pickles them, then fans out per-well feature
construction across worker processes. Output is a single parquet matching
the v9 feature schema.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
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
)


_GLOBAL_PLANE: FormationPlaneKNN | None = None
_GLOBAL_ROW: RowKNN | None = None


def _init_worker(plane, row):
    """Forked-child initializer; shares imputers via copy-on-write.

    Patch RowKNN.impute to use a single thread for cKDTree.query — otherwise
    each worker spawns workers=-1 threads and we OOM/thrash.
    """
    global _GLOBAL_PLANE, _GLOBAL_ROW
    _GLOBAL_PLANE = plane
    _GLOBAL_ROW = row

    # Monkey-patch RowKNN.impute to use workers=1 instead of workers=-1.
    import numpy as _np
    from feature_builder import RowKNN, ROW_K_DEFAULT, ROW_NQ_DEFAULT

    def _impute_solo(self, xy_q, self_wid=None, k=ROW_K_DEFAULT, n_q=ROW_NQ_DEFAULT):
        q = xy_q / self.scale
        n_q = min(n_q, len(self.targets))
        dist, idx = self.tree.query(q, k=n_q, workers=1)  # <-- 1 not -1
        if self_wid is not None:
            mask_self = self.wids[idx] == self_wid
            dist = _np.where(mask_self, _np.inf, dist)
        order = _np.argpartition(dist, kth=min(k - 1, n_q - 1), axis=1)[:, :k]
        d_k = _np.take_along_axis(dist, order, axis=1)
        idx_k = _np.take_along_axis(idx, order, axis=1)
        valid_k = _np.isfinite(d_k)
        w = _np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0)
        sw = w.sum(axis=1)
        no_n = sw < 1e-9
        safe = _np.where(no_n, 1.0, sw)
        f_n = self.targets[idx_k]
        preds = (f_n * w[:, :, None]).sum(axis=1) / safe[:, None]
        if no_n.any():
            global_mean = self.targets.mean(axis=0)
            preds[no_n] = global_mean
        diff_sq = (f_n - preds[:, None, :]) ** 2
        var = (diff_sq * w[:, :, None]).sum(axis=1) / safe[:, None]
        stds = _np.sqrt(_np.maximum(var, 0.0))
        d_finite = _np.where(valid_k, d_k, _np.inf)
        min_dist = d_finite.min(axis=1)
        return (preds.astype(_np.float32),
                stds.astype(_np.float32),
                min_dist.astype(_np.float32))

    RowKNN.impute = _impute_solo


def _worker(p_str: str):
    plane = _GLOBAL_PLANE
    row = _GLOBAL_ROW
    p = Path(p_str)
    wid = p.stem.replace("__horizontal_well", "")
    try:
        h = pd.read_csv(p)
        t = pd.read_csv(p.parent / f"{wid}__typewell.csv")
    except Exception:
        return None
    if "TVT" not in h.columns:
        return None
    feats = build_hidden_features(
        h, t, wid,
        is_train=True,
        formation_imputer=plane,
        row_imputer=row,
        mlp_imputer=None,
        primary_formation="ANCC",
        enable_beam=False,
    )
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    train_dir = ROOT / "data" / "competition" / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    sub = [paths[i] for i in idx[: args.n_wells]]
    print(f">> {len(sub)} of {len(paths)} wells (seed={args.seed})", flush=True)

    t0 = time.perf_counter()
    plane = FormationPlaneKNN.fit(paths)
    print(f"plane fit: {time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    row = RowKNN.fit(paths)
    print(f"row KNN fit: {time.perf_counter() - t0:.1f}s", flush=True)

    work = [str(p) for p in sub]
    parts: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    print(f">> launching {args.workers} workers for {len(work)} wells", flush=True)
    ctx = mp.get_context("fork")  # share imputers via fork (no need to pickle)
    with ctx.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(plane, row),
    ) as pool:
        for i, df in enumerate(pool.imap_unordered(_worker, work)):
            if df is not None:
                parts.append(df)
            if (i + 1) % 10 == 0:
                rate = (time.perf_counter() - t0) / (i + 1)
                rem = rate * (len(work) - (i + 1))
                print(
                    f"   {i+1}/{len(work)}  {rate:.1f}s/well  ETA {rem:.0f}s",
                    flush=True,
                )

    print(f">> assembled {len(parts)} non-null parts in {time.perf_counter()-t0:.1f}s",
          flush=True)
    big = pd.concat(parts, ignore_index=True)
    print(f">> merged shape={big.shape}", flush=True)
    pl.from_pandas(big).write_parquet(args.out)
    print(f">> saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
