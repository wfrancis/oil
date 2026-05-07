"""Cheap OOF that emits CLOSED-FORM TVT-formula predictions for all
training wells, with self-well exclusion via the existing imputers.

No GBM, no MLP. Just the konbu17-style formula:
    tvt_formula_row   = -Z + KNN_ANCC + b_prefix
    tvt_formula_plane = -Z + plane_ANCC + b_prefix
    tvt_formula_mean  = mean of the two

These are (a) deterministic given the train data + self-well exclusion
mask, so no fold-level training is needed, (b) row-level predictions
suitable for stacking, (c) cheap (~20-25 min for 773 wells with no
beam features and no MLP fit/inference).

This script is the *amortized* alternative to running the full v9 OOF
again. When the heavy v9 finishes, this row-level closed-form CSV
gives the stacker an independent prediction column to work with.

Outputs:
    /tmp/formula_oof.csv   columns: prediction_id, well, row_idx,
                          target, last_known_tvt,
                          oof_pred_formula_row, oof_pred_formula_plane,
                          oof_pred_formula_mean
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from feature_builder import (
    FORMATIONS,
    FormationPlaneKNN,
    RowKNN,
    median_b,
)


def _read(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        infer_schema_length=2000,
        null_values=["", "NA", "NaN", "nan", "null"],
        truncate_ragged_lines=True,
    )


def main() -> int:
    t_start = time.perf_counter()
    train_dir = ROOT / "data" / "competition" / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    print(f">> Closed-form formula OOF over {len(paths)} train wells", flush=True)

    print(">> Building plane-fit imputer ...", flush=True)
    t0 = time.perf_counter()
    plane = FormationPlaneKNN.fit(paths)
    print(f"   plane fit: {len(plane.df)} wells, {time.perf_counter() - t0:.1f}s", flush=True)

    print(">> Building row-level KNN imputer ...", flush=True)
    t0 = time.perf_counter()
    row = RowKNN.fit(paths)
    print(f"   row KNN: {len(row.targets):,} rows, {time.perf_counter() - t0:.1f}s", flush=True)

    pred_blocks = []
    f_idx_primary = FORMATIONS.index("ANCC")
    n_done = 0

    for path in paths:
        wid = path.stem.replace("__horizontal_well", "")
        try:
            df = _read(path)
        except Exception:
            continue
        if "TVT" not in df.columns or "TVT_input" not in df.columns:
            continue
        for c in ("MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"):
            if c in df.columns:
                df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))

        x = df["X"].to_numpy().astype(np.float64)
        y = df["Y"].to_numpy().astype(np.float64)
        z = df["Z"].to_numpy().astype(np.float64)
        tvt = df["TVT"].to_numpy().astype(np.float64)
        tvt_in = df["TVT_input"].to_numpy().astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            n_done += 1
            continue

        finite_in = np.isfinite(tvt_in)
        eval_mask = ~finite_in & np.isfinite(tvt)
        eval_idx = np.flatnonzero(eval_mask)
        if eval_idx.size == 0:
            n_done += 1
            continue

        anchor_pos = np.flatnonzero(finite_in)
        if anchor_pos.size < 4:
            n_done += 1
            continue
        last_known_tvt = float(tvt_in[anchor_pos[-1]])

        xy = np.column_stack([x, y])

        # Row-level KNN
        row_preds, row_stds, row_min_dist = row.impute(xy, self_wid=wid)
        ancc_row_full = row_preds[:, f_idx_primary]
        # Plane-fit
        plane_preds, plane_min_dist = plane.impute(xy, self_wid=wid)
        ancc_plane_full = plane_preds[:, f_idx_primary]

        # b_prefix from KNN
        prefix = anchor_pos
        ancc_row_prefix = ancc_row_full[prefix]
        ancc_plane_prefix = ancc_plane_full[prefix]
        b_row = median_b(tvt_in[prefix] + z[prefix] - ancc_row_prefix)
        b_plane = median_b(tvt_in[prefix] + z[prefix] - ancc_plane_prefix)

        tvt_row = -z + ancc_row_full + b_row
        tvt_plane = -z + ancc_plane_full + b_plane

        # delta-anchored predictions (target = TVT - last_known_TVT)
        target = tvt[eval_idx] - last_known_tvt
        delta_row = tvt_row[eval_idx] - last_known_tvt
        delta_plane = tvt_plane[eval_idx] - last_known_tvt
        delta_mean = 0.5 * (delta_row + delta_plane)

        pred_blocks.append(pl.DataFrame({
            "prediction_id": [f"{wid}_{int(i)}" for i in eval_idx],
            "well": [wid] * eval_idx.size,
            "row_idx": eval_idx.astype(np.int32),
            "target": target.astype(np.float64),
            "last_known_tvt": np.full(eval_idx.size, last_known_tvt, dtype=np.float64),
            "oof_pred_formula_row": delta_row.astype(np.float64),
            "oof_pred_formula_plane": delta_plane.astype(np.float64),
            "oof_pred_formula_mean": delta_mean.astype(np.float64),
        }))

        n_done += 1
        if n_done % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = n_done / elapsed
            eta = (len(paths) - n_done) / rate / 60.0
            print(f"   well {n_done}/{len(paths)}  rate={rate:.2f} wells/s  eta={eta:.1f} min", flush=True)

    print(f">> Concatenating {len(pred_blocks)} blocks ...", flush=True)
    out = pl.concat(pred_blocks)
    print(f"   total rows: {out.height:,}", flush=True)

    # Per-predictor RMSE summary
    target = out["target"].to_numpy().astype(np.float64)
    for col in ("oof_pred_formula_row", "oof_pred_formula_plane", "oof_pred_formula_mean"):
        pred = out[col].to_numpy().astype(np.float64)
        err = pred - target
        rmse = float(np.sqrt(np.mean(err * err)))
        print(f"   {col}: RMSE={rmse:.4f}", flush=True)

    out_path = "/tmp/formula_oof.csv"
    out.write_csv(out_path)
    print(f"   saved {out_path}", flush=True)
    print(f"   total wall time: {time.perf_counter() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
