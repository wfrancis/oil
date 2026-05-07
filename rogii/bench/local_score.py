"""Local validation scorer for the ROGII competition workspace.

The scorer uses train wells as a public validation set:

* ``TVT`` is held back as ground truth.
* Predictions are scored only on rows where ``TVT_input`` is missing.
* Predictor inputs are sanitized to look like test data by default:
  horizontal ``TVT`` and formation columns are dropped, and typewell
  ``Geology`` is dropped unless explicitly kept.

This gives us a fast loop for Python experiments and a simple bridge for Rust:
export a validation folder, run the Rust predictor to produce a Kaggle-style
``id,tvt`` CSV, then score it with ``score-csv``.

The ``run-residual`` command is the v5 submit gate: it trains the residual
LightGBM only on non-validation wells, then scores the held-out fold with
test-like horizontal columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_TRAIN_DIR = ROOT / "data" / "competition" / "train"
NULL_VALUES = ["", "NA", "NaN", "nan", "null"]
FORMATION_COLS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
TEST_LIKE_HORIZONTAL_COLS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
RESIDUAL_FEATURE_COLS = (
    "md_delta",
    "row_delta",
    "frac",
    "x_delta",
    "y_delta",
    "z_delta",
    "gr",
    "gr_delta",
    "gr_z",
    "last_tvt",
    "anchor_md",
    "anchor_x",
    "anchor_y",
    "anchor_z",
    "anchor_gr",
    "cased_slope",
    "tvt_std",
    "tvt_range",
    "n_minus_anchor",
    "current_md",
    "current_x",
    "current_y",
    "current_z",
)

logger = logging.getLogger("rogii.local_score")


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        infer_schema_length=2000,
        null_values=NULL_VALUES,
        truncate_ragged_lines=True,
    )


def _well_names(train_dir: Path) -> list[str]:
    horiz = {
        p.name.replace("__horizontal_well.csv", "")
        for p in train_dir.glob("*__horizontal_well.csv")
    }
    typew = {
        p.name.replace("__typewell.csv", "")
        for p in train_dir.glob("*__typewell.csv")
    }
    wells = sorted(horiz & typew)
    if not wells:
        raise FileNotFoundError(f"No paired train wells found under {train_dir}")
    return wells


def _stable_score(well: str, seed: int) -> int:
    payload = f"{seed}:{well}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _select_wells(args: argparse.Namespace) -> list[str]:
    train_dir = Path(args.train_dir)
    wells = _well_names(train_dir)

    if args.wells:
        wanted = [w.strip() for w in args.wells.split(",") if w.strip()]
        known = set(wells)
        missing = sorted(set(wanted) - known)
        if missing:
            raise ValueError(f"Requested wells not found in {train_dir}: {missing[:10]}")
        wells = wanted
    else:
        wells = sorted(wells, key=lambda w: _stable_score(w, int(args.seed)))
        if int(args.n_folds) > 1:
            fold = int(args.fold)
            n_folds = int(args.n_folds)
            if not 0 <= fold < n_folds:
                raise ValueError(f"--fold must be in [0, {n_folds - 1}], got {fold}")
            wells = [w for i, w in enumerate(wells) if i % n_folds == fold]

    if args.limit:
        wells = wells[: int(args.limit)]
    if not wells:
        raise ValueError("Well selection is empty.")
    return wells


def _sanitize_horizontal(df: pl.DataFrame, *, keep_extra_cols: bool) -> pl.DataFrame:
    if keep_extra_cols:
        drops = [c for c in ("TVT", *FORMATION_COLS) if c in df.columns]
        return df.drop(drops) if drops else df
    cols = [c for c in TEST_LIKE_HORIZONTAL_COLS if c in df.columns]
    return df.select(cols)


def _sanitize_typewell(df: pl.DataFrame, *, keep_geology: bool) -> pl.DataFrame:
    if keep_geology or "Geology" not in df.columns:
        return df
    return df.drop("Geology")


def _truth_for_well(well: str, h_raw: pl.DataFrame) -> pd.DataFrame:
    needed = {"TVT", "TVT_input"}
    missing = needed - set(h_raw.columns)
    if missing:
        raise ValueError(f"{well}: horizontal file missing truth columns {sorted(missing)}")

    truth = h_raw.get_column("TVT").to_numpy().astype(np.float64, copy=False)
    tvt_in = h_raw.get_column("TVT_input").to_numpy().astype(np.float64, copy=False)
    n = min(truth.size, tvt_in.size)
    truth = truth[:n]
    tvt_in = tvt_in[:n]
    mask = ~np.isfinite(tvt_in) & np.isfinite(truth)
    idx = np.flatnonzero(mask)
    return pd.DataFrame(
        {
            "id": [f"{well}_{int(i)}" for i in idx],
            "well": well,
            "row": idx.astype(np.int64),
            "tvt": truth[idx].astype(np.float64),
        }
    )


def _truth_for_wells(train_dir: Path, wells: list[str]) -> pd.DataFrame:
    frames = []
    for well in wells:
        h_raw = _read_csv(train_dir / f"{well}__horizontal_well.csv")
        frames.append(_truth_for_well(well, h_raw))
    return pd.concat(frames, ignore_index=True) if frames else _empty_truth()


def _empty_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": pd.Series(dtype=str),
            "well": pd.Series(dtype=str),
            "row": pd.Series(dtype=np.int64),
            "tvt": pd.Series(dtype=np.float64),
        }
    )


def _metric_dict(scored: pd.DataFrame, *, elapsed_s: float | None = None) -> dict[str, Any]:
    if scored.empty:
        return {
            "rows": 0,
            "wells": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "median_ae": float("nan"),
            "p90_ae": float("nan"),
            "mean_well_rmse": float("nan"),
            "elapsed_s": elapsed_s,
        }
    err = scored["pred_tvt"].to_numpy(dtype=np.float64) - scored["true_tvt"].to_numpy(
        dtype=np.float64
    )
    abs_err = np.abs(err)
    per_well_rmse = (
        scored.assign(sq_err=err * err)
        .groupby("well", sort=True)["sq_err"]
        .mean()
        .pow(0.5)
    )
    out = {
        "rows": int(len(scored)),
        "wells": int(scored["well"].nunique()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(abs_err)),
        "bias": float(np.mean(err)),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "mean_well_rmse": float(per_well_rmse.mean()),
        "elapsed_s": elapsed_s,
    }
    if elapsed_s and elapsed_s > 0:
        out["rows_per_s"] = float(len(scored) / elapsed_s)
    return out


def _train_median_tvt(train_dir: Path) -> float:
    """Median finite TVT over train horizontals, used for no-anchor fallback."""
    vals: list[np.ndarray] = []
    for path in train_dir.glob("*__horizontal_well.csv"):
        try:
            df = _read_csv(path)
        except Exception:
            continue
        if "TVT" not in df.columns:
            continue
        arr = df.get_column("TVT").to_numpy().astype(np.float64, copy=False)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return 0.0
    return float(np.median(np.concatenate(vals)))


def _predict_constant(horizontal_df: pl.DataFrame, fallback_tvt: float) -> np.ndarray:
    """Last-known TVT_input baseline, with median-TVt fallback if no anchor exists."""
    n = horizontal_df.height
    if "TVT_input" not in horizontal_df.columns:
        return np.full(n, float(fallback_tvt), dtype=np.float64)
    tvt_in = horizontal_df.get_column("TVT_input").to_numpy().astype(np.float64, copy=False)
    finite = np.isfinite(tvt_in)
    out = np.empty(n, dtype=np.float64)
    if finite.any():
        out[:] = float(tvt_in[np.flatnonzero(finite)[-1]])
        out[finite] = tvt_in[finite]
    else:
        out[:] = float(fallback_tvt)
    return out


def _predict_slope(
    horizontal_df: pl.DataFrame,
    fallback_tvt: float,
    *,
    cap: float,
    n_tail: int,
    min_points: int,
    ci_threshold: float,
    uncertain_shrink: float,
) -> np.ndarray:
    """Small Theil-Sen slope correction layered on top of last-known TVT_input."""
    n = horizontal_df.height
    if "MD" not in horizontal_df.columns or "TVT_input" not in horizontal_df.columns:
        return _predict_constant(horizontal_df, fallback_tvt)

    md = horizontal_df.get_column("MD").to_numpy().astype(np.float64, copy=False)
    tvt_in = horizontal_df.get_column("TVT_input").to_numpy().astype(np.float64, copy=False)
    finite = np.isfinite(tvt_in) & np.isfinite(md)
    if not finite.any():
        return np.full(n, float(fallback_tvt), dtype=np.float64)

    idx = np.flatnonzero(finite)
    last_i = int(idx[-1])
    last_tvt = float(tvt_in[last_i])
    last_md = float(md[last_i])
    slope = 0.0

    use = idx[-int(n_tail):]
    if use.size >= int(min_points):
        try:
            from scipy.stats import theilslopes

            s, _intercept, lo, hi = theilslopes(tvt_in[use], md[use], alpha=0.95)
            s = float(np.clip(s, -float(cap), float(cap)))
            ci_half = 0.5 * float(hi - lo)
            if not np.isfinite(ci_half) or ci_half > float(ci_threshold):
                s *= float(uncertain_shrink)
            slope = s
        except Exception:
            slope = 0.0

    out = last_tvt + slope * (md - last_md)
    out = out.astype(np.float64, copy=False)
    out[finite] = tvt_in[finite]
    return out


def _finite_array_pd(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return df[col].to_numpy(dtype=np.float64, copy=False)


def _build_residual_features(
    df: pd.DataFrame,
    fallback_tvt: float,
) -> dict[str, Any] | None:
    """Build the same test-available residual features used by v5."""
    md = _finite_array_pd(df, "MD")
    x = _finite_array_pd(df, "X")
    y = _finite_array_pd(df, "Y")
    z = _finite_array_pd(df, "Z")
    gr = _finite_array_pd(df, "GR")
    tvt_in = _finite_array_pd(df, "TVT_input")

    finite = np.isfinite(tvt_in)
    eval_idx = np.flatnonzero(~finite)
    n = len(df)
    if eval_idx.size == 0:
        return None
    if not finite.any():
        return {
            "no_anchor": True,
            "eval_idx": eval_idx,
            "last_tvt": float(fallback_tvt),
            "X": None,
        }

    anchor_i = int(np.flatnonzero(finite)[-1])
    last_tvt = float(tvt_in[anchor_i])
    anchor_md = float(md[anchor_i])
    anchor_x = float(x[anchor_i])
    anchor_y = float(y[anchor_i])
    anchor_z = float(z[anchor_i])
    anchor_gr = float(gr[anchor_i])

    finite_idx = np.flatnonzero(finite)
    tail = finite_idx[-min(300, finite_idx.size):]
    if tail.size >= 2:
        mm = md[tail]
        tt = tvt_in[tail]
        dm = mm - np.nanmean(mm)
        den = float(np.nansum(dm * dm))
        slope = (
            float(np.nansum((tt - np.nanmean(tt)) * dm) / den)
            if den > 1e-9
            else 0.0
        )
        slope = float(np.clip(slope, -0.005, 0.005))
        tvt_std = float(np.nanstd(tt))
        tvt_range = float(np.nanmax(tt) - np.nanmin(tt))
    else:
        slope = 0.0
        tvt_std = 0.0
        tvt_range = 0.0

    if np.isfinite(gr).any():
        gr_mean = float(np.nanmean(gr))
        gr_std = max(float(np.nanstd(gr)), 1.0)
    else:
        gr_mean = 0.0
        gr_std = 1.0

    row_delta = eval_idx.astype(np.float64) - float(anchor_i)
    eval_len = max(int(eval_idx[-1]) - anchor_i, 1)
    current_md = md[eval_idx]
    current_x = x[eval_idx]
    current_y = y[eval_idx]
    current_z = z[eval_idx]
    current_gr = gr[eval_idx]

    X = np.column_stack(
        [
            current_md - anchor_md,
            row_delta,
            row_delta / float(eval_len),
            current_x - anchor_x,
            current_y - anchor_y,
            current_z - anchor_z,
            current_gr,
            current_gr - anchor_gr,
            (current_gr - gr_mean) / gr_std,
            np.full(eval_idx.size, last_tvt, dtype=np.float64),
            np.full(eval_idx.size, anchor_md, dtype=np.float64),
            np.full(eval_idx.size, anchor_x, dtype=np.float64),
            np.full(eval_idx.size, anchor_y, dtype=np.float64),
            np.full(eval_idx.size, anchor_z, dtype=np.float64),
            np.full(eval_idx.size, anchor_gr, dtype=np.float64),
            np.full(eval_idx.size, slope, dtype=np.float64),
            np.full(eval_idx.size, tvt_std, dtype=np.float64),
            np.full(eval_idx.size, tvt_range, dtype=np.float64),
            np.full(eval_idx.size, n - anchor_i, dtype=np.float64),
            current_md,
            current_x,
            current_y,
            current_z,
        ]
    ).astype(np.float32, copy=False)
    return {
        "no_anchor": False,
        "eval_idx": eval_idx,
        "last_tvt": last_tvt,
        "X": X,
    }


def _residual_train_wells(train_dir: Path, validation_wells: list[str]) -> list[str]:
    validation = set(validation_wells)
    return [well for well in _well_names(train_dir) if well not in validation]


def _read_horizontal_pd(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _train_residual_model(
    train_dir: Path,
    train_wells: list[str],
    *,
    fallback_tvt: float,
    max_train_rows: int,
    seed: int,
    threads: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        raise RuntimeError(f"LightGBM unavailable: {exc}") from exc

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    used_wells = 0
    for well in train_wells:
        path = train_dir / f"{well}__horizontal_well.csv"
        try:
            df = _read_horizontal_pd(path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        if "TVT" not in df.columns:
            continue
        built = _build_residual_features(df, fallback_tvt)
        if built is None or built["no_anchor"] or built["X"] is None:
            continue
        tvt = _finite_array_pd(df, "TVT")
        target = tvt[built["eval_idx"]] - float(built["last_tvt"])
        good = np.isfinite(target)
        if not good.any():
            continue
        X_blocks.append(built["X"][good])
        y_blocks.append(target[good].astype(np.float32, copy=False))
        used_wells += 1

    if not X_blocks:
        raise RuntimeError("No residual training rows were built.")

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    logger.info("Residual training matrix: X=%s y=%s wells=%d", X.shape, y.shape, used_wells)
    if X.shape[0] > max_train_rows > 0:
        rng = np.random.default_rng(seed)
        take = rng.choice(X.shape[0], int(max_train_rows), replace=False)
        X = X[take]
        y = y[take]
        logger.info("Sampled residual training rows to %d.", X.shape[0])

    model = LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=threads,
        verbose=-1,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
    )
    model.fit(X, y)
    return model


def _print_report(metrics: dict[str, Any], per_well: pd.DataFrame | None, *, as_json: bool) -> None:
    if as_json:
        payload = dict(metrics)
        if per_well is not None:
            payload["per_well"] = per_well.to_dict(orient="records")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "rows={rows} wells={wells} rmse={rmse:.4f} mae={mae:.4f} "
        "bias={bias:.4f} median_ae={median_ae:.4f} p90_ae={p90_ae:.4f} "
        "mean_well_rmse={mean_well_rmse:.4f}".format(**metrics)
    )
    elapsed_s = metrics.get("elapsed_s")
    if elapsed_s is not None:
        rate = metrics.get("rows_per_s")
        if rate is not None:
            print(f"elapsed_s={elapsed_s:.3f} rows_per_s={rate:.1f}")
        else:
            print(f"elapsed_s={elapsed_s:.3f}")
    if per_well is not None and not per_well.empty:
        print("\nWorst wells:")
        print(per_well.sort_values("rmse", ascending=False).head(12).to_string(index=False))


def _per_well_table(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["well", "rows", "rmse", "mae", "bias"])
    grouped = scored.assign(
        err=scored["pred_tvt"] - scored["true_tvt"],
        abs_err=lambda x: x["err"].abs(),
        sq_err=lambda x: x["err"] * x["err"],
    ).groupby("well", sort=True)
    return grouped.agg(
        rows=("id", "size"),
        rmse=("sq_err", lambda s: float(np.sqrt(np.mean(s)))),
        mae=("abs_err", "mean"),
        bias=("err", "mean"),
    ).reset_index()


def cmd_export(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    wells = _select_wells(args)
    export_root = Path(args.export_dir)
    test_dir = export_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    truth_frames = []
    sample_rows = []
    for well in wells:
        h_raw = _read_csv(train_dir / f"{well}__horizontal_well.csv")
        t_raw = _read_csv(train_dir / f"{well}__typewell.csv")
        truth = _truth_for_well(well, h_raw)
        truth_frames.append(truth)
        sample_rows.append(truth[["id"]].assign(tvt=0.0))

        h_out = _sanitize_horizontal(h_raw, keep_extra_cols=bool(args.keep_horizontal_extra_cols))
        t_out = _sanitize_typewell(t_raw, keep_geology=bool(args.keep_typewell_geology))
        h_out.write_csv(test_dir / f"{well}__horizontal_well.csv")
        t_out.write_csv(test_dir / f"{well}__typewell.csv")

    truth_df = pd.concat(truth_frames, ignore_index=True) if truth_frames else _empty_truth()
    truth_df.to_csv(export_root / "truth.csv", index=False)
    sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame({"id": [], "tvt": []})
    sample_df.to_csv(export_root / "sample_submission.csv", index=False)
    (export_root / "selected_wells.txt").write_text("\n".join(wells) + "\n")

    manifest = {
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "truth_csv": str(export_root / "truth.csv"),
        "sample_submission_csv": str(export_root / "sample_submission.csv"),
        "selected_wells": wells,
        "rows": int(len(truth_df)),
        "wells": int(len(wells)),
        "typewell_geology_kept": bool(args.keep_typewell_geology),
        "horizontal_extra_cols_kept": bool(args.keep_horizontal_extra_cols),
    }
    (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


def cmd_run_python(args: argparse.Namespace) -> None:
    from inference import predict_well

    train_dir = Path(args.train_dir)
    wells = _select_wells(args)
    scored_frames = []
    pred_rows = []
    per_well_runtime = []
    start = time.perf_counter()

    for well in wells:
        h_raw = _read_csv(train_dir / f"{well}__horizontal_well.csv")
        t_raw = _read_csv(train_dir / f"{well}__typewell.csv")
        truth = _truth_for_well(well, h_raw)
        if truth.empty:
            continue

        h_pred = _sanitize_horizontal(
            h_raw, keep_extra_cols=bool(args.keep_horizontal_extra_cols)
        )
        t_pred = _sanitize_typewell(t_raw, keep_geology=bool(args.keep_typewell_geology))

        t0 = time.perf_counter()
        pred_all = np.asarray(predict_well(h_pred, t_pred, smoother=args.smoother), dtype=np.float64)
        elapsed = time.perf_counter() - t0

        row_idx = truth["row"].to_numpy(dtype=np.int64)
        if pred_all.size <= int(row_idx.max(initial=-1)):
            raise RuntimeError(
                f"{well}: predictor returned {pred_all.size} rows, "
                f"but validation needs row {int(row_idx.max())}"
            )
        pred_eval = pred_all[row_idx]
        scored = truth.rename(columns={"tvt": "true_tvt"}).copy()
        scored["pred_tvt"] = pred_eval
        scored_frames.append(scored)
        pred_rows.append(pd.DataFrame({"id": truth["id"], "tvt": pred_eval}))
        per_well_runtime.append(
            {"well": well, "elapsed_s": elapsed, "runtime_rows": int(len(truth))}
        )

    elapsed_s = time.perf_counter() - start
    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    metrics = _metric_dict(scored_all, elapsed_s=elapsed_s)
    per_well = _per_well_table(scored_all)
    runtime_df = pd.DataFrame(per_well_runtime)
    if not runtime_df.empty and not per_well.empty:
        per_well = per_well.merge(runtime_df, on="well", how="left")

    if args.output_predictions:
        pred_df = (
            pd.concat(pred_rows, ignore_index=True)
            if pred_rows
            else pd.DataFrame({"id": [], "tvt": []})
        )
        Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.output_predictions, index=False)

    _print_report(metrics, per_well, as_json=bool(args.json))


def cmd_run_baseline(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    wells = _select_wells(args)
    fallback_tvt = float(args.fallback_tvt) if args.fallback_tvt is not None else _train_median_tvt(train_dir)
    scored_frames = []
    pred_rows = []
    per_well_runtime = []
    start = time.perf_counter()

    for well in wells:
        h_raw = _read_csv(train_dir / f"{well}__horizontal_well.csv")
        truth = _truth_for_well(well, h_raw)
        if truth.empty:
            continue

        h_pred = _sanitize_horizontal(
            h_raw, keep_extra_cols=bool(args.keep_horizontal_extra_cols)
        )

        t0 = time.perf_counter()
        if args.strategy == "constant":
            pred_all = _predict_constant(h_pred, fallback_tvt)
        elif args.strategy == "slope":
            pred_all = _predict_slope(
                h_pred,
                fallback_tvt,
                cap=float(args.slope_cap),
                n_tail=int(args.slope_tail),
                min_points=int(args.slope_min_points),
                ci_threshold=float(args.slope_ci_threshold),
                uncertain_shrink=float(args.slope_uncertain_shrink),
            )
        else:
            raise ValueError(f"Unknown baseline strategy {args.strategy!r}")
        elapsed = time.perf_counter() - t0

        row_idx = truth["row"].to_numpy(dtype=np.int64)
        pred_eval = pred_all[row_idx]
        scored = truth.rename(columns={"tvt": "true_tvt"}).copy()
        scored["pred_tvt"] = pred_eval
        scored_frames.append(scored)
        pred_rows.append(pd.DataFrame({"id": truth["id"], "tvt": pred_eval}))
        per_well_runtime.append(
            {"well": well, "elapsed_s": elapsed, "runtime_rows": int(len(truth))}
        )

    elapsed_s = time.perf_counter() - start
    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    metrics = _metric_dict(scored_all, elapsed_s=elapsed_s)
    metrics["strategy"] = args.strategy
    metrics["fallback_tvt"] = fallback_tvt
    if args.strategy == "slope":
        metrics["slope_cap"] = float(args.slope_cap)
        metrics["slope_tail"] = int(args.slope_tail)

    per_well = _per_well_table(scored_all)
    runtime_df = pd.DataFrame(per_well_runtime)
    if not runtime_df.empty and not per_well.empty:
        per_well = per_well.merge(runtime_df, on="well", how="left")

    if args.output_predictions:
        pred_df = (
            pd.concat(pred_rows, ignore_index=True)
            if pred_rows
            else pd.DataFrame({"id": [], "tvt": []})
        )
        Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.output_predictions, index=False)

    _print_report(metrics, per_well, as_json=bool(args.json))


def cmd_run_residual(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    validation_wells = _select_wells(args)
    train_wells = _residual_train_wells(train_dir, validation_wells)
    if int(args.train_limit) > 0:
        train_wells = train_wells[: int(args.train_limit)]
    fallback_tvt = (
        float(args.fallback_tvt)
        if args.fallback_tvt is not None
        else _train_median_tvt(train_dir)
    )

    start = time.perf_counter()
    train_start = time.perf_counter()
    model = _train_residual_model(
        train_dir,
        train_wells,
        fallback_tvt=fallback_tvt,
        max_train_rows=int(args.train_rows),
        seed=int(args.model_seed),
        threads=int(args.threads),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
    )
    train_elapsed_s = time.perf_counter() - train_start

    scored_frames = []
    pred_rows = []
    per_well_runtime = []
    for well in validation_wells:
        path = train_dir / f"{well}__horizontal_well.csv"
        h_raw = _read_horizontal_pd(path)
        h_pred = h_raw[[c for c in TEST_LIKE_HORIZONTAL_COLS if c in h_raw.columns]].copy()
        built = _build_residual_features(h_pred, fallback_tvt)
        if built is None:
            continue

        eval_idx = built["eval_idx"]
        last_tvt = float(built["last_tvt"])
        t0 = time.perf_counter()
        if built["X"] is None:
            pred_eval = np.full(eval_idx.size, last_tvt, dtype=np.float64)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names",
                    category=UserWarning,
                )
                residual = model.predict(built["X"])
            pred_eval = last_tvt + float(args.shrink) * np.asarray(residual, dtype=np.float64)
        predict_elapsed_s = time.perf_counter() - t0

        bad = ~np.isfinite(pred_eval)
        if bad.any():
            pred_eval = pred_eval.copy()
            pred_eval[bad] = last_tvt

        tvt = _finite_array_pd(h_raw, "TVT")
        good = np.isfinite(tvt[eval_idx])
        if not good.any():
            continue
        row_idx = eval_idx[good].astype(np.int64, copy=False)
        pred_good = pred_eval[good]
        scored = pd.DataFrame(
            {
                "id": [f"{well}_{int(i)}" for i in row_idx],
                "well": well,
                "row": row_idx,
                "true_tvt": tvt[row_idx],
                "pred_tvt": pred_good,
            }
        )
        scored_frames.append(scored)
        pred_rows.append(pd.DataFrame({"id": scored["id"], "tvt": pred_good}))
        per_well_runtime.append(
            {
                "well": well,
                "elapsed_s": predict_elapsed_s,
                "runtime_rows": int(len(scored)),
            }
        )

    elapsed_s = time.perf_counter() - start
    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    metrics = _metric_dict(scored_all, elapsed_s=elapsed_s)
    metrics.update(
        {
            "strategy": "residual",
            "fallback_tvt": fallback_tvt,
            "shrink": float(args.shrink),
            "train_rows": int(args.train_rows),
            "train_wells": int(len(train_wells)),
            "validation_wells": int(len(validation_wells)),
            "train_elapsed_s": train_elapsed_s,
            "feature_count": len(RESIDUAL_FEATURE_COLS),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_child_samples": int(args.min_child_samples),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_lambda": float(args.reg_lambda),
        }
    )

    per_well = _per_well_table(scored_all)
    runtime_df = pd.DataFrame(per_well_runtime)
    if not runtime_df.empty and not per_well.empty:
        per_well = per_well.merge(runtime_df, on="well", how="left")

    if args.output_predictions:
        pred_df = (
            pd.concat(pred_rows, ignore_index=True)
            if pred_rows
            else pd.DataFrame({"id": [], "tvt": []})
        )
        Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.output_predictions, index=False)

    _print_report(metrics, per_well, as_json=bool(args.json))


def cmd_score_csv(args: argparse.Namespace) -> None:
    if args.truth_csv:
        truth = pd.read_csv(args.truth_csv)
        required = {"id", "well", "row", "tvt"}
        missing = required - set(truth.columns)
        if missing:
            raise ValueError(f"truth CSV is missing columns: {sorted(missing)}")
    else:
        truth = _truth_for_wells(Path(args.train_dir), _select_wells(args))

    pred = pd.read_csv(args.predictions_csv)
    required_pred = {"id", "tvt"}
    missing_pred = required_pred - set(pred.columns)
    if missing_pred:
        raise ValueError(f"predictions CSV is missing columns: {sorted(missing_pred)}")
    if pred["id"].duplicated().any():
        dupes = pred.loc[pred["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"predictions CSV has duplicate ids, first examples: {dupes}")

    pred = pred[["id", "tvt"]].rename(columns={"tvt": "pred_tvt"})
    truth = truth[["id", "well", "row", "tvt"]].rename(columns={"tvt": "true_tvt"})
    merged = truth.merge(pred, on="id", how="left")
    missing = merged["pred_tvt"].isna()
    if missing.any():
        examples = merged.loc[missing, "id"].head(10).tolist()
        raise ValueError(f"Missing {int(missing.sum())} predictions, first examples: {examples}")
    extra = int((~pred["id"].isin(truth["id"])).sum())
    if extra:
        logger.warning("Ignoring %d prediction rows not present in validation truth.", extra)

    merged["pred_tvt"] = pd.to_numeric(merged["pred_tvt"], errors="coerce")
    bad = ~np.isfinite(merged["pred_tvt"].to_numpy(dtype=np.float64))
    if bad.any():
        examples = merged.loc[bad, "id"].head(10).tolist()
        raise ValueError(f"Non-finite predictions for {int(bad.sum())} rows: {examples}")

    metrics = _metric_dict(merged)
    per_well = _per_well_table(merged)
    _print_report(metrics, per_well, as_json=bool(args.json))


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Limit selected wells after fold selection.")
    parser.add_argument("--wells", default="", help="Comma-separated explicit well ids.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    export = sub.add_parser("export", help="Write a Kaggle-like validation folder and truth.csv.")
    _add_selection_args(export)
    export.add_argument("--export-dir", required=True)
    export.add_argument("--keep-typewell-geology", action="store_true")
    export.add_argument("--keep-horizontal-extra-cols", action="store_true")
    export.set_defaults(func=cmd_export)

    run_py = sub.add_parser("run-python", help="Score the current Python inference pipeline.")
    _add_selection_args(run_py)
    run_py.add_argument("--smoother", default="rts", choices=["rts", "gaussian", "none"])
    run_py.add_argument("--keep-typewell-geology", action="store_true")
    run_py.add_argument("--keep-horizontal-extra-cols", action="store_true")
    run_py.add_argument("--output-predictions", default="")
    run_py.add_argument("--json", action="store_true")
    run_py.set_defaults(func=cmd_run_python)

    run_base = sub.add_parser("run-baseline", help="Score simple constant/slope baselines.")
    _add_selection_args(run_base)
    run_base.add_argument("--strategy", default="constant", choices=["constant", "slope"])
    run_base.add_argument("--fallback-tvt", type=float, default=None)
    run_base.add_argument("--keep-horizontal-extra-cols", action="store_true")
    run_base.add_argument("--slope-cap", type=float, default=0.001)
    run_base.add_argument("--slope-tail", type=int, default=300)
    run_base.add_argument("--slope-min-points", type=int, default=30)
    run_base.add_argument("--slope-ci-threshold", type=float, default=0.002)
    run_base.add_argument("--slope-uncertain-shrink", type=float, default=0.3)
    run_base.add_argument("--output-predictions", default="")
    run_base.add_argument("--json", action="store_true")
    run_base.set_defaults(func=cmd_run_baseline)

    run_resid = sub.add_parser(
        "run-residual",
        help="Train v5-style residual LightGBM on non-validation wells and score the held-out fold.",
    )
    _add_selection_args(run_resid)
    run_resid.add_argument("--fallback-tvt", type=float, default=None)
    run_resid.add_argument("--train-rows", type=int, default=1_200_000)
    run_resid.add_argument("--train-limit", type=int, default=0, help="Debug only: limit residual training wells.")
    run_resid.add_argument("--shrink", type=float, default=0.75)
    run_resid.add_argument("--model-seed", type=int, default=20260506)
    run_resid.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    run_resid.add_argument("--n-estimators", type=int, default=700)
    run_resid.add_argument("--learning-rate", type=float, default=0.035)
    run_resid.add_argument("--num-leaves", type=int, default=63)
    run_resid.add_argument("--min-child-samples", type=int, default=200)
    run_resid.add_argument("--subsample", type=float, default=0.9)
    run_resid.add_argument("--colsample-bytree", type=float, default=0.9)
    run_resid.add_argument("--reg-lambda", type=float, default=2.0)
    run_resid.add_argument("--output-predictions", default="")
    run_resid.add_argument("--json", action="store_true")
    run_resid.set_defaults(func=cmd_run_residual)

    score = sub.add_parser("score-csv", help="Score a Kaggle-style id,tvt prediction CSV.")
    _add_selection_args(score)
    score.add_argument("--predictions-csv", required=True)
    score.add_argument("--truth-csv", default="", help="Use an exported truth.csv instead of selecting wells.")
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=cmd_score_csv)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    quiet_json = bool(getattr(args, "json", False)) and not args.verbose
    logging.basicConfig(
        level=logging.INFO if args.verbose else (logging.CRITICAL if quiet_json else logging.ERROR),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
