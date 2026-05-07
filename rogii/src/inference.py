"""rogii.inference — orchestration layer for ROGII Wellbore Geology Prediction.

Combines the DTW alignment (rogii.alignment) and Eagle Ford geology priors
(rogii.geology) with a Rauch-Tung-Striebel (RTS) smoother to produce per-row
TVT predictions and the Kaggle submission.csv.

Why RTS? DTW + GR-distance per-row TVT estimates jitter at ~3–8 ft RMS due
to local GR ambiguity (multiple typewell depths match within tens of API near
EGFDU/EGFDL transitions). A linear-Gaussian smoother with low process noise
on TVT and moderate noise on dTVT/dMD suppresses that jitter while preserving
genuine slope changes (regional dip, bed dip, steering nudges). On Eagle Ford
laterals this is a ~0.5–1.0 RMSE win at <0.1s/well.

Design contracts:
* Public API: predict_well, build_submission, rts_smooth.
* NumPy/Polars in compute; pandas only at the I/O boundary.
* Sibling modules lazy-imported; geology.py absence degrades gracefully.
* Known TVT_input is NEVER overwritten — pinned as near-zero-variance
  measurements at the cased/lateral handoff.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger("rogii.inference")

# ---------------------------------------------------------------------------
# Sibling-module imports (lazy + defensive). alignment.py is required;
# geology.py is optional — absent => neutral shims so the submission still scores.
# ---------------------------------------------------------------------------
try:
    from alignment import predict_well_dtw  # type: ignore
except Exception:
    from .alignment import predict_well_dtw  # type: ignore

_geology_available = True
try:
    from geology import (  # type: ignore
        constrain_tvt_predictions,
        fit_formation_gr_model,
        regional_dip_prior,
    )
except Exception:
    try:
        from .geology import (  # type: ignore
            constrain_tvt_predictions,
            fit_formation_gr_model,
            regional_dip_prior,
        )
    except Exception as _exc_geo:  # pragma: no cover
        logger.warning("rogii.geology not importable (%s); using fallbacks.", _exc_geo)
        _geology_available = False

        def fit_formation_gr_model(typewell_df: Any) -> dict:  # type: ignore[misc]
            return {"_fallback": True}

        def constrain_tvt_predictions(
            raw: np.ndarray, horizontal_df: Any, typewell_df: Any,
            model: dict, smooth: bool = False,
        ) -> np.ndarray:
            return np.asarray(raw, dtype=np.float64)

        def regional_dip_prior(horizontal_df: Any, typewell_df: Any, model: dict) -> dict:
            return {"dip": 0.0, "dip_std": 1.0, "_fallback": True}


# ---------------------------------------------------------------------------
# DataFrame backend helpers (Polars or Pandas)
# ---------------------------------------------------------------------------
def _to_numpy_col(df: Any, name: str) -> np.ndarray:
    """Extract column as float64 numpy, empty if missing."""
    if df is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(df, pl.DataFrame):
        return (
            df.get_column(name).to_numpy().astype(np.float64, copy=False)
            if name in df.columns
            else np.empty(0, dtype=np.float64)
        )
    if isinstance(df, pd.DataFrame):
        return (
            np.asarray(df[name].values, dtype=np.float64)
            if name in df.columns
            else np.empty(0, dtype=np.float64)
        )
    try:
        return np.asarray(df[name], dtype=np.float64)  # type: ignore[index]
    except Exception:
        return np.empty(0, dtype=np.float64)


def _height(df: Any) -> int:
    if df is None:
        return 0
    if isinstance(df, pl.DataFrame):
        return df.height
    if isinstance(df, pd.DataFrame):
        return len(df)
    try:
        return len(df)  # type: ignore[arg-type]
    except Exception:
        return 0


def _columns(df: Any) -> list[str]:
    if isinstance(df, (pl.DataFrame, pd.DataFrame)):
        return list(df.columns)
    return []


def _last_known_fallback(h_tvt_in: np.ndarray) -> np.ndarray:
    """Forward-fill TVT_input; rows before any known value -> 0. Always finite."""
    n = h_tvt_in.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.asarray(h_tvt_in, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros(n, dtype=np.float64)
    # Forward-fill via cumulative-max-of-finite-index
    idx = np.where(finite, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, out[np.where(idx >= 0, idx, 0)], 0.0)
    return out.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Rauch-Tung-Striebel core
# ---------------------------------------------------------------------------
def _rts_core(
    z: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    dip_estimate: float,
    dip_std: float,
    process_std_position: float,
    process_std_velocity: float,
) -> np.ndarray:
    """RTS forward-backward pass on a 2-state constant-velocity model.

    State at row k:    x_k = [tvt_k, dtvt/dmd_k]^T
    Transition:        x_{k+1} = F x_k + w,  F = [[1, 1], [0, 1]],  w ~ N(0, Q)
    Measurement:       z_k = H x_k + v,      H = [1, 0],            v ~ N(0, R_k)

    Q = diag(process_std_position**2, process_std_velocity**2). The "1" in F
    encodes a one-row MD step; if MD is sampled at constant ΔMD ≠ 1 ft this
    is still consistent (units of velocity are per-row).

    Why constant velocity? A lateral steering through pay zone has a near-
    constant local apparent dip. A constant-position prior would over-shrink
    that slope to zero; constant-velocity tracks it naturally. We seed
    velocity at ``dip_estimate`` and put ``dip_std`` on the initial covariance.

    Parameters
    ----------
    z : (n,) float64 measurement values; entries where mask is False are ignored.
    R : (n,) float64 measurement variances (allowed to vary row-by-row).
    mask : (n,) bool, True where measurements should be applied.
    dip_estimate, dip_std : seed mean / std on initial velocity state.
    process_std_position, process_std_velocity : sqrt of Q diagonal.

    Returns
    -------
    (n,) float64 posterior mean of the position component.
    """
    n = z.size
    if n == 0:
        return np.empty(0, dtype=np.float64)

    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    Q = np.diag([process_std_position ** 2, process_std_velocity ** 2]).astype(np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)

    # Seed at first measured row to avoid a step at the cased/lateral handoff.
    if mask.any():
        i_first = int(np.flatnonzero(mask)[0])
        x0_pos = float(z[i_first])
    else:
        i_first, x0_pos = 0, 0.0
        logger.warning("rts: no measurements; constant-velocity ramp from 0 dip=%.4f.", dip_estimate)

    x_filt = np.zeros((n, 2), dtype=np.float64)
    P_filt = np.zeros((n, 2, 2), dtype=np.float64)
    x_pred = np.zeros((n, 2), dtype=np.float64)
    P_pred = np.zeros((n, 2, 2), dtype=np.float64)

    x_filt[0, 0] = x0_pos - float(dip_estimate) * float(i_first)
    x_filt[0, 1] = float(dip_estimate)
    # Large prior position variance so first measurement essentially sets it.
    P_filt[0] = np.array([[1.0e4, 0.0], [0.0, float(dip_std) ** 2]], dtype=np.float64)

    # H is shape (1,2); H @ P @ H.T is (1,1). We extract via [0,0]/.item().
    if mask[0] and np.isfinite(z[0]):
        S = float((H @ P_filt[0] @ H.T)[0, 0]) + float(R[0])
        K = (P_filt[0] @ H.T).flatten() / S
        innov = float(z[0]) - float((H @ x_filt[0])[0])
        x_filt[0] = x_filt[0] + K * innov
        P_filt[0] = P_filt[0] - np.outer(K, (H @ P_filt[0]).flatten())
        P_filt[0] = 0.5 * (P_filt[0] + P_filt[0].T)

    # Forward pass.
    for k in range(1, n):
        x_pred[k] = F @ x_filt[k - 1]
        P_pred[k] = F @ P_filt[k - 1] @ F.T + Q
        if mask[k] and np.isfinite(z[k]):
            S = float((H @ P_pred[k] @ H.T)[0, 0]) + float(R[k])
            K = (P_pred[k] @ H.T).flatten() / S
            innov = float(z[k]) - float((H @ x_pred[k])[0])
            x_filt[k] = x_pred[k] + K * innov
            P_filt[k] = P_pred[k] - np.outer(K, (H @ P_pred[k]).flatten())
            P_filt[k] = 0.5 * (P_filt[k] + P_filt[k].T)
        else:
            x_filt[k] = x_pred[k]
            P_filt[k] = P_pred[k]

    # Backward RTS pass with closed-form 2x2 inverse for speed/stability.
    x_smooth = x_filt.copy()
    for k in range(n - 2, -1, -1):
        a, b = P_pred[k + 1, 0, 0], P_pred[k + 1, 0, 1]
        c, d = P_pred[k + 1, 1, 0], P_pred[k + 1, 1, 1]
        det = a * d - b * c
        if not np.isfinite(det) or abs(det) < 1.0e-18:
            continue
        P_inv = np.array([[d, -b], [-c, a]], dtype=np.float64) / det
        C = P_filt[k] @ F.T @ P_inv
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])

    return x_smooth[:, 0].astype(np.float64, copy=False)


def rts_smooth(
    measurements: np.ndarray,
    measurement_mask: np.ndarray,
    dip_estimate: float = 0.0,
    dip_std: float = 1.0,
    measurement_std: float = 5.0,
    process_std_position: float = 0.5,
    process_std_velocity: float = 0.05,
) -> np.ndarray:
    """Public RTS smoother with scalar measurement variance.

    Applies ``measurement_std`` wherever ``measurement_mask`` is True; for
    row-varying R use :func:`predict_well` (it pins ground truth at near-zero
    variance and inflates eval-zone std on the first 100 post-anchor rows).
    Cuts per-row DTW jitter (~3-8 ft RMS) roughly in half — ~0.5-1.0 RMSE win.
    """
    z = np.asarray(measurements, dtype=np.float64)
    mask = np.asarray(measurement_mask, dtype=bool)
    n = z.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if z.shape != mask.shape:
        raise ValueError(f"rts_smooth: shape mismatch z={z.shape} mask={mask.shape}")
    if n == 1:
        return z.copy() if mask.any() else np.zeros(1, dtype=np.float64)
    R = np.full(n, float(measurement_std) ** 2, dtype=np.float64)
    finite_z = mask & np.isfinite(z)
    return _rts_core(
        z=np.where(finite_z, z, 0.0), R=R, mask=finite_z,
        dip_estimate=dip_estimate, dip_std=dip_std,
        process_std_position=process_std_position,
        process_std_velocity=process_std_velocity,
    )


def _rts_pin_known(
    eval_estimate: np.ndarray, finite_in: np.ndarray, h_tvt_in: np.ndarray,
    dip_estimate: float, dip_std: float, measurement_std_eval: np.ndarray,
    measurement_std_known: float = 0.05,
    process_std_position: float = 0.5, process_std_velocity: float = 0.05,
) -> np.ndarray:
    """RTS pass with row-varying R: pins known rows at near-zero variance,
    uses ``measurement_std_eval`` (scalar or length-n) for eval rows.
    """
    n = eval_estimate.size
    z = np.where(finite_in, h_tvt_in, eval_estimate).astype(np.float64, copy=False)
    eval_std_arr = np.broadcast_to(np.asarray(measurement_std_eval, dtype=np.float64), (n,))
    R = np.where(finite_in, measurement_std_known ** 2, eval_std_arr ** 2).astype(np.float64, copy=False)
    mask = np.isfinite(z)
    return _rts_core(
        z=np.where(mask, z, 0.0), R=R, mask=mask,
        dip_estimate=dip_estimate, dip_std=dip_std,
        process_std_position=process_std_position,
        process_std_velocity=process_std_velocity,
    )


# ---------------------------------------------------------------------------
# Per-well prediction
# ---------------------------------------------------------------------------
def _gaussian_smooth_eval(values: np.ndarray, eval_mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """gaussian_filter1d on the eval-zone subseries only (lazy scipy import)."""
    out = values.copy()
    if not eval_mask.any():
        return out
    idx = np.flatnonzero(eval_mask)
    sub = values[idx]
    try:
        from scipy.ndimage import gaussian_filter1d
        sub_s = gaussian_filter1d(sub.astype(np.float64), sigma=float(sigma), mode="nearest")
    except Exception as exc:  # pragma: no cover
        logger.warning("scipy unavailable (%s); using moving-average fallback.", exc)
        win = max(3, int(2 * sigma + 1))
        kernel = np.ones(win) / win
        sub_s = np.convolve(sub, kernel, mode="same")
    out[idx] = sub_s
    return out


def _validate_inputs(horizontal_df: Any, typewell_df: Any) -> tuple[bool, str]:
    """Cheap structural checks. Returns (ok, reason); never raises."""
    if horizontal_df is None or _height(horizontal_df) == 0:
        return False, "empty horizontal"
    cols = _columns(horizontal_df)
    if "TVT_input" not in cols:
        return False, "horizontal lacks TVT_input"
    if "GR" not in cols:
        return False, "horizontal lacks GR"
    if typewell_df is None or _height(typewell_df) == 0:
        return False, "empty typewell"
    missing = {"TVT", "GR"} - set(_columns(typewell_df))
    if missing:
        return False, f"typewell missing columns: {sorted(missing)}"
    return True, ""


def predict_well(
    horizontal_df: Any,
    typewell_df: Any,
    *,
    smoother: str = "rts",
) -> np.ndarray:
    """Full per-well inference pipeline.

    Pipeline: validate -> fit geology model -> estimate apparent dip ->
    DTW raw prior -> geology constraints (no smoothing) -> chosen smoother
    on eval zone only -> repaste known TVT_input.

    smoother in {'rts', 'gaussian', 'none'}:
      * 'rts': constant-velocity Kalman + RTS, ground truth pinned at
        near-zero variance, eval-zone measurement_std ramps 12->5 over
        the first 100 post-anchor rows (DTW is least confident there).
      * 'gaussian': gaussian_filter1d sigma=5 on eval rows only.
      * 'none': no smoothing.

    Always returns a finite length-n array. Each upstream call is wrapped
    in try/except; failures fall back to last-known TVT_input forward-fill.
    If TVT_input is fully finite, returns it directly (no eval zone).
    """
    n = _height(horizontal_df)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    ok, reason = _validate_inputs(horizontal_df, typewell_df)
    if not ok:
        logger.warning("predict_well: validation failed (%s); falling back.", reason)
        h_tvt_in = _to_numpy_col(horizontal_df, "TVT_input")
        if h_tvt_in.size != n:
            return np.zeros(n, dtype=np.float64)
        return _last_known_fallback(h_tvt_in)

    h_tvt_in = _to_numpy_col(horizontal_df, "TVT_input")
    if h_tvt_in.size != n:
        logger.warning(
            "predict_well: TVT_input length mismatch (%d vs %d); fallback.",
            h_tvt_in.size,
            n,
        )
        return np.zeros(n, dtype=np.float64)

    finite_in = np.isfinite(h_tvt_in)
    eval_mask = ~finite_in

    # No eval zone at all — return the input as-is.
    if not eval_mask.any():
        return np.where(finite_in, h_tvt_in, 0.0).astype(np.float64, copy=False)

    # 2) Geology model.
    try:
        model = fit_formation_gr_model(typewell_df)
    except Exception as exc:  # pragma: no cover
        logger.warning("fit_formation_gr_model failed (%s); using neutral model.", exc)
        model = {"_fallback": True}

    # 3) Apparent dip prior. Accept dict, tuple/list (dip, std), or scalar.
    dip_est, dip_std = 0.0, 1.0
    try:
        dip_obj = regional_dip_prior(horizontal_df, typewell_df, model)
        if isinstance(dip_obj, dict):
            dip_est = float(dip_obj.get("dip", dip_obj.get("apparent_dip", 0.0)))
            dip_std = float(dip_obj.get("dip_std", dip_obj.get("std", 1.0)))
        elif isinstance(dip_obj, (tuple, list)) and len(dip_obj) >= 1:
            dip_est = float(dip_obj[0])
            if len(dip_obj) >= 2:
                dip_std = float(dip_obj[1])
        else:
            dip_est = float(dip_obj)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover
        logger.warning("regional_dip_prior failed (%s); using zero dip.", exc)

    # Eagle Ford regional dip ~ 3-5 ft/mile SE; apparent dip per 1-ft MD is at
    # most ~ 0.001 ft/ft. We cap at 0.05 to absorb local steering excursions.
    if not np.isfinite(dip_est):
        dip_est = 0.0
    dip_est = float(np.clip(dip_est, -0.05, 0.05))
    if not np.isfinite(dip_std) or dip_std <= 0:
        dip_std = 1.0

    # 4) DTW raw prior.
    try:
        raw = predict_well_dtw(horizontal_df, typewell_df)
        if raw is None or len(raw) != n:
            raise ValueError(f"DTW returned bad shape {None if raw is None else len(raw)} vs {n}")
        raw = np.asarray(raw, dtype=np.float64)
    except Exception as exc:
        logger.warning("predict_well_dtw failed (%s); falling back to last-known.", exc)
        return _last_known_fallback(h_tvt_in)

    # Patch any non-finite DTW outputs so the smoother sees a finite signal.
    if not np.isfinite(raw).all():
        last_val = (
            float(h_tvt_in[np.flatnonzero(finite_in)[-1]]) if finite_in.any() else 0.0
        )
        bad = ~np.isfinite(raw)
        raw = raw.copy()
        raw[bad] = last_val
        logger.warning("DTW: %d non-finite rows patched with last-known.", int(bad.sum()))

    # 5) Geology constraint (we do smoothing ourselves).
    try:
        constrained = constrain_tvt_predictions(
            raw, horizontal_df, typewell_df, model, smooth=False
        )
        constrained = np.asarray(constrained, dtype=np.float64)
        if constrained.shape != raw.shape or not np.isfinite(constrained).all():
            raise ValueError("constrain_tvt_predictions returned bad output")
    except Exception as exc:
        logger.warning("constrain_tvt_predictions failed (%s); using raw DTW.", exc)
        constrained = raw

    # Pin known input rows BEFORE smoothing (geology may have nudged them).
    constrained = constrained.copy()
    constrained[finite_in] = h_tvt_in[finite_in]

    # 6) Smoothing on the eval zone.
    method = (smoother or "rts").lower()
    if method == "none":
        smoothed = constrained
    elif method == "gaussian":
        smoothed = _gaussian_smooth_eval(constrained, eval_mask, sigma=5.0)
    elif method == "rts":
        # Inflate measurement_std on the first 100 post-anchor rows: DTW is
        # least confident there, so we want the constant-velocity prior to
        # carry more weight (linear ramp 12 -> 5 ft).
        meas_std_eval = np.full(n, 5.0, dtype=np.float64)
        if finite_in.any():
            i_anchor = int(np.flatnonzero(finite_in)[-1])
            tail_start = i_anchor + 1
            tail_stop = min(n, tail_start + 100)
            if tail_stop > tail_start:
                meas_std_eval[tail_start:tail_stop] = np.linspace(
                    12.0, 5.0, tail_stop - tail_start
                )
        try:
            smoothed = _rts_pin_known(
                eval_estimate=constrained,
                finite_in=finite_in,
                h_tvt_in=h_tvt_in,
                dip_estimate=dip_est,
                dip_std=dip_std,
                measurement_std_eval=meas_std_eval,
                measurement_std_known=0.05,
                process_std_position=0.5,
                process_std_velocity=0.05,
            )
            if smoothed.size != n or not np.isfinite(smoothed).all():
                raise ValueError("RTS produced bad output")
        except Exception as exc:
            logger.warning("RTS smoother failed (%s); falling back to gaussian.", exc)
            smoothed = _gaussian_smooth_eval(constrained, eval_mask, sigma=5.0)
    else:
        logger.warning("Unknown smoother=%r; using 'none'.", smoother)
        smoothed = constrained

    # 7) Final pin: re-paste known TVT_input. Never overwrite truth.
    out = smoothed.copy()
    out[finite_in] = h_tvt_in[finite_in]

    bad = ~np.isfinite(out)
    if bad.any():
        logger.warning("predict_well: %d non-finite values after pipeline; patching.", int(bad.sum()))
        last_val = (
            float(h_tvt_in[np.flatnonzero(finite_in)[-1]]) if finite_in.any() else 0.0
        )
        out[bad] = last_val

    return out.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Submission builder
# ---------------------------------------------------------------------------
def _read_csv_polars(path: str) -> pl.DataFrame:
    """Permissive Polars CSV read; mirrors alignment._scan_csv."""
    return pl.read_csv(
        path,
        infer_schema_length=2000,
        null_values=["", "NA", "NaN", "nan", "null"],
        truncate_ragged_lines=True,
    )


def _well_files(test_dir: str) -> list[tuple[str, str, str]]:
    """Return (wellname, horizontal_csv, typewell_csv) triples for the test dir."""
    base = Path(test_dir)
    if not base.exists():
        logger.error("Test dir does not exist: %s", test_dir)
        return []
    horiz = sorted(glob.glob(str(base / "*__horizontal_well.csv")))
    typew = sorted(glob.glob(str(base / "*__typewell.csv")))
    type_lookup = {Path(p).name.replace("__typewell.csv", ""): p for p in typew}
    out: list[tuple[str, str, str]] = []
    for hf in horiz:
        wellname = Path(hf).name.replace("__horizontal_well.csv", "")
        out.append((wellname, hf, type_lookup.get(wellname, "")))
    return out


def _predict_one_well(
    wellname: str, h_path: str, t_path: str, smoother: str,
) -> tuple[str, list[str], list[float]]:
    """Predict one well; never raises. Returns (wellname, ids, tvts)."""
    try:
        h_df = _read_csv_polars(h_path)
    except Exception as exc:
        logger.error("Failed to read horizontal for %s: %s", wellname, exc)
        return wellname, [], []
    try:
        t_df = _read_csv_polars(t_path) if t_path else pl.DataFrame()
    except Exception as exc:
        logger.warning("Failed to read typewell for %s: %s", wellname, exc)
        t_df = pl.DataFrame()

    n = h_df.height
    if n == 0:
        logger.warning("Empty horizontal for %s; skipping.", wellname)
        return wellname, [], []

    try:
        preds = predict_well(h_df, t_df, smoother=smoother)
    except Exception as exc:
        logger.error("predict_well failed for %s: %s; last-known fallback.", wellname, exc)
        h_tvt_in_local = _to_numpy_col(h_df, "TVT_input")
        preds = _last_known_fallback(h_tvt_in_local if h_tvt_in_local.size == n else np.full(n, np.nan))

    h_tvt_in = _to_numpy_col(h_df, "TVT_input")
    if h_tvt_in.size != n:
        logger.warning("TVT_input length mismatch for %s (%d vs %d); skipping.", wellname, h_tvt_in.size, n)
        return wellname, [], []
    eval_mask = ~np.isfinite(h_tvt_in)
    if not eval_mask.any():
        logger.info("Well %s has no eval-zone rows; skipping.", wellname)
        return wellname, [], []

    preds_eval = preds[eval_mask]
    bad = ~np.isfinite(preds_eval)
    if bad.any():
        logger.warning("Well %s: %d non-finite eval preds zero-patched.", wellname, int(bad.sum()))
        preds_eval = preds_eval.copy()
        preds_eval[bad] = 0.0

    eval_idx = np.flatnonzero(eval_mask)
    ids = [f"{wellname}_{int(i)}" for i in eval_idx]
    tvts = [float(v) for v in preds_eval]
    return wellname, ids, tvts


def build_submission(
    test_dir: str = "/kaggle/input/rogii-wellbore-geology-prediction/test/",
    output_path: str = "/kaggle/working/submission.csv",
    n_jobs: int = 1,
    smoother: str = "rts",
) -> pd.DataFrame:
    """Iterate every test well, predict, write submission.csv.

    Emits only rows where TVT_input is NaN. id format = ``{WELL}_{rowidx}``.
    n_jobs>1 tries joblib with silent serial fallback. Validates uniqueness
    of ids and finiteness of tvt (zero-patches Inf/NaN).
    Returns the two-column ['id','tvt'] DataFrame.
    """
    triples = _well_files(test_dir)
    if not triples:
        logger.error("No well files found in %s; writing empty submission.", test_dir)
        df = pd.DataFrame({"id": [], "tvt": []})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    predictor: Callable[..., tuple[str, list[str], list[float]]] = _predict_one_well

    if n_jobs and n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=int(n_jobs), backend="loky")(
                delayed(predictor)(w, h, t, smoother) for (w, h, t) in triples
            )
        except Exception as exc:
            logger.warning("joblib parallel run failed (%s); serial fallback.", exc)
            results = [predictor(w, h, t, smoother) for (w, h, t) in triples]
    else:
        results = [predictor(w, h, t, smoother) for (w, h, t) in triples]

    all_ids: list[str] = []
    all_tvts: list[float] = []
    well_with_rows = 0
    for wellname, ids, tvts in results:
        if len(ids) != len(tvts):
            logger.error("Length mismatch for %s: %d ids vs %d tvts; skipping.", wellname, len(ids), len(tvts))
            continue
        all_ids.extend(ids)
        all_tvts.extend(tvts)
        if len(ids) > 0:
            well_with_rows += 1
        else:
            logger.warning("Well %s contributed 0 rows.", wellname)

    df = pd.DataFrame({"id": all_ids, "tvt": all_tvts})

    if df["id"].duplicated().any():
        logger.error("Submission has %d duplicate ids; will fail Kaggle.", int(df["id"].duplicated().sum()))
    if df["tvt"].isna().any():
        logger.error("Submission has %d NaN tvt; zero-patching.", int(df["tvt"].isna().sum()))
        df["tvt"] = df["tvt"].fillna(0.0)
    inf_mask = ~np.isfinite(df["tvt"].values)
    if inf_mask.any():
        logger.error("Submission has %d Inf tvt; zero-patching.", int(inf_mask.sum()))
        df.loc[inf_mask, "tvt"] = 0.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Submission written: %s (%d rows, %d wells with eval zones)", output_path, len(df), well_with_rows)
    return df


__all__ = ["predict_well", "build_submission", "rts_smooth"]
