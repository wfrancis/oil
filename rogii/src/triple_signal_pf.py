"""Faithful port of the public triple-signal kernel's two particle filters.

Source notebook:
  /Users/william/drilling_oil_gas/rogii/research/public_kernels/
  triple-signal-beam-search-dual-pf-lightgbm.ipynb (LB 11.284, 2026-05-07)

This module ports cells 5 (TVT-PF with Z-velocity coupling) and 6 (ANCC-PF
tracking S = TVT + Z) verbatim — same hyperparameters, same update equations,
same RNG sequencing per well. Constants live next to the functions for
readability; they exactly mirror the source notebook's cell-2 block.

Two callers are exposed:

* :func:`run_pf_z_velocity`, :func:`run_pf_ancc` — single-well functions that
  take pandas-equivalent column arrays and return ``(pred, std)`` numpy
  vectors over the eval rows. Drop-in for the source.
* :func:`run_pfs_for_wells` — multiprocessing fan-out across wells. Returns
  ``{well_id: {pf_z_pred, pf_z_std, pf_ancc_pred, pf_ancc_std, eval_idx}}``.
  Each worker re-seeds ``np.random.seed(42)`` before each well's PFs so the
  per-well outputs are reproducible regardless of worker scheduling.

Project policy: polars-first. The source notebook uses pandas; we accept
polars at the I/O boundary and only materialise numpy arrays inside the PF
loops (which is what the source does internally too).

This is a faithful port. Do not "improve" the PFs here — the LightGBM
upstream that consumes these signals has been calibrated to their exact
biases. Tune in a sibling module if you need to.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Constants — mirror cell 2 of the source notebook EXACTLY.
# ---------------------------------------------------------------------------

# TVT Particle Filter
PF_N_PARTICLES = 500
PF_MOMENTUM_ALPHA = 0.993
PF_Z_SIGMA_FLOOR = 0.005
PF_Z_SIGMA_SCALE = 2.0
PF_VELOCITY_NOISE_STD = 0.005
PF_POSITION_NOISE_STD = 0.01
PF_INIT_VELOCITY_STD = 0.02
PF_GR_SIGMA_MIN = 10.0
PF_GR_SIGMA_MAX = 60.0
PF_GR_SIGMA_DEFAULT = 30.0
PF_INIT_SPREAD_STD = 0.5
PF_RESAMPLE_THRESHOLD = 0.5
PF_ROUGHENING_STD_POS = 0.2
PF_ROUGHENING_STD_VEL = 0.003
PF_GR_ROLLING_WINDOW = 5
PF_GR_ROLLING_WEIGHT = 0.3

# ANCC Particle Filter
ANCC_ALPHA = 0.998
ANCC_RATE_NOISE_STD = 0.002
ANCC_POS_NOISE_STD = 0.005
ANCC_INIT_RATE_STD = 0.01
ANCC_INIT_SPREAD_STD = 0.3
ANCC_ROUGHENING_STD_POS = 0.1
ANCC_ROUGHENING_STD_RATE = 0.001
ANCC_N_PARTICLES = 500

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# pandas <-> numpy adaptors. The source PF code reads columns by name, uses
# .notna() masks and .iloc[-1]. Internally we work in numpy; everything below
# accepts a small struct of pre-extracted arrays so the same helpers feed
# both the pandas-shim entry points (for parity with the source notebook) and
# the polars-driven parallel driver.
# ---------------------------------------------------------------------------


class _HWArrays:
    """Pre-extracted numpy view of one horizontal well.

    Attributes mirror the columns the source code touches plus a precomputed
    smoothed-GR array (matching the source's pandas rolling window). Carrying
    them as numpy keeps the inner PF loops free of any DataFrame ops.

    The ``known_mask`` / ``eval_mask`` are computed once from ``TVT_input``
    being non-null, exactly as ``hw[hw['TVT_input'].notna()]`` does in the
    source notebook (note: nan==nan is False, polars/pandas treat NaN and
    null both as "missing" for this purpose; we replicate by using
    np.isnan on float64).
    """

    __slots__ = (
        "md", "x", "y", "z", "gr", "tvt_input",
        "gr_smooth_full", "known_mask", "eval_mask",
    )

    def __init__(
        self,
        md: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        gr: np.ndarray,
        tvt_input: np.ndarray,
    ):
        self.md = md
        self.x = x
        self.y = y
        self.z = z
        self.gr = gr
        self.tvt_input = tvt_input
        # mirror pandas Series.rolling(window, center=True, min_periods=1).mean()
        self.gr_smooth_full = _rolling_mean_centered(gr, PF_GR_ROLLING_WINDOW)
        self.known_mask = ~_is_missing(tvt_input)
        self.eval_mask = _is_missing(tvt_input)


def _is_missing(arr: np.ndarray) -> np.ndarray:
    """True where the value is NaN (the source uses .notna() on pandas)."""
    return np.isnan(arr)


def _rolling_mean_centered(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with min_periods=1, matching pandas exactly.

    The source uses ``pd.Series.rolling(window, center=True, min_periods=1).mean()``
    which:
      * treats NaN as missing (skips them in the mean),
      * returns NaN only if the entire window is NaN,
      * for window=W centers at i-(W-1)//2 ... i+W//2 (pandas convention).

    We implement using a masked cumulative sum which is O(N) and exact.
    """
    n = arr.shape[0]
    if n == 0:
        return arr.copy()
    a = arr.astype(np.float64, copy=False)
    valid = ~np.isnan(a)
    a_filled = np.where(valid, a, 0.0)
    # cumulative sums with 0 prefix for clean window slicing
    csum = np.concatenate(([0.0], np.cumsum(a_filled)))
    cnt = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    half_lo = (window - 1) // 2
    half_hi = window // 2
    idx = np.arange(n)
    lo = np.maximum(idx - half_lo, 0)
    hi = np.minimum(idx + half_hi + 1, n)
    s = csum[hi] - csum[lo]
    c = cnt[hi] - cnt[lo]
    out = np.full(n, np.nan, dtype=np.float64)
    nz = c > 0
    out[nz] = s[nz] / c[nz]
    return out


def _hw_from_polars(df: pl.DataFrame) -> _HWArrays:
    """Materialise the columns the PFs read from a polars DataFrame."""
    return _HWArrays(
        md=df["MD"].to_numpy().astype(np.float64),
        x=df["X"].to_numpy().astype(np.float64) if "X" in df.columns else np.zeros(df.height),
        y=df["Y"].to_numpy().astype(np.float64) if "Y" in df.columns else np.zeros(df.height),
        z=df["Z"].to_numpy().astype(np.float64),
        gr=df["GR"].to_numpy().astype(np.float64),
        tvt_input=df["TVT_input"].to_numpy().astype(np.float64),
    )


def _tw_from_polars(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract TVT and GR arrays from a typewell polars DataFrame."""
    return (
        df["TVT"].to_numpy().astype(np.float64),
        df["GR"].to_numpy().astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Inner numpy implementations of the four helpers + two PF runners.
# These take pre-extracted arrays so they can be shared between the
# polars-driven driver and the pandas-shim entry points below.
# ---------------------------------------------------------------------------


def _calibrate_gr_sigma(hw: _HWArrays, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    """Source: pf_calibrate_gr_sigma."""
    known = hw.known_mask
    known_gr_valid = known & ~np.isnan(hw.gr)
    if int(known_gr_valid.sum()) < 20:
        return PF_GR_SIGMA_DEFAULT
    tw_func = interp1d(
        tw_tvt, tw_gr, bounds_error=False,
        fill_value=(tw_gr[0], tw_gr[-1]),
    )
    expected = tw_func(hw.tvt_input[known_gr_valid])
    residuals = hw.gr[known_gr_valid] - expected
    return float(np.clip(np.std(residuals), PF_GR_SIGMA_MIN, PF_GR_SIGMA_MAX))


def _estimate_init_velocity(hw: _HWArrays) -> float:
    """Source: pf_estimate_init_velocity."""
    known_idx = np.flatnonzero(hw.known_mask)
    if known_idx.size < 10:
        return 0.0
    tail = known_idx[-20:]
    if tail.size < 5:
        return 0.0
    dtvt = np.diff(hw.tvt_input[tail])
    dmd = np.diff(hw.md[tail])
    mask = dmd > 0
    if int(mask.sum()) < 3:
        return 0.0
    return float(np.median(dtvt[mask] / dmd[mask]))


def _learn_z_beta(hw: _HWArrays) -> tuple[float, float, float]:
    """Source: pf_learn_z_beta. Returns (beta, intercept, sigma)."""
    known_idx = np.flatnonzero(hw.known_mask)
    if known_idx.size < 30:
        return -1.0, 0.0, 0.1
    z_known = hw.z[known_idx]
    tvt_known = hw.tvt_input[known_idx]
    md_known = hw.md[known_idx]
    dz = np.diff(z_known)
    dtvt = np.diff(tvt_known)
    dmd = np.diff(md_known)
    mask = dmd > 0
    if int(mask.sum()) < 10:
        return -1.0, 0.0, 0.1
    vz = dz[mask] / dmd[mask]
    vt = dtvt[mask] / dmd[mask]
    A = np.column_stack([vz, np.ones_like(vz)])
    coef, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
    residuals = vt - (coef[0] * vz + coef[1])
    sigma = max(float(np.std(residuals)), 0.001)
    return float(coef[0]), float(coef[1]), float(sigma)


def _ancc_estimate_init_rate(hw: _HWArrays) -> float:
    """Source: ancc_estimate_init_rate."""
    known_idx = np.flatnonzero(hw.known_mask)
    if known_idx.size < 10:
        return 0.0
    tail = known_idx[-30:]
    dtvt = np.diff(hw.tvt_input[tail])
    dz = np.diff(hw.z[tail])
    dmd = np.diff(hw.md[tail])
    dancc = dtvt + dz
    mask = dmd > 0
    if int(mask.sum()) < 3:
        return 0.0
    return float(np.median(dancc[mask] / dmd[mask]))


def _run_pf_z_velocity_inner(
    hw: _HWArrays,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    n_particles: int = PF_N_PARTICLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Source: run_pf_z_velocity (cell 5).

    Mirrors the source line-for-line, including the order of np.random
    draws — this matters because the caller seeds np.random.seed(42)
    before invoking, so we must consume the same number of randoms in the
    same sequence.
    """
    tw_func_point = interp1d(
        tw_tvt, tw_gr, bounds_error=False,
        fill_value=(tw_gr[0], tw_gr[-1]),
    )
    # Source: pd.Series(tw_gr).rolling(W, center=True, min_periods=1).mean()
    tw_smooth_gr = _rolling_mean_centered(tw_gr, PF_GR_ROLLING_WINDOW)
    tw_func_smooth = interp1d(
        tw_tvt, tw_smooth_gr, bounds_error=False,
        fill_value=(tw_smooth_gr[0], tw_smooth_gr[-1]),
    )
    tvt_min, tvt_max = float(tw_tvt.min()), float(tw_tvt.max())
    gr_sigma = _calibrate_gr_sigma(hw, tw_tvt, tw_gr)
    beta, intercept, z_sigma = _learn_z_beta(hw)

    eval_idx = np.flatnonzero(hw.eval_mask)
    known_idx = np.flatnonzero(hw.known_mask)
    if eval_idx.size == 0:
        return np.array([]), np.array([])

    last_known_pos = int(known_idx[-1])
    last_tvt = float(hw.tvt_input[last_known_pos])
    positions = last_tvt + np.random.normal(0, PF_INIT_SPREAD_STD, n_particles)
    init_v = _estimate_init_velocity(hw)
    velocities = init_v + np.random.normal(0, PF_INIT_VELOCITY_STD, n_particles)
    weights = np.ones(n_particles) / n_particles

    md_vals = hw.md[eval_idx]
    gr_vals = hw.gr[eval_idx]
    z_vals = hw.z[eval_idx]
    gr_smooth_eval = hw.gr_smooth_full[eval_idx]

    prev_md = float(hw.md[last_known_pos])
    prev_z = float(hw.z[last_known_pos])

    pred_tvts = np.empty(eval_idx.size)
    pred_stds = np.empty(eval_idx.size)

    for i in range(eval_idx.size):
        d_md = md_vals[i] - prev_md
        if d_md <= 0:
            d_md = 1.0
        dz_dmd = (z_vals[i] - prev_z) / d_md
        v_expected = beta * dz_dmd + intercept

        velocities = (
            PF_MOMENTUM_ALPHA * velocities
            + np.random.normal(0, PF_VELOCITY_NOISE_STD, n_particles)
        )
        positions = (
            positions + velocities * d_md
            + np.random.normal(0, PF_POSITION_NOISE_STD, n_particles)
        )
        positions = np.clip(positions, tvt_min - 50, tvt_max + 50)

        if not np.isnan(gr_vals[i]):
            gr_smooth = gr_smooth_eval[i]
            expected_point = tw_func_point(positions)
            diff_point = gr_vals[i] - expected_point
            lik_point = np.exp(-0.5 * (diff_point / gr_sigma) ** 2)
            if not np.isnan(gr_smooth):
                expected_smooth = tw_func_smooth(positions)
                diff_smooth = gr_smooth - expected_smooth
                lik_smooth = np.exp(-0.5 * (diff_smooth / (gr_sigma * 1.5)) ** 2)
                likelihood = (
                    (1 - PF_GR_ROLLING_WEIGHT) * lik_point
                    + PF_GR_ROLLING_WEIGHT * lik_smooth
                )
            else:
                likelihood = lik_point
            likelihood = np.maximum(likelihood, 1e-300)
            weights = weights * likelihood
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights[:] = 1.0 / n_particles

        z_sig = max(z_sigma * PF_Z_SIGMA_SCALE, PF_Z_SIGMA_FLOOR)
        diff_v = velocities - v_expected
        lik_z = np.exp(-0.5 * (diff_v / z_sig) ** 2)
        lik_z = np.maximum(lik_z, 1e-300)
        weights = weights * lik_z
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights[:] = 1.0 / n_particles

        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            cum = np.cumsum(weights)
            pos_resample = (np.arange(n_particles) + np.random.uniform()) / n_particles
            indices = np.searchsorted(cum, pos_resample)
            positions = positions[indices]
            velocities = velocities[indices]
            weights[:] = 1.0 / n_particles
            positions += np.random.normal(0, PF_ROUGHENING_STD_POS, n_particles)
            velocities += np.random.normal(0, PF_ROUGHENING_STD_VEL, n_particles)

        pred_tvts[i] = np.average(positions, weights=weights)
        pred_stds[i] = np.sqrt(
            np.average((positions - pred_tvts[i]) ** 2, weights=weights)
        )
        prev_md = md_vals[i]
        prev_z = z_vals[i]

    return pred_tvts, pred_stds


def _run_pf_ancc_inner(
    hw: _HWArrays,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    n_particles: int = ANCC_N_PARTICLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Source: run_pf_ancc (cell 6). Tracks state S = TVT + Z."""
    tvt_min, tvt_max = float(tw_tvt.min()), float(tw_tvt.max())
    gr_sigma = _calibrate_gr_sigma(hw, tw_tvt, tw_gr)

    eval_idx = np.flatnonzero(hw.eval_mask)
    known_idx = np.flatnonzero(hw.known_mask)
    if eval_idx.size == 0:
        return np.array([]), np.array([])
    last_known_pos = int(known_idx[-1])
    last_state = float(hw.tvt_input[last_known_pos]) + float(hw.z[last_known_pos])
    init_rate = _ancc_estimate_init_rate(hw)
    pos = last_state + np.random.normal(0, ANCC_INIT_SPREAD_STD, n_particles)
    rate = init_rate + np.random.normal(0, ANCC_INIT_RATE_STD, n_particles)
    w = np.ones(n_particles) / n_particles

    md_vals = hw.md[eval_idx]
    z_vals = hw.z[eval_idx]
    gr_vals = hw.gr[eval_idx]
    prev_md = float(hw.md[last_known_pos])

    pred_tvts = np.empty(eval_idx.size)
    pred_stds = np.empty(eval_idx.size)

    for i in range(eval_idx.size):
        d_md = md_vals[i] - prev_md
        if d_md <= 0:
            d_md = 1.0
        rate = ANCC_ALPHA * rate + np.random.normal(0, ANCC_RATE_NOISE_STD, n_particles)
        pos = pos + rate * d_md + np.random.normal(0, ANCC_POS_NOISE_STD, n_particles)
        tvt_est = pos - z_vals[i]
        tvt_clipped = np.clip(tvt_est, tvt_min - 50, tvt_max + 50)
        pos = tvt_clipped + z_vals[i]
        if not np.isnan(gr_vals[i]):
            expected_gr = np.interp(tvt_clipped, tw_tvt, tw_gr)
            diff = gr_vals[i] - expected_gr
            lik = np.exp(-0.5 * (diff / gr_sigma) ** 2)
            lik = np.maximum(lik, 1e-300)
            w *= lik
            w_sum = w.sum()
            if w_sum > 0:
                w /= w_sum
            else:
                w[:] = 1.0 / n_particles
        n_eff = 1.0 / np.sum(w ** 2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            cum = np.cumsum(w)
            u = (np.arange(n_particles) + np.random.uniform()) / n_particles
            idx = np.searchsorted(cum, u)
            pos = pos[idx]
            rate = rate[idx]
            w[:] = 1.0 / n_particles
            pos += np.random.normal(0, ANCC_ROUGHENING_STD_POS, n_particles)
            rate += np.random.normal(0, ANCC_ROUGHENING_STD_RATE, n_particles)
        tvt_weighted = np.average(pos - z_vals[i], weights=w)
        pred_tvts[i] = tvt_weighted
        pred_stds[i] = np.sqrt(
            np.average((pos - z_vals[i] - tvt_weighted) ** 2, weights=w)
        )
        prev_md = md_vals[i]

    return pred_tvts, pred_stds


# ---------------------------------------------------------------------------
# Public single-well API. Keep the source notebook signatures so a port can
# call us as a drop-in replacement. The source passes pandas DataFrames; we
# accept either pandas or polars by sniffing for the polars columns API.
# ---------------------------------------------------------------------------


def _to_hw_arrays(hw: Any) -> _HWArrays:
    """Accept a polars DataFrame, a pandas DataFrame, or an _HWArrays."""
    if isinstance(hw, _HWArrays):
        return hw
    # polars duck-typing
    if isinstance(hw, pl.DataFrame):
        return _hw_from_polars(hw)
    # pandas: avoid an explicit import to keep the module pandas-free in
    # production; rely on attribute presence.
    if hasattr(hw, "values") and hasattr(hw, "columns"):
        # pandas frame
        return _HWArrays(
            md=np.asarray(hw["MD"].values, dtype=np.float64),
            x=np.asarray(hw["X"].values, dtype=np.float64) if "X" in hw.columns else np.zeros(len(hw)),
            y=np.asarray(hw["Y"].values, dtype=np.float64) if "Y" in hw.columns else np.zeros(len(hw)),
            z=np.asarray(hw["Z"].values, dtype=np.float64),
            gr=np.asarray(hw["GR"].values, dtype=np.float64),
            tvt_input=np.asarray(hw["TVT_input"].values, dtype=np.float64),
        )
    raise TypeError(f"Unsupported hw type: {type(hw)!r}")


def run_pf_z_velocity(
    hw: Any,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    n_particles: int = PF_N_PARTICLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-well TVT particle filter with Z-velocity coupling.

    Faithful port of cell 5 of the source notebook. Returns (pred, std)
    arrays of length len(eval_zone). Caller is responsible for seeding
    ``np.random`` before invoking if reproducibility is desired.
    """
    hw_arr = _to_hw_arrays(hw)
    return _run_pf_z_velocity_inner(
        hw_arr,
        np.asarray(tw_tvt, dtype=np.float64),
        np.asarray(tw_gr, dtype=np.float64),
        n_particles=n_particles,
    )


def run_pf_ancc(
    hw: Any,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    n_particles: int = ANCC_N_PARTICLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-well ANCC particle filter tracking S = TVT + Z.

    Faithful port of cell 6. Returns (pred, std) over the eval rows.
    """
    hw_arr = _to_hw_arrays(hw)
    return _run_pf_ancc_inner(
        hw_arr,
        np.asarray(tw_tvt, dtype=np.float64),
        np.asarray(tw_gr, dtype=np.float64),
        n_particles=n_particles,
    )


# Re-export the helpers under their source names for parity tests / drop-in:
pf_calibrate_gr_sigma = _calibrate_gr_sigma
pf_estimate_init_velocity = _estimate_init_velocity
pf_learn_z_beta = _learn_z_beta
ancc_estimate_init_rate = _ancc_estimate_init_rate


# ---------------------------------------------------------------------------
# Parallel-over-wells driver.
#
# The driver pickles the per-well arrays (small — typical well is < 200 KB)
# to worker processes and runs both PFs back-to-back per well. Each worker
# re-seeds np.random.seed(42) before each well, so output is independent of
# scheduling.
#
# We use the "fork" start method on Linux/macOS for low launch overhead.
# Workers do *not* need to pickle the typewell since each well has its own
# typewell.
# ---------------------------------------------------------------------------


def _worker_one_well(payload: tuple[str, dict, dict, int, int]) -> tuple[str, dict[str, np.ndarray]]:
    """Worker: run both PFs on one well.

    payload is (well_id, hw_dict, tw_dict, n_particles, seed).
    hw_dict / tw_dict carry pre-extracted numpy arrays so the worker does
    no DataFrame work.
    """
    well_id, hw_dict, tw_dict, n_particles, seed = payload
    hw = _HWArrays(
        md=hw_dict["md"],
        x=hw_dict.get("x", np.zeros_like(hw_dict["md"])),
        y=hw_dict.get("y", np.zeros_like(hw_dict["md"])),
        z=hw_dict["z"],
        gr=hw_dict["gr"],
        tvt_input=hw_dict["tvt_input"],
    )
    tw_tvt = tw_dict["tvt"]
    tw_gr = tw_dict["gr"]

    # Per-well determinism. Both PFs share the same seed; this matches the
    # source notebook which calls np.random.seed(42) once at the top of cell
    # 2 and lets the PFs share the global stream. By reseeding per well we
    # decouple wells from each other (so the output is independent of
    # processing order) while still being reproducible.
    np.random.seed(seed)
    pf_z_pred, pf_z_std = _run_pf_z_velocity_inner(hw, tw_tvt, tw_gr, n_particles=n_particles)

    np.random.seed(seed)
    pf_ancc_pred, pf_ancc_std = _run_pf_ancc_inner(hw, tw_tvt, tw_gr, n_particles=n_particles)

    eval_idx = np.flatnonzero(hw.eval_mask)
    return well_id, {
        "pf_z_pred": pf_z_pred.astype(np.float32),
        "pf_z_std": pf_z_std.astype(np.float32),
        "pf_ancc_pred": pf_ancc_pred.astype(np.float32),
        "pf_ancc_std": pf_ancc_std.astype(np.float32),
        "eval_idx": eval_idx.astype(np.int32),
    }


def _build_payload(
    wid: str,
    hw_df: pl.DataFrame,
    tw_df: pl.DataFrame,
    n_particles: int,
    seed: int,
) -> tuple[str, dict, dict, int, int]:
    hw_dict = {
        "md": hw_df["MD"].to_numpy().astype(np.float64),
        "z": hw_df["Z"].to_numpy().astype(np.float64),
        "gr": hw_df["GR"].to_numpy().astype(np.float64),
        "tvt_input": hw_df["TVT_input"].to_numpy().astype(np.float64),
    }
    if "X" in hw_df.columns:
        hw_dict["x"] = hw_df["X"].to_numpy().astype(np.float64)
    if "Y" in hw_df.columns:
        hw_dict["y"] = hw_df["Y"].to_numpy().astype(np.float64)
    tw_dict = {
        "tvt": tw_df["TVT"].to_numpy().astype(np.float64),
        "gr": tw_df["GR"].to_numpy().astype(np.float64),
    }
    return wid, hw_dict, tw_dict, n_particles, seed


def run_pfs_for_wells(
    well_dfs: dict[str, pl.DataFrame],
    typewell_dfs: dict[str, pl.DataFrame],
    n_workers: int = -1,
    n_particles: int = PF_N_PARTICLES,
    seed: int = RANDOM_STATE,
    chunksize: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """Run both particle filters across many wells in parallel.

    Parameters
    ----------
    well_dfs:
        Mapping of well_id -> horizontal well polars DataFrame (must contain
        MD, Z, GR, TVT_input columns; X and Y are read if present).
    typewell_dfs:
        Mapping of well_id -> typewell polars DataFrame (TVT, GR columns).
        A well_id present in ``well_dfs`` but missing here is skipped.
    n_workers:
        Worker count. ``-1`` means ``os.cpu_count()``. ``0`` or ``1`` runs
        sequentially in-process (useful for debugging / pickling-sensitive
        callers).
    n_particles:
        Particle count for both PFs. Default 500 matches the source.
    seed:
        Seed used inside each worker before each well. Default 42 (source).
    chunksize:
        ``imap_unordered`` chunksize. 1 keeps progress reporting fine-
        grained; larger amortises per-task overhead for very fast wells.

    Returns
    -------
    dict
        ``{well_id: {pf_z_pred, pf_z_std, pf_ancc_pred, pf_ancc_std, eval_idx}}``.
        Wells with empty eval zones still appear, with empty arrays.
    """
    if n_workers is None or n_workers < 0:
        n_workers = os.cpu_count() or 1

    payloads: list[tuple[str, dict, dict, int, int]] = []
    for wid, hw_df in well_dfs.items():
        tw_df = typewell_dfs.get(wid)
        if tw_df is None or tw_df.height < 2:
            continue
        if not {"TVT", "GR"}.issubset(set(tw_df.columns)):
            continue
        if not {"MD", "Z", "GR", "TVT_input"}.issubset(set(hw_df.columns)):
            continue
        payloads.append(_build_payload(wid, hw_df, tw_df, n_particles, seed))

    if n_workers <= 1:
        out: dict[str, dict[str, np.ndarray]] = {}
        for payload in payloads:
            wid, res = _worker_one_well(payload)
            out[wid] = res
        return out

    ctx = mp.get_context("fork")
    out = {}
    with ctx.Pool(n_workers) as pool:
        for wid, res in pool.imap_unordered(_worker_one_well, payloads, chunksize=chunksize):
            out[wid] = res
    return out


# ---------------------------------------------------------------------------
# Convenience: load wells from a directory (matches the kernel's data layout)
# ---------------------------------------------------------------------------


def load_wells_from_dir(
    data_dir: Path,
    well_ids: list[str] | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Load the {wid}__horizontal_well.csv / {wid}__typewell.csv pairs.

    If ``well_ids`` is None, loads every horizontal_well CSV in ``data_dir``.
    """
    data_dir = Path(data_dir)
    if well_ids is None:
        well_ids = sorted(
            p.name.split("__", 1)[0]
            for p in data_dir.glob("*__horizontal_well.csv")
        )
    hw_dfs: dict[str, pl.DataFrame] = {}
    tw_dfs: dict[str, pl.DataFrame] = {}
    for wid in well_ids:
        hw_path = data_dir / f"{wid}__horizontal_well.csv"
        tw_path = data_dir / f"{wid}__typewell.csv"
        if not (hw_path.exists() and tw_path.exists()):
            continue
        hw_dfs[wid] = pl.read_csv(
            hw_path,
            null_values=["", "NA", "NaN", "nan", "null"],
            schema_overrides={
                "MD": pl.Float64, "X": pl.Float64, "Y": pl.Float64,
                "Z": pl.Float64, "GR": pl.Float64, "TVT_input": pl.Float64,
                "TVT": pl.Float64,
            },
            infer_schema_length=2000,
        )
        tw_dfs[wid] = pl.read_csv(
            tw_path,
            null_values=["", "NA", "NaN", "nan", "null"],
            schema_overrides={"TVT": pl.Float64, "GR": pl.Float64},
            infer_schema_length=2000,
        )
    return hw_dfs, tw_dfs


__all__ = [
    "run_pf_z_velocity",
    "run_pf_ancc",
    "run_pfs_for_wells",
    "load_wells_from_dir",
    "pf_calibrate_gr_sigma",
    "pf_estimate_init_velocity",
    "pf_learn_z_beta",
    "ancc_estimate_init_rate",
    "PF_N_PARTICLES",
    "ANCC_N_PARTICLES",
    "RANDOM_STATE",
]
