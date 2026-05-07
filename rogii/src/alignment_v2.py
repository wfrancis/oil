"""rogii.alignment_v2 — DTW v2 with bias correction and multi-anchor warping.

Diagnostic motivation
---------------------
Per-well audit showed v3 RMSE is bias-dominated (mean ~63 ft of which +53 is
mean bias). Two structural fixes:

1. **Per-well GR baseline calibration.** The cased section is a free supervised
   probe: at every finite ``TVT_input``, the typewell GR at that TVT is known
   by interpolation. ``offset = median(h_GR_cased) - median(t_GR_interp_cased)``
   is subtracted from the horizontal trace before DTW. This neutralises
   tool/mud/borehole GR scaling that window-z-score does not fully cancel.

2. **Multi-anchor DTW.** v3 anchors only at the *last* finite TVT_input row,
   throwing away hundreds-thousands of supervised matches. v2 treats every
   finite TVT_input row as a *hard* tie-point. The lateral is split into
   anchor-bracketed segments; DTW runs per segment with FIXED start AND end
   typewell indices. Open tail past the last anchor uses forward-only DTW.

3. **Optional extended Itakura step pattern.** Default off; enables steps
   ``{(1,0),(1,1),(1,2),(1,3)}`` for laterals with rapid TVT change.

Public API: ``predict_well_dtw_v2`` (drop-in for v3),
            ``multi_anchor_dtw``, ``calibrate_gr_offset``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl

from alignment import (  # noqa: F401  reuse helpers from v3
    _HAS_NUMBA,
    _INF,
    _njit,
    _safe_window_zscore,
    _to_numpy_col,
    dtw_align_gr,
    predict_well_dtw,
)

logger = logging.getLogger("rogii.alignment_v2")

# Min cased anchors for calibration to engage; below this the median is too
# noisy to estimate a reliable global offset.
_MIN_CALIB_ANCHORS = 20

# Min anchors for multi-anchor DTW; below this defer to v3 single-anchor.
_MIN_MULTI_ANCHOR_ANCHORS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nearest_typewell_index(
    target_tvt: np.ndarray, sorted_typewell_tvt: np.ndarray
) -> np.ndarray:
    """Vectorised nearest-neighbour index in a sorted typewell TVT array.

    NaN targets map to ``-1``. For each finite target returns the index ``j``
    minimising ``|sorted_typewell_tvt[j] - target_tvt|``.
    """
    target_tvt = np.asarray(target_tvt, dtype=np.float64)
    n_t = sorted_typewell_tvt.shape[0]
    if n_t == 0 or target_tvt.size == 0:
        return np.full(target_tvt.shape, -1, dtype=np.int64)

    out = np.full(target_tvt.shape, -1, dtype=np.int64)
    finite = np.isfinite(target_tvt)
    if not finite.any():
        return out

    pos = np.searchsorted(sorted_typewell_tvt, target_tvt[finite], side="left")
    pos = np.clip(pos, 0, n_t - 1)
    pos_left = np.clip(pos - 1, 0, n_t - 1)
    right_better = np.abs(sorted_typewell_tvt[pos] - target_tvt[finite]) <= np.abs(
        sorted_typewell_tvt[pos_left] - target_tvt[finite]
    )
    out[finite] = np.where(right_better, pos, pos_left).astype(np.int64, copy=False)
    return out


def _prepare_typewell(
    typewell_gr: np.ndarray, typewell_tvt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sort typewell by TVT ascending; drop non-finite + duplicate-TVT rows.

    Mirrors the prep block in ``alignment.dtw_align_gr`` so calibration and
    multi-anchor DTW share identical preprocessing.
    """
    t_gr = np.asarray(typewell_gr, dtype=np.float64)
    t_tvt = np.asarray(typewell_tvt, dtype=np.float64)
    if t_tvt.size != t_gr.size:
        m = min(t_tvt.size, t_gr.size)
        t_tvt = t_tvt[:m]
        t_gr = t_gr[:m]
    if t_tvt.size == 0:
        return t_gr, t_tvt

    order = np.argsort(t_tvt, kind="stable")
    t_tvt = t_tvt[order]
    t_gr = t_gr[order]
    finite = np.isfinite(t_tvt) & np.isfinite(t_gr)
    t_tvt = t_tvt[finite]
    t_gr = t_gr[finite]
    if t_tvt.size >= 2:
        keep = np.concatenate(([True], np.diff(t_tvt) > 0))
        t_tvt = t_tvt[keep]
        t_gr = t_gr[keep]
    return t_gr, t_tvt


def _linear_fill_segment(
    out: np.ndarray, i_a: int, i_b: int, tvt_a: float, tvt_b: float
) -> None:
    """In-place linear TVT fill between two anchor rows (interior only)."""
    L = i_b - i_a
    if L <= 1:
        return
    for k in range(1, L):
        out[i_a + k] = tvt_a + (tvt_b - tvt_a) * (k / L)


# ---------------------------------------------------------------------------
# 1) GR baseline calibration
# ---------------------------------------------------------------------------
def calibrate_gr_offset(
    horizontal_gr: np.ndarray,
    horizontal_tvt_input: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray,
) -> float:
    """Additive GR offset to subtract from the horizontal GR trace.

    Cased-section probe: at every finite ``TVT_input`` row the corresponding
    typewell GR is recoverable by linear interpolation. The constant offset
    between the two sources is estimated as
    ``median(h_GR - t_GR_at_matched_TVT)`` over those rows.

    Robustness
    ----------
    * Returns ``0.0`` (no calibration) if fewer than ``_MIN_CALIB_ANCHORS``
      finite-TVT_input rows are available.
    * Skips rows whose matched TVT lies outside the typewell range — interp
      saturation at the boundary is unreliable for a baseline.
    * Skips rows with non-finite GR at either source.
    """
    h_gr = np.asarray(horizontal_gr, dtype=np.float64)
    h_tvt = np.asarray(horizontal_tvt_input, dtype=np.float64)
    t_gr_in = np.asarray(typewell_gr, dtype=np.float64)
    t_tvt_in = np.asarray(typewell_tvt, dtype=np.float64)

    if h_gr.size == 0 or t_gr_in.size == 0 or t_tvt_in.size == 0:
        return 0.0

    t_gr, t_tvt = _prepare_typewell(t_gr_in, t_tvt_in)
    if t_tvt.size < 2:
        return 0.0

    cased = np.isfinite(h_tvt) & np.isfinite(h_gr)
    if cased.sum() < _MIN_CALIB_ANCHORS:
        if cased.sum() > 0:
            logger.info(
                "calibrate_gr_offset: only %d cased anchors (< %d); skip.",
                int(cased.sum()),
                _MIN_CALIB_ANCHORS,
            )
        return 0.0

    h_tvt_cased = h_tvt[cased]
    h_gr_cased = h_gr[cased]

    in_range = (h_tvt_cased >= t_tvt[0]) & (h_tvt_cased <= t_tvt[-1])
    if in_range.sum() < _MIN_CALIB_ANCHORS:
        logger.info(
            "calibrate_gr_offset: only %d in-range anchors; skip.",
            int(in_range.sum()),
        )
        return 0.0

    t_gr_at_h = np.interp(h_tvt_cased[in_range], t_tvt, t_gr)
    deltas = h_gr_cased[in_range] - t_gr_at_h
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return 0.0

    offset = float(np.median(deltas))
    logger.info(
        "calibrate_gr_offset: offset=%.3f over %d anchors (std=%.3f).",
        offset,
        deltas.size,
        float(np.std(deltas)),
    )
    return offset


# ---------------------------------------------------------------------------
# 2) Constrained-segment DTW kernel: BOTH start AND end anchors fixed.
# ---------------------------------------------------------------------------
@_njit(cache=True, fastmath=True)
def _dtw_segment(
    h: np.ndarray,
    t: np.ndarray,
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
    band: int,
    extended: bool,
) -> np.ndarray:
    """DTW on h[i_start..i_end] with FIXED endpoints in the typewell.

    Path constraints:
        path[0]  = (i_start, j_start)
        path[-1] = (i_end,   j_end)
    Step pattern (i, j) -> {(i+1, j), (i+1, j+1), (i+1, j+2)} (extended adds
    (i+1, j+3)).

    Returns int64 ``j_path`` of length ``i_end - i_start + 1`` whose k-th
    entry is the warped typewell index for horizontal row ``i_start + k``.

    Returns a length-matching all-(-1) array if the segment is infeasible
    (e.g. j_end - j_start exceeds reachable steps, or DP found no path).
    """
    n_h = h.shape[0]
    n_t = t.shape[0]
    out_len = i_end - i_start + 1
    if out_len <= 0 or n_h == 0 or n_t == 0:
        return np.full(0, -1, dtype=np.int64)

    out = np.full(out_len, -1, dtype=np.int64)

    if (
        i_start < 0
        or i_start >= n_h
        or i_end < 0
        or i_end >= n_h
        or j_start < 0
        or j_start >= n_t
        or j_end < 0
        or j_end >= n_t
    ):
        return out

    delta = j_end - j_start
    if delta < 0:
        return out  # backward typewell motion — caller falls back

    per_step_max = 3 if extended else 2
    L = out_len
    if L == 1:
        if j_start == j_end:
            out[0] = j_start
        return out

    if delta > per_step_max * (L - 1):
        return out  # cannot reach j_end from j_start in L-1 steps

    D = np.full((L, n_t), _INF, dtype=np.float64)
    P = np.full((L, n_t), -1, dtype=np.int8)

    diff0 = h[i_start] - t[j_start]
    D[0, j_start] = diff0 * diff0

    denom = L - 1
    slope_num = delta

    for k in range(1, L):
        jc = j_start + (slope_num * k) // denom

        j_lo = jc - band
        if j_lo < j_start:
            j_lo = j_start

        j_hi = jc + band
        # Forward reachability: at most per_step_max * k from start.
        j_max_reach = j_start + per_step_max * k
        if j_hi > j_max_reach:
            j_hi = j_max_reach

        if k == L - 1:
            # Last row: pin to end anchor.
            j_lo = j_end
            j_hi = j_end

        if j_hi >= n_t:
            j_hi = n_t - 1

        if k < L - 1:
            # Reverse reachability: from row k we have (L-1-k) steps to reach
            # j_end with step <= per_step_max, and the path is non-decreasing,
            # so j must be in [j_end - per_step_max*(L-1-k), j_end].
            steps_left = L - 1 - k
            j_lo_back = j_end - per_step_max * steps_left
            if j_lo_back > j_lo:
                j_lo = j_lo_back
            if j_hi > j_end:
                j_hi = j_end

        if j_lo > j_hi:
            return np.full(out_len, -1, dtype=np.int64)

        h_val = h[i_start + k]

        for j in range(j_lo, j_hi + 1):
            best = _INF
            arg = -1

            v0 = D[k - 1, j]
            if v0 < best:
                best = v0
                arg = 0
            if j - 1 >= 0:
                v1 = D[k - 1, j - 1]
                if v1 < best:
                    best = v1
                    arg = 1
            if j - 2 >= 0:
                v2 = D[k - 1, j - 2]
                if v2 < best:
                    best = v2
                    arg = 2
            if extended and j - 3 >= 0:
                v3 = D[k - 1, j - 3]
                if v3 < best:
                    best = v3
                    arg = 3

            if best >= _INF:
                continue

            diff = h_val - t[j]
            D[k, j] = best + diff * diff
            P[k, j] = arg

    if D[L - 1, j_end] >= _INF:
        return np.full(out_len, -1, dtype=np.int64)

    j_cur = j_end
    out[L - 1] = j_cur
    for k in range(L - 1, 0, -1):
        step = P[k, j_cur]
        if step == 0:
            j_prev = j_cur
        elif step == 1:
            j_prev = j_cur - 1
        elif step == 2:
            j_prev = j_cur - 2
        elif step == 3:
            j_prev = j_cur - 3
        else:
            return np.full(out_len, -1, dtype=np.int64)
        out[k - 1] = j_prev
        j_cur = j_prev

    if out[0] != j_start:
        return np.full(out_len, -1, dtype=np.int64)

    return out


# ---------------------------------------------------------------------------
# 2b) Forward DTW with optional extended step pattern (open-tail kernel).
# ---------------------------------------------------------------------------
@_njit(cache=True, fastmath=True)
def _dtw_forward_ext(
    h: np.ndarray,
    t: np.ndarray,
    i0: int,
    j0: int,
    band: int,
    extended: bool,
    slope_num_override: int = -1,
) -> np.ndarray:
    """Forward DTW from (i0, j0) through end of horizontal series.

    Mirrors ``alignment._dtw_forward`` but with optional extended Itakura
    step pattern controlled by ``extended``. Used for the open tail past
    the last anchor in multi-anchor mode.

    ``slope_num_override``: when >= 0, use this as ``slope_num`` (numerator of
    the band-centre slope). When < 0 (default sentinel), fall back to the v3
    behaviour of ``slope_num = n_t - 1 - j_seed`` (which forces the warp to
    end at the last typewell row — only correct if the lateral truly traverses
    the entire remaining typewell column, which Eagle Ford laterals DO NOT).
    The integrator should pass a dip-derived slope (typically 0 for in-zone
    steering) for v4.
    """
    n_h = h.shape[0]
    n_t = t.shape[0]
    L = n_h - i0
    if L <= 0 or n_t <= 0:
        return np.full(n_h, -1, dtype=np.int64)

    D = np.full((L, n_t), _INF, dtype=np.float64)
    P = np.full((L, n_t), -1, dtype=np.int8)

    j_seed = j0 if 0 <= j0 < n_t else 0
    diff0 = h[i0] - t[j_seed]
    D[0, j_seed] = diff0 * diff0

    per_step_max = 3 if extended else 2

    denom = L - 1 if L > 1 else 1
    if slope_num_override >= 0:
        slope_num = slope_num_override
    else:
        slope_num = n_t - 1 - j_seed
    if slope_num < 0:
        slope_num = 0
    if slope_num > per_step_max * denom:
        slope_num = per_step_max * denom

    for k in range(1, L):
        jc = j_seed + (slope_num * k) // denom
        j_lo = jc - band
        if j_lo < j_seed:
            j_lo = j_seed
        j_hi = jc + band
        j_max_reach = j_seed + per_step_max * k
        if j_hi > j_max_reach:
            j_hi = j_max_reach
        if j_hi >= n_t:
            j_hi = n_t - 1

        h_val = h[i0 + k]
        for j in range(j_lo, j_hi + 1):
            best = _INF
            arg = -1
            v0 = D[k - 1, j]
            if v0 < best:
                best = v0
                arg = 0
            if j - 1 >= 0:
                v1 = D[k - 1, j - 1]
                if v1 < best:
                    best = v1
                    arg = 1
            if j - 2 >= 0:
                v2 = D[k - 1, j - 2]
                if v2 < best:
                    best = v2
                    arg = 2
            if extended and j - 3 >= 0:
                v3 = D[k - 1, j - 3]
                if v3 < best:
                    best = v3
                    arg = 3

            if best >= _INF:
                continue
            diff = h_val - t[j]
            D[k, j] = best + diff * diff
            P[k, j] = arg

    out = np.full(n_h, -1, dtype=np.int64)
    last_row = D[L - 1]
    j_end = -1
    best_end = _INF
    for j in range(n_t):
        v = last_row[j]
        if v < best_end:
            best_end = v
            j_end = j
    if j_end < 0:
        return out

    j_cur = j_end
    out[i0 + L - 1] = j_cur
    for k in range(L - 1, 0, -1):
        step = P[k, j_cur]
        if step == 0:
            j_prev = j_cur
        elif step == 1:
            j_prev = j_cur - 1
        elif step == 2:
            j_prev = j_cur - 2
        elif step == 3:
            j_prev = j_cur - 3
        else:
            for kk in range(k):
                out[i0 + kk] = -1
            return out
        out[i0 + k - 1] = j_prev
        j_cur = j_prev
    return out


# ---------------------------------------------------------------------------
# 3) Multi-anchor DTW orchestrator
# ---------------------------------------------------------------------------
def multi_anchor_dtw(
    horizontal_gr: np.ndarray,
    horizontal_tvt_input: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    band_pct: float = 0.25,
    extended_step_pattern: bool = False,
) -> np.ndarray:
    """DTW with hard tie-points at every finite ``TVT_input`` row.

    Anchors A_0 < ... < A_{m-1} are the indices in the horizontal series
    where ``TVT_input`` is finite. Each anchor maps to its nearest typewell
    index via TVT lookup. For every consecutive pair we run a constrained
    DTW segment with FIXED start and end. The open tail past A_{m-1} runs
    forward-only DTW. Pre-A_0 rows are linearly extrapolated.

    Fallbacks
    ---------
    * 0 or 1 anchor -> defer to v3 single-anchor ``dtw_align_gr``.
    * Backward typewell motion implied by anchors -> linear-interp segment.
    * Segment infeasible / DP failed -> linear-interp segment.

    Final guarantee: known TVT_input is never overwritten.
    """
    h_gr = np.asarray(horizontal_gr, dtype=np.float64)
    h_tvt = np.asarray(horizontal_tvt_input, dtype=np.float64)
    n_h = h_gr.size
    if n_h == 0:
        return np.empty(0, dtype=np.float64)

    t_gr_clean, t_tvt_clean = _prepare_typewell(typewell_gr, typewell_tvt)
    if t_tvt_clean.size < 2:
        logger.warning(
            "multi_anchor_dtw: typewell unusable (size=%d); v3 fallback.",
            int(t_tvt_clean.size),
        )
        return dtw_align_gr(h_gr, h_tvt, typewell_gr, typewell_tvt, band_pct=band_pct)

    finite_h = np.isfinite(h_tvt)
    n_anchors = int(finite_h.sum())

    out = np.full(n_h, np.nan, dtype=np.float64)
    if n_anchors < _MIN_MULTI_ANCHOR_ANCHORS:
        logger.info(
            "multi_anchor_dtw: only %d anchor(s); v3 fallback.", n_anchors
        )
        return dtw_align_gr(h_gr, h_tvt, typewell_gr, typewell_tvt, band_pct=band_pct)

    anchor_idx = np.flatnonzero(finite_h).astype(np.int64)
    anchor_tvt = h_tvt[anchor_idx]
    anchor_j = _nearest_typewell_index(anchor_tvt, t_tvt_clean)

    win = max(11, t_gr_clean.size // 50, n_h // 200)
    h_z = _safe_window_zscore(h_gr, win)
    t_z = _safe_window_zscore(t_gr_clean, win)

    # Band must be in typewell-index units; using max(n_h, n_t) often makes
    # the band exceed the entire typewell, neutralising the slope constraint.
    # Use t_z.size (typewell length) only and cap at t_z.size // 4 so the band
    # is genuinely a constraint. v3-style band restored only when explicitly
    # asked via band_pct >= 0.5 (keeps regression tests valid).
    if band_pct >= 0.5:
        band = max(8, int(band_pct * max(n_h, t_z.size)))
    else:
        band = max(20, min(int(band_pct * t_z.size), t_z.size // 4))

    # Pre-fill anchors with their known TVT.
    out[anchor_idx] = anchor_tvt

    n_seg_ok = 0
    n_seg_fallback = 0
    for k in range(n_anchors - 1):
        i_a, i_b = int(anchor_idx[k]), int(anchor_idx[k + 1])
        j_a, j_b = int(anchor_j[k]), int(anchor_j[k + 1])

        if i_b - i_a <= 1:
            continue  # adjacent anchors — nothing between them

        if j_b < j_a:
            logger.warning(
                "multi_anchor_dtw: anchor pair %d->%d implies backward typewell "
                "motion (j %d -> %d); linear-interp.",
                i_a, i_b, j_a, j_b,
            )
            _linear_fill_segment(out, i_a, i_b, anchor_tvt[k], anchor_tvt[k + 1])
            n_seg_fallback += 1
            continue

        seg = _dtw_segment(
            h_z, t_z, i_a, i_b, j_a, j_b, band, bool(extended_step_pattern)
        )
        if seg.size == 0 or seg[0] < 0:
            logger.warning(
                "multi_anchor_dtw: segment %d->%d (j %d->%d, L=%d) infeasible; "
                "linear-interp.",
                i_a, i_b, j_a, j_b, i_b - i_a + 1,
            )
            _linear_fill_segment(out, i_a, i_b, anchor_tvt[k], anchor_tvt[k + 1])
            n_seg_fallback += 1
            continue

        # Map warped indices to TVT for the segment interior.
        for off in range(1, i_b - i_a):
            out[i_a + off] = float(t_tvt_clean[int(seg[off])])
        n_seg_ok += 1

    # Open tail past the last anchor.
    last_i = int(anchor_idx[-1])
    last_j = int(anchor_j[-1])
    if last_i < n_h - 1:
        # Compute dip-derived slope from the trailing anchors (Fix 1 in
        # v4_strategy.md). v3's default `slope_num = n_t-1-j_seed` forces the
        # warp to the last typewell row, mechanically producing the +30-120 ft
        # bias we observed. Eagle Ford laterals stay in zone (~5-15 ft of TVT
        # over thousands of ft of MD), so the typewell-index slope is small.
        L_tail = n_h - last_i  # tail length in horizontal rows
        denom_tail = max(L_tail - 1, 1)
        # Trailing dip from last K anchors: median dTVT / dMD over recent anchors.
        K = min(n_anchors, 10)
        if K >= 2:
            recent_tvt = anchor_tvt[-K:]
            recent_idx = anchor_idx[-K:]
            d_tvt = float(recent_tvt[-1] - recent_tvt[0])
            d_idx = float(recent_idx[-1] - recent_idx[0]) or 1.0
            dip_per_row = d_tvt / d_idx
        else:
            dip_per_row = 0.0
        # Median typewell row spacing in TVT units.
        if t_tvt_clean.size >= 2:
            dtvt_per_t_row = float(np.median(np.abs(np.diff(t_tvt_clean))))
            if dtvt_per_t_row <= 0:
                dtvt_per_t_row = 1.0
        else:
            dtvt_per_t_row = 1.0
        # j-progress per horizontal row.
        slope_j_per_row = abs(dip_per_row) / dtvt_per_t_row
        slope_num_override = int(np.clip(round(slope_j_per_row * denom_tail), 0,
                                         3 * denom_tail))
        logger.info(
            "multi_anchor_dtw: tail dip_per_row=%.5f, dtvt_per_t_row=%.4f, "
            "slope_num_override=%d (denom=%d)",
            dip_per_row, dtvt_per_t_row, slope_num_override, denom_tail,
        )
        tail = _dtw_forward_ext(
            h_z, t_z, last_i, last_j, band, bool(extended_step_pattern),
            slope_num_override,
        )
        if (tail >= 0).any():
            valid_idx = np.flatnonzero(tail >= 0)
            for i in valid_idx:
                if i <= last_i:
                    continue
                out[i] = float(t_tvt_clean[int(tail[i])])
        else:
            logger.warning(
                "multi_anchor_dtw: open-tail DTW failed; constant from last anchor."
            )
            out[last_i + 1 :] = float(anchor_tvt[-1])

    # Pre-first-anchor region (rare): linear-extrap from first slope.
    first_i = int(anchor_idx[0])
    if first_i > 0:
        if n_anchors >= 2 and (anchor_idx[1] - anchor_idx[0]) > 0:
            slope = (anchor_tvt[1] - anchor_tvt[0]) / float(
                anchor_idx[1] - anchor_idx[0]
            )
            for i in range(first_i):
                out[i] = float(anchor_tvt[0]) + slope * (i - first_i)
        else:
            out[:first_i] = float(anchor_tvt[0])

    # Pass-through guarantee.
    out[finite_h] = h_tvt[finite_h]

    logger.info(
        "multi_anchor_dtw: anchors=%d ok=%d fallback=%d tail=%d band=%d ext=%s",
        n_anchors, n_seg_ok, n_seg_fallback, n_h - 1 - last_i, band,
        bool(extended_step_pattern),
    )
    return out


# ---------------------------------------------------------------------------
# 4) End-to-end public predictor
# ---------------------------------------------------------------------------
def predict_well_dtw_v2(
    horizontal_df: pl.DataFrame | pd.DataFrame,
    typewell_df: pl.DataFrame | pd.DataFrame,
    *,
    band_pct: float = 0.25,
    use_baseline_calibration: bool = True,
    use_multi_anchor: bool = True,
    extended_step_pattern: bool = False,
) -> np.ndarray:
    """Drop-in replacement for ``predict_well_dtw`` with bias fixes.

    Pipeline
    --------
    1. Extract numpy columns (``GR``, ``TVT_input`` on horizontal; ``GR``,
       ``TVT`` on typewell).
    2. (Optional) Compute per-well GR offset and subtract from horizontal GR.
    3. Multi-anchor DTW (or v3 single-anchor as fallback).
    4. Interpolate any residual NaNs against finite predictions.
    5. Final pass-through guarantee on TVT_input.
    """
    h_gr = _to_numpy_col(horizontal_df, "GR").astype(np.float64, copy=False).copy()
    h_tvt_in = (
        _to_numpy_col(horizontal_df, "TVT_input")
        .astype(np.float64, copy=False)
        .copy()
    )
    t_gr = _to_numpy_col(typewell_df, "GR").astype(np.float64, copy=False)
    t_tvt = _to_numpy_col(typewell_df, "TVT").astype(np.float64, copy=False)

    n = h_gr.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    finite_in = np.isfinite(h_tvt_in)

    # Hard fallback: empty/tiny typewell.
    if t_gr.size < 5 or t_tvt.size < 5:
        logger.warning(
            "predict_well_dtw_v2: typewell too small (gr=%d tvt=%d); "
            "last-known constant fallback.",
            t_gr.size, t_tvt.size,
        )
        if finite_in.any():
            last_val = float(h_tvt_in[np.flatnonzero(finite_in)[-1]])
            out[:] = last_val
            out[finite_in] = h_tvt_in[finite_in]
        else:
            out[:] = 0.0
        return out

    if not finite_in.any():
        logger.warning(
            "predict_well_dtw_v2: TVT_input entirely NaN; defer to v3."
        )
        return predict_well_dtw(horizontal_df, typewell_df)

    # (2) GR baseline calibration.
    if use_baseline_calibration:
        offset = calibrate_gr_offset(h_gr, h_tvt_in, t_gr, t_tvt)
        if offset != 0.0:
            h_gr = h_gr - offset
            logger.info("predict_well_dtw_v2: applied GR offset %.3f.", offset)

    # (3) DTW.
    if use_multi_anchor:
        pred = multi_anchor_dtw(
            horizontal_gr=h_gr,
            horizontal_tvt_input=h_tvt_in,
            typewell_gr=t_gr,
            typewell_tvt=t_tvt,
            band_pct=band_pct,
            extended_step_pattern=extended_step_pattern,
        )
    else:
        pred = dtw_align_gr(
            horizontal_gr=h_gr,
            horizontal_known_tvt=h_tvt_in,
            typewell_gr=t_gr,
            typewell_tvt=t_tvt,
            band_pct=band_pct,
        )

    # (4) NaN-fill via linear interp against finite predictions.
    still_nan = ~np.isfinite(pred)
    if still_nan.any():
        idx = np.arange(n, dtype=np.float64)
        finite_pred = np.isfinite(pred)
        if finite_pred.sum() >= 2:
            pred = pred.copy()
            pred[still_nan] = np.interp(
                idx[still_nan], idx[finite_pred], pred[finite_pred]
            )
        elif finite_in.any():
            const = float(h_tvt_in[np.flatnonzero(finite_in)[-1]])
            pred = pred.copy()
            pred[still_nan] = const

    # (5) Pass-through guarantee.
    pred = pred.copy()
    pred[finite_in] = h_tvt_in[finite_in]
    return pred


__all__ = [
    "calibrate_gr_offset",
    "multi_anchor_dtw",
    "predict_well_dtw_v2",
]
