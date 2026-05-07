"""rogii.alignment — DTW-based TVT prediction for ROGII Wellbore Geology Prediction.

This module implements a constrained Dynamic Time Warping (DTW) aligner that
matches a horizontal well's Gamma Ray (GR) trace to a vertical typewell's GR
trace, then reads off the typewell's TVT (true vertical thickness, ft) at the
warped index to obtain a per-row TVT prediction for the horizontal well.

Design highlights
-----------------
* Sakoe-Chiba banded DTW with an Itakura-style step pattern
  ``{(1,0), (1,1), (1,2)}`` enforcing monotonicity in TVT along the lateral.
* Anchor at the last known ``TVT_input`` value (start of the eval zone) so the
  warp is *forward only* — we don't need to backsolve over the cased section.
* Window z-score on both GR series to neutralise per-well baseline drift and
  scale differences (a known issue in lateral logs).
* Numba JIT for the inner DP loop with a pure-NumPy fallback if Numba is
  unavailable on the target environment.
* Polars-first I/O for the data lake builder; the only pandas usage is at the
  user's discretion in calling code.

This module deliberately does NOT handle multi-fault scenarios (allowing the
warp to step backward across a fault). Faults are deferred to a later
iteration; the contract here is monotonic forward warping post-anchor.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger("rogii.alignment")

# ---------------------------------------------------------------------------
# Numba-or-NumPy import shim. Numba is on the Kaggle Python image, but we keep
# a fallback so unit tests on slim environments still run.
# ---------------------------------------------------------------------------
try:  # pragma: no cover — environment-dependent
    import numba

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    numba = None  # type: ignore[assignment]
    _HAS_NUMBA = False


_INF = np.float64(1e18)  # finite sentinel; avoids NaN propagation in argmin


def _njit(*args: Any, **kwargs: Any):
    """Return ``numba.njit`` if available, else an identity decorator."""
    if _HAS_NUMBA:
        return numba.njit(*args, **kwargs)  # type: ignore[union-attr]

    def _identity(fn):
        return fn

    return _identity


# ---------------------------------------------------------------------------
# GR pre-processing
# ---------------------------------------------------------------------------
def _safe_window_zscore(x: np.ndarray, win: int) -> np.ndarray:
    """Window-z-score a 1-D series with reflect padding.

    Replaces NaNs with the global median first to avoid contaminating window
    stats. Returns a float64 array the same length as ``x``. A small floor on
    the rolling std (1e-3) prevents divide-by-zero on flat segments.

    Complexity: O(n) thanks to cumulative-sum trick.
    Edge cases: all-NaN input -> all-zero output; constant input -> zero output.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x

    # 1) NaN replacement using global median (robust to outliers).
    finite = np.isfinite(x)
    if not finite.all():
        if not finite.any():
            return np.zeros_like(x)
        med = float(np.median(x[finite]))
        x = np.where(finite, x, med)

    win = max(3, min(int(win), n))

    # Reflect-pad so the window is well-defined at boundaries.
    pad = win // 2
    xp = np.pad(x, pad, mode="reflect")

    # Rolling mean / std via cumulative sums (O(n)).
    cs = np.concatenate(([0.0], np.cumsum(xp)))
    cs2 = np.concatenate(([0.0], np.cumsum(xp * xp)))
    # window from i to i+win in the padded array gives a centered window for x[i].
    s1 = cs[win : win + n] - cs[:n]
    s2 = cs2[win : win + n] - cs2[:n]
    mean = s1 / win
    var = np.maximum(s2 / win - mean * mean, 0.0)
    std = np.sqrt(var) + 1e-3
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Constrained DTW kernel (Sakoe-Chiba band + Itakura forward steps)
# ---------------------------------------------------------------------------
@_njit(cache=True, fastmath=True)
def _dtw_forward(
    h: np.ndarray,  # horizontal GR (z-scored), length N_h
    t: np.ndarray,  # typewell GR (z-scored), length N_t
    i0: int,        # anchor index in horizontal
    j0: int,        # anchor index in typewell
    band: int,      # half-width of Sakoe-Chiba band (in typewell index units)
) -> np.ndarray:
    """Forward DTW from the anchor through the end of the horizontal series.

    Step pattern (i, j) -> {(i+1, j), (i+1, j+1), (i+1, j+2)}.
    Returns the warped-typewell-index array of length ``N_h``; entries before
    ``i0`` are filled with -1 (caller fills them with known TVT_input).

    Complexity: O((N_h - i0) * (2 * band + 1)) time and memory.
    """
    n_h = h.shape[0]
    n_t = t.shape[0]

    # Tail length of the warp.
    L = n_h - i0
    if L <= 0 or n_t <= 0:
        out = np.full(n_h, -1, dtype=np.int64)
        return out

    # Cost matrix is dense over the eval tail; band is enforced via INF mask.
    D = np.full((L, n_t), _INF, dtype=np.float64)
    # Backpointer: 0=stay (i+1,j), 1=diag (i+1,j+1), 2=skip (i+1,j+2).
    P = np.full((L, n_t), -1, dtype=np.int8)

    # Seed.
    j_seed = j0 if 0 <= j0 < n_t else 0
    diff0 = h[i0] - t[j_seed]
    D[0, j_seed] = diff0 * diff0

    # Linear band centre: j_centre(k) = j0 + k * (n_t - j0 - 1) / max(L - 1, 1).
    # We march k = 0..L-1 (k = i - i0). The step pattern caps j-advance at 2
    # per horizontal step, so the centre slope must be clipped to [0, 2] and
    # j must lie inside the reachable triangle [j_seed, j_seed + 2*k].
    denom = L - 1 if L > 1 else 1
    slope_num = n_t - 1 - j_seed  # how far typewell can travel total
    if slope_num < 0:
        slope_num = 0
    # Clip slope to <= 2 to stay inside the reachable cone.
    if slope_num > 2 * denom:
        slope_num = 2 * denom

    for k in range(1, L):
        # Centre of band at this row.
        jc = j_seed + (slope_num * k) // denom
        j_lo = jc - band
        if j_lo < j_seed:
            j_lo = j_seed  # cannot go backward in typewell
        j_hi = jc + band
        # Cap at reachable: max j after k steps of size 2 is j_seed + 2*k.
        j_max_reach = j_seed + 2 * k
        if j_hi > j_max_reach:
            j_hi = j_max_reach
        if j_hi >= n_t:
            j_hi = n_t - 1

        h_val = h[i0 + k]
        for j in range(j_lo, j_hi + 1):
            # Three predecessors at row k-1: (j), (j-1), (j-2).
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

            if best >= _INF:
                continue
            diff = h_val - t[j]
            D[k, j] = best + diff * diff
            P[k, j] = arg

    # Backtrack from the cheapest cell on the last row.
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
        # All-INF row — DTW failed; signal with -1 sentinel.
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
        else:
            # No backpointer recorded — bail.
            for kk in range(k):
                out[i0 + kk] = -1
            return out
        out[i0 + k - 1] = j_prev
        j_cur = j_prev
    return out


# ---------------------------------------------------------------------------
# Public DTW alignment API
# ---------------------------------------------------------------------------
def dtw_align_gr(
    horizontal_gr: np.ndarray,
    horizontal_known_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    band_pct: float = 0.15,
) -> np.ndarray:
    """Align a horizontal GR trace to a typewell GR trace and return TVT/row.

    Algorithm
    ---------
    1. Anchor: locate the last horizontal index whose ``TVT_input`` is finite
       (``i_anchor``) and the typewell index whose TVT is closest to that
       value (``j_anchor``). For rows ``i <= i_anchor`` we trust the input.
    2. Run constrained DTW from ``(i_anchor, j_anchor)`` to the end of the
       horizontal series with a Sakoe-Chiba band of half-width
       ``band_pct * max(N_h, N_t)`` and Itakura step pattern
       ``{(1,0), (1,1), (1,2)}`` (no backward typewell motion).
    3. Cost = squared difference of window-z-scored GR values (window size
       max(11, N_t // 50)).
    4. Backtrack the warp; for each horizontal index ``i`` in the eval tail
       return ``typewell_tvt[j*]``.

    Complexity
    ----------
    Time / memory: O((N_h - i_anchor) * band) -- typically << O(N_h * N_t).

    Edge cases handled
    ------------------
    * Empty horizontal or typewell -> all-NaN return.
    * No finite ``TVT_input`` (all-NaN) -> anchor at row 0, typewell index 0.
    * Non-monotonic typewell TVT -> we sort it ascending before the lookup;
      duplicate TVTs collapse to the first occurrence.
    * Typewell shorter than horizontal eval zone -> the band clamps to the
      typewell tail; the last horizontal rows all map to the typewell's last
      sample (which is the right limit-behavior).
    * NaNs in GR -> replaced with the per-series median before z-scoring.
    """
    h_gr = np.asarray(horizontal_gr, dtype=np.float64)
    h_tvt = np.asarray(horizontal_known_tvt, dtype=np.float64)
    t_gr = np.asarray(typewell_gr, dtype=np.float64)
    t_tvt = np.asarray(typewell_tvt, dtype=np.float64)

    n_h = h_gr.size
    if n_h == 0:
        return np.empty(0, dtype=np.float64)
    if t_gr.size == 0 or t_tvt.size == 0:
        logger.warning("Empty typewell — returning NaN predictions.")
        return np.full(n_h, np.nan, dtype=np.float64)

    # Coerce typewell to ascending-monotonic in TVT (some real files are not).
    if t_tvt.size != t_gr.size:
        m = min(t_tvt.size, t_gr.size)
        t_tvt = t_tvt[:m]
        t_gr = t_gr[:m]
    order = np.argsort(t_tvt, kind="stable")
    t_tvt = t_tvt[order]
    t_gr = t_gr[order]
    # Drop duplicate / non-finite TVT samples.
    finite_t = np.isfinite(t_tvt) & np.isfinite(t_gr)
    t_tvt = t_tvt[finite_t]
    t_gr = t_gr[finite_t]
    if t_tvt.size >= 2:
        # Strictly increasing — keep first of any duplicate run.
        keep = np.concatenate(([True], np.diff(t_tvt) > 0))
        t_tvt = t_tvt[keep]
        t_gr = t_gr[keep]
    if t_tvt.size == 0:
        logger.warning("Typewell TVT had no usable samples; returning NaN.")
        return np.full(n_h, np.nan, dtype=np.float64)

    # Anchor selection: last finite TVT_input.
    finite_h = np.isfinite(h_tvt)
    if finite_h.any():
        # last True position
        i_anchor = int(np.flatnonzero(finite_h)[-1])
        anchor_tvt = float(h_tvt[i_anchor])
        j_anchor = int(np.searchsorted(t_tvt, anchor_tvt))
        if j_anchor >= t_tvt.size:
            j_anchor = t_tvt.size - 1
        # nearest neighbour, not just bisect-right.
        if j_anchor > 0 and (
            abs(t_tvt[j_anchor - 1] - anchor_tvt) <= abs(t_tvt[j_anchor] - anchor_tvt)
        ):
            j_anchor -= 1
    else:
        # Pathological: no anchor at all. Start at row 0, typewell row 0.
        i_anchor = 0
        j_anchor = 0

    # Window size: physical scale ~ 11–~typewell/50 samples — enough to wash
    # out a few sand/shale beds but keep formation-scale contrast.
    win = max(11, t_gr.size // 50, n_h // 200)
    h_z = _safe_window_zscore(h_gr, win)
    t_z = _safe_window_zscore(t_gr, win)

    band = max(8, int(band_pct * max(n_h, t_z.size)))

    j_path = _dtw_forward(h_z, t_z, i_anchor, j_anchor, band)

    out = np.full(n_h, np.nan, dtype=np.float64)
    # Pre-fill cased portion with known TVT_input.
    if finite_h.any():
        out[finite_h] = h_tvt[finite_h]

    # Fill the warp tail.
    valid = j_path >= 0
    if valid.any():
        out[valid] = t_tvt[j_path[valid]]
    else:
        logger.warning("DTW returned no valid path; falling back to last anchor.")
        # Constant-fill the tail with the anchor TVT.
        if finite_h.any():
            out[i_anchor:] = float(h_tvt[i_anchor])

    # Don't ever override known input.
    if finite_h.any():
        out[finite_h] = h_tvt[finite_h]

    return out


# ---------------------------------------------------------------------------
# Per-well end-to-end wrapper
# ---------------------------------------------------------------------------
def _to_numpy_col(df: Any, name: str) -> np.ndarray:
    """Return df[name] as a numpy array regardless of pl/pd backend."""
    if isinstance(df, pl.DataFrame):
        if name not in df.columns:
            return np.empty(0, dtype=np.float64)
        return df.get_column(name).to_numpy()
    # pandas
    if name not in df.columns:
        return np.empty(0, dtype=np.float64)
    return np.asarray(df[name].values)


def predict_well_dtw(
    horizontal_df: pl.DataFrame | pd.DataFrame,
    typewell_df: pl.DataFrame | pd.DataFrame,
) -> np.ndarray:
    """End-to-end DTW prediction for a single well.

    Inputs may be Polars or Pandas DataFrames. We extract the columns we need
    and dispatch to ``dtw_align_gr``. Known ``TVT_input`` values are passed
    through unchanged; only NaN rows are predicted.

    Robustness
    ----------
    * If the typewell has fewer than 5 GR samples, falls back to last-known
      TVT_input (constant extrapolation).
    * If ``TVT_input`` is entirely NaN, returns zeros (the contest evaluates
      against ``TVT`` which is positive; a zero baseline is harmless and
      flagged via a warning).
    * Interior gaps (NaN regions before the eval zone) are still filled by
      DTW because the anchor is the *last* finite value, but the resulting
      predictions for those interior gaps are linearly interpolated against
      the surrounding finite values — DTW only writes to positions whose
      TVT_input is NaN AND which sit in the post-anchor tail.
    """
    h_gr = _to_numpy_col(horizontal_df, "GR").astype(np.float64, copy=False)
    h_tvt_in = _to_numpy_col(horizontal_df, "TVT_input").astype(np.float64, copy=False)
    t_gr = _to_numpy_col(typewell_df, "GR").astype(np.float64, copy=False)
    t_tvt = _to_numpy_col(typewell_df, "TVT").astype(np.float64, copy=False)

    n = h_gr.size
    out = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return out

    finite_in = np.isfinite(h_tvt_in)

    # Hard fallback: empty/tiny typewell -> last-known constant or zero.
    if t_gr.size < 5 or t_tvt.size < 5:
        logger.warning(
            "Typewell has < 5 samples (gr=%d tvt=%d); using last-known fallback.",
            t_gr.size,
            t_tvt.size,
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
            "TVT_input is entirely NaN for this well; returning zeros. "
            "Caller should treat this well specially."
        )
        out[:] = 0.0
        return out

    pred = dtw_align_gr(
        horizontal_gr=h_gr,
        horizontal_known_tvt=h_tvt_in,
        typewell_gr=t_gr,
        typewell_tvt=t_tvt,
    )

    # Linear interpolation of any *interior* NaN gap that DTW didn't fill
    # (e.g. NaN strictly before the anchor and not contiguous with the eval
    # tail). We do this only on positions still NaN.
    still_nan = ~np.isfinite(pred)
    if still_nan.any() and finite_in.any():
        idx = np.arange(n, dtype=np.float64)
        finite_pred = np.isfinite(pred)
        if finite_pred.sum() >= 2:
            pred = pred.copy()
            pred[still_nan] = np.interp(
                idx[still_nan], idx[finite_pred], pred[finite_pred]
            )
        else:
            # Not enough anchors to interp — constant fill.
            const = float(h_tvt_in[np.flatnonzero(finite_in)[-1]])
            pred = pred.copy()
            pred[still_nan] = const

    # Don't override known input under any circumstances.
    pred = pred.copy()
    pred[finite_in] = h_tvt_in[finite_in]
    return pred


# ---------------------------------------------------------------------------
# Data lake builder
# ---------------------------------------------------------------------------
def _scan_csv(path: str) -> pl.DataFrame:
    """Read a CSV with permissive schema inference; returns a Polars df."""
    return pl.read_csv(
        path,
        infer_schema_length=2000,
        null_values=["", "NA", "NaN", "nan", "null"],
        truncate_ragged_lines=True,
    )


def build_data_lake(
    train_dir: str = "/kaggle/input/rogii-wellbore-geology-prediction/train/",
    test_dir: str = "/kaggle/input/rogii-wellbore-geology-prediction/test/",
    output_path: str = "/tmp/rogii_lake.parquet",
) -> dict:
    """Scan all per-well CSVs into one Parquet and return a lookup dict.

    Output dict shape:
        { WELLNAME: { 'split': 'train'|'test',
                       'horizontal': pl.DataFrame,
                       'typewell':   pl.DataFrame } }

    The Parquet stack is written as one tall file with a ``__split`` and
    ``__kind`` (``horizontal``/``typewell``) column so subsequent runs can
    skip CSV parsing. We use Polars throughout because pandas is ~10x slower
    for this many small files. If ``train_dir`` doesn't exist, the train
    portion is silently skipped (handles inference-only Kaggle environments).

    Edge cases
    ----------
    * Missing directories -> warn and skip.
    * A horizontal file without a matching typewell -> the dict entry's
      ``typewell`` will be an empty DataFrame.
    * Output Parquet failure -> log warning and continue (the dict is the
      primary deliverable; parquet is a cache).
    """
    out: dict[str, dict[str, Any]] = {}
    frames: list[pl.DataFrame] = []

    for split, base in (("train", train_dir), ("test", test_dir)):
        base_path = Path(base)
        if not base_path.exists():
            logger.warning("Data dir missing: %s (skipping %s split)", base, split)
            continue

        # Two file kinds per well.
        horiz_files = sorted(glob.glob(str(base_path / "*__horizontal_well.csv")))
        type_files = sorted(glob.glob(str(base_path / "*__typewell.csv")))
        type_lookup = {
            Path(p).name.replace("__typewell.csv", ""): p for p in type_files
        }

        for hf in horiz_files:
            wellname = Path(hf).name.replace("__horizontal_well.csv", "")
            try:
                h_df = _scan_csv(hf)
            except Exception as exc:  # pragma: no cover — depends on real files
                logger.warning("Failed to read horizontal %s: %s", hf, exc)
                continue
            tf = type_lookup.get(wellname)
            if tf is not None:
                try:
                    t_df = _scan_csv(tf)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to read typewell %s: %s", tf, exc)
                    t_df = pl.DataFrame()
            else:
                logger.warning("No typewell file for well %s in %s", wellname, split)
                t_df = pl.DataFrame()

            out[wellname] = {"split": split, "horizontal": h_df, "typewell": t_df}

            # Tag and stash for parquet.
            if h_df.height > 0:
                frames.append(
                    h_df.with_columns(
                        pl.lit(wellname).alias("WELLNAME_KEY"),
                        pl.lit(split).alias("__split"),
                        pl.lit("horizontal").alias("__kind"),
                    )
                )
            if t_df.height > 0:
                frames.append(
                    t_df.with_columns(
                        pl.lit(wellname).alias("WELLNAME_KEY"),
                        pl.lit(split).alias("__split"),
                        pl.lit("typewell").alias("__kind"),
                    )
                )

    if frames:
        try:
            # diagonal_relaxed handles heterogeneous columns across splits/kinds.
            stacked = pl.concat(frames, how="diagonal_relaxed")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            stacked.write_parquet(output_path, compression="zstd")
            logger.info(
                "Wrote data lake: %s (%d rows, %d wells)",
                output_path,
                stacked.height,
                len(out),
            )
        except Exception as exc:
            logger.warning("Failed to write parquet cache: %s", exc)
    else:
        logger.warning("No frames collected; skipping parquet write.")

    return out


__all__ = ["dtw_align_gr", "predict_well_dtw", "build_data_lake"]
