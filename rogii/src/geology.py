"""rogii.geology — Stratigraphic priors for ROGII Wellbore Geology Prediction.

Encodes the domain knowledge a Maverick-Basin Eagle Ford geosteerer would
actually use to evaluate a TVT prediction along a horizontal lateral. Sits
between the raw alignment output (e.g. DTW from :mod:`rogii.alignment`) and
the final submission, applying soft Bayesian + hard physical constraints
that reflect the South Texas Cretaceous shelf stratigraphy.

Why a separate module?
----------------------
DTW treats GR as a generic time series. But the EGFDL→BUDA contact is the
single highest-information event in the play: GR drops from ~150–250 API to
~10–25 API over a few feet. Operators essentially never drill more than
~10–20 ft into the Buda (tight micrite, low porosity; laterals land in the
basal EGFDL hot shale). The predicted TVT must respect that floor — soft,
not hard, because steerers do nick the contact during corrections.
Regional dip in the Maverick Basin is 3–5° SE; the *apparent* dip seen by a
lateral depends on hole azimuth and is fittable from the cased section.

Stratigraphic order (top→bottom in TVT, i.e. shallowest→deepest):

    ANCC   — Anacacho Lst (Campanian; often above logged interval)
    ASTNU  — Austin Chalk Upper (Coniacian–Santonian; carbonate, low GR)
    ASTNL  — Austin Chalk Lower (carbonate, low–moderate GR)
    EGFDU  — Eagle Ford Upper (Turonian; mixed marl, intermediate GR)
    EGFDL  — Eagle Ford Lower (Cenomanian–Turonian; organic shale, HIGH GR)
    BUDA   — Buda Lst (Cenomanian; tight micrite, LOW GR — the floor)

Convention: TVT increases downward (deeper = larger TVT). Verified at runtime;
inverted typewells are detected and sign-flipped internally.

Public API: :func:`fit_formation_gr_model`, :func:`gr_to_formation_logprob`,
:func:`constrain_tvt_predictions`, :func:`regional_dip_prior`,
:func:`fault_jump_detector`. All accept Polars *or* Pandas DataFrames, log via
``rogii.geology``, and degrade gracefully when the typewell lacks landmarks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import ndimage, signal, stats

logger = logging.getLogger("rogii.geology")

# ---------------------------------------------------------------------------
# Stratigraphic constants — the ground truth of the play.
# ---------------------------------------------------------------------------
# Shallow → deep. This order is enforced; reordering breaks physical layering.
FORMATION_ORDER: tuple[str, ...] = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
)

# Typical S. Texas Eagle Ford GR ranges (API), used only as weak fallbacks
# when the typewell has insufficient data for a formation. These are NOT
# substituted blindly — the typewell-derived stats win whenever available.
_FALLBACK_GR_RANGES: dict[str, tuple[float, float]] = {
    "ANCC":  ( 30.0,  90.0),   # marly chalk
    "ASTNU": ( 20.0,  60.0),   # clean chalk
    "ASTNL": ( 25.0,  80.0),   # somewhat marlier
    "EGFDU": ( 60.0, 130.0),   # intermediate marl
    "EGFDL": (130.0, 250.0),   # organic-rich hot shale
    "BUDA":  ( 10.0,  35.0),   # tight micrite — sharp drop
}

# Soft slack (ft) for the floor and ceiling clips in
# :func:`constrain_tvt_predictions`. Encoded as physically reasonable
# excursions, not numerical fudge factors:
#   * 10 ft below EGFDL/BUDA contact = the deepest a steerer would ever
#     allow the bit before pulling up. Real drilling logs occasionally go
#     this far during fault crossings or steering corrections.
#   * 50 ft of slack above the shallowest typewell point absorbs the case
#     where the typewell does not log Anacacho but the lateral is in it.
_FLOOR_SLACK_FT: float = 10.0
_CEILING_SLACK_FT: float = 50.0

# Default smoother sigma in *rows* (1 row ≈ 1 ft along MD). σ=5 keeps bed-scale
# detail while killing single-sample alignment jitter.
_SMOOTH_SIGMA_DEFAULT: float = 5.0

# Posterior-nudge gain: how aggressively we move toward the formation-implied
# TVT range when the GR strongly disagrees with the predicted formation. 0
# disables; 1.0 snaps. 0.25 is a soft Bayesian update — empirically a good
# trade-off because GR alone is *not* sufficient to identify a formation
# (overlap is real), so we want gentle pressure, not snapping.
_POSTERIOR_NUDGE_GAIN: float = 0.25

# Log-prob floor for downstream stability (avoid -inf when GR falls outside
# every formation's support).
_LOGPROB_FLOOR: float = -1e6


# ---------------------------------------------------------------------------
# I/O shim: accept Polars or Pandas at the boundary.
# ---------------------------------------------------------------------------
def _as_numpy(df: Any, name: str) -> np.ndarray:
    """Return ``df[name]`` as float64 numpy regardless of pl/pd backend.

    Returns an empty array if the column doesn't exist. Strings are kept as
    object arrays — the only string column we touch is ``Geology`` and we
    cast it explicitly elsewhere.
    """
    if df is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(df, pl.DataFrame):
        if name not in df.columns:
            return np.empty(0, dtype=np.float64)
        return df.get_column(name).to_numpy()
    # pandas
    if name not in getattr(df, "columns", []):
        return np.empty(0, dtype=np.float64)
    return np.asarray(df[name].values)


def _as_str_array(df: Any, name: str) -> np.ndarray:
    """Return ``df[name]`` as a numpy object array of strings (or empty)."""
    arr = _as_numpy(df, name)
    if arr.size == 0:
        return arr
    # Coerce to str. NaNs (floats) become "nan" — we filter those downstream.
    out = np.array([str(v) for v in arr], dtype=object)
    return out


# ---------------------------------------------------------------------------
# 1. Fit the per-formation GR model from the typewell.
# ---------------------------------------------------------------------------
def fit_formation_gr_model(typewell_df: pl.DataFrame | pd.DataFrame) -> dict:
    """Fit per-formation GR distributions, contact depth, and landing TVT.

    The typewell is the contest-provided Rosetta stone: labelled (TVT, GR,
    Geology) samples in the same region as the lateral. We extract:

    1. Per-formation GR stats (mean, std, p10, p90) — Gaussian fallback.
    2. Per-formation gaussian_kde — non-parametric density (handles e.g.
       bimodal EGFDL hot streaks). Fit only if ≥ 8 samples + non-degenerate σ.
    3. TVT range per formation — observed (min, max) envelope.
    4. EGFDL/BUDA contact TVT — the sharpest GR drop in the play; localised
       via max-negative dGR/dTVT in the EGFDL/BUDA interface (Sav-Gol filter).
       This contact is the cornerstone of every downstream constraint.
    5. Landing-zone TVT — typically 30–80 ft above the contact. Estimated
       as ``contact − 50`` (sign-flipped for inverted typewells), else
       median TVT of the deepest 25 % of EGFDL samples.

    Convention is auto-detected by comparing median TVT of EGFDL vs ASTNU:
    if EGFDL has the smaller median, the typewell stores TVT increasing-up
    and we sign-flip internally (returned values stay in the original sign).

    Edge cases: empty typewell → ``fit=False`` (downstream short-circuits);
    missing formations → absent from maps (callers fall back to
    ``_FALLBACK_GR_RANGES``); EGFDL or BUDA absent → contact ``None`` and
    floor clip is skipped; non-monotonic TVT → sorted before stats.
    """
    model: dict[str, Any] = {
        "formation_order": list(FORMATION_ORDER),
        "tvt_ranges": {},
        "gr_stats": {},
        "gr_kde": {},
        "egfdl_buda_contact_tvt": None,
        "landing_zone_tvt": None,
        "tvt_convention": "increasing_down",  # filled in
        "formation_priors": {},  # P(formation), thickness-weighted
        "fit": False,
    }

    tvt = _as_numpy(typewell_df, "TVT").astype(np.float64, copy=False)
    gr = _as_numpy(typewell_df, "GR").astype(np.float64, copy=False)
    geo = _as_str_array(typewell_df, "Geology")

    if tvt.size == 0 or gr.size == 0 or geo.size == 0:
        logger.warning(
            "fit_formation_gr_model: empty typewell columns "
            "(tvt=%d gr=%d geo=%d); returning unfit model.",
            tvt.size, gr.size, geo.size,
        )
        return model
    n = min(tvt.size, gr.size, geo.size)
    tvt, gr, geo = tvt[:n], gr[:n], geo[:n]

    finite = np.isfinite(tvt) & np.isfinite(gr)
    valid_geo = np.array([g in FORMATION_ORDER for g in geo], dtype=bool)
    keep = finite & valid_geo
    if keep.sum() < 5:
        logger.warning(
            "fit_formation_gr_model: only %d valid samples; returning unfit model.",
            int(keep.sum()),
        )
        return model
    tvt, gr, geo = tvt[keep], gr[keep], geo[keep]

    # Detect convention by comparing median TVT of a known-shallow vs known-deep
    # formation. EGFDL is deeper than ASTNU in physical reality; if our values
    # disagree, the typewell stores TVT inverted (increasing-up).
    deep_label = "EGFDL"
    shallow_label = "ASTNU"
    deep_tvt = tvt[geo == deep_label]
    shallow_tvt = tvt[geo == shallow_label]
    if deep_tvt.size > 0 and shallow_tvt.size > 0:
        if np.median(deep_tvt) < np.median(shallow_tvt):
            model["tvt_convention"] = "increasing_up"
            logger.info(
                "Typewell TVT appears to increase upward "
                "(EGFDL median %.1f < ASTNU median %.1f). Internal sign-flip applied.",
                float(np.median(deep_tvt)), float(np.median(shallow_tvt)),
            )

    # Internal working axis: always increasing-down. We store contact and
    # ranges in the *original* convention by mapping back at the end.
    sign = -1.0 if model["tvt_convention"] == "increasing_up" else 1.0
    tvt_w = sign * tvt  # working depth axis: deeper = larger value

    # ----- Per-formation statistics ------------------------------------------
    total_thickness = 0.0
    raw_thickness: dict[str, float] = {}
    for fm in FORMATION_ORDER:
        mask = geo == fm
        n_fm = int(mask.sum())
        if n_fm == 0:
            continue

        gr_fm = gr[mask]
        tvt_fm = tvt[mask]      # original-convention values for ranges
        tvt_fm_w = tvt_w[mask]  # working axis
        if gr_fm.size < 1:
            continue

        # TVT range in original convention.
        model["tvt_ranges"][fm] = (float(np.min(tvt_fm)), float(np.max(tvt_fm)))

        # Thickness in working-axis units (always positive).
        thick = float(np.max(tvt_fm_w) - np.min(tvt_fm_w))
        raw_thickness[fm] = thick
        total_thickness += thick

        # Robust GR stats.
        mean = float(np.mean(gr_fm))
        std = float(np.std(gr_fm)) if gr_fm.size >= 2 else 5.0
        std = max(std, 1.0)  # floor: 1 API unit. Real tools have ~1 API noise.
        p10 = float(np.percentile(gr_fm, 10)) if gr_fm.size >= 5 else mean - std
        p90 = float(np.percentile(gr_fm, 90)) if gr_fm.size >= 5 else mean + std
        model["gr_stats"][fm] = {
            "mean": mean, "std": std, "p10": p10, "p90": p90,
            "count": n_fm,
        }

        # KDE — only if we have ≥ 8 samples and non-degenerate variance.
        # gaussian_kde requires at least one data point but practically needs
        # several to be informative; we threshold at 8 to avoid wild bandwidths.
        if n_fm >= 8 and np.std(gr_fm) > 1e-6:
            try:
                model["gr_kde"][fm] = stats.gaussian_kde(gr_fm)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("KDE fit failed for %s: %s", fm, exc)

    # ----- Formation priors (thickness-weighted) -----------------------------
    # P(formation) ∝ thickness in the typewell. This is a weak prior; in real
    # geosteering you'd weight by along-MD probability instead, but absent a
    # spatial model the typewell thickness ratio is the right marginal.
    if total_thickness > 0:
        for fm, thick in raw_thickness.items():
            model["formation_priors"][fm] = max(thick / total_thickness, 1e-4)
    # Make sure every fitted formation has *some* prior (uniform fallback).
    for fm in model["gr_stats"]:
        if fm not in model["formation_priors"]:
            model["formation_priors"][fm] = 1.0 / max(len(FORMATION_ORDER), 1)

    # ----- EGFDL/BUDA contact ------------------------------------------------
    contact_orig = _detect_egfdl_buda_contact(tvt, tvt_w, gr, geo, sign)
    model["egfdl_buda_contact_tvt"] = contact_orig

    # ----- Landing-zone TVT (original convention) ----------------------------
    # If contact found, landing is ~50 ft above (shallower) it. "Above" means
    # smaller TVT in the increasing-down convention, i.e. ``contact - 50``;
    # but when the typewell is increasing-up the user-facing convention has
    # "shallower" = larger numerical value, so the operation flips sign.
    if contact_orig is not None:
        if model["tvt_convention"] == "increasing_down":
            model["landing_zone_tvt"] = contact_orig - 50.0
        else:
            model["landing_zone_tvt"] = contact_orig + 50.0
    else:
        # Fallback: median TVT of the deepest 25 % of EGFDL samples.
        if "EGFDL" in model["tvt_ranges"]:
            mask_egfdl = geo == "EGFDL"
            tvt_egfdl_w = tvt_w[mask_egfdl]
            if tvt_egfdl_w.size >= 4:
                cutoff = np.percentile(tvt_egfdl_w, 75)
                deep_25 = tvt_egfdl_w[tvt_egfdl_w >= cutoff]
                if deep_25.size > 0:
                    landing_w = float(np.median(deep_25))
                    model["landing_zone_tvt"] = sign * landing_w  # back to orig

    model["fit"] = True
    logger.info(
        "Fitted formation GR model: %d formations, contact=%s, landing=%s",
        len(model["gr_stats"]),
        f"{model['egfdl_buda_contact_tvt']:.2f}" if contact_orig is not None else "None",
        f"{model['landing_zone_tvt']:.2f}"
        if model["landing_zone_tvt"] is not None else "None",
    )
    return model


def _detect_egfdl_buda_contact(
    tvt_orig: np.ndarray,
    tvt_w: np.ndarray,
    gr: np.ndarray,
    geo: np.ndarray,
    sign: float,
) -> float | None:
    """Locate the EGFDL/BUDA contact TVT (returned in original convention).

    Method: sort EGFDL+BUDA samples by working-axis TVT, compute a Sav-Gol
    smoothed dGR/dTVT, return the row of maximum *negative* slope — the
    steepest GR drop with depth, i.e. the contact (textbook 150–250 →
    10–25 API drop). Search restricted to the deeper 2/3 of the interval
    so EGFDL hot-shale spikes don't confound the maximum.

    Fallback ladder when derivative is unreliable:
      - Only one of {EGFDL, BUDA} labelled → use deepest EGFDL TVT, or
        shallowest BUDA TVT, as the contact proxy.
      - < 5 samples in interval → midpoint of the deepest EGFDL row and
        the next deeper row (label transition).
      - No EGFDL and no BUDA → return ``None``.

    Identical-TVT duplicates are filtered before differentiation.
    """
    has_egfdl = np.any(geo == "EGFDL")
    has_buda = np.any(geo == "BUDA")
    if not has_egfdl and not has_buda:
        return None

    # Order by working axis (deeper = larger).
    order = np.argsort(tvt_w, kind="stable")
    tvt_w_s = tvt_w[order]
    gr_s = gr[order]
    geo_s = geo[order]

    # Indices in the EGFDL+BUDA interval. We want the contiguous slice from the
    # first EGFDL (or first BUDA, whichever is shallower) to the last BUDA
    # (or last EGFDL).
    in_interval = (geo_s == "EGFDL") | (geo_s == "BUDA")
    if not np.any(in_interval):
        return None

    idx_int = np.flatnonzero(in_interval)
    i_lo, i_hi = idx_int[0], idx_int[-1]
    tvt_int = tvt_w_s[i_lo : i_hi + 1]
    gr_int = gr_s[i_lo : i_hi + 1]
    geo_int = geo_s[i_lo : i_hi + 1]

    # Both formations needed for a derivative-based contact. If only one,
    # fall back to label boundary.
    has_both = np.any(geo_int == "EGFDL") and np.any(geo_int == "BUDA")
    if not has_both:
        # Fallback: boundary between most common formation here and the next.
        if np.any(geo_int == "EGFDL"):
            # Use the deepest EGFDL TVT as a proxy contact.
            egfdl_max = float(np.max(tvt_w_s[geo_s == "EGFDL"]))
            return float(sign * egfdl_max)
        # only BUDA
        buda_min = float(np.min(tvt_w_s[geo_s == "BUDA"]))
        return float(sign * buda_min)

    # De-duplicate identical TVTs (sort is stable; keep first).
    if tvt_int.size >= 2:
        keep = np.concatenate(([True], np.diff(tvt_int) > 0))
        tvt_int = tvt_int[keep]
        gr_int = gr_int[keep]
        geo_int = geo_int[keep]

    if tvt_int.size < 5:
        # Not enough to differentiate — fall back to label transition.
        # Find the deepest EGFDL row; contact is right after it.
        egfdl_idx = np.flatnonzero(geo_int == "EGFDL")
        if egfdl_idx.size > 0:
            last_egfdl = egfdl_idx[-1]
            if last_egfdl + 1 < tvt_int.size:
                tvt_contact = 0.5 * (tvt_int[last_egfdl] + tvt_int[last_egfdl + 1])
            else:
                tvt_contact = float(tvt_int[last_egfdl])
            return float(sign * tvt_contact)
        return None

    # Derivative-based detection. SG filter requires window ≤ data and odd.
    win = min(tvt_int.size, 11)
    if win % 2 == 0:
        win -= 1
    win = max(5, win)
    try:
        # dGR/dTVT — note we need uniform spacing or close enough; the SG
        # delta is set to the median spacing for an order-of-magnitude
        # correct slope. Sign of the slope is what we care about.
        dt = float(np.median(np.diff(tvt_int)))
        if dt <= 0:
            dt = 1.0
        d_gr = signal.savgol_filter(
            gr_int, window_length=win, polyorder=1, deriv=1, delta=dt, mode="interp"
        )
    except Exception as exc:  # pragma: no cover — SG can fail on tiny windows
        logger.warning("Savitzky-Golay derivative failed: %s; using np.gradient.", exc)
        d_gr = np.gradient(gr_int, tvt_int)

    # Most negative slope (largest GR drop with depth). Restrict the search to
    # the interface region: the deepest 30 % of the interval, where geology
    # tells us the contact lives. This guards against confounding spikes
    # higher in the EGFDL hot zones.
    n_int = tvt_int.size
    search_start = max(0, n_int // 3)  # search the deeper 2/3
    search_slice = slice(search_start, n_int)
    j_min = int(np.argmin(d_gr[search_slice])) + search_start
    tvt_contact_w = float(tvt_int[j_min])
    return float(sign * tvt_contact_w)


# ---------------------------------------------------------------------------
# 2. P(formation | GR) using KDEs / Gaussians and a thickness prior.
# ---------------------------------------------------------------------------
def gr_to_formation_logprob(gr_value: float, model: dict) -> dict:
    """Return ``{formation: log P(formation | GR)}`` (unnormalised, additive const).

    Bayes:  log P(F|GR) = log P(GR|F) + log P(F) − log P(GR). The marginal
    log P(GR) is constant across F so we drop it — callers (argmax /
    log-ratio) don't need the normalisation.

    Likelihood: KDE if fitted (≥ 8 samples), else Gaussian from per-formation
    stats. If a formation is absent from the model, a Gaussian centred at the
    midpoint of ``_FALLBACK_GR_RANGES[F]`` with σ = (hi − lo) / 4 is used.

    Prior: thickness-weighted, P(F) ∝ TVT-extent of F in the typewell.

    Edge cases: non-finite ``gr_value`` → uniform over known formations;
    KDE evaluation failure → silent fallback to Gaussian; underflow →
    ``_LOGPROB_FLOOR``.
    """
    out = {fm: _LOGPROB_FLOOR for fm in FORMATION_ORDER}

    if not model.get("fit"):
        # No info; uniform.
        return {fm: 0.0 for fm in FORMATION_ORDER}

    if not np.isfinite(gr_value):
        # Uniform over formations the model knows about.
        for fm in FORMATION_ORDER:
            if fm in model["gr_stats"] or fm in _FALLBACK_GR_RANGES:
                out[fm] = 0.0
        return out

    priors = model.get("formation_priors", {})
    stats_map = model.get("gr_stats", {})
    kdes = model.get("gr_kde", {})

    for fm in FORMATION_ORDER:
        # Likelihood
        if fm in kdes:
            try:
                p = float(kdes[fm].evaluate(np.asarray([gr_value]))[0])
                ll = np.log(max(p, 1e-30))
            except Exception:
                # Fall back to Gaussian.
                s = stats_map.get(fm)
                if s is None:
                    ll = _gaussian_loglik_fallback(gr_value, fm)
                else:
                    ll = _gaussian_loglik(gr_value, s["mean"], s["std"])
        elif fm in stats_map:
            s = stats_map[fm]
            ll = _gaussian_loglik(gr_value, s["mean"], s["std"])
        else:
            ll = _gaussian_loglik_fallback(gr_value, fm)

        # Prior (uniform across known formations if no thickness info).
        prior = priors.get(fm, 1.0 / max(len(FORMATION_ORDER), 1))
        log_prior = np.log(max(prior, 1e-6))

        out[fm] = float(max(ll + log_prior, _LOGPROB_FLOOR))
    return out


def _gaussian_loglik(x: float, mean: float, std: float) -> float:
    """Log N(x; mean, std). Std is floored at 1 API to avoid spikes."""
    s = max(float(std), 1.0)
    z = (float(x) - float(mean)) / s
    return -0.5 * z * z - np.log(s) - 0.5 * np.log(2.0 * np.pi)


def _gaussian_loglik_fallback(x: float, fm: str) -> float:
    """Use the static fallback range when no typewell stats exist for ``fm``."""
    if fm not in _FALLBACK_GR_RANGES:
        return _LOGPROB_FLOOR
    lo, hi = _FALLBACK_GR_RANGES[fm]
    mean = 0.5 * (lo + hi)
    std = max((hi - lo) / 4.0, 5.0)
    return _gaussian_loglik(x, mean, std)


# ---------------------------------------------------------------------------
# 3. Apply geological priors to a raw TVT prediction sequence.
# ---------------------------------------------------------------------------
def constrain_tvt_predictions(
    raw_tvt_predictions: np.ndarray,
    horizontal_df: pl.DataFrame | pd.DataFrame,
    typewell_df: pl.DataFrame | pd.DataFrame,
    model: dict,
    *,
    smooth: bool = True,
    smooth_sigma: float = _SMOOTH_SIGMA_DEFAULT,
    posterior_nudge_gain: float = _POSTERIOR_NUDGE_GAIN,
) -> np.ndarray:
    """Apply geological priors to a raw TVT prediction sequence.

    Pipeline (each step on the eval-zone tail only; cased rows are never
    touched):

    1. Pass-through known TVT: rows with finite ``TVT_input`` left exact.
    2. Soft floor (Buda): clip predictions deeper than
       ``contact + _FLOOR_SLACK_FT`` (operators nick the Buda but rarely go
       more than ~10 ft in).
    3. Soft ceiling: clip predictions shallower than the typewell's
       shallowest top minus ``_CEILING_SLACK_FT`` (catches alignment failures
       landing above the logged interval).
    4. Local smoothness: σ ≈ 5-row Gaussian smoother per contiguous eval
       run (reflect edges; NaN-safe via linear interp pre-smooth). Never
       crosses into known TVT_input. Toggle with ``smooth=False``.
    5. Formation-consistency nudge: for each eval row, compute
       :func:`gr_to_formation_logprob` from its GR. If the *implied*
       formation (whichever range contains the predicted TVT) has a
       log-prob ≥ 2 nats below the argmax formation, soft-pull TVT
       toward the argmax formation's centre by
       ``posterior_nudge_gain · tanh(gap/2)``. tanh saturates so a 100-API
       mismatch doesn't pull 100 ft.

    Convention is read from ``model['tvt_convention']``; horizontal
    ``TVT_input`` is assumed to share it (true for contest data). Floor /
    ceiling sign flips automatically.
    """
    pred = np.asarray(raw_tvt_predictions, dtype=np.float64).copy()
    n = pred.size
    if n == 0:
        return pred

    h_tvt_in = _as_numpy(horizontal_df, "TVT_input").astype(np.float64, copy=False)
    h_gr = _as_numpy(horizontal_df, "GR").astype(np.float64, copy=False)

    # Guard against shape mismatches at the boundary.
    if h_tvt_in.size != n:
        # If horizontal has extra/fewer rows, align by leading n. Not ideal but
        # the contract is that pred lines up with horizontal_df rows.
        h_tvt_in = _resize_to_match(h_tvt_in, n)
    if h_gr.size != n:
        h_gr = _resize_to_match(h_gr, n)

    finite_in = np.isfinite(h_tvt_in)
    eval_mask = ~finite_in  # rows where we are predicting

    # Convention (sign of "deeper" — larger or smaller TVT value).
    convention = model.get("tvt_convention", "increasing_down")
    sign_deeper = 1.0 if convention == "increasing_down" else -1.0

    # ---- Step 1: pass-through known TVT_input. -----------------------------
    pred[finite_in] = h_tvt_in[finite_in]

    # ---- Step 2: soft floor (Buda). ----------------------------------------
    contact = model.get("egfdl_buda_contact_tvt")
    if contact is not None and np.any(eval_mask):
        # Floor TVT in the original convention:
        #   increasing_down: floor_tvt = contact + slack    (slack is positive)
        #   increasing_up:   floor_tvt = contact - slack
        floor_tvt = contact + sign_deeper * _FLOOR_SLACK_FT
        # "Below the floor" depends on convention.
        if convention == "increasing_down":
            too_deep = pred > floor_tvt
        else:
            too_deep = pred < floor_tvt
        clip_mask = eval_mask & too_deep & np.isfinite(pred)
        n_clip = int(clip_mask.sum())
        if n_clip > 0:
            logger.info(
                "constrain_tvt_predictions: clipping %d rows below "
                "EGFDL/BUDA contact + slack (%.1f ft).",
                n_clip, _FLOOR_SLACK_FT,
            )
            pred[clip_mask] = floor_tvt

    # ---- Step 3: soft ceiling. ---------------------------------------------
    # Shallowest formation top in original convention.
    tvt_ranges = model.get("tvt_ranges", {})
    if tvt_ranges:
        all_mins = []
        all_maxes = []
        for lo, hi in tvt_ranges.values():
            all_mins.append(lo)
            all_maxes.append(hi)
        # In increasing_down: shallowest = smallest TVT.
        # In increasing_up:   shallowest = largest TVT.
        if convention == "increasing_down":
            shallowest = float(min(all_mins))
            ceil_tvt = shallowest - _CEILING_SLACK_FT
            too_shallow = pred < ceil_tvt
        else:
            shallowest = float(max(all_maxes))
            ceil_tvt = shallowest + _CEILING_SLACK_FT
            too_shallow = pred > ceil_tvt
        clip_mask = eval_mask & too_shallow & np.isfinite(pred)
        n_clip = int(clip_mask.sum())
        if n_clip > 0:
            logger.info(
                "constrain_tvt_predictions: clipping %d rows above shallowest "
                "typewell top minus slack (%.1f ft).",
                n_clip, _CEILING_SLACK_FT,
            )
            pred[clip_mask] = ceil_tvt

    # ---- Step 4: local smoothness. -----------------------------------------
    if smooth and smooth_sigma > 0 and np.any(eval_mask):
        # Smooth ONLY the eval-zone tail. We do this by extracting the eval
        # tail as a contiguous block (the largest contiguous run of NaN
        # TVT_input) and applying gaussian_filter1d with reflect mode. If
        # the eval zone is non-contiguous (rare but possible if there are
        # NaN strips earlier), each contiguous run is smoothed independently.
        for lo, hi in _contiguous_runs(eval_mask):
            seg_len = hi - lo
            if seg_len < 3:
                continue
            sigma = float(smooth_sigma)
            # Gaussian kernel half-width is ~3σ. Cap σ so kernel ≤ seg/2.
            sigma = min(sigma, max(seg_len / 6.0, 1.0))
            seg = pred[lo:hi]
            finite_seg = np.isfinite(seg)
            if finite_seg.sum() < 3:
                continue
            # Replace any NaN inside the eval segment with linear-interp before
            # smoothing — gaussian_filter1d is NOT NaN-aware.
            seg_clean = seg.copy()
            if not finite_seg.all():
                idx = np.arange(seg_len, dtype=np.float64)
                seg_clean[~finite_seg] = np.interp(
                    idx[~finite_seg], idx[finite_seg], seg[finite_seg]
                )
            smoothed = ndimage.gaussian_filter1d(seg_clean, sigma=sigma, mode="reflect")
            pred[lo:hi] = smoothed

        # Restore known input precisely (smoothing is segment-only, so this is
        # already guaranteed; we re-assert for robustness).
        pred[finite_in] = h_tvt_in[finite_in]

    # ---- Step 5: formation-consistency nudge. ------------------------------
    if posterior_nudge_gain > 0 and np.any(eval_mask) and h_gr.size == n:
        pred = _formation_consistency_nudge(
            pred, h_gr, eval_mask, model, posterior_nudge_gain, sign_deeper
        )
        # Re-clip to floor/ceiling after the nudge — nudges should never
        # push us back into the Buda or out of the typewell range.
        if contact is not None:
            floor_tvt = contact + sign_deeper * _FLOOR_SLACK_FT
            if convention == "increasing_down":
                pred[eval_mask] = np.minimum(pred[eval_mask], floor_tvt)
            else:
                pred[eval_mask] = np.maximum(pred[eval_mask], floor_tvt)
        # Final pass-through guarantee.
        pred[finite_in] = h_tvt_in[finite_in]

    return pred


def _resize_to_match(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad with NaN or truncate ``arr`` to length ``n``."""
    if arr.size == n:
        return arr
    if arr.size > n:
        return arr[:n]
    pad = np.full(n - arr.size, np.nan, dtype=np.float64)
    return np.concatenate([arr, pad])


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(lo, hi), ...] of contiguous True runs in ``mask`` (hi exclusive)."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(diff == 1) + 1)
    ends = list(np.flatnonzero(diff == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def _formation_consistency_nudge(
    pred: np.ndarray,
    h_gr: np.ndarray,
    eval_mask: np.ndarray,
    model: dict,
    gain: float,
    sign_deeper: float,
) -> np.ndarray:
    """Soft posterior pull toward GR-implied formation TVT range.

    For each eval row: implied formation = whichever ``tvt_ranges`` slab
    contains the current prediction (else closest centre); argmax formation
    = ``argmax_F log P(F | GR)``. If the log-prob gap ≥ 2 nats (≈ 7×
    likelihood ratio) and they disagree, nudge by
    ``gain · tanh(gap/2) · (target_centre − tvt_pred)``. tanh saturates so
    huge mismatches don't yank us across the entire section.
    """
    pred_out = pred.copy()
    tvt_ranges = model.get("tvt_ranges", {})
    if not tvt_ranges:
        return pred_out

    # Pre-compute centres and bounds in the original convention.
    centres: dict[str, float] = {fm: 0.5 * (lo + hi) for fm, (lo, hi) in tvt_ranges.items()}
    eval_idx = np.flatnonzero(eval_mask)
    if eval_idx.size == 0:
        return pred_out

    # We only nudge a *fraction* of rows — those with a strong log-prob gap.
    nudged_count = 0
    for i in eval_idx:
        gr_i = float(h_gr[i])
        tvt_i = float(pred_out[i])
        if not (np.isfinite(gr_i) and np.isfinite(tvt_i)):
            continue
        log_p = gr_to_formation_logprob(gr_i, model)
        # Best (most likely) formation by log-posterior.
        best_fm = max(log_p, key=log_p.get)
        best_lp = log_p[best_fm]
        # Implied formation by current TVT (whichever range contains it,
        # else the closest by centre).
        implied_fm = _formation_for_tvt(tvt_i, tvt_ranges, sign_deeper)
        if implied_fm is None:
            continue
        implied_lp = log_p.get(implied_fm, _LOGPROB_FLOOR)
        gap = best_lp - implied_lp
        if gap < 2.0:
            continue  # not strong enough to act on
        if best_fm == implied_fm:
            continue
        if best_fm not in centres:
            continue

        target = centres[best_fm]
        delta = target - tvt_i
        # Saturating soft pull.
        scale = float(np.tanh(gap / 2.0))
        pred_out[i] = tvt_i + gain * scale * delta
        nudged_count += 1

    if nudged_count > 0:
        logger.info(
            "constrain_tvt_predictions: applied formation-consistency nudge "
            "to %d / %d eval rows (gain=%.2f).",
            nudged_count, int(eval_mask.sum()), gain,
        )
    return pred_out


def _formation_for_tvt(
    tvt: float,
    tvt_ranges: dict[str, tuple[float, float]],
    sign_deeper: float,
) -> str | None:
    """Return the formation whose TVT range contains ``tvt``, else closest centre."""
    # Containment first.
    for fm, (lo, hi) in tvt_ranges.items():
        if lo <= tvt <= hi:
            return fm
    # Closest centre.
    best_fm = None
    best_d = np.inf
    for fm, (lo, hi) in tvt_ranges.items():
        c = 0.5 * (lo + hi)
        d = abs(tvt - c)
        if d < best_d:
            best_d = d
            best_fm = fm
    return best_fm


# ---------------------------------------------------------------------------
# 4. Regional dip from the cased section.
# ---------------------------------------------------------------------------
def regional_dip_prior(
    horizontal_df: pl.DataFrame | pd.DataFrame,
    typewell_df: pl.DataFrame | pd.DataFrame,
    model: dict,
) -> tuple[float, float]:
    """Robust apparent-dip estimate from the cased section.

    Apparent dip = dTVT / dMD along the wellbore. Maverick Basin regional
    structural dip is 3–5° SE; *apparent* dip on a horizontal depends on
    azimuth (a downdip lateral sees full dip; along-strike sees ~ 0).

    Method: take the contiguous finite-TVT_input block immediately before
    the first NaN (the PS marker), capped at 500 rows (most recent —
    closest to PS in MD). Fit ``TVT = a · MD + b`` with Theil-Sen — chosen
    over RANSAC because it is parameter-free, has 29 % breakdown, and is
    optimal for the few-outlier regime expected here. Slope ``a`` is dip
    in ft/ft (0.05 ≈ 3°). 1-σ from the 95 % CI half-width / 1.96.

    Defaults: no finite TVT_input → ``(0.0, 2.0)``, flat with wide σ so
    consumers won't blindly trust it. Fewer than 10 samples in block → same.
    OLS is the inner fallback if Theil-Sen errors.
    """
    md = _as_numpy(horizontal_df, "MD").astype(np.float64, copy=False)
    tvt_in = _as_numpy(horizontal_df, "TVT_input").astype(np.float64, copy=False)

    if md.size == 0 or tvt_in.size == 0:
        return 0.0, 2.0
    n = min(md.size, tvt_in.size)
    md, tvt_in = md[:n], tvt_in[:n]

    finite = np.isfinite(md) & np.isfinite(tvt_in)
    if not finite.any():
        logger.warning(
            "regional_dip_prior: no finite TVT_input. Returning flat default (0, 2)."
        )
        return 0.0, 2.0

    # Pick the run of finite TVT immediately preceding the first NaN (the PS
    # marker). If there is no NaN at all, use the whole finite block.
    nan_idx = np.flatnonzero(~finite)
    if nan_idx.size == 0:
        end = n
    else:
        end = int(nan_idx[0])
    # Scan backward from `end` to find the contiguous finite block.
    start = end
    while start > 0 and finite[start - 1]:
        start -= 1
    block_md = md[start:end]
    block_tvt = tvt_in[start:end]

    # Window cap: 1–500 rows is the spec. We use the most recent (latest)
    # 500 rows — they are closest in MD to the PS marker and best reflect
    # the local apparent dip going forward.
    if block_md.size > 500:
        block_md = block_md[-500:]
        block_tvt = block_tvt[-500:]

    # Need at least ~10 samples for a meaningful Theil-Sen fit.
    if block_md.size < 10:
        logger.warning(
            "regional_dip_prior: only %d samples in pre-PS block; using flat default.",
            block_md.size,
        )
        return 0.0, 2.0

    # Theil-Sen via scipy.stats.theilslopes — returns (slope, intercept, lo, hi)
    # where lo/hi are the 95 % CI on the slope. We convert to a 1-σ std.
    try:
        slope, _intercept, lo_slope, hi_slope = stats.theilslopes(
            block_tvt, block_md, alpha=0.95
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Theil-Sen failed: %s; falling back to OLS.", exc)
        if block_md.size < 2 or np.std(block_md) == 0:
            return 0.0, 2.0
        slope, _ = np.polyfit(block_md, block_tvt, 1)
        # Crude std from residuals.
        resid = block_tvt - (slope * block_md + np.mean(block_tvt - slope * block_md))
        std = float(np.std(resid)) / max(np.std(block_md), 1e-6)
        return float(slope), max(std, 1e-3)

    # 95 % CI half-width / 1.96 ≈ 1-σ.
    ci_half = 0.5 * (hi_slope - lo_slope)
    std = max(float(ci_half) / 1.96, 1e-4)
    return float(slope), float(std)


# ---------------------------------------------------------------------------
# 5. Fault-jump detector — interface only for first submission.
# ---------------------------------------------------------------------------
def fault_jump_detector(
    horizontal_df: pl.DataFrame | pd.DataFrame,
    typewell_df: pl.DataFrame | pd.DataFrame,
    raw_tvt_predictions: np.ndarray,
    model: dict,
) -> np.ndarray:
    """Detect candidate fault crossings as alignment-cost discontinuities.

    Iteration 1: returns an all-False array of the correct length. The
    signature is locked so iteration 2 can drop in detection without
    touching the integrating notebook. False positives are expensive
    (they license non-monotonic jumps in downstream smoothing) so a
    no-op default is safer than a tunable detector with poor priors.

    Iteration-2 plan: at each row build the expected typewell GR window
    around the current TVT prediction (± 10 ft); cross-correlate vs the
    actual horizontal GR window; flag rows where (running median − current)
    correlation drops > 0.4 *and* |dZ/dMD| spikes. Downstream callers
    should skip smoothing across flagged indices.

    Edge cases: empty inputs → empty bool; length mismatch → ``min`` length.
    """
    md = _as_numpy(horizontal_df, "MD")
    n_h = md.size
    n_p = int(np.asarray(raw_tvt_predictions).size)
    n = min(n_h, n_p) if (n_h and n_p) else max(n_h, n_p)
    if n <= 0:
        return np.empty(0, dtype=bool)
    return np.zeros(n, dtype=bool)


__all__ = [
    "FORMATION_ORDER",
    "fit_formation_gr_model",
    "gr_to_formation_logprob",
    "constrain_tvt_predictions",
    "regional_dip_prior",
    "fault_jump_detector",
]
