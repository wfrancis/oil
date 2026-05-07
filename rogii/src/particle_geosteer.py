"""Particle-filter geosteering predictor.

The competition is framed as geosteering: along the lateral, the operator
keeps the bit in a target geological zone (e.g., LBHL inside EGFDL) by
adjusting trajectory. TVT is therefore a hidden Markov state evolving
along MD, and horizontal GR(MD) compared to typewell GR at the predicted
TVT is the natural observation.

konbu17's beam-search Viterbi (in feature_builder.py) is a discrete
approximation of this with a fixed +-1 typewell-row step per MD increment.
A continuous particle filter is more flexible and can naturally fold in:

  * The v8 spatial prediction as a soft prior on TVT_t
  * Multiple typewells (sister wells' prefixes) as additional observations
  * Regime detection (flat / steering up / steering down) via a discrete
    sub-state with mixture-of-Gaussians transitions

Empirical TVT regime in the lateral (50 train wells):
  eval-TVT range:    median 26 ft, max 66 ft
  drift_from_anchor: median +0.1 ft, p10 -12.3, p90 +12.3
So the lateral is mostly *flat* with occasional ~30 ft excursions. The
particle filter's transition kernel should reflect this: small per-MD
step in TVT plus rare regime changes.

This module is a stand-alone predictor; it can also produce features
(filter mean, filter std, particle-mean delta from formula) for v8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        infer_schema_length=2000,
        null_values=["", "NA", "NaN", "nan", "null"],
        truncate_ragged_lines=True,
    )


def _interp_safe(x_q: float, xs: np.ndarray, ys: np.ndarray) -> float:
    if not np.isfinite(x_q):
        return float("nan")
    if x_q <= xs[0]:
        return float(ys[0])
    if x_q >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x_q, xs, ys))


@dataclass
class ParticleGeosteerConfig:
    n_particles: int = 4000
    sigma_init: float = 8.0           # initial spread (ft) around last_known_tvt
    sigma_step: float = 0.6           # per-row TVT step Gaussian noise (ft)
    sigma_step_jump: float = 30.0     # rare large step (regime change) (ft)
    p_jump: float = 0.002             # probability of a regime-change step per row
    obs_sigma: float = 12.0           # GR observation likelihood scale (GR API units)
    prior_sigma: float = 30.0         # spatial-prior likelihood scale (ft)
    use_prior: bool = True
    resample_eff_threshold: float = 0.5
    seed: int = 42


def particle_filter_well(
    horizontal: pl.DataFrame,
    typewell: pl.DataFrame,
    *,
    config: ParticleGeosteerConfig | None = None,
    spatial_prior: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run a particle filter on one horizontal well.

    Parameters
    ----------
    horizontal : pl.DataFrame
        Must contain MD, GR, TVT_input.
    typewell : pl.DataFrame
        Must contain TVT, GR.
    spatial_prior : optional (N,) array
        Soft prior on TVT_t at each MD row. v8's `tvt_formula_row` is a good
        choice. If None, the filter has no prior beyond the prefix anchor
        and the GR likelihood.

    Returns
    -------
    dict with keys:
        ``tvt``      : (N,) posterior mean
        ``tvt_std``  : (N,) posterior std
        ``logprob``  : (N,) log normaliser (sanity)
        ``ess``      : (N,) effective sample size at each row
    """
    cfg = config or ParticleGeosteerConfig()
    rng = np.random.default_rng(cfg.seed)

    md = horizontal["MD"].to_numpy().astype(np.float64)
    gr_h = horizontal["GR"].to_numpy().astype(np.float64)
    tvt_in = (
        horizontal["TVT_input"].to_numpy().astype(np.float64)
        if "TVT_input" in horizontal.columns
        else np.full(md.size, np.nan, dtype=np.float64)
    )

    tw_tvt = typewell["TVT"].to_numpy().astype(np.float64)
    tw_gr = typewell["GR"].to_numpy().astype(np.float64)
    tw_ok = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if tw_ok.sum() < 16:
        return {"tvt": np.full(md.size, np.nan), "tvt_std": np.full(md.size, np.nan),
                "logprob": np.full(md.size, np.nan), "ess": np.full(md.size, np.nan)}
    tw_tvt = tw_tvt[tw_ok]
    tw_gr = tw_gr[tw_ok]
    # Sort typewell by TVT
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]

    n = md.size
    finite_in = np.isfinite(tvt_in)
    if not finite_in.any():
        # No anchor — abort
        return {"tvt": np.full(n, np.nan), "tvt_std": np.full(n, np.nan),
                "logprob": np.full(n, np.nan), "ess": np.full(n, np.nan)}

    last_anchor_idx = int(np.flatnonzero(finite_in)[-1])
    last_tvt = float(tvt_in[last_anchor_idx])

    # Initialize particles at the last finite anchor
    P = cfg.n_particles
    parts = rng.normal(loc=last_tvt, scale=cfg.sigma_init, size=P)
    log_w = np.zeros(P, dtype=np.float64)

    out_mean = np.full(n, np.nan, dtype=np.float64)
    out_std = np.full(n, np.nan, dtype=np.float64)
    out_lp = np.full(n, np.nan, dtype=np.float64)
    out_ess = np.full(n, np.nan, dtype=np.float64)

    out_mean[last_anchor_idx] = last_tvt
    out_std[last_anchor_idx] = cfg.sigma_init
    out_ess[last_anchor_idx] = float(P)

    last_md = md[last_anchor_idx]

    # Walk forward from the last anchor, predicting TVT for the eval rows
    for i in range(last_anchor_idx + 1, n):
        if not np.isfinite(md[i]):
            out_mean[i] = float(np.average(parts, weights=np.exp(log_w - log_w.max())))
            continue
        dmd = max(md[i] - last_md, 1e-3)

        # Transition: small step + occasional jump
        step = rng.normal(loc=0.0, scale=cfg.sigma_step * np.sqrt(dmd / 1.0), size=P)
        jumps = (rng.random(P) < cfg.p_jump).astype(np.float64)
        jump_steps = rng.normal(loc=0.0, scale=cfg.sigma_step_jump, size=P) * jumps
        parts = parts + step + jump_steps
        last_md = md[i]

        # Observation: typewell-GR likelihood
        if np.isfinite(gr_h[i]):
            # Gaussian likelihood around |gr_h - typewell_GR(parts_TVT)|
            gr_at_parts = np.interp(parts, tw_tvt, tw_gr,
                                    left=tw_gr[0], right=tw_gr[-1])
            log_w_i_obs = -0.5 * ((gr_h[i] - gr_at_parts) / cfg.obs_sigma) ** 2
        else:
            log_w_i_obs = np.zeros(P, dtype=np.float64)

        # Spatial prior likelihood (v8's formula)
        if cfg.use_prior and spatial_prior is not None and np.isfinite(spatial_prior[i]):
            log_w_i_prior = -0.5 * ((parts - spatial_prior[i]) / cfg.prior_sigma) ** 2
        else:
            log_w_i_prior = np.zeros(P, dtype=np.float64)

        log_w = log_w + log_w_i_obs + log_w_i_prior

        # Normalize and resample if ESS is too low
        log_w_norm = log_w - log_w.max()
        w = np.exp(log_w_norm)
        wsum = w.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            # Re-init: completely lost — fall back to anchor
            parts = rng.normal(loc=last_tvt, scale=cfg.sigma_init, size=P)
            log_w = np.zeros(P)
            continue
        w = w / wsum
        ess = 1.0 / float((w * w).sum())

        if ess < cfg.resample_eff_threshold * P:
            # Systematic resampling
            cum = np.cumsum(w)
            u = (rng.random() + np.arange(P)) / P
            new_idx = np.searchsorted(cum, u).clip(0, P - 1)
            parts = parts[new_idx]
            log_w = np.zeros(P)
            ess = float(P)

        out_mean[i] = float((parts * w).sum() / w.sum())
        out_std[i] = float(np.sqrt((((parts - out_mean[i]) ** 2) * w).sum() / w.sum()))
        out_ess[i] = ess
        out_lp[i] = float(np.log(wsum) + log_w.max() if np.isfinite(wsum) else np.nan)

    # Pin the prefix to TVT_input (perfect anchor)
    out_mean = np.where(finite_in, tvt_in, out_mean)
    return {
        "tvt": out_mean,
        "tvt_std": out_std,
        "logprob": out_lp,
        "ess": out_ess,
    }


def predict_well_pf(
    horizontal: pl.DataFrame,
    typewell: pl.DataFrame,
    *,
    config: ParticleGeosteerConfig | None = None,
    spatial_prior: np.ndarray | None = None,
) -> np.ndarray:
    return particle_filter_well(
        horizontal, typewell,
        config=config, spatial_prior=spatial_prior,
    )["tvt"]
