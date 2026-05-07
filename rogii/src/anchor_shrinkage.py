"""Anchor-shrinkage post-processor for v10.

The outlier diagnosis on v9 (OOF max well RMSE 165, 8/11 catastrophic
wells = MODEL DRIFT, not real geosteering action) shows that for these
wells a trivial baseline -- "predict the rolling mean of the last 100
prefix TVT_input values for every eval row" -- would score 5-15 ft
RMSE versus our model's 53-166 ft. The model is *over-confidently
moving* predictions away from the anchor when there's no geological
reason to.

This post-processor blends the model's predicted delta with zero
(zero = "stay at the anchor", since target = TVT - last_known_TVT):

    delta_final = alpha * delta_model
    TVT_final  = last_known_TVT + delta_final

with alpha < 1. James-Stein-style multiplicative shrinkage.

The optimal alpha balances:
  - The cost of shrinking GOOD predictions (reduces signal on real-
    motion wells where target is non-trivial).
  - The benefit of damping CATASTROPHIC predictions (where target is
    near zero but model says ±50ft).

Empirically calibrated against population baselines (median eval-
offset 0.84 ft, p95 20.4, p99 37.7): shrinking by 0.6-0.8 should
reduce catastrophic max-well-RMSE substantially while losing little
on the typical median.

A more sophisticated variant (`gated_shrinkage`) uses a per-row
confidence signal -- KNN neighbor distance, MLP-vs-KNN disagreement,
neighbor std -- to set alpha per row. Higher confidence -> alpha
closer to 1. Lower confidence -> alpha closer to 0.

This module is a stand-alone applied to ANY OOF prediction array
(v9, v8, stacker output). It pairs cleanly with the meta-stacker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def constant_shrinkage(
    delta_pred: np.ndarray,
    *,
    alpha: float = 0.7,
) -> np.ndarray:
    """Multiplicative shrinkage toward zero (i.e., toward the anchor).

    Parameters
    ----------
    delta_pred : (N,) np.ndarray
        Model's predicted delta = TVT - last_known_TVT.
    alpha : float
        Shrinkage factor. 1.0 = no shrinkage; 0.0 = predict anchor for all.
        Recommended starting value 0.7 (audit on full v9 OOF).

    Returns
    -------
    shrunk : (N,) np.ndarray
        alpha * delta_pred. To recover absolute TVT, add last_known_TVT.
    """
    return alpha * delta_pred


def hard_cap(
    delta_pred: np.ndarray,
    *,
    band: float = 30.0,
) -> np.ndarray:
    """Hard-cap predicted delta to [-band, +band] (in ft).

    Population p95 of |eval_offset_from_anchor| is ~20 ft, p99 ~38 ft.
    Capping at 30 ft preserves real motion in 99% of typical wells while
    chopping the catastrophic tail.
    """
    return np.clip(delta_pred, -band, band)


def gated_shrinkage(
    delta_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    alpha_min: float = 0.4,
    alpha_max: float = 1.0,
    confidence_clip: tuple[float, float] | None = None,
) -> np.ndarray:
    """Per-row shrinkage with alpha modulated by a confidence signal.

    confidence \in [0, 1]: 0 = totally untrusted (collapse to anchor),
    1 = fully trusted (no shrinkage). The mapping is linear:
        alpha = alpha_min + (alpha_max - alpha_min) * confidence
    so a row with confidence=0 gets alpha=alpha_min (maximum shrinkage).
    """
    c = np.asarray(confidence, dtype=np.float64)
    if confidence_clip is not None:
        lo, hi = confidence_clip
        c = (c - lo) / max(hi - lo, 1e-12)
    c = np.clip(c, 0.0, 1.0)
    alpha = alpha_min + (alpha_max - alpha_min) * c
    return alpha * delta_pred


@dataclass
class ShrinkageReport:
    overall_rmse: float
    overall_mae: float
    overall_bias: float
    median_well_rmse: float
    p90_well_rmse: float
    max_well_rmse: float


def evaluate_shrinkage(
    delta_pred: np.ndarray,
    target: np.ndarray,
    well_ids: np.ndarray,
) -> ShrinkageReport:
    """Score a shrunk prediction on the OOF target."""
    err = np.asarray(delta_pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    well_ids = np.asarray(well_ids)
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    well_unique = np.unique(well_ids)
    well_rmse = np.zeros(well_unique.size, dtype=np.float64)
    for i, w in enumerate(well_unique):
        mask = well_ids == w
        e = err[mask]
        well_rmse[i] = float(np.sqrt(np.mean(e * e))) if e.size else float("nan")
    return ShrinkageReport(
        overall_rmse=rmse,
        overall_mae=mae,
        overall_bias=bias,
        median_well_rmse=float(np.median(well_rmse)),
        p90_well_rmse=float(np.percentile(well_rmse, 90)),
        max_well_rmse=float(well_rmse.max()),
    )
