"""Anisotropic local kriging for formation-top surfaces.

konbu17 uses isotropic IDW (with X/Y std-scaling) on an ~5M-row spatial
grid. The audit attributes 0.3-0.6 RMSE potential to replacing this with
anisotropic kriging that respects the regional NE-SW Eagle Ford strike.

This module is the v9 starting point. It is **not wired into v8**; the v9
bench will plug it in by replacing ``RowKNN.impute`` in feature_builder.

Design decisions:
  * Anisotropy via a 2x2 SPD whitening matrix W applied to (X, Y) before
    kdtree lookup. W is estimated empirically from the ANCC gradient
    field of train data (PCA on (dANCC/dX, dANCC/dY)) or set explicitly
    by user. The "long" axis of W is the stable-direction (along strike).
  * Local kriging weights: Gaussian kernel
        w_i = exp(- 0.5 * ((x_i - x_q)^T W^T W (x_i - x_q)) )
    (with a small ridge to keep the kriging matrix well-conditioned).
  * Predictive variance is exposed for the GBM to use as a feature.

Compute budget: still O(K) per query after kdtree narrowing, so the
overhead vs IDW is tiny.

Reference (free of jargon):
    For a stationary, normally-distributed surface the optimal linear
    estimator at a query point is a weighted average of nearby samples
    where the weights solve the kriging system
        K w = k_q
    K_ij = covariance(x_i, x_j),  k_q,i = covariance(x_q, x_i).
    With a Gaussian/exponential kernel + a tiny ridge (nugget), this is
    a 20x20 linear solve per query at K=20 — essentially free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def estimate_anisotropy_from_field(
    xy: np.ndarray,
    z: np.ndarray,
    *,
    radius_quantile: float = 0.05,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate anisotropy axis & length scales from a noisy spatial field.

    Idea: gradients of the field point in the high-variation direction.
    PCA on the local gradient distribution yields the anisotropy ellipse.

    Parameters
    ----------
    xy : (N, 2) float64
    z  : (N,)   float64       sampled values of the surface at xy
    radius_quantile : neighborhood radius for local-gradient estimate as a
        quantile of the pairwise-NN distance distribution.

    Returns
    -------
    R : (2, 2) rotation matrix; columns are anisotropy axes
    sigma : (2,) length scales (large axis first)
    """
    if xy.shape[0] != z.shape[0] or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2), z must be (N,)")

    tree = cKDTree(xy)
    nn_d, _ = tree.query(xy, k=2)
    radius = float(np.quantile(nn_d[:, 1], radius_quantile))
    radius = max(radius, 50.0)  # guard against pathological clusters

    # Estimate gradients via local linear fit: solve z ~ a + bx + cy on
    # neighbors. Sub-sample queries to avoid 5M solves.
    n_sub = min(20_000, xy.shape[0])
    rng = np.random.default_rng(20260507)
    idx_sub = rng.choice(xy.shape[0], n_sub, replace=False)

    grad_xy = np.zeros((n_sub, 2), dtype=np.float64)
    for k, i in enumerate(idx_sub):
        nbr_idx = tree.query_ball_point(xy[i], r=radius)
        if len(nbr_idx) < 6:
            grad_xy[k] = np.nan
            continue
        nbr = np.asarray(nbr_idx, dtype=np.int64)
        A = np.column_stack([np.ones(nbr.size), xy[nbr, 0] - xy[i, 0], xy[nbr, 1] - xy[i, 1]])
        b = z[nbr]
        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            grad_xy[k] = coef[1:3]
        except np.linalg.LinAlgError:
            grad_xy[k] = np.nan

    grad_xy = grad_xy[np.isfinite(grad_xy).all(axis=1)]
    if grad_xy.shape[0] < 100:
        # Fallback: identity (isotropic)
        return np.eye(2), np.array([1.0, 1.0])

    # Robust covariance via MAD-style scaling
    g_med = np.median(grad_xy, axis=0)
    centered = grad_xy - g_med
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    # PCA: principal axes = eigenvectors; smaller eigenvalue == "low-gradient"
    # axis (along strike).
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(2))
    # Sort descending so vecs[:, 0] is high-gradient (perpendicular-to-strike)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Length scales: inversely proportional to gradient magnitude in each axis
    sigma = 1.0 / np.sqrt(np.maximum(vals, eps))
    sigma = sigma / sigma.min()    # normalize: short axis = 1

    # We want the rotation matrix R such that R.T (xy - center) gives
    # coordinates in the (high-gradient, along-strike) frame.
    R = vecs
    return R, sigma


@dataclass
class AnisoFormationKNN:
    """Anisotropic local kriging predictor for one formation top.

    Build once on all train rows; query per test row.
    """

    xy: np.ndarray              # (N, 2) original coords
    z: np.ndarray               # (N,) target values
    well_ids: np.ndarray        # (N,) integer well-ids
    well_index: list[str]
    R: np.ndarray               # (2, 2) rotation
    sigma: np.ndarray           # (2,) length scales
    L: np.ndarray               # (2, 2) whitening = R / sigma (per axis)
    tree: cKDTree
    nugget: float
    range_scale: float          # multiplier for sigma -> kriging length

    def well_to_int(self, wid: str) -> int:
        try:
            return self.well_index.index(wid)
        except ValueError:
            return -1

    @classmethod
    def fit(
        cls,
        xy: np.ndarray,
        z: np.ndarray,
        well_ids: np.ndarray,
        well_index: list[str],
        *,
        anisotropy: tuple[np.ndarray, np.ndarray] | None = None,
        nugget: float = 1e-6,
        range_scale: float = 1.0,
    ) -> "AnisoFormationKNN":
        if anisotropy is None:
            R, sigma = estimate_anisotropy_from_field(xy, z)
        else:
            R, sigma = anisotropy
        # Whitening: project onto axes, scale each by 1/sigma so that the
        # kdtree's Euclidean distance corresponds to anisotropic Mahalanobis.
        L = R @ np.diag(1.0 / (sigma * range_scale))
        xy_white = xy @ L
        tree = cKDTree(xy_white)
        return cls(xy=xy, z=z, well_ids=well_ids, well_index=well_index,
                   R=R, sigma=sigma, L=L, tree=tree,
                   nugget=nugget, range_scale=range_scale)

    def query(
        self,
        xy_q: np.ndarray,
        *,
        k: int = 20,
        n_q: int = 4_000,
        exclude_well: str | None = None,
        kernel: str = "gaussian",   # "gaussian" | "exponential"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean_pred, std_pred, min_dist)."""
        excl_int = self.well_to_int(exclude_well) if exclude_well else -2
        q_white = xy_q @ self.L
        n_q = min(n_q, self.xy.shape[0])
        dist, idx = self.tree.query(q_white, k=n_q, workers=-1)
        if exclude_well:
            mask_self = self.well_ids[idx] == excl_int
            dist = np.where(mask_self, np.inf, dist)

        order = np.argpartition(dist, kth=min(k - 1, n_q - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)

        if kernel == "gaussian":
            cov = np.where(valid_k, np.exp(-0.5 * d_k * d_k), 0.0)
        elif kernel == "exponential":
            cov = np.where(valid_k, np.exp(-d_k), 0.0)
        else:
            raise ValueError(f"unknown kernel {kernel!r}")

        means = np.full(xy_q.shape[0], np.nan, dtype=np.float64)
        stds = np.full(xy_q.shape[0], np.nan, dtype=np.float64)
        min_dist = np.where(valid_k, d_k, np.inf).min(axis=1)

        for i in range(xy_q.shape[0]):
            ix_i = idx_k[i][valid_k[i]]
            if ix_i.size < 3:
                continue
            d_i = d_k[i][valid_k[i]]
            c_i = cov[i][valid_k[i]]
            # Build the kriging system on neighbors with their pairwise distance.
            # We compute the neighbor-neighbor whitened distance from raw coords
            # via the same L matrix.
            xy_n = self.xy[ix_i] @ self.L
            diffs = xy_n[:, None, :] - xy_n[None, :, :]
            dn = np.sqrt(np.sum(diffs * diffs, axis=-1))
            if kernel == "gaussian":
                K = np.exp(-0.5 * dn * dn)
            else:
                K = np.exp(-dn)
            K = K + self.nugget * np.eye(K.shape[0])
            try:
                w = np.linalg.solve(K, c_i)
            except np.linalg.LinAlgError:
                w = c_i / max(c_i.sum(), 1e-12)
            wsum = w.sum()
            if abs(wsum) < 1e-12:
                continue
            mean = float((self.z[ix_i] * w).sum() / wsum)
            # Ordinary-kriging variance approximation:
            var = max(1.0 - float((c_i * w).sum()), 0.0)
            means[i] = mean
            stds[i] = np.sqrt(var)

        return means, stds, min_dist


def fit_aniso_for_formations(
    train_paths: list[Path],
    formations: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"),
    *,
    range_scale: float = 1.0,
) -> dict[str, AnisoFormationKNN]:
    """Build one AnisoFormationKNN per formation. The anisotropy direction
    is estimated independently per formation; in practice they should be
    similar for parallel formation tops.
    """
    cols = ["X", "Y", *formations]
    xs, ys = [], []
    f_arrs: list[np.ndarray] = []
    wid_arr: list[str] = []
    for p in train_paths:
        wid = p.stem.replace("__horizontal_well", "")
        try:
            df = pd.read_csv(p, usecols=cols).dropna()
        except Exception:
            continue
        if df.empty:
            continue
        xs.append(df["X"].to_numpy())
        ys.append(df["Y"].to_numpy())
        f_arrs.append(df[list(formations)].to_numpy(dtype=np.float64))
        wid_arr.extend([wid] * len(df))

    xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    f_targets = np.vstack(f_arrs)
    well_index = sorted(set(wid_arr))
    well_pos = {w: i for i, w in enumerate(well_index)}
    well_ids = np.array([well_pos[w] for w in wid_arr], dtype=np.int32)

    out: dict[str, AnisoFormationKNN] = {}
    for j, fname in enumerate(formations):
        z = f_targets[:, j]
        out[fname] = AnisoFormationKNN.fit(
            xy, z, well_ids, well_index,
            range_scale=range_scale,
        )
    return out
