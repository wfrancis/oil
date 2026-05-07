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

    Notes
    -----
    The whitening matrix L = R @ diag(1 / (sigma * range_scale * L_norm)).
    L_norm is an overall length scale (median nearest-neighbor distance in
    raw whitened space) that ensures the kernel argument is O(1) at typical
    inter-row distances. range_scale further tightens (<1) or loosens (>1)
    the kernel decay.
    """

    xy: np.ndarray              # (N, 2) original coords
    z: np.ndarray               # (N,) target values
    well_ids: np.ndarray        # (N,) integer well-ids
    well_index: list[str]
    R: np.ndarray               # (2, 2) rotation
    sigma: np.ndarray           # (2,) length scales
    L: np.ndarray               # (2, 2) whitening = R / (sigma * range_scale * L_norm)
    L_norm: float               # overall length scale
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
        nugget: float = 1e-4,
        range_scale: float = 1.0,
    ) -> "AnisoFormationKNN":
        if anisotropy is None:
            R, sigma = estimate_anisotropy_from_field(xy, z)
        else:
            R, sigma = anisotropy

        # First-pass whitening (just rotation + sigma): used to learn L_norm
        # so kernel argument is O(1) at typical neighbor distance.
        L_pre = R @ np.diag(1.0 / sigma)
        xy_pre = xy @ L_pre
        # Estimate the characteristic INTER-CLUSTER (between-track / between-
        # well) length scale, not intra-well row spacing. We do this by
        # subsampling sparsely (so within-well dense rows can't dominate)
        # and taking the median 1-NN distance in that thinned point set.
        rng = np.random.default_rng(20260507)
        n_sub = min(2_000, xy.shape[0])
        idx_sub = rng.choice(xy.shape[0], n_sub, replace=False)
        tree_thin = cKDTree(xy_pre[idx_sub])
        d_thin, _ = tree_thin.query(xy_pre[idx_sub], k=2)
        L_norm = float(np.median(d_thin[:, 1]))
        L_norm = max(L_norm, 1e-9)

        # Final whitening: rotate, anisotropy-scale, then divide by overall
        # length scale * range_scale.
        # L = L_pre / (L_norm * range_scale) so kernel arg ~1 near typical NN.
        L = L_pre / (L_norm * range_scale)
        xy_white = xy @ L
        tree = cKDTree(xy_white)
        return cls(xy=xy, z=z, well_ids=well_ids, well_index=well_index,
                   R=R, sigma=sigma, L=L, L_norm=L_norm, tree=tree,
                   nugget=nugget, range_scale=range_scale)

    def query(
        self,
        xy_q: np.ndarray,
        *,
        k: int = 20,
        kernel: str = "gaussian",   # "gaussian" | "exponential"
        batch_size: int = 200_000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean_pred, std_pred, min_dist).

        Self-well exclusion is intentionally NOT done here. For benchmark/
        OOF use, the caller is expected to have built this object with only
        the train-fold rows, so leakage is impossible by construction.
        """
        if kernel not in ("gaussian", "exponential"):
            raise ValueError(f"unknown kernel {kernel!r}")
        n = xy_q.shape[0]
        means = np.full(n, np.nan, dtype=np.float64)
        stds = np.full(n, np.nan, dtype=np.float64)
        min_dist = np.full(n, np.inf, dtype=np.float64)

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            xy_b = xy_q[start:stop]
            q_white = xy_b @ self.L
            d_k, idx_k = self.tree.tree.query(q_white, k=k, workers=-1) \
                if False else self.tree.query(q_white, k=k, workers=-1)
            # cKDTree returns (B,) arrays for k=1; ensure 2-D.
            if d_k.ndim == 1:
                d_k = d_k[:, None]
                idx_k = idx_k[:, None]
            valid_k = np.isfinite(d_k)
            min_dist[start:stop] = np.where(valid_k, d_k, np.inf).min(axis=1)

            if kernel == "gaussian":
                c_i = np.where(valid_k, np.exp(-0.5 * d_k * d_k), 0.0)
            else:
                c_i = np.where(valid_k, np.exp(-d_k), 0.0)

            # Batched kriging system. Build (B, K, K) Gram matrix from neighbor
            # whitened coords. self.xy is raw, so whitening must be applied.
            xy_n = self.xy[idx_k] @ self.L                # (B, K, 2)
            diffs = xy_n[:, :, None, :] - xy_n[:, None, :, :]  # (B, K, K, 2)
            dn = np.sqrt(np.sum(diffs * diffs, axis=-1))
            if kernel == "gaussian":
                K_mat = np.exp(-0.5 * dn * dn)
            else:
                K_mat = np.exp(-dn)
            K_mat = K_mat + self.nugget * np.eye(k)[None, :, :]

            # Solve K_mat[i] @ w[i] = c_i[i]  (B systems of size K)
            try:
                w = np.linalg.solve(K_mat, c_i[..., None]).squeeze(-1)
            except np.linalg.LinAlgError:
                w = np.full_like(c_i, np.nan)

            # Numerically-degenerate rows: weights all sub-ULP or non-finite.
            # Fall back to IDW; if even IDW row-sum is tiny, use uniform 1/K.
            wsum = w.sum(axis=1)
            bad_solve = (~np.isfinite(w).all(axis=1)) | (np.abs(wsum) < 1e-12)
            if bad_solve.any():
                row_sum = c_i.sum(axis=1, keepdims=True)
                tiny = row_sum < 1e-12
                row_sum_safe = np.where(tiny, 1.0, row_sum)
                w_fallback = np.where(tiny, 1.0 / k, c_i / row_sum_safe)
                w = np.where(bad_solve[:, None], w_fallback, w)
                wsum = w.sum(axis=1)

            wsum_safe = np.where(np.abs(wsum) < 1e-12, 1.0, wsum)
            z_n = self.z[idx_k]                              # (B, K)
            means_b = (z_n * w).sum(axis=1) / wsum_safe
            # Variance: 1 - c.T @ w  (clipped)
            var_b = np.clip(1.0 - (c_i * w).sum(axis=1), 0.0, None)
            std_b = np.sqrt(var_b)

            no_neigh = ~np.any(valid_k, axis=1)
            means_b = np.where(no_neigh, np.nan, means_b)
            std_b = np.where(no_neigh, np.nan, std_b)
            means[start:stop] = means_b
            stds[start:stop] = std_b

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
