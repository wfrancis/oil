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
    cell_size: float | None = None,
    n_cells_per_axis: int = 60,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate anisotropy axis & length scales from a noisy spatial field.

    Idea: gradients of the spatial trend in z(x, y) reveal the anisotropy.
    Naively fitting `z ~ a + bx + cy` on raw rows fails because well
    trajectories sample the same XY very densely along the trajectory,
    and the local linear fit picks up intra-track row-level noise rather
    than the broad spatial gradient. We therefore aggregate to a coarse
    spatial grid first (mean z per cell), then fit local linear gradients
    over moderate-radius neighborhoods of cell centroids.

    Parameters
    ----------
    xy : (N, 2) float64
    z  : (N,)   float64       sampled values of the surface at xy
    cell_size : explicit cell size in raw XY units (e.g. ft). If None,
        bbox_span / n_cells_per_axis is used.
    n_cells_per_axis : grid resolution when cell_size is None.

    Returns
    -------
    R : (2, 2) rotation matrix; columns = (high-gradient, along-strike) axes
    sigma : (2,) length scales (along-strike axis is the larger one)
    """
    if xy.shape[0] != z.shape[0] or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2), z must be (N,)")

    bbox_min = xy.min(axis=0)
    bbox_max = xy.max(axis=0)
    bbox_span = np.maximum(bbox_max - bbox_min, 1.0)
    if cell_size is None:
        cell_size = float(bbox_span.mean()) / float(n_cells_per_axis)
    cell_size = max(cell_size, 1e-6)

    cell_idx = np.floor((xy - bbox_min) / cell_size).astype(np.int64)
    keys = cell_idx[:, 0] * (cell_idx[:, 1].max() + 2) + cell_idx[:, 1]
    # Group by cell, take mean(z) and mean(xy)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    xy_s = xy[order]
    z_s = z[order]
    boundaries = np.flatnonzero(np.diff(keys_s) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(keys_s)]])

    n_cells = len(starts)
    cell_xy = np.zeros((n_cells, 2), dtype=np.float64)
    cell_z = np.zeros(n_cells, dtype=np.float64)
    for k in range(n_cells):
        s, e = starts[k], ends[k]
        cell_xy[k] = xy_s[s:e].mean(axis=0)
        cell_z[k] = z_s[s:e].mean()

    if n_cells < 20:
        return np.eye(2), np.array([1.0, 1.0])

    # Local-gradient radius = a few cell sizes
    radius = 3.0 * cell_size
    tree = cKDTree(cell_xy)

    grad_xy = np.zeros((n_cells, 2), dtype=np.float64)
    grad_xy[:] = np.nan
    for i in range(n_cells):
        nbr_idx = tree.query_ball_point(cell_xy[i], r=radius)
        if len(nbr_idx) < 6:
            continue
        nbr = np.asarray(nbr_idx, dtype=np.int64)
        A = np.column_stack([
            np.ones(nbr.size),
            cell_xy[nbr, 0] - cell_xy[i, 0],
            cell_xy[nbr, 1] - cell_xy[i, 1],
        ])
        b = cell_z[nbr]
        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            grad_xy[i] = coef[1:3]
        except np.linalg.LinAlgError:
            pass

    grad_xy = grad_xy[np.isfinite(grad_xy).all(axis=1)]
    if grad_xy.shape[0] < 20:
        return np.eye(2), np.array([1.0, 1.0])

    # Robust covariance via median-centered outer product
    g_med = np.median(grad_xy, axis=0)
    centered = grad_xy - g_med
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(2))
    # Descending: vecs[:, 0] = high-gradient (dip-perpendicular)
    order_s = np.argsort(vals)[::-1]
    vals = vals[order_s]
    vecs = vecs[:, order_s]

    sigma = 1.0 / np.sqrt(np.maximum(vals, eps))
    sigma = sigma / sigma.min()
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

        # Set L_norm to the inter-well length scale, NOT the intra-well row
        # spacing. Each well has thousands of dense rows along its track,
        # so "median NN over all rows" is misleadingly small. The relevant
        # scale for held-out queries is the median distance from one well
        # CENTROID to its nearest other well centroid in the rotated frame.
        # This is on the order of typical well spacing.
        unique_wids = np.unique(well_ids)
        if len(unique_wids) >= 4:
            centroids = np.array([
                xy_pre[well_ids == wid].mean(axis=0) for wid in unique_wids
            ])
            tree_c = cKDTree(centroids)
            d_c, _ = tree_c.query(centroids, k=2)
            L_norm = float(np.median(d_c[:, 1]))
        else:
            bbox_min = xy_pre.min(axis=0)
            bbox_max = xy_pre.max(axis=0)
            bbox_span = float(np.maximum(bbox_max - bbox_min, 1.0).mean())
            L_norm = bbox_span / 30.0
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
