"""Per-well feature builder for the GBM stack.

Adapted from the konbu17 LB-11.912 kernel (rogii-plane-fit-formation-top-knn),
re-implemented as a clean module with several targeted enhancements:

  * **Primary formation switchable**: konbu17 uses ANCC; the multi-formation
    study showed EGFDL is spatially smoothest at 0-10 mi and ANCC has a 0.9%
    coverage gap. ``primary_formation`` controls which one drives the
    closed-form ``tvt_formula`` feature. Other formations are still imputed.

  * **Multi-formation b_well features**: per-formation ``b_F`` is computed
    from prefix and exposed alongside ANCC-based one. The GBM can learn
    when to trust each.

  * **Huber-anchored b_well variant**: ``b_huber_F`` for the primary
    formation, in addition to the ``median``-based one. ~0.05-0.15 RMSE in
    the literature.

The output is a long-form DataFrame with ``well``, ``prediction_id``,
``row_idx``, ``last_known_tvt``, ``target`` (train only), and ~80 numeric
features. Identical schema for train and test except for ``target``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
PLANE_K_DEFAULT = 10
ROW_K_DEFAULT = 20
# konbu17 default n_q=12000. Empirical well sizes: median 6576, p90 8056,
# p99 11032, max 12141. After self-exclusion we need n_q > self_well_rows
# + K=20. 8000 gives us safety for ~85% of wells; the remaining ~15%
# fall back to the global formation mean for sparse-neighbour rows.
# Per-well row KNN cost is O(n_query * n_q); 12k -> 8k is a 1.5x speedup.
ROW_NQ_DEFAULT = 8_000


# ---------------------------------------------------------------------------
# MLP imputer (v9 lever)
# ---------------------------------------------------------------------------

class MLPAnccImputer:
    """Wraps a multi-output ANCC field MLP behind a (X, Y) -> (N, F) API.

    Training once on the union of train wells produces a global smooth
    surface that complements konbu17's row-level KNN. In the v9 GBM stack
    we pass BOTH KNN and MLP predictions as features and let the boosted
    trees learn the gate (KNN dominates dense-neighbor wells; MLP
    dominates the sparse-neighbour catastrophic-outlier tail).

    For local OOF this needs per-fold retraining (self-well exclusion)
    since the MLP doesn't have a natural neighbor-exclusion mechanism
    like KNN does. For the Kaggle submission path test wells are not in
    train, so a single fit on all train rows suffices.

    v10: ``nets`` (a list of trained ANCC nets) supports multi-seed
    averaging at imputer time. Empirically a 3-seed ensemble gives
    -18 ft on the worst-well TVT RMSE (the 165-ft outlier 059c8f24)
    while costing only ~6 min extra training time. Recommended setting
    for production private-LB stability.
    """

    def __init__(self, ancc_net=None, nets=None, formations: tuple[str, ...] = FORMATIONS):
        if nets is None:
            nets = [ancc_net] if ancc_net is not None else []
        if not nets:
            raise ValueError("MLPAnccImputer requires at least one net")
        self.nets = nets
        self.net = nets[0]   # back-compat
        self.formations = formations

    @classmethod
    def fit(
        cls,
        train_paths,
        *,
        formations: tuple[str, ...] = FORMATIONS,
        exclude_wids: set[str] | None = None,
        num_freqs: int = 8,
        hidden: int = 256,
        epochs: int = 12,
        rows_per_epoch: int = 500_000,
        seed: int = 42,
        seeds: list[int] | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> "MLPAnccImputer":
        """Fit one or several ANCC MLPs on the train rows.

        ``seeds``: if provided, fits one MLP per seed and impute()
        returns the average across them. Empirically a 3-seed ensemble
        cuts the worst-well TVT RMSE by ~18 ft (the 165-ft outlier
        collapses to ~148 ft) at the cost of 3x training. Recommended:
        ``seeds=[42, 7, 123]`` for production v10.
        """
        from neural_ancc import AnccNet, TrainConfig, load_train_rows

        if exclude_wids:
            train_paths = [p for p in train_paths
                           if p.stem.replace("__horizontal_well", "") not in exclude_wids]

        xy, targets, _wids = load_train_rows(
            train_dir=None, formations=formations, paths=train_paths,
        )

        seed_list = list(seeds) if seeds else [seed]
        nets = []
        for s in seed_list:
            cfg = TrainConfig(
                num_freqs=num_freqs,
                hidden=hidden,
                out_dim=len(formations),
                rows_per_epoch=rows_per_epoch,
                epochs=epochs,
                seed=s,
            )
            if device is not None:
                cfg.device = device
            net = AnccNet(cfg)
            net.fit(xy, targets, verbose=verbose)
            nets.append(net)
        return cls(nets=nets, formations=tuple(formations))

    def impute(self, xy_q: np.ndarray) -> np.ndarray:
        """Predict (M, F) formation values at each (X, Y) query.

        With multiple nets (multi-seed), returns the simple mean.
        """
        if len(self.nets) == 1:
            return self.nets[0].predict(xy_q)
        preds = [net.predict(xy_q) for net in self.nets]
        return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Anisotropic-kriging imputer (v11 lever)
# ---------------------------------------------------------------------------

class AnisoFormationImputer:
    """Wraps a per-formation anisotropic-kriging predictor with the same
    ``impute(xy) -> (M, F), (M, F) stds, (M,) min_dist`` API as RowKNN.

    Agent benchmark on the full 765-well 5-fold OOF found
    ``aniso_exponential`` (K=20, range_scale=1.0, sigma_ratio=3) beats
    konbu17's row-level KNN on ANCC pool RMSE (23.29 vs 30.74) and TVT
    median (10.87 vs 12.30 ft). MLP+PE-L8 still wins on TVT max-well
    RMSE (165.66 vs 275.54), so v11 keeps both spatial layers and
    lets the GBM gate.
    """

    def __init__(self, predictors: dict, formations: tuple[str, ...] = FORMATIONS,
                 kernel: str = "exponential", k: int = 20):
        self.predictors = predictors
        self.formations = tuple(formations)
        self.kernel = kernel
        self.k = k

    @classmethod
    def fit(
        cls,
        train_paths,
        *,
        formations: tuple[str, ...] = FORMATIONS,
        exclude_wids: set[str] | None = None,
        kernel: str = "exponential",
        range_scale: float = 1.0,
        k: int = 20,
    ) -> "AnisoFormationImputer":
        from aniso_kriging import fit_aniso_for_formations

        if exclude_wids:
            train_paths = [p for p in train_paths
                           if p.stem.replace("__horizontal_well", "") not in exclude_wids]
        predictors = fit_aniso_for_formations(
            train_paths, formations=tuple(formations), range_scale=range_scale,
        )
        return cls(predictors, formations=tuple(formations),
                   kernel=kernel, k=k)

    def impute(self, xy_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (preds (M, F), stds (M, F), min_dist (M,))."""
        n = xy_q.shape[0]
        preds = np.full((n, len(self.formations)), np.nan, dtype=np.float64)
        stds = np.full((n, len(self.formations)), np.nan, dtype=np.float64)
        min_dist = np.full(n, np.inf, dtype=np.float64)
        for j, fname in enumerate(self.formations):
            p = self.predictors.get(fname)
            if p is None:
                continue
            m, s, d = p.query(xy_q, k=self.k, kernel=self.kernel)
            preds[:, j] = m
            stds[:, j] = s
            # Use the smallest formation min_dist as the imputer's min_dist
            min_dist = np.minimum(min_dist, d)
        return preds, stds, min_dist


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

def median_b(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else 0.0


def huber_b(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad <= 0:
        return med
    scale = 1.4826 * mad
    k = 1.345 * scale
    r = a - med
    r_clipped = np.clip(r, -k, k)
    return float(med + r_clipped.mean())


# ---------------------------------------------------------------------------
# Spatial imputers (konbu17-faithful)
# ---------------------------------------------------------------------------

@dataclass
class FormationPlaneKNN:
    """K nearest non-self centroids, weighted 2D plane fit per row."""

    df: pd.DataFrame
    wid_idx: dict[str, int]
    tree: cKDTree
    scale: np.ndarray
    x_arr: np.ndarray
    y_arr: np.ndarray
    formation_arr: np.ndarray
    formations: tuple[str, ...]

    @classmethod
    def fit(cls, train_paths: Iterable[Path], formations: tuple[str, ...] = FORMATIONS) -> "FormationPlaneKNN":
        rows = []
        for p in train_paths:
            wid = p.stem.replace("__horizontal_well", "")
            try:
                df = pd.read_csv(p, usecols=["X", "Y", *formations]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in formations:
                row[f"{c}_med"] = float(df[c].median())
            rows.append(row)
        df = pd.DataFrame(rows)
        wid_idx = {w: i for i, w in enumerate(df["wid"].to_numpy())}
        xy = df[["x", "y"]].to_numpy()
        scale = xy.std(axis=0)
        scale = np.where(scale < 1e-3, 1.0, scale)
        tree = cKDTree(xy / scale)
        x_arr = df["x"].to_numpy()
        y_arr = df["y"].to_numpy()
        formation_arr = df[[f"{c}_med" for c in formations]].to_numpy(dtype=np.float64)
        return cls(df=df, wid_idx=wid_idx, tree=tree, scale=scale,
                   x_arr=x_arr, y_arr=y_arr, formation_arr=formation_arr,
                   formations=formations)

    def impute(self, xy_q: np.ndarray, self_wid: str | None = None, k: int = PLANE_K_DEFAULT
               ) -> tuple[np.ndarray, np.ndarray]:
        q = xy_q / self.scale
        n_q = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=n_q)
        if self_wid is not None and self_wid in self.wid_idx:
            self_i = self.wid_idx[self_wid]
            mask_self = idx == self_i
            dist = np.where(mask_self, np.inf, dist)
        order = np.argpartition(dist, kth=min(k - 1, n_q - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)
        w = np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0).astype(np.float64)
        x_n = self.x_arr[idx_k]
        y_n = self.y_arr[idx_k]
        wx = w * x_n
        wy = w * y_n
        ATWA_xx = (wx * x_n).sum(axis=1)
        ATWA_xy = (wx * y_n).sum(axis=1)
        ATWA_xc = wx.sum(axis=1)
        ATWA_yy = (wy * y_n).sum(axis=1)
        ATWA_yc = wy.sum(axis=1)
        ATWA_cc = w.sum(axis=1)
        ATWA = np.zeros((len(xy_q), 3, 3))
        ATWA[:, 0, 0] = ATWA_xx
        ATWA[:, 0, 1] = ATWA_xy
        ATWA[:, 0, 2] = ATWA_xc
        ATWA[:, 1, 0] = ATWA_xy
        ATWA[:, 1, 1] = ATWA_yy
        ATWA[:, 1, 2] = ATWA_yc
        ATWA[:, 2, 0] = ATWA_xc
        ATWA[:, 2, 1] = ATWA_yc
        ATWA[:, 2, 2] = ATWA_cc
        ATWA[:, 0, 0] += 1e-9
        ATWA[:, 1, 1] += 1e-9
        ATWA[:, 2, 2] += 1e-9
        f_n = self.formation_arr[idx_k]
        ATWb_x = (wx[:, :, None] * f_n).sum(axis=1)
        ATWb_y = (wy[:, :, None] * f_n).sum(axis=1)
        ATWb_c = (w[:, :, None] * f_n).sum(axis=1)
        rhs = np.stack([ATWb_x, ATWb_y, ATWb_c], axis=1)
        try:
            coef = np.linalg.solve(ATWA, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((len(xy_q), 3, len(self.formations)))
            for r in range(len(xy_q)):
                try:
                    coef[r] = np.linalg.pinv(ATWA[r]) @ rhs[r]
                except Exception:
                    coef[r] = 0
        X_q = xy_q[:, 0]
        Y_q = xy_q[:, 1]
        formations = (X_q[:, None] * coef[:, 0, :]
                      + Y_q[:, None] * coef[:, 1, :]
                      + coef[:, 2, :]).astype(np.float32)
        no_n = (~valid_k).all(axis=1)
        if no_n.any():
            global_mean = self.formation_arr.mean(axis=0)
            formations[no_n] = global_mean
        d_finite = np.where(valid_k, d_k, np.inf)
        min_dist = d_finite.min(axis=1).astype(np.float32)
        return formations, min_dist


@dataclass
class RowKNN:
    """All-rows (X, Y, formation) KNN. konbu17 uses ANCC; we expose all six."""

    xy: np.ndarray
    targets: np.ndarray         # (N, F) float32
    wids: np.ndarray            # (N,) object str
    scale: np.ndarray
    tree: cKDTree
    formations: tuple[str, ...]

    @classmethod
    def fit(cls, train_paths: Iterable[Path], formations: tuple[str, ...] = FORMATIONS) -> "RowKNN":
        xs, ys = [], []
        f_blocks: list[np.ndarray] = []
        wid_arr = []
        cols = ["X", "Y", *formations]
        for p in train_paths:
            wid = p.stem.replace("__horizontal_well", "")
            try:
                df = pd.read_csv(p, usecols=cols).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            xs.append(df["X"].to_numpy())
            ys.append(df["Y"].to_numpy())
            f_blocks.append(df[list(formations)].to_numpy(dtype=np.float32))
            wid_arr.extend([wid] * len(df))
        xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        targets = np.vstack(f_blocks)
        wids = np.array(wid_arr)
        scale = xy.std(axis=0)
        scale = np.where(scale < 1e-3, 1.0, scale)
        tree = cKDTree(xy / scale)
        return cls(xy=xy, targets=targets, wids=wids, scale=scale,
                   tree=tree, formations=formations)

    def impute(self, xy_q: np.ndarray, self_wid: str | None = None,
               k: int = ROW_K_DEFAULT, n_q: int = ROW_NQ_DEFAULT
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (preds (M,F), stds (M,F), min_dist (M,)) for all formations."""
        q = xy_q / self.scale
        n_q = min(n_q, len(self.targets))
        dist, idx = self.tree.query(q, k=n_q, workers=-1)
        if self_wid is not None:
            mask_self = self.wids[idx] == self_wid
            dist = np.where(mask_self, np.inf, dist)
        order = np.argpartition(dist, kth=min(k - 1, n_q - 1), axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)
        w = np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0)
        sw = w.sum(axis=1)
        no_n = sw < 1e-9
        safe = np.where(no_n, 1.0, sw)
        # (M, K, F) target tensor
        f_n = self.targets[idx_k]                              # (M, K, F)
        preds = (f_n * w[:, :, None]).sum(axis=1) / safe[:, None]
        if no_n.any():
            global_mean = self.targets.mean(axis=0)
            preds[no_n] = global_mean
        diff_sq = (f_n - preds[:, None, :]) ** 2
        var = (diff_sq * w[:, :, None]).sum(axis=1) / safe[:, None]
        stds = np.sqrt(np.maximum(var, 0.0))
        d_finite = np.where(valid_k, d_k, np.inf)
        min_dist = d_finite.min(axis=1)
        return (preds.astype(np.float32),
                stds.astype(np.float32),
                min_dist.astype(np.float32))


# ---------------------------------------------------------------------------
# Per-row feature construction
# ---------------------------------------------------------------------------

def _recent_mean_diff(values: np.ndarray, window: int) -> float:
    v = values[-(window + 1):]
    if len(v) < 2:
        return 0.0
    return float(np.diff(v).mean())


def _recent_slope(y: np.ndarray, x: np.ndarray, window: int) -> float:
    y = y[-window:]
    x = x[-window:]
    if len(y) < 2:
        return 0.0
    cx = x - x.mean()
    d = float(np.dot(cx, cx))
    return 0.0 if d == 0.0 else float(np.dot(cx, y - y.mean()) / d)


def _nearest_index(sorted_values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(sorted_values, target, side="left"))
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    if idx > 0 and abs(sorted_values[idx - 1] - target) <= abs(sorted_values[idx] - target):
        return idx - 1
    return idx


def _fill_smooth_gr(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    s = pd.Series(values, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius <= 0:
        return s.to_numpy(dtype=np.float32)
    return s.rolling(radius * 2 + 1, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def _beam_predict(gr_values: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                  start_tvt: float, beam_size: int, move_cost: float,
                  emit_scale: float, radius: int) -> np.ndarray:
    """Beam-search Viterbi alignment of GR to typewell GR (konbu17)."""
    start_idx = _nearest_index(tw_tvt, start_tvt)
    smoothed = _fill_smooth_gr(gr_values, float(np.nanmean(tw_gr)), radius)
    states: dict[int, float] = {start_idx: 0.0}
    backpointers: list[dict[int, int]] = []
    for gr_value in smoothed:
        candidates: dict[int, float] = {}
        parents: dict[int, int] = {}
        for idx, cost in states.items():
            for delta in (-1, 0, 1):
                ni = idx + delta
                if ni < 0 or ni >= len(tw_tvt):
                    continue
                emit = ((gr_value - tw_gr[ni]) ** 2) / emit_scale
                tot = cost + emit + move_cost * abs(delta)
                prev = candidates.get(ni)
                if prev is None or tot < prev:
                    candidates[ni] = tot
                    parents[ni] = idx
        kept = sorted(candidates.items(), key=lambda kv: kv[1])[:beam_size]
        states = {idx: cost for idx, cost in kept}
        backpointers.append({idx: parents[idx] for idx, _ in kept})
    if not states:
        return np.full(len(smoothed), tw_tvt[start_idx], dtype=np.float32)
    final_idx = min(states, key=states.get)
    path = [final_idx]
    for step in range(len(backpointers) - 1, 0, -1):
        path.append(backpointers[step][path[-1]])
    path.reverse()
    return tw_tvt[np.asarray(path, dtype=np.int32)]


def _gr_fft_features(gr_post: np.ndarray) -> tuple[float, float]:
    valid = gr_post[~np.isnan(gr_post)]
    if len(valid) < 32:
        return 0.0, 0.0
    centered = valid - valid.mean()
    spec = np.abs(np.fft.rfft(centered)) ** 2
    if len(spec) < 3:
        return 0.0, 0.0
    dom = int(np.argmax(spec[1:])) + 1
    return float(dom / len(valid)), float(np.log1p(spec[dom]))


def build_hidden_features(
    h: pd.DataFrame,
    t: pd.DataFrame,
    wid: str,
    *,
    is_train: bool,
    formation_imputer: FormationPlaneKNN,
    row_imputer: RowKNN,
    mlp_imputer: "MLPAnccImputer | None" = None,
    aniso_imputer: "AnisoFormationImputer | None" = None,
    pf_results: dict | None = None,
    primary_formation: str = "ANCC",
    formations: tuple[str, ...] = FORMATIONS,
    enable_beam: bool = True,
) -> pd.DataFrame | None:
    """Build the per-row feature DataFrame for one well's hidden segment.

    Hidden segment = rows where TVT_input is NaN. Returns None if there's no
    visible prefix or no hidden segment to predict.
    """
    f_idx_primary = formations.index(primary_formation)

    mask = h["TVT_input"].isna().to_numpy()
    if not mask.any():
        return None
    mask_start = int(np.flatnonzero(mask)[0])
    if mask_start == 0:
        return None
    known = h.iloc[:mask_start].copy()
    hidden = h.iloc[mask_start:].copy()
    last_known = known.iloc[-1]

    tw_tvt = t["TVT"].to_numpy(dtype=np.float32)
    tw_gr = t["GR"].to_numpy(dtype=np.float32)

    gr_full = h["GR"].interpolate(limit_direction="both")
    if gr_full.isna().any():
        gr_full = gr_full.fillna(float(np.nanmean(tw_gr)))

    gr_roll5 = gr_full.rolling(5, center=True, min_periods=1).mean()
    gr_roll21 = gr_full.rolling(21, center=True, min_periods=1).mean()
    gr_grad = gr_full.diff().fillna(0.0)
    gr_std5 = gr_full.rolling(5, center=True, min_periods=1).std().fillna(0.0)
    gr_std21 = gr_full.rolling(21, center=True, min_periods=1).std().fillna(0.0)
    gr_lag1 = gr_full.shift(1).bfill()
    gr_lead1 = gr_full.shift(-1).ffill()
    gr_lag5 = gr_full.shift(5).bfill()
    gr_lead5 = gr_full.shift(-5).ffill()
    gr_cumsum = gr_full.cumsum()

    known_tvt = known["TVT_input"].to_numpy(dtype=np.float32)
    known_md = known["MD"].to_numpy(dtype=np.float32)
    known_z = known["Z"].to_numpy(dtype=np.float32)

    prefix_tw_gr = np.interp(known_tvt, tw_tvt, tw_gr)
    prefix_gr = gr_full.iloc[:mask_start].to_numpy(dtype=np.float32)
    prefix_residual = prefix_gr - prefix_tw_gr
    prefix_tw_rmse = float(np.sqrt(np.mean(prefix_residual ** 2)))
    prefix_tw_mae = float(np.mean(np.abs(prefix_residual)))

    last_known_tvt = float(last_known["TVT_input"])
    hidden_gr = hidden["GR"].to_numpy(dtype=np.float32)

    if enable_beam:
        beam_cons = _beam_predict(hidden_gr, tw_tvt, tw_gr, last_known_tvt, 10, 20.0, 144.0, 2)
        beam_loose = _beam_predict(hidden_gr, tw_tvt, tw_gr, last_known_tvt, 10, 8.0, 64.0, 2)
    else:
        beam_cons = np.full(len(hidden), last_known_tvt, dtype=np.float32)
        beam_loose = np.full(len(hidden), last_known_tvt, dtype=np.float32)

    hidden_gr_filled = gr_full.iloc[mask_start:].to_numpy(dtype=np.float32)
    offsets = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], dtype=np.float32)
    offset_diffs = {
        f"tw_diff_{int(off)}": hidden_gr_filled
        - np.float32(np.interp(last_known_tvt + float(off), tw_tvt, tw_gr))
        for off in offsets
    }

    # ---- spatial features ------------------------------------------------
    xy_full = h[["X", "Y"]].to_numpy(dtype=np.float64)
    self_wid_for_train = wid if is_train else None

    plane_full, plane_min_dist_full = formation_imputer.impute(
        xy_full, self_wid=self_wid_for_train
    )
    plane_post = plane_full[mask_start:]
    plane_min_dist_post = plane_min_dist_full[mask_start:]
    z_full = h["Z"].to_numpy(dtype=np.float32)
    z_post = hidden["Z"].to_numpy(dtype=np.float32)

    # b_well per formation from prefix using PLANE imputation
    b_plane_per_F: dict[str, float] = {}
    b_plane_huber_per_F: dict[str, float] = {}
    for fi, fname in enumerate(formations):
        per_row = known_tvt + known_z - plane_full[:mask_start, fi]
        b_plane_per_F[fname] = median_b(per_row)
        b_plane_huber_per_F[fname] = huber_b(per_row)

    tvt_formula_plane_primary = (
        -z_post + plane_post[:, f_idx_primary] + b_plane_per_F[primary_formation]
    )

    # Row-level KNN, all formations
    row_preds_full, row_stds_full, row_min_dist_full = row_imputer.impute(
        xy_full, self_wid=self_wid_for_train
    )
    row_preds_post = row_preds_full[mask_start:]
    row_stds_post = row_stds_full[mask_start:]
    row_min_dist_post = row_min_dist_full[mask_start:]

    b_row_per_F: dict[str, float] = {}
    b_row_huber_per_F: dict[str, float] = {}
    for fi, fname in enumerate(formations):
        per_row = known_tvt + known_z - row_preds_full[:mask_start, fi]
        b_row_per_F[fname] = median_b(per_row)
        b_row_huber_per_F[fname] = huber_b(per_row)

    tvt_formula_row_primary = (
        -z_post + row_preds_post[:, f_idx_primary] + b_row_per_F[primary_formation]
    )

    # Multi-formation row-formula ensemble (inverse-variance over std)
    cand_T = []
    cand_W = []
    for fi, fname in enumerate(formations):
        b = b_row_per_F[fname]
        tvt_f = -z_post + row_preds_post[:, fi] + b
        std_f = row_stds_post[:, fi]
        std_f = np.where(np.isfinite(std_f), std_f, 1.0)
        std_f = np.maximum(std_f, 1e-3)
        cand_T.append(tvt_f)
        cand_W.append(1.0 / (std_f * std_f))
    T = np.stack(cand_T, axis=1)
    W = np.stack(cand_W, axis=1)
    valid = np.isfinite(T) & np.isfinite(W)
    T = np.where(valid, T, 0.0)
    W = np.where(valid, W, 0.0)
    wsum = W.sum(axis=1)
    tvt_formula_row_ensemble = np.where(
        wsum > 0, (T * W).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan
    )

    # ---- assemble feature dict (build once, DataFrame-ify at end) -------
    # Pandas DataFrames suffer O(N^2) memory copies when many columns are
    # added one at a time on a wide frame ("highly fragmented" warning).
    # We collect everything in a dict and call pd.DataFrame ONCE at the end.
    fd: dict = {
        "well": wid,
        "prediction_id": [f"{wid}_{i}" for i in hidden.index],
        "row_idx": hidden.index.to_numpy(dtype=np.int32),
        "last_known_tvt": np.float32(last_known_tvt),
        "known_len": np.int32(mask_start),
        "hidden_len": np.int32(len(hidden)),
        "frac_hidden": ((hidden.index - mask_start) / max(len(hidden) - 1, 1)).astype(np.float32),
        "md": hidden["MD"].to_numpy(dtype=np.float32),
        "z": z_post,
        "x": hidden["X"].to_numpy(dtype=np.float32),
        "y": hidden["Y"].to_numpy(dtype=np.float32),
        "gr": hidden_gr_filled,
        "gr_missing": hidden["GR"].isna().to_numpy(dtype=np.int8),
        "gr_roll5": gr_roll5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_roll21": gr_roll21.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_grad": gr_grad.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_std5": gr_std5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_std21": gr_std21.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lag1": gr_lag1.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lead1": gr_lead1.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lag5": gr_lag5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_lead5": gr_lead5.iloc[mask_start:].to_numpy(dtype=np.float32),
        "gr_cumsum": (gr_cumsum.iloc[mask_start:] - gr_cumsum.iloc[mask_start - 1]).to_numpy(dtype=np.float32),
        "dmd": (hidden["MD"] - float(last_known["MD"])).to_numpy(dtype=np.float32),
        "dz": (hidden["Z"] - float(last_known["Z"])).to_numpy(dtype=np.float32),
        "dx": (hidden["X"] - float(last_known["X"])).to_numpy(dtype=np.float32),
        "dy": (hidden["Y"] - float(last_known["Y"])).to_numpy(dtype=np.float32),
        "dx_dmd": ((hidden["X"] - float(last_known["X"]))
                   / np.maximum(hidden["MD"] - float(last_known["MD"]), 1e-5)).to_numpy(dtype=np.float32),
        "dy_dmd": ((hidden["Y"] - float(last_known["Y"]))
                   / np.maximum(hidden["MD"] - float(last_known["MD"]), 1e-5)).to_numpy(dtype=np.float32),
        "dz_dmd": ((hidden["Z"] - float(last_known["Z"]))
                   / np.maximum(hidden["MD"] - float(last_known["MD"]), 1e-5)).to_numpy(dtype=np.float32),
        "dist_xy": np.sqrt((hidden["X"] - float(last_known["X"])) ** 2
                           + (hidden["Y"] - float(last_known["Y"])) ** 2).to_numpy(dtype=np.float32),
        "dist_xyz": np.sqrt((hidden["X"] - float(last_known["X"])) ** 2
                            + (hidden["Y"] - float(last_known["Y"])) ** 2
                            + (hidden["Z"] - float(last_known["Z"])) ** 2).to_numpy(dtype=np.float32),
        "prefix_tvt_step20": np.float32(_recent_mean_diff(known_tvt, 20)),
        "prefix_tvt_step100": np.float32(_recent_mean_diff(known_tvt, 100)),
        "prefix_tvt_md_slope100": np.float32(_recent_slope(known_tvt, known_md, 100)),
        "prefix_tvt_z_slope100": np.float32(_recent_slope(known_tvt, known_z, 100)),
        "prefix_tw_rmse": np.float32(prefix_tw_rmse),
        "prefix_tw_mae": np.float32(prefix_tw_mae),
        "beam_cons_delta": (beam_cons - np.float32(last_known_tvt)).astype(np.float32),
        "beam_loose_delta": (beam_loose - np.float32(last_known_tvt)).astype(np.float32),
        "beam_gap": (beam_loose - beam_cons).astype(np.float32),
    }
    for name, vals in offset_diffs.items():
        fd[name] = vals.astype(np.float32)

    # NCC-style typewell shift estimate
    slc = (tw_tvt >= last_known_tvt - 40.0) & (tw_tvt <= last_known_tvt + 40.0)
    if slc.sum() >= 5 and (~np.isnan(hidden_gr)).any():
        gr_ok = hidden_gr[~np.isnan(hidden_gr)]
        tvt_s, gr_s = tw_tvt[slc], tw_gr[slc]
        d = np.abs(gr_ok[:, None] - gr_s[None, :])
        nn = np.argmin(d, axis=1)
        matched = tvt_s[nn]
        fd["ncc_med_shift_well"] = np.float32(np.median(matched) - last_known_tvt)
        fd["ncc_mean_shift_well"] = np.float32(np.mean(matched) - last_known_tvt)
    else:
        fd["ncc_med_shift_well"] = np.float32(0.0)
        fd["ncc_mean_shift_well"] = np.float32(0.0)

    fft_freq, fft_pow = _gr_fft_features(hidden_gr)
    fd["gr_fft_dom_freq"] = np.float32(fft_freq)
    fd["gr_fft_dom_power"] = np.float32(fft_pow)

    if len(tw_tvt):
        tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
        fd["anchor_t_pos"] = np.float32((last_known_tvt - tmin) / max(tmax - tmin, 1e-3))
    else:
        fd["anchor_t_pos"] = np.float32(0.0)
    fd["spatial_knn_delta"] = np.float32(0.0)

    # Plane formation features (anchored deltas + dz)
    for fi, fname in enumerate(formations):
        fd[f"fk_{fname}"] = plane_post[:, fi].astype(np.float32)
        fd[f"fk_{fname}_dz"] = (z_post - plane_post[:, fi]).astype(np.float32)
        fd[f"fk_b_{fname}"] = np.float32(b_plane_per_F[fname])
        fd[f"fk_b_huber_{fname}"] = np.float32(b_plane_huber_per_F[fname])
        tvt_F = -z_post + plane_post[:, fi] + b_plane_per_F[fname]
        fd[f"fk_tvt_formula_{fname}"] = (tvt_F - np.float32(last_known_tvt)).astype(np.float32)
    fd["fk_min_dist"] = plane_min_dist_post.astype(np.float32)
    fd["fk_tvt_formula"] = (
        tvt_formula_plane_primary - np.float32(last_known_tvt)
    ).astype(np.float32)

    # Row-level features (per formation), anchored deltas
    for fi, fname in enumerate(formations):
        fd[f"knn_row_{fname}"] = row_preds_post[:, fi].astype(np.float32)
        fd[f"knn_row_{fname}_dz"] = (z_post - row_preds_post[:, fi]).astype(np.float32)
        fd[f"knn_row_{fname}_std"] = row_stds_post[:, fi].astype(np.float32)
        fd[f"knn_row_b_{fname}"] = np.float32(b_row_per_F[fname])
        fd[f"knn_row_b_huber_{fname}"] = np.float32(b_row_huber_per_F[fname])
        tvt_F = -z_post + row_preds_post[:, fi] + b_row_per_F[fname]
        fd[f"knn_row_tvt_pred_delta_{fname}"] = (
            tvt_F - np.float32(last_known_tvt)
        ).astype(np.float32)
    fd["knn_row_dist"] = row_min_dist_post.astype(np.float32)
    fd["knn_row_tvt_pred_delta"] = (
        tvt_formula_row_primary - np.float32(last_known_tvt)
    ).astype(np.float32)

    fd["knn_row_tvt_ensemble_delta"] = (
        tvt_formula_row_ensemble - np.float32(last_known_tvt)
    ).astype(np.float32)

    fd["fk_vs_row_primary_diff"] = (
        plane_post[:, f_idx_primary] - row_preds_post[:, f_idx_primary]
    ).astype(np.float32)
    fd["fk_vs_row_primary_tvt_diff"] = (
        tvt_formula_plane_primary - tvt_formula_row_primary
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # v9 MLP-global-ANCC features (optional)
    # ------------------------------------------------------------------
    if mlp_imputer is not None:
        mlp_preds_full = mlp_imputer.impute(xy_full)
        mlp_preds_post = mlp_preds_full[mask_start:]
        b_mlp_per_F: dict[str, float] = {}
        b_mlp_huber_per_F: dict[str, float] = {}
        for fi, fname in enumerate(formations):
            per_row = known_tvt + known_z - mlp_preds_full[:mask_start, fi]
            b_mlp_per_F[fname] = median_b(per_row)
            b_mlp_huber_per_F[fname] = huber_b(per_row)
        for fi, fname in enumerate(formations):
            fd[f"mlp_{fname}"] = mlp_preds_post[:, fi].astype(np.float32)
            fd[f"mlp_{fname}_dz"] = (z_post - mlp_preds_post[:, fi]).astype(np.float32)
            fd[f"mlp_b_{fname}"] = np.float32(b_mlp_per_F[fname])
            fd[f"mlp_b_huber_{fname}"] = np.float32(b_mlp_huber_per_F[fname])
            tvt_F_mlp = -z_post + mlp_preds_post[:, fi] + b_mlp_per_F[fname]
            fd[f"mlp_tvt_formula_{fname}"] = (
                tvt_F_mlp - np.float32(last_known_tvt)
            ).astype(np.float32)
        tvt_formula_mlp_primary = (
            -z_post + mlp_preds_post[:, f_idx_primary] + b_mlp_per_F[primary_formation]
        )
        fd["mlp_tvt_formula"] = (
            tvt_formula_mlp_primary - np.float32(last_known_tvt)
        ).astype(np.float32)
        fd["mlp_vs_row_primary_diff"] = (
            mlp_preds_post[:, f_idx_primary] - row_preds_post[:, f_idx_primary]
        ).astype(np.float32)
        fd["mlp_vs_row_primary_tvt_diff"] = (
            tvt_formula_mlp_primary - tvt_formula_row_primary
        ).astype(np.float32)
        fd["mlp_vs_plane_primary_diff"] = (
            mlp_preds_post[:, f_idx_primary] - plane_post[:, f_idx_primary]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # v11 Aniso-exponential kriging features (optional)
    # Agent benchmark: aniso_exponential (K=20, range_scale=1.0,
    # sigma_ratio=3) beats KNN on ANCC pool RMSE (23.29 vs 30.74) and
    # TVT median (10.87 vs 12.30). MLP still owns max-well-RMSE so
    # we keep BOTH spatial layers as features.
    # ------------------------------------------------------------------
    if aniso_imputer is not None:
        aniso_preds_full, aniso_stds_full, aniso_min_dist_full = aniso_imputer.impute(xy_full)
        aniso_preds_post = aniso_preds_full[mask_start:]
        aniso_stds_post = aniso_stds_full[mask_start:]
        aniso_min_dist_post = aniso_min_dist_full[mask_start:]
        b_aniso_per_F: dict[str, float] = {}
        b_aniso_huber_per_F: dict[str, float] = {}
        for fi, fname in enumerate(formations):
            per_row = known_tvt + known_z - aniso_preds_full[:mask_start, fi]
            b_aniso_per_F[fname] = median_b(per_row)
            b_aniso_huber_per_F[fname] = huber_b(per_row)
        for fi, fname in enumerate(formations):
            fd[f"aniso_{fname}"] = aniso_preds_post[:, fi].astype(np.float32)
            fd[f"aniso_{fname}_dz"] = (z_post - aniso_preds_post[:, fi]).astype(np.float32)
            fd[f"aniso_{fname}_std"] = aniso_stds_post[:, fi].astype(np.float32)
            fd[f"aniso_b_{fname}"] = np.float32(b_aniso_per_F[fname])
            fd[f"aniso_b_huber_{fname}"] = np.float32(b_aniso_huber_per_F[fname])
            tvt_F_a = -z_post + aniso_preds_post[:, fi] + b_aniso_per_F[fname]
            fd[f"aniso_tvt_formula_{fname}"] = (
                tvt_F_a - np.float32(last_known_tvt)
            ).astype(np.float32)
        tvt_formula_aniso_primary = (
            -z_post + aniso_preds_post[:, f_idx_primary] + b_aniso_per_F[primary_formation]
        )
        fd["aniso_min_dist"] = aniso_min_dist_post.astype(np.float32)
        fd["aniso_tvt_formula"] = (
            tvt_formula_aniso_primary - np.float32(last_known_tvt)
        ).astype(np.float32)
        fd["aniso_vs_row_primary_diff"] = (
            aniso_preds_post[:, f_idx_primary] - row_preds_post[:, f_idx_primary]
        ).astype(np.float32)
        fd["aniso_vs_row_primary_tvt_diff"] = (
            tvt_formula_aniso_primary - tvt_formula_row_primary
        ).astype(np.float32)
        if mlp_imputer is not None:
            fd["aniso_vs_mlp_primary_diff"] = (
                aniso_preds_post[:, f_idx_primary] - mlp_preds_post[:, f_idx_primary]
            ).astype(np.float32)

    # ------------------------------------------------------------------
    # v12: Triple-Signal particle filter features (optional).
    # `pf_results` is a dict[wid -> {pf_z_pred, pf_z_std, pf_ancc_pred,
    # pf_ancc_std}] produced by triple_signal_pf.run_pfs_for_wells. Each
    # array has length = N_eval_rows (the hidden segment).
    #
    # Per the public top notebook (LB 11.284): the ANCC-PF (S=TVT+Z) is
    # the primary PF signal; Z-velocity TVT-PF is the fallback.
    # ------------------------------------------------------------------
    if pf_results is not None and wid in pf_results:
        pf = pf_results[wid]
        pf_z_pred = np.asarray(pf.get("pf_z_pred", []), dtype=np.float64)
        pf_z_std = np.asarray(pf.get("pf_z_std", []), dtype=np.float64)
        pf_ancc_pred = np.asarray(pf.get("pf_ancc_pred", []), dtype=np.float64)
        pf_ancc_std = np.asarray(pf.get("pf_ancc_std", []), dtype=np.float64)

        # Pick the primary PF (ANCC if available + finite, else Z-vel).
        if (pf_ancc_pred.size == len(hidden)
                and np.isfinite(pf_ancc_pred).all()):
            pf_pred = pf_ancc_pred
            pf_std = pf_ancc_std
            pf_source = "ancc"
        elif (pf_z_pred.size == len(hidden)
                and np.isfinite(pf_z_pred).all()):
            pf_pred = pf_z_pred
            pf_std = pf_z_std
            pf_source = "zvel"
        else:
            pf_pred = np.full(len(hidden), last_known_tvt, dtype=np.float64)
            pf_std = np.full(len(hidden), 1.0, dtype=np.float64)
            pf_source = "fallback"

        fd["pf_pred"] = pf_pred.astype(np.float32)
        fd["pf_std"] = pf_std.astype(np.float32)
        fd["pf_delta"] = (pf_pred - last_known_tvt).astype(np.float32)
        # Std trend / ratio relative to first-row PF std (gauges drift)
        s0 = float(pf_std[0]) if pf_std.size else 1.0
        s0_safe = max(s0, 0.01)
        fd["pf_std_trend"] = (pf_std - s0).astype(np.float32)
        fd["pf_std_ratio"] = (pf_std / s0_safe).astype(np.float32)

        # Both PFs as separate features (the GBM can use Z-vel even when
        # ANCC is the primary)
        if pf_z_pred.size == len(hidden):
            fd["pf_z_pred"] = pf_z_pred.astype(np.float32)
            fd["pf_z_std"] = pf_z_std.astype(np.float32)
            fd["pf_z_delta"] = (pf_z_pred - last_known_tvt).astype(np.float32)
        if pf_ancc_pred.size == len(hidden):
            fd["pf_ancc_pred"] = pf_ancc_pred.astype(np.float32)
            fd["pf_ancc_std"] = pf_ancc_std.astype(np.float32)
            fd["pf_ancc_delta"] = (pf_ancc_pred - last_known_tvt).astype(np.float32)

        # Cross-comparisons with other spatial signals
        fd["pf_vs_row_primary_diff"] = (pf_pred - tvt_formula_row_primary).astype(np.float32)
        fd["pf_vs_row_primary_abs"] = np.abs(pf_pred - tvt_formula_row_primary).astype(np.float32)
        fd["pf_vs_plane_primary_diff"] = (pf_pred - tvt_formula_plane_primary).astype(np.float32)
        if mlp_imputer is not None:
            fd["pf_vs_mlp_primary_diff"] = (pf_pred - tvt_formula_mlp_primary).astype(np.float32)

        # Typewell GR at predicted TVT (signal: how plausible is the PF
        # prediction given the typewell's GR profile?)
        tw_gr_at_pf = np.interp(pf_pred, tw_tvt, tw_gr,
                                left=tw_gr[0], right=tw_gr[-1])
        fd["tw_gr_at_pf"] = tw_gr_at_pf.astype(np.float32)
        fd["gr_minus_tw_at_pf"] = (
            hidden_gr_filled - tw_gr_at_pf
        ).astype(np.float32)
        for offset in [-60, 60]:
            tw_off = np.interp(pf_pred + offset, tw_tvt, tw_gr,
                               left=tw_gr[0], right=tw_gr[-1])
            fd[f"gr_tw_off_{offset}"] = (
                hidden_gr_filled - tw_off
            ).astype(np.float32)

        # Slope-based baseline cross-check
        if "prefix_tvt_md_slope100" in fd:
            slope_recent = float(fd["prefix_tvt_md_slope100"])
            md_eval = hidden["MD"].to_numpy(dtype=np.float64)
            md_anchor = float(last_known["MD"])
            baseline = last_known_tvt + slope_recent * (md_eval - md_anchor)
            fd["pf_minus_slope"] = (pf_pred - baseline).astype(np.float32)
            fd["spatial_minus_slope"] = (
                tvt_formula_row_primary - baseline
            ).astype(np.float32)
            if mlp_imputer is not None:
                fd["mlp_minus_slope"] = (
                    tvt_formula_mlp_primary - baseline
                ).astype(np.float32)

    if is_train:
        fd["target"] = (hidden["TVT"].to_numpy(dtype=np.float32)
                        - np.float32(last_known_tvt)).astype(np.float32)

    # Single DataFrame allocation — no fragmentation.
    feats = pd.DataFrame(fd)
    return feats


def build_dataset(
    paths: list[Path],
    formation_imputer: FormationPlaneKNN,
    row_imputer: RowKNN,
    *,
    is_train: bool,
    mlp_imputer: "MLPAnccImputer | None" = None,
    aniso_imputer: "AnisoFormationImputer | None" = None,
    pf_results: dict | None = None,
    primary_formation: str = "ANCC",
    formations: tuple[str, ...] = FORMATIONS,
    enable_beam: bool = True,
    label: str = "data",
    progress_every: int = 100,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for i, p in enumerate(paths):
        wid = p.stem.replace("__horizontal_well", "")
        h = pd.read_csv(p)
        try:
            t = pd.read_csv(p.parent / f"{wid}__typewell.csv")
        except Exception:
            continue
        if is_train and "TVT" not in h.columns:
            continue
        feats = build_hidden_features(
            h, t, wid,
            is_train=is_train,
            formation_imputer=formation_imputer,
            row_imputer=row_imputer,
            mlp_imputer=mlp_imputer,
            aniso_imputer=aniso_imputer,
            pf_results=pf_results,
            primary_formation=primary_formation,
            formations=formations,
            enable_beam=enable_beam,
        )
        if feats is not None:
            parts.append(feats)
        if (i + 1) % progress_every == 0:
            print(f"  {label}: {i + 1}/{len(paths)}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
