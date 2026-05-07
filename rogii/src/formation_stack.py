"""Formation-surface TVT predictor for the ROGII Eagle Ford competition.

Load-bearing identity (verified empirically across 766 train wells):

    TVT = -Z + ANCC + b_well

with intra-well std of the residual = 0.0065 ft (median). The whole problem
reduces to predicting ``ANCC(X, Y)`` for the test rows, then anchoring
``b_well`` from the visible ``TVT_input`` prefix. b_well varies ~2866 ft
between wells so it MUST be re-anchored per-well; the formula has no
free parameter once ANCC and b are known.

Two complementary spatial estimators (matching konbu17 LB-11.912 setup):

  1. **Row-level KNN** on all ~3.8M (X, Y, ANCC) training rows. K=20,
     inverse-distance weights ``1 / (d + 1e-3)``. Self-well excluded
     during local validation only.
  2. **Per-well centroid weighted plane fit**. For each query, K=10 nearest
     centroids; solve weighted ``f = a + b·X + c·Y`` for all 6 formations
     simultaneously. Used as an independent estimator and as a far-from-
     neighbors fallback.

Both are kept *separately* — they are features, not blended. A downstream
GBM (or simple ensemble) decides per-row how much to trust each.

Improvements in flight (later commits):
  - Robust ``b_well`` (Huber over the prefix) — small gain.
  - Multi-formation ensemble of ``-Z + F + b_F`` (uses all six tops). Whether
    this is worth the complexity is being measured by the formation-stats
    study agent (output: rogii/bench/formation_stats.json).
  - Anisotropic-kernel ANCC predictor (Eagle Ford strikes NE-SW).
  - Per-well EWM smoothing post-process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.neighbors import BallTree


FORMATION_COLS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
SPATIAL_COLS = ("X", "Y")
TEST_LIKE_COLS = ("MD", "X", "Y", "Z", "GR", "TVT_input")


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def _well_id_from_path(path: Path) -> str:
    return path.name.replace("__horizontal_well.csv", "")


def load_train_horizontals(
    train_dir: Path,
    *,
    formations: tuple[str, ...] = FORMATION_COLS,
    require_tvt: bool = True,
) -> dict[str, pl.DataFrame]:
    """Load every ``*__horizontal_well.csv`` under ``train_dir``.

    Only retains rows where every requested formation column AND TVT (if
    ``require_tvt``) plus X, Y, Z are finite. Wells with fewer than 16
    usable rows are dropped.
    """
    out: dict[str, pl.DataFrame] = {}
    paths = sorted(Path(train_dir).glob("*__horizontal_well.csv"))
    for path in paths:
        wid = _well_id_from_path(path)
        df = pl.read_csv(
            path,
            infer_schema_length=2000,
            null_values=["", "NA", "NaN", "nan", "null"],
            truncate_ragged_lines=True,
        )
        needed = set(SPATIAL_COLS) | {"Z"} | set(formations)
        if require_tvt:
            needed.add("TVT")
        missing = needed - set(df.columns)
        if missing:
            continue
        # Some wells have formation columns inferred as Utf8 due to sentinel
        # tokens. Coerce all numeric columns explicitly.
        cast_cols = list(needed)
        cast_exprs = [
            pl.col(c).cast(pl.Float64, strict=False)
            for c in cast_cols
            if c in df.columns
        ]
        df = df.with_columns(cast_exprs)
        keep = (
            df["X"].is_not_null() & df["X"].is_finite()
            & df["Y"].is_not_null() & df["Y"].is_finite()
            & df["Z"].is_not_null() & df["Z"].is_finite()
        )
        if require_tvt:
            keep = keep & df["TVT"].is_not_null() & df["TVT"].is_finite()
        for f in formations:
            keep = keep & df[f].is_not_null() & df[f].is_finite()
        df = df.filter(keep)
        if df.height >= 16:
            out[wid] = df
    return out


def load_test_horizontals(test_dir: Path) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for path in sorted(Path(test_dir).glob("*__horizontal_well.csv")):
        wid = _well_id_from_path(path)
        df = pl.read_csv(
            path,
            infer_schema_length=2000,
            null_values=["", "NA", "NaN", "nan", "null"],
            truncate_ragged_lines=True,
        )
        out[wid] = df
    return out


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

def robust_center(x: np.ndarray, method: str = "median") -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if method == "median":
        return float(np.median(x))
    if method == "trimmed_mean":
        if x.size < 4:
            return float(np.median(x))
        lo, hi = np.percentile(x, [10, 90])
        keep = x[(x >= lo) & (x <= hi)]
        if keep.size == 0:
            return float(np.median(x))
        return float(keep.mean())
    if method == "huber":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = max(1.4826 * mad, 1e-9)
        k = 1.345 * scale
        r = x - med
        r = np.clip(r, -k, k)
        return float(med + r.mean())
    raise ValueError(f"unknown robust_center method: {method!r}")


# ---------------------------------------------------------------------------
# Row-level KNN over all (X, Y, formation) samples
# ---------------------------------------------------------------------------

@dataclass
class _RowLevelKNN:
    xy: np.ndarray              # (N, 2) float64
    targets: np.ndarray         # (N, F) float64
    well_ids: np.ndarray        # (N,) int32 — index into well_index
    well_index: list[str]
    formations: tuple[str, ...]
    tree: BallTree

    def well_to_int(self, wid: str) -> int:
        try:
            return self.well_index.index(wid)
        except ValueError:
            return -1

    def query(
        self,
        xy_query: np.ndarray,
        *,
        k: int,
        exclude_well: str | None = None,
        weight_power: float = 1.0,
        eps: float = 1e-3,
        oversample: int = 6,
        scale: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """IDW KNN. Returns per-formation predicted value, neighbor std,
        mean neighbor distance, and used-K count.
        """
        excl_int = self.well_to_int(exclude_well) if exclude_well else -2

        # Scale query coords to match the tree's coordinate frame.
        if scale is None:
            xy_q = xy_query
        else:
            xy_q = xy_query / scale

        kq = min(int(k * oversample), self.xy.shape[0])
        dists, idx = self.tree.query(xy_q, k=kq)
        dists = dists.astype(np.float64, copy=False)
        idx = idx.astype(np.int64, copy=False)

        if exclude_well:
            mask = self.well_ids[idx] != excl_int
        else:
            mask = np.ones_like(idx, dtype=bool)

        n_targets = self.targets.shape[1]
        M = xy_query.shape[0]
        pred = np.full((M, n_targets), np.nan, dtype=np.float64)
        nbr_std = np.full((M, n_targets), np.nan, dtype=np.float64)
        nbr_d = np.full(M, np.nan, dtype=np.float64)
        n_used = np.zeros(M, dtype=np.int32)

        for i in range(M):
            valid = mask[i]
            if not valid.any():
                continue
            d_i = dists[i, valid][:k]
            ix_i = idx[i, valid][:k]
            n_used[i] = d_i.size
            w = 1.0 / np.power(d_i + eps, weight_power)
            w_sum = w.sum()
            if w_sum <= 0:
                continue
            t = self.targets[ix_i]
            mean = (w[:, None] * t).sum(axis=0) / w_sum
            pred[i] = mean
            diff = t - mean[None, :]
            nbr_std[i] = np.sqrt(
                ((w[:, None] * diff * diff).sum(axis=0) / w_sum).clip(min=0)
            )
            nbr_d[i] = d_i.mean()

        out = {}
        for j, f in enumerate(self.formations):
            out[f"row_{f}"] = pred[:, j]
            out[f"row_std_{f}"] = nbr_std[:, j]
        out["row_mean_dist"] = nbr_d
        out["row_n_used"] = n_used
        return out


# ---------------------------------------------------------------------------
# Per-well centroid weighted plane fit
# ---------------------------------------------------------------------------

@dataclass
class _CentroidPlaneFit:
    centroids: np.ndarray       # (W, 2)
    targets: np.ndarray         # (W, F)
    well_index: list[str]
    formations: tuple[str, ...]
    tree: BallTree

    def well_to_int(self, wid: str) -> int:
        try:
            return self.well_index.index(wid)
        except ValueError:
            return -1

    def query(
        self,
        xy_query: np.ndarray,
        *,
        k: int = 10,
        exclude_well: str | None = None,
        weight_power: float = 1.0,
        eps: float = 1e-3,
        scale: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        excl_int = self.well_to_int(exclude_well) if exclude_well else -2

        if scale is None:
            xy_q = xy_query
        else:
            xy_q = xy_query / scale

        kq = min(int(k + 6), self.centroids.shape[0])
        dists, idx = self.tree.query(xy_q, k=kq)
        dists = dists.astype(np.float64, copy=False)
        idx = idx.astype(np.int64, copy=False)

        n_targets = self.targets.shape[1]
        M = xy_query.shape[0]
        pred = np.full((M, n_targets), np.nan, dtype=np.float64)
        nbr_d = np.full(M, np.nan, dtype=np.float64)

        for i in range(M):
            ix_i = idx[i]
            d_i = dists[i]
            if exclude_well:
                keep_mask = ix_i != excl_int
                ix_i = ix_i[keep_mask]
                d_i = d_i[keep_mask]
            ix_i = ix_i[:k]
            d_i = d_i[:k]
            if ix_i.size < 3:
                continue
            # Plane fit uses ORIGINAL (un-scaled) X, Y coords for interpretable
            # coefficients. The scaling is only for KNN distance ranking.
            xy_n = self.centroids[ix_i]
            t_n = self.targets[ix_i]
            w = 1.0 / np.power(d_i + eps, weight_power)
            A = np.column_stack([np.ones(ix_i.size), xy_n[:, 0], xy_n[:, 1]])
            wA = A * w[:, None]
            ata = wA.T @ A
            ata[0, 0] += 1e-9
            ata[1, 1] += 1e-9
            ata[2, 2] += 1e-9
            atb = wA.T @ t_n
            try:
                coefs = np.linalg.solve(ata, atb)
            except np.linalg.LinAlgError:
                pred[i] = (w[:, None] * t_n).sum(axis=0) / w.sum()
                nbr_d[i] = d_i.mean()
                continue
            pred[i] = coefs[0] + coefs[1] * xy_query[i, 0] + coefs[2] * xy_query[i, 1]
            nbr_d[i] = d_i.mean()

        out = {}
        for j, f in enumerate(self.formations):
            out[f"plane_{f}"] = pred[:, j]
        out["plane_mean_dist"] = nbr_d
        return out


# ---------------------------------------------------------------------------
# Top-level predictor
# ---------------------------------------------------------------------------

@dataclass
class FormationStackPredictor:
    """Fits two spatial estimators on train horizontals; produces features
    and a "naive formula" prediction usable as either the final answer or
    the input to a downstream residual model.

    The naive formula prediction:
        TVT = -Z + ANCC_row_knn + b_well
    where b_well = robust_center(TVT_input - (-Z + ANCC_row_knn))
    over the visible-prefix rows. This is the konbu17 step-1 baseline.

    For the test set, ``predict_well`` always emits ``tvt_formula_row`` and
    ``tvt_formula_plane`` plus all spatial features. A downstream model can
    learn the residual from these.
    """

    train_wells: dict[str, pl.DataFrame] = field(default_factory=dict)
    formations: tuple[str, ...] = FORMATION_COLS
    k_row: int = 20
    k_plane: int = 10
    weight_power_row: float = 1.0
    weight_power_plane: float = 1.0
    eps: float = 1e-3
    b_method: str = "median"               # "median" | "huber" | "trimmed_mean"
    primary_formation: str = "EGFDL"       # EGFDL is spatially smoothest (3-5% gain over ANCC)
    _row: _RowLevelKNN | None = field(default=None, init=False, repr=False)
    _plane: _CentroidPlaneFit | None = field(default=None, init=False, repr=False)

    def fit(self) -> "FormationStackPredictor":
        if not self.train_wells:
            raise ValueError("train_wells is empty")
        well_index = sorted(self.train_wells)
        well_pos = {w: i for i, w in enumerate(well_index)}

        xy_blocks = []
        target_blocks = []
        wid_blocks = []
        for wid in well_index:
            df = self.train_wells[wid]
            xy_blocks.append(np.column_stack([
                df["X"].to_numpy().astype(np.float64, copy=False),
                df["Y"].to_numpy().astype(np.float64, copy=False),
            ]))
            target_blocks.append(np.column_stack([
                df[f].to_numpy().astype(np.float64, copy=False)
                for f in self.formations
            ]))
            wid_blocks.append(np.full(df.height, well_pos[wid], dtype=np.int32))
        xy = np.vstack(xy_blocks)
        targets = np.vstack(target_blocks)
        wids = np.concatenate(wid_blocks)

        # X/Y std-scaling (konbu17): row-level uses scale of the full point cloud.
        scale = xy.std(axis=0)
        scale = np.where(scale < 1e-3, 1.0, scale)
        self._row_scale = scale

        self._row = _RowLevelKNN(
            xy=xy, targets=targets, well_ids=wids,
            well_index=well_index, formations=self.formations,
            tree=BallTree(xy / scale, leaf_size=64, metric="euclidean"),
        )

        cent_xy = np.zeros((len(well_index), 2), dtype=np.float64)
        cent_tgt = np.zeros((len(well_index), len(self.formations)), dtype=np.float64)
        for i, wid in enumerate(well_index):
            df = self.train_wells[wid]
            cent_xy[i] = (df["X"].mean(), df["Y"].mean())
            for j, f in enumerate(self.formations):
                cent_tgt[i, j] = df[f].median()
        # Plane fit uses centroid-cloud std-scaling.
        cent_scale = cent_xy.std(axis=0)
        cent_scale = np.where(cent_scale < 1e-3, 1.0, cent_scale)
        self._plane_scale = cent_scale
        self._plane = _CentroidPlaneFit(
            centroids=cent_xy, targets=cent_tgt,
            well_index=well_index, formations=self.formations,
            tree=BallTree(cent_xy / cent_scale, leaf_size=32, metric="euclidean"),
        )
        return self

    def features_for_well(
        self,
        horizontal: pl.DataFrame,
        *,
        well_id: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute spatial features for all rows of one horizontal well.

        Returned keys:
          ``row_<F>``, ``row_std_<F>`` — KNN ANCC mean & weighted std
          ``plane_<F>`` — plane-fit prediction
          ``row_mean_dist``, ``plane_mean_dist``, ``row_n_used``
          ``tvt_formula_row``, ``tvt_formula_plane``
          ``b_row``, ``b_plane`` (scalars broadcast to row)
          ``has_anchor`` (bool, broadcast)
        """
        assert self._row is not None and self._plane is not None, "call fit() first"
        df = horizontal
        n = df.height
        x = df["X"].to_numpy().astype(np.float64, copy=False)
        y = df["Y"].to_numpy().astype(np.float64, copy=False)
        z = df["Z"].to_numpy().astype(np.float64, copy=False)
        xy_query = np.column_stack([x, y])

        row_feats = self._row.query(
            xy_query, k=self.k_row, exclude_well=well_id,
            weight_power=self.weight_power_row, eps=self.eps,
            scale=getattr(self, "_row_scale", None),
        )
        plane_feats = self._plane.query(
            xy_query, k=self.k_plane, exclude_well=well_id,
            weight_power=self.weight_power_plane, eps=self.eps,
            scale=getattr(self, "_plane_scale", None),
        )

        feats: dict[str, np.ndarray] = {}
        feats.update(row_feats)
        feats.update(plane_feats)

        # Anchor estimation per estimator using the primary formation
        tvt_in = (
            df["TVT_input"].to_numpy().astype(np.float64, copy=False)
            if "TVT_input" in df.columns else np.full(n, np.nan)
        )
        finite = np.isfinite(tvt_in)
        feats["has_anchor"] = np.full(n, finite.any(), dtype=bool)

        F0 = self.primary_formation
        ancc_row = feats[f"row_{F0}"]
        ancc_plane = feats[f"plane_{F0}"]

        if finite.any():
            okr = finite & np.isfinite(ancc_row)
            okp = finite & np.isfinite(ancc_plane)
            b_row = robust_center(
                tvt_in[okr] - (-z[okr] + ancc_row[okr]),
                method=self.b_method,
            ) if okr.sum() >= 4 else float("nan")
            b_plane = robust_center(
                tvt_in[okp] - (-z[okp] + ancc_plane[okp]),
                method=self.b_method,
            ) if okp.sum() >= 4 else float("nan")
        else:
            b_row = float("nan")
            b_plane = float("nan")

        feats["b_row"] = np.full(n, b_row, dtype=np.float64)
        feats["b_plane"] = np.full(n, b_plane, dtype=np.float64)

        # Closed-form formula predictions
        feats["tvt_formula_row"] = -z + ancc_row + b_row
        feats["tvt_formula_plane"] = -z + ancc_plane + b_plane

        # Multi-formation: closed-form per formation, weighted by 1/std^2
        # (informational; downstream models can use these)
        cand_tvt = []
        cand_w = []
        for f in self.formations:
            fp_row = feats[f"row_{f}"]
            okr = finite & np.isfinite(fp_row)
            if okr.sum() < 4:
                continue
            b_f = robust_center(
                tvt_in[okr] - (-z[okr] + fp_row[okr]),
                method=self.b_method,
            )
            tvt_f = -z + fp_row + b_f
            std_f = feats.get(f"row_std_{f}", np.full(n, 1.0))
            std_f = np.where(np.isfinite(std_f), std_f, 1.0)
            std_f = np.maximum(std_f, 1e-3)
            cand_tvt.append(tvt_f)
            cand_w.append(1.0 / (std_f * std_f))
            feats[f"tvt_formula_row_{f}"] = tvt_f
            feats[f"b_row_{f}"] = np.full(n, b_f, dtype=np.float64)
        if cand_tvt:
            T = np.stack(cand_tvt, axis=1)
            W = np.stack(cand_w, axis=1)
            valid = np.isfinite(T) & np.isfinite(W)
            T = np.where(valid, T, 0.0)
            W = np.where(valid, W, 0.0)
            wsum = W.sum(axis=1)
            ens = np.where(wsum > 0, (T * W).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan)
            feats["tvt_formula_row_ensemble"] = ens
        else:
            feats["tvt_formula_row_ensemble"] = np.full(n, np.nan)

        return feats

    def predict_well(
        self,
        horizontal: pl.DataFrame,
        *,
        well_id: str | None = None,
        train_median_tvt: float = 11354.51,
        strategy: str = "row_only",
    ) -> np.ndarray:
        """Return final TVT prediction (length N).

        ``strategy``:
          - ``"row_only"`` — TVT_pred = -Z + ANCC_row_knn + b_row (konbu17 step 1)
          - ``"plane_only"`` — TVT_pred = -Z + ANCC_plane + b_plane
          - ``"row_avg_plane"`` — simple mean of the two formula predictions
          - ``"formation_ensemble"`` — inverse-variance ensemble over all six
            formations using row-level KNN.
        """
        feats = self.features_for_well(horizontal, well_id=well_id)
        n = horizontal.height
        tvt_in = (
            horizontal["TVT_input"].to_numpy().astype(np.float64, copy=False)
            if "TVT_input" in horizontal.columns else np.full(n, np.nan)
        )
        finite = np.isfinite(tvt_in)

        if strategy == "row_only":
            pred = feats["tvt_formula_row"]
        elif strategy == "plane_only":
            pred = feats["tvt_formula_plane"]
        elif strategy == "row_avg_plane":
            r = feats["tvt_formula_row"]
            p = feats["tvt_formula_plane"]
            both = np.isfinite(r) & np.isfinite(p)
            pred = np.where(both, 0.5 * (r + p), np.where(np.isfinite(r), r, p))
        elif strategy == "formation_ensemble":
            pred = feats["tvt_formula_row_ensemble"]
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        # No-anchor fallback
        if not finite.any():
            pred = np.full(n, train_median_tvt, dtype=np.float64)

        bad = ~np.isfinite(pred)
        if bad.any():
            # Fall back to row formula → plane formula → constant median
            r = feats["tvt_formula_row"]
            p = feats["tvt_formula_plane"]
            pred = np.where(bad & np.isfinite(r), r, pred)
            pred = np.where(bad & np.isfinite(p), p, pred)
            still = ~np.isfinite(pred)
            if still.any():
                pred = pred.copy()
                pred[still] = train_median_tvt

        # Hard-pin prefix to TVT_input
        pred = np.where(finite, tvt_in, pred)
        return pred


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def predict_submission(
    train_dir: Path,
    test_dir: Path,
    *,
    formations: Iterable[str] = FORMATION_COLS,
    k_row: int = 20,
    k_plane: int = 10,
    strategy: str = "row_only",
    b_method: str = "median",
) -> tuple[list[str], list[float]]:
    train_wells = load_train_horizontals(Path(train_dir), formations=tuple(formations))
    if not train_wells:
        raise RuntimeError(f"No train wells loaded from {train_dir}")
    pred = FormationStackPredictor(
        train_wells=train_wells,
        formations=tuple(formations),
        k_row=k_row, k_plane=k_plane,
        b_method=b_method,
    ).fit()

    median_tvt = float(np.median(np.concatenate([
        df["TVT"].to_numpy().astype(np.float64) for df in train_wells.values()
    ])))

    ids: list[str] = []
    tvts: list[float] = []
    for wid, df in load_test_horizontals(Path(test_dir)).items():
        tvt_arr = pred.predict_well(
            df, well_id=None,
            train_median_tvt=median_tvt,
            strategy=strategy,
        )
        if "TVT_input" in df.columns:
            mask = ~(df["TVT_input"].is_finite() & df["TVT_input"].is_not_null()).to_numpy()
        else:
            mask = np.ones(df.height, dtype=bool)
        eval_idx = np.flatnonzero(mask)
        for i in eval_idx:
            ids.append(f"{wid}_{int(i)}")
            tvts.append(float(tvt_arr[i]))
    return ids, tvts
