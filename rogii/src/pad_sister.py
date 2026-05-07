"""Pad-sister retrieval predictor for the ROGII competition.

In Eagle Ford operations multiple horizontal laterals are drilled from a
single surface pad to the same target zone. Their lateral trajectories
are roughly parallel (250-1000 ft inter-well spacing) and they intersect
nearly the same geology. Their TVT(MD) trajectories are therefore
geologically constrained to be very similar — much more so than the
typical "two random wells in the basin" pair.

Empirically, in the ROGII train set:
  53.6% of wells have a sister within 500 ft of centroid
  82.5% within 1000 ft
  94.3% within 2000 ft

konbu17's row-level KNN captures this only incidentally — it averages
many nearby point samples but doesn't leverage *whole-trajectory*
consistency. The pad-sister retrieval below does.

Algorithm:
  1. For each query well, score all candidate train wells by:
       - Centroid distance (X, Y)
       - Lateral azimuth similarity (parallel laterals score high)
       - Prefix-GR similarity (Pearson on the visible prefix of test
         vs training rows at matching MD)
       - Z-range overlap (rules out vertical-displaced wells)
  2. Pick top K sisters.
  3. Project each sister's TVT(MD) onto the query well's MD frame by
     interpolation, optionally re-anchored at the query's last known
     TVT_input.
  4. Robust mean (median or weighted-median) of the K aligned curves
     gives the prediction.

This module is intentionally orthogonal to the v8 GBM stack — it can
be used as: (a) a stand-alone predictor for ablation, (b) an extra
feature in v8/v9 (sister-mean TVT, sister-spread, sister-count), or
(c) a fall-back for the catastrophic-outlier wells where v8's max
well RMSE blows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.neighbors import BallTree


def _well_id_from_path(path: Path) -> str:
    return path.name.replace("__horizontal_well.csv", "")


def _read_horizontal(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        infer_schema_length=2000,
        null_values=["", "NA", "NaN", "nan", "null"],
        truncate_ragged_lines=True,
    )


def _well_summary(df: pl.DataFrame, wid: str) -> dict:
    """Summarize a horizontal well into the features we need for sister scoring."""
    needed = {"X", "Y", "Z", "MD", "GR"}
    if not needed.issubset(df.columns):
        return None
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()
    md = df["MD"].to_numpy()
    gr = df["GR"].to_numpy()

    # Lateral azimuth: vector from first to last (X, Y). Approx with PCA later.
    finite_xy = np.isfinite(x) & np.isfinite(y)
    if finite_xy.sum() < 16:
        return None
    x0, y0 = x[finite_xy][0], y[finite_xy][0]
    x1, y1 = x[finite_xy][-1], y[finite_xy][-1]
    azimuth = float(np.arctan2(y1 - y0, x1 - x0))    # radians

    # Trajectory PCA — capture the strongest axis (lateral direction)
    xy = np.column_stack([x[finite_xy], y[finite_xy]])
    centred = xy - xy.mean(axis=0)
    cov = centred.T @ centred / max(len(centred) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    primary_axis = vecs[:, np.argmax(vals)]
    primary_angle = float(np.arctan2(primary_axis[1], primary_axis[0]))

    out = {
        "wid": wid,
        "x_med": float(np.median(x[finite_xy])),
        "y_med": float(np.median(y[finite_xy])),
        "x_start": float(x0),
        "y_start": float(y0),
        "x_end": float(x1),
        "y_end": float(y1),
        "z_p05": float(np.quantile(z[np.isfinite(z)], 0.05)) if np.isfinite(z).any() else float("nan"),
        "z_p95": float(np.quantile(z[np.isfinite(z)], 0.95)) if np.isfinite(z).any() else float("nan"),
        "md_min": float(np.nanmin(md)),
        "md_max": float(np.nanmax(md)),
        "azimuth": azimuth,
        "primary_angle": primary_angle,
        "n_rows": int(df.height),
    }
    return out


def _angular_diff(a: float, b: float) -> float:
    """Smallest unsigned angle between two directions in radians, mod pi.

    Two parallel laterals with opposite drill direction are the same lateral
    geologically, so we mod by pi.
    """
    d = abs(a - b)
    d = d % np.pi
    return float(min(d, np.pi - d))


def _gr_signature_at_prefix(
    h_query: pl.DataFrame,
    h_sister: pl.DataFrame,
    *,
    n_target: int = 64,
) -> float:
    """Approximate Pearson similarity of the prefix-GR profile against a
    sister's GR at matching MD. Returns a value in [0, 1] (clamped & nan-safe).
    """
    if "GR" not in h_query.columns or "GR" not in h_sister.columns:
        return 0.0
    if "TVT_input" not in h_query.columns:
        return 0.0
    md_q = h_query["MD"].to_numpy()
    gr_q = h_query["GR"].to_numpy()
    tvt_in = h_query["TVT_input"].to_numpy()
    md_s = h_sister["MD"].to_numpy()
    gr_s = h_sister["GR"].to_numpy()

    finite_prefix = np.isfinite(tvt_in) & np.isfinite(gr_q) & np.isfinite(md_q)
    if finite_prefix.sum() < 16:
        return 0.0
    md_q_pref = md_q[finite_prefix]
    gr_q_pref = gr_q[finite_prefix]

    # Down-sample prefix to n_target evenly-spaced rows
    if md_q_pref.size > n_target:
        idx = np.linspace(0, md_q_pref.size - 1, n_target).astype(int)
        md_q_pref = md_q_pref[idx]
        gr_q_pref = gr_q_pref[idx]

    # Interpolate sister GR at those MDs
    finite_s = np.isfinite(md_s) & np.isfinite(gr_s)
    if finite_s.sum() < 16:
        return 0.0
    gr_s_at_q = np.interp(
        md_q_pref, md_s[finite_s], gr_s[finite_s],
        left=np.nan, right=np.nan,
    )
    valid = np.isfinite(gr_s_at_q)
    if valid.sum() < 16:
        return 0.0
    a = gr_q_pref[valid] - gr_q_pref[valid].mean()
    b = gr_s_at_q[valid] - gr_s_at_q[valid].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-9:
        return 0.0
    return float(np.clip((a * b).sum() / den, -1.0, 1.0))


@dataclass
class PadSisterIndex:
    summaries: dict[str, dict]
    train_paths: dict[str, Path]
    centroid_xy: np.ndarray
    wid_order: list[str]
    tree: BallTree

    @classmethod
    def fit(cls, train_dir: Path) -> "PadSisterIndex":
        summaries: dict[str, dict] = {}
        train_paths: dict[str, Path] = {}
        for path in sorted(Path(train_dir).glob("*__horizontal_well.csv")):
            wid = _well_id_from_path(path)
            df = _read_horizontal(path)
            s = _well_summary(df, wid)
            if s is None:
                continue
            summaries[wid] = s
            train_paths[wid] = path

        wid_order = sorted(summaries)
        centroid_xy = np.array([
            (summaries[w]["x_med"], summaries[w]["y_med"]) for w in wid_order
        ])
        tree = BallTree(centroid_xy, leaf_size=32)
        return cls(
            summaries=summaries, train_paths=train_paths,
            centroid_xy=centroid_xy, wid_order=wid_order, tree=tree,
        )

    def find_sisters(
        self,
        query_summary: dict,
        h_query: pl.DataFrame,
        *,
        k: int = 5,
        candidate_radius_ft: float = 4000.0,
        max_candidates: int = 32,
        weights: dict | None = None,
        exclude_wid: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Return up to ``k`` ranked sisters with (wid, score, components)."""
        if weights is None:
            weights = {
                "centroid": 0.4,
                "azimuth": 0.2,
                "z_overlap": 0.2,
                "gr_corr": 0.2,
            }

        qx = query_summary["x_med"]
        qy = query_summary["y_med"]
        cand_idx = self.tree.query_radius(
            np.array([[qx, qy]]), r=candidate_radius_ft, return_distance=True,
        )
        idxs, dists = cand_idx[0][0], cand_idx[1][0]
        order = np.argsort(dists)
        idxs = idxs[order][:max_candidates]
        dists = dists[order][:max_candidates]

        scored: list[tuple[str, float, dict]] = []
        q_az = query_summary["primary_angle"]
        q_z_lo = query_summary["z_p05"]
        q_z_hi = query_summary["z_p95"]

        for i, d in zip(idxs, dists):
            wid = self.wid_order[i]
            if exclude_wid and wid == exclude_wid:
                continue
            s = self.summaries[wid]
            az_diff = _angular_diff(q_az, s["primary_angle"])
            # Centroid score: 1.0 at d=0, ~0 at d=4000ft
            c_score = float(np.exp(-d / 1500.0))
            # Azimuth score: 1.0 if parallel, falls off
            az_score = float(np.exp(-(az_diff / (np.pi / 8)) ** 2))
            # Z-range overlap (in feet)
            zlo = max(q_z_lo, s["z_p05"])
            zhi = min(q_z_hi, s["z_p95"])
            z_overlap = max(0.0, zhi - zlo)
            z_total = max((q_z_hi - q_z_lo), (s["z_p95"] - s["z_p05"]), 1.0)
            z_score = float(z_overlap / z_total)
            # GR-correlation score
            try:
                h_sister = _read_horizontal(self.train_paths[wid])
                gr_corr = _gr_signature_at_prefix(h_query, h_sister)
            except Exception:
                gr_corr = 0.0
            gr_score = float(0.5 * (gr_corr + 1.0))  # remap [-1,1] → [0,1]

            score = (
                weights["centroid"] * c_score
                + weights["azimuth"] * az_score
                + weights["z_overlap"] * z_score
                + weights["gr_corr"] * gr_score
            )
            comps = {
                "centroid_dist_ft": float(d),
                "azimuth_diff_rad": az_diff,
                "z_overlap_score": z_score,
                "gr_corr": gr_corr,
            }
            scored.append((wid, float(score), comps))

        scored.sort(key=lambda t: -t[1])
        return scored[:k]

    def predict_well(
        self,
        h_query: pl.DataFrame,
        query_wid: str | None = None,
        *,
        k: int = 5,
        exclude_wid: str | None = None,
        re_anchor: bool = True,
    ) -> dict[str, np.ndarray]:
        """Predict TVT for ``h_query`` by averaging sister TVT trajectories
        warped to the query MD frame.

        Returns dict with:
            ``tvt`` : (N,) predicted TVT
            ``tvt_sisters`` : (k, N) per-sister predictions (NaN where outside MD overlap)
            ``sister_wids`` : list[str]
            ``sister_scores`` : list[float]
            ``coverage`` : (N,) bool — True where at least 1 sister covered the row
        """
        n = h_query.height
        s = _well_summary(h_query, query_wid or "__query__")
        if s is None:
            return {"tvt": np.full(n, np.nan), "tvt_sisters": np.zeros((0, n)),
                    "sister_wids": [], "sister_scores": [], "coverage": np.zeros(n, dtype=bool)}
        sisters = self.find_sisters(
            s, h_query, k=k, exclude_wid=exclude_wid,
        )
        if not sisters:
            return {"tvt": np.full(n, np.nan), "tvt_sisters": np.zeros((0, n)),
                    "sister_wids": [], "sister_scores": [], "coverage": np.zeros(n, dtype=bool)}

        md_q = h_query["MD"].to_numpy().astype(np.float64)
        z_q = h_query["Z"].to_numpy().astype(np.float64)
        tvt_in = (
            h_query["TVT_input"].to_numpy().astype(np.float64)
            if "TVT_input" in h_query.columns else np.full(n, np.nan)
        )

        # b_query: anchor offset between query and each sister, computed on
        # the visible prefix.
        sister_wids = [w for w, _, _ in sisters]
        sister_scores = [sc for _, sc, _ in sisters]

        per_sister = np.full((len(sisters), n), np.nan, dtype=np.float64)
        for j, (wid, _score, _comps) in enumerate(sisters):
            df_s = _read_horizontal(self.train_paths[wid])
            if "TVT" not in df_s.columns:
                continue
            md_s = df_s["MD"].to_numpy().astype(np.float64)
            tvt_s = df_s["TVT"].to_numpy().astype(np.float64)
            ok = np.isfinite(md_s) & np.isfinite(tvt_s)
            if ok.sum() < 8:
                continue
            # Warp sister TVT to query's MD frame
            tvt_warped = np.interp(md_q, md_s[ok], tvt_s[ok], left=np.nan, right=np.nan)
            per_sister[j] = tvt_warped

        coverage = np.isfinite(per_sister).any(axis=0)

        # Re-anchor: shift each sister's TVT so its median over the query
        # prefix matches the query's TVT_input median.
        if re_anchor:
            finite_q = np.isfinite(tvt_in)
            if finite_q.sum() >= 4:
                q_anchor = float(np.median(tvt_in[finite_q]))
                for j in range(len(sisters)):
                    sis_at_prefix = per_sister[j][finite_q]
                    valid = np.isfinite(sis_at_prefix)
                    if valid.sum() >= 4:
                        sis_anchor = float(np.median(sis_at_prefix[valid]))
                        per_sister[j] = per_sister[j] + (q_anchor - sis_anchor)

        # Score-weighted nan-aware mean
        weights = np.array(sister_scores, dtype=np.float64)
        weights = np.maximum(weights, 1e-6)
        tvt_pred = np.full(n, np.nan)
        for i in range(n):
            col = per_sister[:, i]
            valid = np.isfinite(col)
            if not valid.any():
                continue
            w = weights[valid]
            tvt_pred[i] = float(np.average(col[valid], weights=w))

        # Pin prefix to TVT_input
        finite_q = np.isfinite(tvt_in)
        if finite_q.any():
            tvt_pred[finite_q] = tvt_in[finite_q]

        return {
            "tvt": tvt_pred,
            "tvt_sisters": per_sister,
            "sister_wids": sister_wids,
            "sister_scores": sister_scores,
            "coverage": coverage,
        }
