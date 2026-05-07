"""rogii.geology_classifier — GR-only Geology classifier for ROGII Wellbore Geology Prediction.

Why this module exists
======================
The downstream :mod:`rogii.geology` module fits per-formation GR distributions,
locates the EGFDL/BUDA contact, and applies stratigraphic priors to TVT
predictions. It depends on the typewell having a populated ``Geology`` column
with labels from the canonical six-formation set.

* **Train typewells** ship with ``Geology`` labelled (~70 % of rows on average,
  18 unique labels — see ``train/*__typewell.csv``).
* **Test typewells** ship with ``Geology`` empty (all-null).

Without filling the labels on the test side, every geological constraint
short-circuits to no-op and the test pipeline degenerates to a label-free DTW.
This module trains a LightGBM multiclass classifier on the train typewells and
predicts labels for any typewell whose ``Geology`` column is missing or empty.

Design choices (defended)
-------------------------
1. **Per-row classification**, not per-segment. The downstream code consumes a
   ``Geology`` value at every row, so we predict at every row and let
   per-formation aggregation happen in :func:`fit_formation_gr_model`.
2. **GroupKFold by WELLNAME** — never split rows from the same well across
   train/val. GR baselines drift between operators / log vendors, and a
   row-level random split would inflate accuracy by ~10 %.
3. **Coarse class set by default** (``main_six``: ``ANCC, ASTNU, ASTNL, EGFDU,
   EGFDL, BUDA``). Sub-zones (``LBHL``, ``UEGFD TGT`` etc.) carry only ~1–3 %
   of rows and 95 %+ of them roll up cleanly to the parents — using them at
   classification time would double the model size for a marginal gain that
   the consumer (:mod:`rogii.geology`) cannot use anyway (its FORMATION_ORDER
   is the six parents).
4. **Position-within-typewell features carry most of the information.** Pure
   GR is genuinely degenerate: ANCC mean ≈ 78 ≈ ASTNU mean ≈ 77, and ASTNL
   (~51) overlaps EGFDU (~68) almost completely. The ordered context
   (``z(TVT)`` within the well, distance to the well's deepest GR-drop, etc.)
   is what separates them. We engineer these explicitly.
5. **LightGBM**, not a neural net, because the relevant non-linearities are
   tabular and locally axis-aligned (GR thresholds, Sav-Gol features), the
   feature count is small (~25), and we need fast inference (one .joblib
   ships in the Kaggle code submission).

Geological label mapping (this module's authoritative version)
--------------------------------------------------------------

Sub-zone → parent (used in ``main_six``):

* ``ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`` → kept as-is.
* ``LBHL`` (Lower Bottom Hot Lime, sub-zone of Lower Eagle Ford) → ``EGFDL``.
* ``LTGT, LTHL`` (Lower Eagle Ford target / top hot lime) → ``EGFDL``.
* ``UEGFD BHL, UEGFD TGT, UEGFD THL`` (Upper Eagle Ford sub-zones)
  → ``EGFDU``.
* ``AC_UEF_BHL, AC_UEF_TRGT, AC_UEF_THL`` (operator-specific Upper Eagle Ford
  Austin Chalk-adjacent layering) → ``EGFDU``.
* ``UTGT, UTHL, UBHL, UPSN`` (Upper sub-zones — empirical GR ≈ 60–80,
  position above EGFDL) → ``EGFDU``.
* ``Clay Rich Interval`` (high GR ≈ 120, sits in the Upper Eagle Ford
  marl section) → ``EGFDU``.
* ``MNSS`` (Maness Shale; GR ≈ 130, basal Eagle Ford / sub-EGFL organic
  shale in S. Texas) → ``EGFDL``.
* ``OLMOS`` (Campanian shoreface, regionally shallower than ANCC but
  above ASTN; the only main_six neighbour is ANCC) → ``ANCC``.

Set ``label_set='all'`` to keep the fine-grained labels in training. The
.joblib metadata records whichever set was used so :func:`predict_geology`
returns labels in the same vocabulary.
"""

from __future__ import annotations

import glob
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy import signal
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

logger = logging.getLogger("rogii.geology_classifier")


# ---------------------------------------------------------------------------
# Label mapping — module-public so the integrator can audit it.
# ---------------------------------------------------------------------------
MAIN_SIX: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")

#: Maps every observed train-typewell label to a parent in MAIN_SIX. Labels not
#: in this dict are dropped from training (and treated as 'unknown' at predict
#: time, where they should never appear).
SUBZONE_TO_PARENT: dict[str, str] = {
    # main six pass through identity
    "ANCC": "ANCC", "ASTNU": "ASTNU", "ASTNL": "ASTNL",
    "EGFDU": "EGFDU", "EGFDL": "EGFDL", "BUDA": "BUDA",
    # Lower Eagle Ford sub-zones → EGFDL
    "LBHL": "EGFDL", "LTGT": "EGFDL", "LTHL": "EGFDL",
    # Upper Eagle Ford sub-zones → EGFDU
    "UEGFD BHL": "EGFDU", "UEGFD TGT": "EGFDU", "UEGFD THL": "EGFDU",
    "AC_UEF_BHL": "EGFDU", "AC_UEF_TRGT": "EGFDU", "AC_UEF_THL": "EGFDU",
    "UTGT": "EGFDU", "UTHL": "EGFDU", "UBHL": "EGFDU", "UPSN": "EGFDU",
    # Marl / clay horizon within Upper EGF
    "Clay Rich Interval": "EGFDU",
    # Maness Shale — basal/sub-EGFL organic shale in S. Texas
    "MNSS": "EGFDL",
    # Olmos Formation — regionally above Anacacho; closest main_six neighbour
    "OLMOS": "ANCC",
}

#: When ``label_set='all'`` the classifier learns this expanded vocabulary
#: directly. Anything outside is folded into the parent above, so
#: ``len(ALL_LABELS) ≤ len(SUBZONE_TO_PARENT)`` and labels with too few rows
#: are pruned at runtime by ``min_samples_per_class``.
ALL_LABELS: tuple[str, ...] = tuple(SUBZONE_TO_PARENT.keys())


def _coarsen_labels(labels: np.ndarray, label_set: str) -> np.ndarray:
    """Apply the canonical sub-zone → parent mapping (or pass through)."""
    if label_set == "main_six":
        # Map each entry. Labels not in SUBZONE_TO_PARENT become object 'UNK'
        # which we filter out before fitting (these are extremely rare).
        out = np.array(
            [SUBZONE_TO_PARENT.get(str(l), "UNK") for l in labels],
            dtype=object,
        )
        return out
    if label_set == "all":
        # Keep as-is, but drop labels we don't recognise so we don't blow up
        # the class set with one-off typos.
        out = np.array(
            [str(l) if str(l) in SUBZONE_TO_PARENT else "UNK" for l in labels],
            dtype=object,
        )
        return out
    raise ValueError(
        f"label_set must be 'main_six' or 'all'; got {label_set!r}"
    )


# ---------------------------------------------------------------------------
# Feature engineering — applied identically at train and predict time.
# ---------------------------------------------------------------------------
#: Canonical feature order. Anything that touches the model uses this list so
#: train/predict can never get out of sync.
FEATURE_NAMES: tuple[str, ...] = (
    "GR",
    "TVT",
    "tvt_z",                # standardised TVT within typewell
    "tvt_frac",             # 0..1 position from shallowest to deepest
    "GR_roll5_mean",        # ±5-row mean
    "GR_roll5_std",
    "GR_roll11_mean",       # ±11-row (~22-ft) mean
    "GR_roll11_std",
    "GR_roll21_mean",       # ±21-row (~42-ft) longer-context mean
    "GR_roll21_std",
    "GR_minus_well_mean",   # de-trend GR by per-well baseline
    "GR_minus_well_median",
    "GR_z_well",            # standardised GR within well
    "dGR_dTVT",             # Sav-Gol first derivative
    "d2GR_dTVT2",           # Sav-Gol second derivative
    "dGR_smooth",           # smoothed first derivative (window=21)
    "GR_p_in_well_low",     # row's GR percentile within typewell
    "GR_p_in_well_high",    # complementary high
    "tvt_above_min_gr",     # ft from the well's max-GR row (proxy for EGFDL crest)
    "tvt_above_max_neg_dgr",# ft from the well's max-negative dGR row
                            # (proxy for EGFDL/BUDA contact)
    "tvt_below_min_gr",     # ft below the well's min-GR row
    "GR_lag1",              # GR one row above (shallower)
    "GR_lag5",              # GR five rows above
    "GR_lead1",             # GR one row below
    "GR_lead5",
)


def _safe_savgol(
    x: np.ndarray, window: int, polyorder: int, deriv: int, delta: float
) -> np.ndarray:
    """Sav-Gol filter with safe window/order. Returns zeros if data is too short."""
    n = x.size
    if n < polyorder + 2:
        return np.zeros_like(x, dtype=np.float64)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder:
        # Need window > polyorder.
        return np.zeros_like(x, dtype=np.float64)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        # Fall back to numpy gradient for tiny arrays.
        if deriv == 1:
            return np.gradient(x, max(delta, 1e-9))
        return np.zeros_like(x, dtype=np.float64)
    try:
        return signal.savgol_filter(
            x, window_length=w, polyorder=polyorder, deriv=deriv,
            delta=max(delta, 1e-9), mode="interp",
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("savgol_filter failed (n=%d w=%d p=%d): %s", n, w, polyorder, exc)
        if deriv == 1:
            return np.gradient(x, max(delta, 1e-9))
        return np.zeros_like(x, dtype=np.float64)


def _rolling_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered rolling mean and std with edge clipping (no NaNs at edges)."""
    n = x.size
    if n == 0:
        return x.copy(), x.copy()
    half = max(1, window // 2)
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            means[i] = 0.0
            stds[i] = 0.0
        else:
            means[i] = float(np.mean(finite))
            stds[i] = float(np.std(finite)) if finite.size >= 2 else 0.0
    return means, stds


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Each entry's percentile rank within the (finite) array, in [0, 1]."""
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float64)
    ranks = np.full(x.shape, 0.5, dtype=np.float64)
    finite_vals = x[finite]
    order = np.argsort(finite_vals, kind="stable")
    pr = np.empty_like(finite_vals, dtype=np.float64)
    pr[order] = np.linspace(0.0, 1.0, finite_vals.size, endpoint=False)
    ranks[finite] = pr
    return ranks


def _engineer_typewell_features(
    tvt: np.ndarray, gr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the FEATURE_NAMES matrix for one typewell.

    Returns
    -------
    X : (n, F) float64 — feature matrix in FEATURE_NAMES order.
    valid : (n,) bool — rows where every feature is finite enough to score.
    """
    tvt = np.asarray(tvt, dtype=np.float64)
    gr = np.asarray(gr, dtype=np.float64)
    n = tvt.size

    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), np.empty(0, dtype=bool)

    # Sort by TVT (shallowest first → row 0 is shallowest in increasing-down).
    # We undo this at the end so output rows match the input order.
    order = np.argsort(tvt, kind="stable")
    tvt_s = tvt[order]
    gr_s = gr[order]

    # Median TVT step for derivative scaling.
    diffs = np.diff(tvt_s)
    finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt = float(np.median(finite_diffs)) if finite_diffs.size > 0 else 0.5

    # Replace non-finite GR with the per-well median for feature stability.
    gr_med = float(np.nanmedian(gr_s)) if np.isfinite(gr_s).any() else 0.0
    gr_filled = np.where(np.isfinite(gr_s), gr_s, gr_med)

    # Per-well stats.
    finite_mask = np.isfinite(gr_s) & np.isfinite(tvt_s)
    if finite_mask.sum() >= 2:
        gr_mean = float(np.mean(gr_filled[finite_mask]))
        gr_std = float(np.std(gr_filled[finite_mask]))
        gr_med_full = float(np.median(gr_filled[finite_mask]))
        tvt_mean = float(np.mean(tvt_s[finite_mask]))
        tvt_std = float(np.std(tvt_s[finite_mask]))
    else:
        gr_mean = gr_med_full = float(gr_med)
        gr_std = 1.0
        tvt_mean = float(np.nanmean(tvt_s)) if np.isfinite(tvt_s).any() else 0.0
        tvt_std = 1.0
    gr_std = max(gr_std, 1.0)
    tvt_std = max(tvt_std, 1.0)

    # TVT positional features.
    tvt_z = (tvt_s - tvt_mean) / tvt_std
    if np.isfinite(tvt_s).any():
        tvt_min = float(np.nanmin(tvt_s))
        tvt_max = float(np.nanmax(tvt_s))
        span = max(tvt_max - tvt_min, 1.0)
        tvt_frac = (tvt_s - tvt_min) / span
    else:
        tvt_frac = np.zeros_like(tvt_s)

    # Rolling stats — three windows (±2, ±5, ±10 ⇒ 5/11/21 row windows).
    r5_mean, r5_std = _rolling_mean_std(gr_filled, 5)
    r11_mean, r11_std = _rolling_mean_std(gr_filled, 11)
    r21_mean, r21_std = _rolling_mean_std(gr_filled, 21)

    # Per-well baselines.
    gr_minus_mean = gr_filled - gr_mean
    gr_minus_med = gr_filled - gr_med_full
    gr_z = (gr_filled - gr_mean) / gr_std

    # Sav-Gol derivatives. dGR/dTVT for the contact, d²GR/dTVT² for sharp edges.
    dgr = _safe_savgol(gr_filled, window=11, polyorder=2, deriv=1, delta=dt)
    d2gr = _safe_savgol(gr_filled, window=11, polyorder=2, deriv=2, delta=dt)
    dgr_sm = _safe_savgol(gr_filled, window=21, polyorder=2, deriv=1, delta=dt)

    # Within-well GR percentile (low and high).
    p_low = _percentile_rank(gr_filled)
    p_high = 1.0 - p_low

    # Anchor features: distance to per-well GR landmarks.
    if finite_mask.any():
        idx_max = int(np.argmax(gr_filled))
        idx_min = int(np.argmin(gr_filled))
        # max-negative dGR (steepest GR drop with depth — EGFDL/BUDA proxy).
        if dgr_sm.size > 0:
            idx_negdgr = int(np.argmin(dgr_sm))
        else:
            idx_negdgr = idx_max
        tvt_above_max = tvt_s - tvt_s[idx_max]
        tvt_above_negdgr = tvt_s - tvt_s[idx_negdgr]
        tvt_below_min_gr = tvt_s - tvt_s[idx_min]
    else:
        tvt_above_max = np.zeros_like(tvt_s)
        tvt_above_negdgr = np.zeros_like(tvt_s)
        tvt_below_min_gr = np.zeros_like(tvt_s)

    # Lagged GR (shifts in row order). Edge values clamped to nearest sample.
    def _shift(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.empty_like(arr)
        if k > 0:
            out[:k] = arr[0]
            out[k:] = arr[:-k]
        elif k < 0:
            out[k:] = arr[-1]
            out[:k] = arr[-k:]
        else:
            out[:] = arr
        return out

    gr_lag1 = _shift(gr_filled, 1)
    gr_lag5 = _shift(gr_filled, 5)
    gr_lead1 = _shift(gr_filled, -1)
    gr_lead5 = _shift(gr_filled, -5)

    # Stack in FEATURE_NAMES order.
    X_sorted = np.column_stack([
        gr_filled, tvt_s, tvt_z, tvt_frac,
        r5_mean, r5_std, r11_mean, r11_std, r21_mean, r21_std,
        gr_minus_mean, gr_minus_med, gr_z,
        dgr, d2gr, dgr_sm,
        p_low, p_high,
        tvt_above_max, tvt_above_negdgr, tvt_below_min_gr,
        gr_lag1, gr_lag5, gr_lead1, gr_lead5,
    ]).astype(np.float64, copy=False)

    if X_sorted.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature engineering mismatch: got {X_sorted.shape[1]} columns, "
            f"FEATURE_NAMES has {len(FEATURE_NAMES)}."
        )

    # Validity mask — drop rows whose original TVT or GR was non-finite, or
    # where features came out non-finite for any reason.
    valid_sorted = np.isfinite(X_sorted).all(axis=1) & finite_mask

    # Un-sort to match input row order.
    inverse = np.argsort(order, kind="stable")
    X = X_sorted[inverse]
    valid = valid_sorted[inverse]
    return X, valid


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _to_pandas(df: Any) -> pd.DataFrame:
    """Coerce a Polars or Pandas DataFrame to Pandas (cheap if already pd)."""
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    raise TypeError(
        f"typewell_df must be Polars or Pandas DataFrame; got {type(df).__name__}"
    )


def _read_typewell(path: Path) -> pd.DataFrame | None:
    """Read a typewell CSV. Returns None if it's empty / lacks needed columns."""
    try:
        df = pl.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    if df.height == 0:
        return None
    needed = {"TVT", "GR", "Geology"}
    if not needed.issubset(df.columns):
        logger.warning("%s missing one of %s; skipping.", path.name, needed)
        return None
    return df.to_pandas()


def _wellname_from_filename(path: Path) -> str:
    """Extract WELLNAME from ``{name}__typewell.csv``."""
    name = path.name
    if name.endswith("__typewell.csv"):
        return name[: -len("__typewell.csv")]
    return path.stem


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_geology_classifier(
    train_dir: str,
    output_path: str = "/kaggle/working/geology_clf.joblib",
    *,
    label_set: str = "main_six",
    min_samples_per_class: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Train a GR-only Geology classifier from labelled train typewells.

    Parameters
    ----------
    train_dir : str
        Directory containing ``{wellname}__typewell.csv`` files. Each must
        have ``TVT, GR, Geology`` columns. Anything else is ignored.
    output_path : str
        Where to write the joblib bundle. Parent dir is created on demand.
    label_set : {'main_six', 'all'}
        ``main_six`` collapses sub-zones to the six parent formations
        (recommended, and required for downstream :mod:`rogii.geology`).
        ``all`` keeps the fine-grained labels in :data:`SUBZONE_TO_PARENT`.
    min_samples_per_class : int
        Classes with fewer rows than this across the whole training set are
        dropped (or, in 'main_six' mode, would never appear). Skipped wells
        are logged.
    n_splits : int
        GroupKFold splits — folds split by WELLNAME so a row from a given
        well never appears in both train and val.
    random_state : int
        LightGBM seed.

    Returns
    -------
    dict
        ``{'model_path': str, 'classes': [...], 'metadata': {...}}``.
        The metadata bundle includes per-class accuracy, the confusion
        matrix, the feature names, the label mapping, and the folds' OOF
        accuracy. The persisted joblib has the trained Booster, the
        LabelEncoder, the feature list, and metadata.
    """
    train_path = Path(train_dir)
    if not train_path.exists() or not train_path.is_dir():
        raise FileNotFoundError(
            f"train_dir {train_dir!r} does not exist or is not a directory."
        )

    files = sorted(train_path.glob("*__typewell.csv"))
    if not files:
        # Some Kaggle layouts use forward-slash globs; try the legacy form.
        files = sorted(map(Path, glob.glob(str(train_path / "*__typewell.csv"))))
    if not files:
        raise FileNotFoundError(
            f"No '*__typewell.csv' files found under {train_dir!r}."
        )
    logger.info(
        "train_geology_classifier: found %d typewells in %s.",
        len(files), train_path,
    )

    # ----- Build the global training matrix -----------------------------------
    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    g_blocks: list[np.ndarray] = []  # WELLNAME group ids
    n_skipped = 0
    n_used = 0
    for path in files:
        wellname = _wellname_from_filename(path)
        df = _read_typewell(path)
        if df is None:
            n_skipped += 1
            continue
        # Drop rows without Geology — labelling fraction is ~70 %, the rest
        # would only add noise.
        df = df[df["Geology"].notna()]
        if df.empty:
            n_skipped += 1
            continue
        labels_raw = df["Geology"].astype(str).to_numpy()
        labels = _coarsen_labels(labels_raw, label_set)
        keep = labels != "UNK"
        if not keep.any():
            n_skipped += 1
            continue
        df = df.loc[keep].reset_index(drop=True)
        labels = labels[keep]
        X, valid = _engineer_typewell_features(
            df["TVT"].to_numpy(dtype=np.float64),
            df["GR"].to_numpy(dtype=np.float64),
        )
        if not valid.any():
            n_skipped += 1
            continue
        X_blocks.append(X[valid])
        y_blocks.append(labels[valid])
        g_blocks.append(np.full(int(valid.sum()), wellname, dtype=object))
        n_used += 1

    if not X_blocks:
        raise RuntimeError(
            "No usable rows after feature engineering; cannot train classifier."
        )
    logger.info(
        "Used %d typewells, skipped %d (no Geology / no usable rows).",
        n_used, n_skipped,
    )

    X = np.concatenate(X_blocks, axis=0)
    y_str = np.concatenate(y_blocks, axis=0)
    groups = np.concatenate(g_blocks, axis=0)

    # ----- Class pruning ------------------------------------------------------
    counter = Counter(y_str.tolist())
    keep_classes = {c for c, n in counter.items() if n >= min_samples_per_class}
    if len(keep_classes) < 2:
        raise RuntimeError(
            f"After pruning to ≥{min_samples_per_class} samples, "
            f"only {len(keep_classes)} class(es) remain: {keep_classes}. "
            f"Lower min_samples_per_class or check the input data."
        )
    if len(keep_classes) < len(counter):
        dropped = sorted(set(counter) - keep_classes)
        logger.info(
            "Dropping %d under-represented class(es): %s",
            len(dropped),
            {c: counter[c] for c in dropped},
        )
    keep_mask = np.array([c in keep_classes for c in y_str], dtype=bool)
    X = X[keep_mask]
    y_str = y_str[keep_mask]
    groups = groups[keep_mask]

    classes = sorted(keep_classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int32)

    logger.info(
        "Training matrix: X=%s, classes=%d (%s)",
        X.shape, len(classes), classes,
    )

    # ----- LightGBM params ----------------------------------------------------
    n_classes = len(classes)
    base_params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "seed": random_state,
        "num_threads": 0,  # use all cores
    }
    n_estimators = 500

    # ----- Out-of-fold validation --------------------------------------------
    n_splits_eff = max(2, min(n_splits, len(np.unique(groups))))
    if n_splits_eff < n_splits:
        logger.warning(
            "Requested %d folds but only %d unique groups; using %d.",
            n_splits, len(np.unique(groups)), n_splits_eff,
        )
    gkf = GroupKFold(n_splits=n_splits_eff)
    oof_pred = np.zeros((y.size, n_classes), dtype=np.float64)
    oof_filled = np.zeros(y.size, dtype=bool)
    fold_accs: list[float] = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        dtrain = lgb.Dataset(X[tr], label=y[tr], feature_name=list(FEATURE_NAMES))
        dval = lgb.Dataset(X[va], label=y[va], reference=dtrain,
                           feature_name=list(FEATURE_NAMES))
        booster = lgb.train(
            base_params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=25, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        prob_va = booster.predict(X[va], num_iteration=booster.best_iteration)
        oof_pred[va] = prob_va
        oof_filled[va] = True
        pred_va = np.argmax(prob_va, axis=1)
        acc_va = float(accuracy_score(y[va], pred_va))
        fold_accs.append(acc_va)
        logger.info("Fold %d/%d: val acc=%.4f (best_iter=%d)",
                    fold + 1, n_splits_eff, acc_va, booster.best_iteration or n_estimators)

    # OOF metrics (rows actually scored).
    if not oof_filled.any():
        raise RuntimeError("No OOF rows produced — GroupKFold yielded no folds.")
    yp = np.argmax(oof_pred[oof_filled], axis=1)
    yt = y[oof_filled]
    oof_acc = float(accuracy_score(yt, yp))
    cm = confusion_matrix(yt, yp, labels=list(range(n_classes)))
    per_class_acc: dict[str, float] = {}
    for i, c in enumerate(classes):
        denom = int(cm[i].sum())
        per_class_acc[c] = float(cm[i, i] / denom) if denom > 0 else float("nan")
    logger.info("Out-of-fold accuracy: %.4f (mean per-fold %.4f ± %.4f)",
                oof_acc, float(np.mean(fold_accs)), float(np.std(fold_accs)))
    for c, a in per_class_acc.items():
        logger.info("  %-25s %.4f  (n=%d)", c, a, int(cm[classes.index(c)].sum()))

    # ----- Final model: train on ALL rows ------------------------------------
    final_n_iter = int(np.median([
        # Reuse fold OOF guidance: we don't have direct best_iter list per
        # fold (we logged it), but using n_estimators is OK because early
        # stopping during folds tells us the model converges. For the final
        # model, we let it run a fixed number with no val.
        n_estimators
    ]))
    full_dtrain = lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES))
    booster_final = lgb.train(
        base_params,
        full_dtrain,
        num_boost_round=final_n_iter,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # ----- Persist ------------------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "label_set": label_set,
        "classes": classes,
        "feature_names": list(FEATURE_NAMES),
        "subzone_to_parent": dict(SUBZONE_TO_PARENT),
        "oof_accuracy": oof_acc,
        "fold_accuracies": fold_accs,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "class_counts": {c: int((y_str == c).sum()) for c in classes},
        "n_train_rows": int(y.size),
        "n_train_wells": int(np.unique(groups).size),
        "n_skipped_wells": n_skipped,
        "n_iterations": booster_final.current_iteration(),
        "fallback_label": "EGFDL",
    }

    bundle = {
        "model": booster_final.model_to_string(),
        "classes": classes,
        "feature_names": list(FEATURE_NAMES),
        "label_set": label_set,
        "metadata": metadata,
        "subzone_to_parent": dict(SUBZONE_TO_PARENT),
    }
    joblib.dump(bundle, out_path)
    logger.info("Saved classifier bundle to %s (%d classes).", out_path, n_classes)

    return {
        "model_path": str(out_path),
        "classes": classes,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
_BUNDLE_CACHE: dict[str, dict] = {}


def _load_bundle(model_path: str) -> dict:
    """Load and cache the joblib bundle. Reconstitutes the LightGBM Booster."""
    cached = _BUNDLE_CACHE.get(model_path)
    if cached is not None:
        return cached
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Geology classifier model not found at {model_path!r}. "
            f"Did you run train_geology_classifier()?"
        )
    bundle = joblib.load(p)
    if "model" not in bundle or "classes" not in bundle:
        raise RuntimeError(
            f"Bundle at {model_path!r} is malformed (keys={list(bundle.keys())})."
        )
    booster = lgb.Booster(model_str=bundle["model"])
    bundle["_booster"] = booster
    _BUNDLE_CACHE[model_path] = bundle
    return bundle


def predict_geology(
    typewell_df: Any,
    model_path: str = "/kaggle/working/geology_clf.joblib",
) -> np.ndarray:
    """Apply the trained classifier to a typewell.

    Returns
    -------
    np.ndarray
        String array of formation labels — one per row of ``typewell_df``,
        in the original row order. Rows where the classifier cannot score
        (non-finite TVT or GR) are filled with ``'EGFDL'`` (the fallback,
        recorded in the bundle metadata) so the downstream code never sees
        a null.
    """
    df = _to_pandas(typewell_df)
    n = len(df)
    if n == 0:
        return np.array([], dtype=object)
    if "TVT" not in df.columns or "GR" not in df.columns:
        raise ValueError("typewell_df must contain 'TVT' and 'GR' columns.")

    bundle = _load_bundle(model_path)
    booster: lgb.Booster = bundle["_booster"]
    classes: list[str] = bundle["classes"]
    feature_names: list[str] = bundle["feature_names"]
    fallback = str(bundle.get("metadata", {}).get("fallback_label", "EGFDL"))

    if feature_names != list(FEATURE_NAMES):
        raise RuntimeError(
            "Trained model's feature_names disagree with current FEATURE_NAMES; "
            "this bundle was trained with a different module version."
        )

    X, valid = _engineer_typewell_features(
        df["TVT"].to_numpy(dtype=np.float64),
        df["GR"].to_numpy(dtype=np.float64),
    )

    out = np.array([fallback] * n, dtype=object)
    if not valid.any():
        logger.warning(
            "predict_geology: no valid rows for typewell of size %d; "
            "returning fallback '%s' for all rows.", n, fallback,
        )
        return out

    proba = booster.predict(X[valid])
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = np.array([classes[i] for i in pred_idx], dtype=object)
    out[valid] = pred_labels

    # Log a per-class summary so the integrator can spot weird outputs.
    counts = Counter(out.tolist())
    logger.info(
        "predict_geology: n=%d, valid=%d, distribution=%s",
        n, int(valid.sum()),
        {c: counts.get(c, 0) for c in classes},
    )
    return out


def fill_missing_geology(
    typewell_df: Any,
    model_path: str | None = None,
) -> Any:
    """Drop-in replacement: fill ``Geology`` if missing/empty, else pass-through.

    Behaviour
    ---------
    * If ``Geology`` column is present and at least one row is non-null, the
      DataFrame is returned unchanged. (We trust real labels over predictions.)
    * If ``Geology`` is absent or entirely null, a NEW DataFrame is returned
      with the column populated by :func:`predict_geology`. The original
      backend (Polars / Pandas) is preserved.

    The model is loaded only when needed.
    """
    is_polars = isinstance(typewell_df, pl.DataFrame)
    is_pandas = isinstance(typewell_df, pd.DataFrame)
    if not (is_polars or is_pandas):
        raise TypeError(
            f"typewell_df must be Polars or Pandas; got {type(typewell_df).__name__}"
        )

    pdf = typewell_df.to_pandas() if is_polars else typewell_df.copy()
    has_col = "Geology" in pdf.columns
    if has_col:
        non_null = pdf["Geology"].notna().sum()
        if non_null > 0:
            logger.info(
                "fill_missing_geology: %d/%d rows already have Geology; "
                "returning input unchanged.", int(non_null), len(pdf),
            )
            # Return the original (untouched) input — preserve backend.
            return typewell_df

    # Need to predict.
    if model_path is None:
        model_path = "/kaggle/working/geology_clf.joblib"
    logger.info(
        "fill_missing_geology: Geology missing/empty (n=%d); predicting via %s.",
        len(pdf), model_path,
    )
    labels = predict_geology(pdf, model_path=model_path)
    pdf = pdf.copy()
    pdf["Geology"] = labels.astype(object)

    if is_polars:
        return pl.from_pandas(pdf)
    return pdf


__all__ = [
    "MAIN_SIX",
    "ALL_LABELS",
    "SUBZONE_TO_PARENT",
    "FEATURE_NAMES",
    "train_geology_classifier",
    "predict_geology",
    "fill_missing_geology",
]
