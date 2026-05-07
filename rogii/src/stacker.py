"""GroupKFold OOF stacker for ROGII TVT predictors.

This module implements the meta-stacker requested for the v9 + (v8 + ...)
ensemble. The intended use:

  predictions = {
      "v9":         oof_v9,             # (N,) np.float32/64
      "constant":   np.zeros(N),        # baseline: predict last_known_TVT
      ...
  }
  out = stack_oof(predictions, target=target, groups=well_ids)

  out["best"]            -> "ridge_nn" | "simple_mean" | "single:<name>"
  out["best_oof"]        -> (N,) the chosen OOF predictions
  out["best_rmse"]       -> float overall RMSE of the chosen ensemble
  out["ridge_weights"]   -> (5, K) per-fold non-negative ridge weights
  out["mean_ridge_weights"] -> (K,) average across folds (informational)
  out["per_fold_rmse"]   -> dict[str, list[float]] for each predictor / ensemble
  out["per_well_better"] -> dict[name, fraction of wells the ridge stack beats name on]

The split is GroupKFold(5) with groups=well; we mimic the project's
existing `random_state=42, shuffle=True` convention used in
bench/full_score.py.

Non-negative ridge: solved with scipy.optimize.nnls (closed-form
non-negative least squares with Tikhonov augmentation), since
sklearn doesn't expose `positive=True` for plain Ridge in older
versions and we want zero-bias intercept (the targets and
predictors are all in delta-TVT space).

Note on scoring: the OOF "RMSE" reported here is computed on the
target = (TVT - last_known_TVT_input) residual scale. This is
equivalent (up to a constant) to the leaderboard scale because
the constant offset cancels under L2 — but be careful when
comparing absolute numbers to LB; LB is on TVT directly.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold


__all__ = ["stack_oof"]


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    diff = pred.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _per_well_rmse(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    err = pred - y
    err2 = err * err
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=err2)
    counts = np.bincount(inv).astype(np.float64)
    rmses = np.sqrt(sums / np.maximum(counts, 1.0))
    for w, r in zip(uniq, rmses):
        out[str(w)] = float(r)
    return out


def _nn_ridge_weights(
    X: np.ndarray, y: np.ndarray, alpha: float
) -> np.ndarray:
    """Closed-form non-negative ridge via Tikhonov-augmented NNLS.

    min_{w >= 0} ||X w - y||^2 + alpha * ||w||^2
    is equivalent to NNLS on
        Xa = [[X], [sqrt(alpha) * I]],   ya = [y; 0]
    """
    n, k = X.shape
    Xa = np.vstack([X, np.sqrt(alpha) * np.eye(k, dtype=X.dtype)])
    ya = np.concatenate([y, np.zeros(k, dtype=y.dtype)])
    w, _ = nnls(Xa, ya)
    return w.astype(np.float64)


def stack_oof(
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    alpha: float = 1.0,
) -> dict:
    """Fit a non-negative ridge stacker per fold and return diagnostics.

    Parameters
    ----------
    predictions : mapping of name -> (N,) array
        OOF predictions from each base learner, all on the same row order.
    target : (N,)
        True residual target (TVT - last_known_TVT_input).
    groups : (N,)
        Well IDs for GroupKFold.
    n_splits : int, default 5
    seed : int, default 42 -- random_state for GroupKFold(shuffle=True).
    alpha : float, default 1.0
        L2 ridge penalty for the meta-stacker.

    Returns
    -------
    dict with keys:
        names                 list of predictor names (in column order)
        ridge_weights         (n_splits, K) per-fold weights
        mean_ridge_weights    (K,) average weights
        ridge_oof             (N,) per-row stacked predictions (each row
                              uses the weights from the fold where that
                              row was the validation set)
        ridge_oof_rmse        float
        simple_mean_oof       (N,)
        simple_mean_rmse      float
        single_rmse           dict[name, rmse]
        per_well_better       dict[name, frac_wells_ridge_beats_name]
        per_well_max_rmse     dict[name, max_well_rmse]
        per_well_max_rmse_ridge float
        best                  "ridge_nn" | "simple_mean" | "single:<name>"
        best_oof              (N,)
        best_rmse             float
        per_fold_rmse         dict with keys "ridge", "simple_mean",
                              and each base name; values are length-n_splits
                              lists of fold RMSEs.
    """
    names = list(predictions.keys())
    if not names:
        raise ValueError("predictions must be non-empty")

    K = len(names)
    cols = [np.asarray(predictions[n], dtype=np.float64).reshape(-1) for n in names]
    N = len(cols[0])
    for c, n in zip(cols, names):
        if len(c) != N:
            raise ValueError(f"length mismatch for {n}: {len(c)} vs {N}")

    X = np.column_stack(cols)  # (N, K)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if len(y) != N or len(g) != N:
        raise ValueError("target / groups length mismatch")

    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(gkf.split(X, y, groups=g))

    fold_w = np.zeros((n_splits, K), dtype=np.float64)
    ridge_oof = np.zeros(N, dtype=np.float64)
    fold_rmse_ridge: list[float] = []
    fold_rmse_simple: list[float] = []
    fold_rmse_single: dict[str, list[float]] = {n: [] for n in names}

    simple_mean_oof = X.mean(axis=1)

    for fi, (tr, va) in enumerate(splits):
        w = _nn_ridge_weights(X[tr], y[tr], alpha=alpha)
        fold_w[fi] = w
        ridge_va = X[va] @ w
        ridge_oof[va] = ridge_va
        fold_rmse_ridge.append(_rmse(ridge_va, y[va]))
        fold_rmse_simple.append(_rmse(simple_mean_oof[va], y[va]))
        for ki, n in enumerate(names):
            fold_rmse_single[n].append(_rmse(X[va, ki], y[va]))

    ridge_rmse = _rmse(ridge_oof, y)
    simple_mean_rmse = _rmse(simple_mean_oof, y)
    single_rmse = {n: _rmse(X[:, ki], y) for ki, n in enumerate(names)}

    # per-well comparisons
    pw_ridge = _per_well_rmse(ridge_oof, y, g)
    per_well_better: dict[str, float] = {}
    per_well_max_rmse: dict[str, float] = {}
    for ki, n in enumerate(names):
        pw_n = _per_well_rmse(X[:, ki], y, g)
        wells = list(pw_ridge.keys())
        wins = sum(1 for w in wells if pw_ridge[w] < pw_n[w])
        per_well_better[n] = wins / max(len(wells), 1)
        per_well_max_rmse[n] = max(pw_n.values()) if pw_n else float("nan")
    per_well_max_rmse_ridge = max(pw_ridge.values()) if pw_ridge else float("nan")

    # winner
    best = "ridge_nn"
    best_oof = ridge_oof
    best_rmse = ridge_rmse
    if simple_mean_rmse < best_rmse:
        best = "simple_mean"
        best_oof = simple_mean_oof
        best_rmse = simple_mean_rmse
    for n, r in single_rmse.items():
        if r < best_rmse:
            best = f"single:{n}"
            best_oof = X[:, names.index(n)]
            best_rmse = r

    return {
        "names": names,
        "ridge_weights": fold_w,
        "mean_ridge_weights": fold_w.mean(axis=0),
        "ridge_oof": ridge_oof,
        "ridge_oof_rmse": ridge_rmse,
        "simple_mean_oof": simple_mean_oof,
        "simple_mean_rmse": simple_mean_rmse,
        "single_rmse": single_rmse,
        "per_well_better": per_well_better,
        "per_well_max_rmse": per_well_max_rmse,
        "per_well_max_rmse_ridge": per_well_max_rmse_ridge,
        "best": best,
        "best_oof": best_oof,
        "best_rmse": best_rmse,
        "per_fold_rmse": {
            "ridge": fold_rmse_ridge,
            "simple_mean": fold_rmse_simple,
            **fold_rmse_single,
        },
    }
