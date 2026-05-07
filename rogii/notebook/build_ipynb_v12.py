"""Build v12 = v11 + Triple-Signal's two particle filters as features.

The new public top kernel (LB 11.284) uses Z-velocity-coupled TVT-PF
+ ANCC-PF (S = TVT + Z tracker) feeding into a single LightGBM. We
keep our richer feature stack (v11: KNN + plane + MLP + aniso) and add
the two PFs as extra signals — strictly more informative than either
notebook alone.

Architecture: 6 spatial signals -> ridge-stacked GBMs -> shrinkage cap
-> EWM smoothing.

  1. Spatial plane fit KNN K=10                    (konbu17)
  2. Row KNN K=20 (n_q=8000)                       (konbu17)
  3. Multi-output MLP+PE-L8, 3-seed ensemble       (v9 -> v11)
  4. Anisotropic-exponential kriging               (v11)
  5. TVT Particle Filter (Z-velocity coupling)     (v12 NEW)
  6. ANCC Particle Filter (S = TVT + Z tracker)    (v12 NEW)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT_PY = ROOT / "notebook" / "kaggle_cell_v12.py"
OUT_IPYNB_V12 = ROOT / "notebook" / "submission_v12.ipynb"
OUT_IPYNB_ACTIVE = ROOT / "notebook" / "submission.ipynb"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


feature_builder_b64 = b64(SRC / "feature_builder.py")
neural_ancc_b64 = b64(SRC / "neural_ancc.py")
aniso_kriging_b64 = b64(SRC / "aniso_kriging.py")
anchor_shrinkage_b64 = b64(SRC / "anchor_shrinkage.py")
triple_signal_pf_b64 = b64(SRC / "triple_signal_pf.py")


CELL = f'''# ROGII Wellbore Geology Prediction - v12 submission notebook
#
# v12 = v11 + Triple-Signal's two particle filters.
#
# Architecture (6 spatial signals -> LGB x 5 + XGB -> ridge -> shrinkage -> EWM):
#   1. Spatial plane fit KNN  (konbu17)
#   2. Row KNN K=20           (konbu17, n_q=8000)
#   3. MLP+PE-L8 multi-out, 3-seed ensemble (v9 -> v11)
#   4. Anisotropic-exponential kriging (v11)
#   5. TVT-PF, Z-velocity coupled (v12 NEW)
#   6. ANCC-PF, S = TVT + Z tracker (v12 NEW)
#
# The two PFs are run in PARALLEL via multiprocessing.Pool over wells
# (0.15 sec/well measured on M1 Pro 10-core; expect ~3 min for 773 wells
# on Kaggle 4-core).

import os
import sys
import base64
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rogii.v12")

# 1) Modules
SRC_DIR = "/kaggle/working/rogii_src"
os.makedirs(SRC_DIR, exist_ok=True)

_MODULES = {{
    "feature_builder.py": "{feature_builder_b64}",
    "neural_ancc.py": "{neural_ancc_b64}",
    "aniso_kriging.py": "{aniso_kriging_b64}",
    "anchor_shrinkage.py": "{anchor_shrinkage_b64}",
    "triple_signal_pf.py": "{triple_signal_pf_b64}",
}}
for _name, _payload in _MODULES.items():
    with open(os.path.join(SRC_DIR, _name), "wb") as _f:
        _f.write(base64.b64decode(_payload))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 2) Data root
INPUT_ROOT = "/kaggle/input"
DATA_ROOT = None
if os.path.isdir(INPUT_ROOT):
    for root, dirs, _files in os.walk(INPUT_ROOT):
        depth = root.replace(INPUT_ROOT, "").count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        if "test" in dirs and "train" in dirs:
            DATA_ROOT = root
            logger.info("Found competition data at %s", DATA_ROOT)
            break
if DATA_ROOT is None:
    raise SystemExit("FATAL: could not locate competition test/+train/ directories")

TRAIN_DIR = Path(DATA_ROOT) / "train"
TEST_DIR = Path(DATA_ROOT) / "test"

# 3) Imports
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as _e:
    HAS_XGB = False
    logger.warning("XGBoost unavailable: %s", _e)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from feature_builder import (
    FORMATIONS,
    FormationPlaneKNN,
    RowKNN,
    MLPAnccImputer,
    AnisoFormationImputer,
    build_dataset,
)
from anchor_shrinkage import constant_shrinkage, hard_cap
from triple_signal_pf import run_pfs_for_wells

# 4) Config
PRIMARY_FORMATION = "ANCC"
N_SPLITS = 5
SPLIT_SEED = 42
LGB_SEEDS = [42, 7, 123, 999, 31337]
ENABLE_BEAM = True
EWM_SPAN = 4.0
USE_GPU = True

SHRINK_ALPHA = 1.0
HARD_CAP_BAND = 40.0

MLP_NUM_FREQS = 8
MLP_HIDDEN = 256
MLP_EPOCHS = 12
MLP_ROWS_PER_EPOCH = 500_000
MLP_SEEDS = [42, 7, 123]

ANISO_KERNEL = "exponential"
ANISO_K = 20
ANISO_RANGE_SCALE = 1.0

PF_N_PARTICLES = 500
PF_SEED = 42

OUTPUT = Path("/kaggle/working/submission.csv")
OOF_OUT = Path("/kaggle/working/oof.csv")

# 5) Spatial imputers
train_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
logger.info("train paths=%d test paths=%d", len(train_paths), len(test_paths))

logger.info("Building plane-fit imputer ...")
formation_imputer = FormationPlaneKNN.fit(train_paths, formations=FORMATIONS)

logger.info("Building row-level KNN imputer ...")
row_imputer = RowKNN.fit(train_paths, formations=FORMATIONS)

logger.info("Building anisotropic-%s kriging imputer ...", ANISO_KERNEL)
aniso_imputer = AnisoFormationImputer.fit(
    train_paths, formations=FORMATIONS,
    kernel=ANISO_KERNEL, range_scale=ANISO_RANGE_SCALE, k=ANISO_K,
)

logger.info("Training %d-seed MLP ensemble ...", len(MLP_SEEDS))
mlp_imputer = MLPAnccImputer.fit(
    train_paths, formations=FORMATIONS,
    num_freqs=MLP_NUM_FREQS, hidden=MLP_HIDDEN,
    epochs=MLP_EPOCHS, rows_per_epoch=MLP_ROWS_PER_EPOCH,
    seeds=MLP_SEEDS, verbose=False,
)


# 6) Particle filters in parallel (across wells)
def _load_pl(path):
    return pl.read_csv(str(path), infer_schema_length=2000,
                       null_values=["", "NA", "NaN", "nan", "null"],
                       truncate_ragged_lines=True)


def _build_pf_input(paths_dir):
    well_dfs = {{}}
    typewell_dfs = {{}}
    for p in sorted(paths_dir.glob("*__horizontal_well.csv")):
        wid = p.stem.replace("__horizontal_well", "")
        tw_path = paths_dir / f"{{wid}}__typewell.csv"
        if not tw_path.exists():
            continue
        well_dfs[wid] = _load_pl(p)
        typewell_dfs[wid] = _load_pl(tw_path)
    return well_dfs, typewell_dfs


logger.info("Running TVT-PF + ANCC-PF on train wells (parallel) ...")
t0 = time.perf_counter()
train_well_dfs, train_typewell_dfs = _build_pf_input(TRAIN_DIR)
train_pf_results = run_pfs_for_wells(
    train_well_dfs, train_typewell_dfs,
    n_workers=-1, n_particles=PF_N_PARTICLES, seed=PF_SEED,
)
logger.info("  train PF done: %d wells in %.1fs",
            len(train_pf_results), time.perf_counter() - t0)

logger.info("Running TVT-PF + ANCC-PF on test wells (parallel) ...")
t0 = time.perf_counter()
test_well_dfs, test_typewell_dfs = _build_pf_input(TEST_DIR)
test_pf_results = run_pfs_for_wells(
    test_well_dfs, test_typewell_dfs,
    n_workers=-1, n_particles=PF_N_PARTICLES, seed=PF_SEED,
)
logger.info("  test PF done: %d wells in %.1fs",
            len(test_pf_results), time.perf_counter() - t0)
del train_well_dfs, train_typewell_dfs, test_well_dfs, test_typewell_dfs

# 7) Features
logger.info("Building train features ...")
train_df = build_dataset(
    train_paths, formation_imputer, row_imputer,
    is_train=True,
    mlp_imputer=mlp_imputer, aniso_imputer=aniso_imputer,
    pf_results=train_pf_results,
    primary_formation=PRIMARY_FORMATION,
    formations=FORMATIONS, enable_beam=ENABLE_BEAM, label="train",
)
logger.info("  train shape: %s", train_df.shape)

logger.info("Building test features ...")
test_df = build_dataset(
    test_paths, formation_imputer, row_imputer,
    is_train=False,
    mlp_imputer=mlp_imputer, aniso_imputer=aniso_imputer,
    pf_results=test_pf_results,
    primary_formation=PRIMARY_FORMATION,
    formations=FORMATIONS, enable_beam=ENABLE_BEAM, label="test",
)
logger.info("  test shape: %s", test_df.shape)

if train_df.empty or test_df.empty:
    raise SystemExit("FATAL: empty feature matrix")

feature_cols = [c for c in train_df.columns if c not in {{"well", "prediction_id", "target"}}]
pf_cols = [c for c in feature_cols if c.startswith("pf_") or c.startswith("gr_tw_off_") or c == "tw_gr_at_pf" or c == "gr_minus_tw_at_pf"]
logger.info("  #features: %d (PF features: %d)", len(feature_cols), len(pf_cols))

# 8) Splits
gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
splits = list(gkf.split(train_df, train_df["target"], groups=train_df["well"]))

# 9) Models
LGB_BASE = dict(
    boosting_type="gbdt", learning_rate=0.06, num_leaves=89,
    min_child_samples=10, min_child_weight=0.5,
    n_estimators=5000, n_jobs=-1,
    reg_alpha=2.03, reg_lambda=87.28,
    subsample=0.645, subsample_freq=1, colsample_bytree=0.821,
    objective="regression", metric="rmse", verbose=-1,
)
if USE_GPU:
    LGB_BASE.update(device_type="gpu", gpu_use_dp=False, max_bin=255)


def train_lgb(seed):
    logger.info("LGB seed=%d", seed)
    params = dict(LGB_BASE); params["random_state"] = seed
    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(train_df.iloc[tr][feature_cols], label=train_df.iloc[tr]["target"])
        dva = lgb.Dataset(train_df.iloc[va][feature_cols], label=train_df.iloc[va]["target"], reference=dtr)
        m = lgb.train(params, dtr, valid_sets=[dva],
                      num_boost_round=params["n_estimators"],
                      callbacks=[lgb.early_stopping(125, verbose=False),
                                 lgb.log_evaluation(period=500)])
        oof[va] = m.predict(train_df.iloc[va][feature_cols], num_iteration=m.best_iteration).astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof[va] - train_df.iloc[va]["target"].values) ** 2)))
        logger.info("  fold %d: rmse=%.4f best_iter=%d", fold, rmse, m.best_iteration)
        test_pred += m.predict(test_df[feature_cols], num_iteration=m.best_iteration).astype(np.float32) / N_SPLITS
    overall = float(np.sqrt(np.mean((oof - train_df["target"].values) ** 2)))
    logger.info("LGB seed=%d: OOF rmse=%.4f", seed, overall)
    return oof, test_pred, overall


XGB_BASE = dict(
    objective="reg:squarederror", eval_metric="rmse",
    learning_rate=0.06, max_depth=8, min_child_weight=10,
    subsample=0.7, colsample_bytree=0.85,
    reg_alpha=1.0, reg_lambda=20.0,
    tree_method="hist", n_jobs=-1,
)
if USE_GPU:
    XGB_BASE.update(device="cuda")


def train_xgb(seed):
    if not HAS_XGB:
        return None, None, None
    logger.info("XGB seed=%d", seed)
    params = dict(XGB_BASE); params["seed"] = seed
    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = xgb.DMatrix(train_df.iloc[tr][feature_cols].values, label=train_df.iloc[tr]["target"].values)
        dva = xgb.DMatrix(train_df.iloc[va][feature_cols].values, label=train_df.iloc[va]["target"].values)
        m = xgb.train(params, dtr, num_boost_round=5000,
                      evals=[(dva, "val")], early_stopping_rounds=125, verbose_eval=500)
        oof[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1)).astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof[va] - train_df.iloc[va]["target"].values) ** 2)))
        logger.info("  fold %d: rmse=%.4f best_iter=%d", fold, rmse, m.best_iteration)
        dte = xgb.DMatrix(test_df[feature_cols].values)
        test_pred += m.predict(dte, iteration_range=(0, m.best_iteration + 1)).astype(np.float32) / N_SPLITS
    overall = float(np.sqrt(np.mean((oof - train_df["target"].values) ** 2)))
    logger.info("XGB seed=%d: OOF rmse=%.4f", seed, overall)
    return oof, test_pred, overall


results = {{}}
for seed in LGB_SEEDS:
    oof, tp, score = train_lgb(seed)
    results[f"lgb_{{seed}}"] = {{"oof": oof, "test": tp, "rmse": score}}

if HAS_XGB:
    oof_xgb, test_xgb, rmse_xgb = train_xgb(42)
    if oof_xgb is not None:
        results["xgb_42"] = {{"oof": oof_xgb, "test": test_xgb, "rmse": rmse_xgb}}

# 10) Ensemble
oof_avg = np.mean([r["oof"] for r in results.values()], axis=0)
test_avg = np.mean([r["test"] for r in results.values()], axis=0)
rmse_avg = float(np.sqrt(np.mean((oof_avg - train_df["target"].values) ** 2)))

stack_X = np.column_stack([r["oof"] for r in results.values()])
ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge.fit(stack_X, train_df["target"].values)
stack_oof = ridge.predict(stack_X)
rmse_stack = float(np.sqrt(np.mean((stack_oof - train_df["target"].values) ** 2)))
weights = ridge.coef_ / max(ridge.coef_.sum(), 1e-9)
logger.info("simple avg=%.4f  ridge=%.4f  weights=%s", rmse_avg, rmse_stack,
            {{k: float(round(w, 3)) for k, w in zip(results.keys(), weights)}})
stack_test = ridge.predict(np.column_stack([r["test"] for r in results.values()]))

if rmse_avg <= rmse_stack:
    final_test = test_avg; final_oof = oof_avg; final_rmse = rmse_avg
else:
    final_test = stack_test; final_oof = stack_oof; final_rmse = rmse_stack
logger.info("Final raw OOF rmse: %.4f", final_rmse)

# 11) Anchor-shrinkage post-process
shrunk_delta = constant_shrinkage(final_test.astype(np.float64), alpha=SHRINK_ALPHA)
shrunk_delta = hard_cap(shrunk_delta, band=HARD_CAP_BAND)
shrunk_oof = constant_shrinkage(final_oof.astype(np.float64), alpha=SHRINK_ALPHA)
shrunk_oof = hard_cap(shrunk_oof, band=HARD_CAP_BAND)
shrunk_oof_rmse = float(np.sqrt(np.mean((shrunk_oof - train_df["target"].values) ** 2)))
logger.info("Post-shrink OOF rmse: %.4f (alpha=%.2f, band=%.1f ft)",
            shrunk_oof_rmse, SHRINK_ALPHA, HARD_CAP_BAND)

# 12) Reconstruct + EWM
test_anchor = test_df["last_known_tvt"].to_numpy(dtype=np.float64)
test_tvt = test_anchor + shrunk_delta

submission = pd.DataFrame({{
    "well": test_df["well"].values,
    "row_idx": test_df["row_idx"].astype(np.int32).values,
    "id": test_df["prediction_id"].values,
    "tvt": test_tvt,
}}).sort_values(["well", "row_idx"]).reset_index(drop=True)


def _apply_ewm(group):
    g = group.copy()
    g["tvt"] = g["tvt"].ewm(span=EWM_SPAN, adjust=False).mean()
    return g


submission = submission.groupby("well", group_keys=False).apply(_apply_ewm)

submission_out = submission[["id", "tvt"]].copy()
if submission_out["tvt"].isna().any():
    bad = submission_out["tvt"].isna()
    submission_out.loc[bad, "tvt"] = test_anchor[bad.to_numpy()]
if not np.isfinite(submission_out["tvt"]).all():
    median_tvt = float(np.median(test_anchor[np.isfinite(test_anchor)]))
    bad = ~np.isfinite(submission_out["tvt"])
    submission_out.loc[bad, "tvt"] = median_tvt

submission_out.to_csv(OUTPUT, index=False)
logger.info("Wrote %s (%d rows)", OUTPUT, len(submission_out))
print(f"Submission: {{len(submission_out)}} rows, {{submission_out['id'].nunique()}} unique ids")
print(submission_out["tvt"].describe())
print(f"Final raw OOF rmse:    {{final_rmse:.4f}}")
print(f"Final shrunk OOF rmse: {{shrunk_oof_rmse:.4f}}")
'''

OUT_PY.write_text(CELL)
print(f"Wrote {OUT_PY} ({len(CELL):,} chars)")

nb = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL.splitlines(keepends=True),
    }],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT_IPYNB_V12.write_text(json.dumps(nb, indent=1))
# Don't auto-update active submission.ipynb -- user has v9 currently running
print(f"Wrote {OUT_IPYNB_V12}")
print("(NOT updating submission.ipynb to avoid disrupting the running v9 kernel)")
