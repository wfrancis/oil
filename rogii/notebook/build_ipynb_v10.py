"""Build notebook/submission_v10.ipynb from feature_builder.py +
neural_ancc.py + anchor_shrinkage.py + a thin orchestrator.

v10 = v9 + (a) 5 LGB seeds + (b) 3 MLP-ANCC seeds + (c) anchor-shrinkage
post-process + (d) EWM(span=4) per-well smoothing.

The strategic case: ROGII's public LB is 26% of test, private is 74%.
The reference shake-up shows top-12 private rankings come from rank
700-1000+ on public when the metric is tightly clustered. The robustness
lever for private LB is max-well-RMSE, not overall RMSE. v10 trades
~0.05-0.20 RMSE on the median for a substantial reduction in
catastrophic-tail wells.

Key v9 -> v10 changes:
  LGB_SEEDS [42,7,123] -> [42,7,123,999,31337]   (mult-seed agent rec)
  POST-EWM: anchor-shrinkage (alpha tunable; default 0.85 = mild)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT_PY = ROOT / "notebook" / "kaggle_cell_v10.py"
OUT_IPYNB_V10 = ROOT / "notebook" / "submission_v10.ipynb"
OUT_IPYNB_ACTIVE = ROOT / "notebook" / "submission.ipynb"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


feature_builder_b64 = b64(SRC / "feature_builder.py")
neural_ancc_b64 = b64(SRC / "neural_ancc.py")
anchor_shrinkage_b64 = b64(SRC / "anchor_shrinkage.py")


CELL = f'''# ROGII Wellbore Geology Prediction - v10 submission notebook
#
# v10 = v9 + 5-seed LGB + 3-seed MLP + anchor-shrinkage post-process.
#
# v9 baseline (with v8 feature builder + MLP-ANCC features stacked):
#   OOF projected ~11.5-12 with KNN+MLP+GBM, max-well-RMSE ~40-60 ft.
#
# v10 motivations (in order of expected private-LB impact):
#
#   1) ANCHOR-SHRINKAGE post-process (the big private-LB win):
#      v9 outlier diagnosis found 8/11 catastrophic-tail wells have
#      DRIFT failure mode -- model is over-confidently moving
#      predictions away from anchor when there's no geology to support
#      it. For these wells the trivial constant-anchor baseline scores
#      5-15 ft RMSE while the model scores 53-166 ft (5-10x worse).
#      Multiplicative shrinkage of the predicted delta by alpha < 1
#      pulls catastrophic deltas toward zero (= toward anchor), with
#      minimal cost on typical wells.
#
#   2) MLP MULTI-SEED at imputer level (concentrated tail rescue):
#      Empirically (full 765-well 5-fold OOF in
#      bench/neural_ancc_results.json), 3-seed MLP ensemble cuts the
#      worst-well TVT RMSE by 18 ft (the 165-ft outlier 059c8f24
#      collapses to 148 ft) at +6 min training cost. This is the
#      cheapest concentrated-tail intervention available.
#
#   3) LGB MULTI-SEED averaging (small variance reduction):
#      Empirical seed-to-seed correlation rho=0.93 caps reduction at
#      ~5% std even infinite. With 5 seeds the agent measured -0.39
#      overall RMSE (2.8%) and -0.99 ft max-well (3.1%) on the v8
#      pilot. Tiny but free since LGB is fast.
#
# Strategic context: public LB is 26% of test, private 74%. Reference
# Kaggle competitions with this split routinely show top-12 private
# rankings emerging from rank 700-1000+ on public. The metric
# competition optimizes for is overall RMSE, but the metric that wins
# private LB stability is max-well-RMSE. A single 300-ft-RMSE well
# doubles overall RMSE on the private 74%; v10 explicitly trades
# slightly-higher mean for substantially-lower max.

import os
import sys
import base64
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rogii.v10")

# ---------------------------------------------------------------------------
# 1) Write modules and import.
# ---------------------------------------------------------------------------
SRC_DIR = "/kaggle/working/rogii_src"
os.makedirs(SRC_DIR, exist_ok=True)

_MODULES = {{
    "feature_builder.py": "{feature_builder_b64}",
    "neural_ancc.py": "{neural_ancc_b64}",
    "anchor_shrinkage.py": "{anchor_shrinkage_b64}",
}}
for _name, _payload in _MODULES.items():
    with open(os.path.join(SRC_DIR, _name), "wb") as _f:
        _f.write(base64.b64decode(_payload))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# 2) Discover competition data root.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3) Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as _xgb_exc:
    logger.warning("XGBoost unavailable: %s", _xgb_exc)
    HAS_XGB = False

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from feature_builder import (
    FORMATIONS, FormationPlaneKNN, RowKNN, MLPAnccImputer, build_dataset,
)
from anchor_shrinkage import constant_shrinkage, hard_cap

# ---------------------------------------------------------------------------
# 4) Config
# ---------------------------------------------------------------------------
PRIMARY_FORMATION = "ANCC"
N_SPLITS = 5
SPLIT_SEED = 42
LGB_SEEDS = [42, 7, 123, 999, 31337]    # v10: 5 seeds (was 3 in v9)
ENABLE_BEAM = True
EWM_SPAN = 4.0
USE_GPU = True

# Anchor-shrinkage config (CALIBRATED on v9 OOF, 2026-05-07):
# Sweep showed multiplicative shrinkage HURTS overall RMSE at every
# alpha < 1.0 (the GBM already absorbed most of the catastrophic-tail
# drift; further shrinkage damages legitimate motion predictions).
# Hard cap at +-40 ft is the only intervention that improves overall
# RMSE (-0.04) without raising max-well-RMSE. p99 of population
# eval_offset_from_anchor is 37.7 ft so 40 ft exactly clips the tail.
SHRINK_ALPHA = 1.0     # disabled (calibrated on v9 OOF)
HARD_CAP_BAND = 40.0   # ft; clips ~p99 of legitimate eval motion

# Neural-ANCC config (matches the OOF-validated MLP+PE-L8 multi-output)
# v10: 3-seed ensemble at imputer level cuts worst-well RMSE by 18 ft
# per multi-seed agent's full 765-well OOF measurement.
MLP_NUM_FREQS = 8
MLP_HIDDEN = 256
MLP_EPOCHS = 12
MLP_ROWS_PER_EPOCH = 500_000
MLP_SEEDS = [42, 7, 123]

OUTPUT = Path("/kaggle/working/submission.csv")
OOF_OUT = Path("/kaggle/working/oof.csv")

# ---------------------------------------------------------------------------
# 5) Spatial imputers + MLP fit (full data, no fold logic for inference)
# ---------------------------------------------------------------------------
train_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
logger.info("train paths=%d test paths=%d", len(train_paths), len(test_paths))

logger.info("Building plane-fit imputer ...")
formation_imputer = FormationPlaneKNN.fit(train_paths, formations=FORMATIONS)
logger.info("  %d wells", len(formation_imputer.df))

logger.info("Building row-level KNN imputer ...")
row_imputer = RowKNN.fit(train_paths, formations=FORMATIONS)
logger.info("  %d rows", len(row_imputer.targets))

logger.info("Training %d-seed neural-ANCC ensemble (MLP+PE L=%d, hidden=%d, %d epochs) ...",
            len(MLP_SEEDS), MLP_NUM_FREQS, MLP_HIDDEN, MLP_EPOCHS)
mlp_imputer = MLPAnccImputer.fit(
    train_paths, formations=FORMATIONS,
    num_freqs=MLP_NUM_FREQS, hidden=MLP_HIDDEN,
    epochs=MLP_EPOCHS, rows_per_epoch=MLP_ROWS_PER_EPOCH,
    seeds=MLP_SEEDS, verbose=False,
)
logger.info("  MLP ensemble fit done (%d nets averaged)", len(mlp_imputer.nets))

# ---------------------------------------------------------------------------
# 6) Build feature matrices
# ---------------------------------------------------------------------------
logger.info("Building train features ...")
train_df = build_dataset(
    train_paths, formation_imputer, row_imputer,
    is_train=True, mlp_imputer=mlp_imputer,
    primary_formation=PRIMARY_FORMATION,
    formations=FORMATIONS, enable_beam=ENABLE_BEAM, label="train",
)
logger.info("  train shape: %s", train_df.shape)

logger.info("Building test features ...")
test_df = build_dataset(
    test_paths, formation_imputer, row_imputer,
    is_train=False, mlp_imputer=mlp_imputer,
    primary_formation=PRIMARY_FORMATION,
    formations=FORMATIONS, enable_beam=ENABLE_BEAM, label="test",
)
logger.info("  test shape: %s", test_df.shape)

if train_df.empty or test_df.empty:
    raise SystemExit("FATAL: empty feature matrix")

feature_cols = [c for c in train_df.columns if c not in {{"well", "prediction_id", "target"}}]
logger.info("  #features: %d", len(feature_cols))

# ---------------------------------------------------------------------------
# 7) GroupKFold splits
# ---------------------------------------------------------------------------
gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
splits = list(gkf.split(train_df, train_df["target"], groups=train_df["well"]))

# ---------------------------------------------------------------------------
# 8) Models (LGB x 5 seeds + XGB)
# ---------------------------------------------------------------------------
LGB_BASE = dict(
    boosting_type="gbdt",
    learning_rate=0.06,
    num_leaves=89,
    min_child_samples=10,
    min_child_weight=0.5,
    n_estimators=5000,
    n_jobs=-1,
    reg_alpha=2.03,
    reg_lambda=87.28,
    subsample=0.645,
    subsample_freq=1,
    colsample_bytree=0.821,
    objective="regression",
    metric="rmse",
    verbose=-1,
)
if USE_GPU:
    LGB_BASE.update(device_type="gpu", gpu_use_dp=False, max_bin=255)


def train_lgb(seed):
    logger.info("LGB seed=%d", seed)
    params = dict(LGB_BASE)
    params["random_state"] = seed
    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    for fold, (tr, va) in enumerate(splits):
        dtr = lgb.Dataset(train_df.iloc[tr][feature_cols], label=train_df.iloc[tr]["target"])
        dva = lgb.Dataset(train_df.iloc[va][feature_cols], label=train_df.iloc[va]["target"], reference=dtr)
        m = lgb.train(
            params, dtr, valid_sets=[dva],
            num_boost_round=params["n_estimators"],
            callbacks=[lgb.early_stopping(125, verbose=False),
                       lgb.log_evaluation(period=500)],
        )
        oof[va] = m.predict(train_df.iloc[va][feature_cols], num_iteration=m.best_iteration).astype(np.float32)
        rmse = float(np.sqrt(np.mean((oof[va] - train_df.iloc[va]["target"].values) ** 2)))
        logger.info("  fold %d: rmse=%.4f best_iter=%d", fold, rmse, m.best_iteration)
        test_pred += m.predict(test_df[feature_cols], num_iteration=m.best_iteration).astype(np.float32) / N_SPLITS
    overall = float(np.sqrt(np.mean((oof - train_df["target"].values) ** 2)))
    logger.info("LGB seed=%d: OOF rmse=%.4f", seed, overall)
    return oof, test_pred, overall


XGB_BASE = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.06,
    max_depth=8,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.85,
    reg_alpha=1.0,
    reg_lambda=20.0,
    tree_method="hist",
    n_jobs=-1,
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

# ---------------------------------------------------------------------------
# 9) Ensemble: simple average vs ridge stack
# ---------------------------------------------------------------------------
oof_avg = np.mean([r["oof"] for r in results.values()], axis=0)
test_avg = np.mean([r["test"] for r in results.values()], axis=0)
rmse_avg = float(np.sqrt(np.mean((oof_avg - train_df["target"].values) ** 2)))
logger.info("simple avg OOF rmse = %.4f", rmse_avg)

stack_X = np.column_stack([r["oof"] for r in results.values()])
ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
ridge.fit(stack_X, train_df["target"].values)
stack_oof = ridge.predict(stack_X)
rmse_stack = float(np.sqrt(np.mean((stack_oof - train_df["target"].values) ** 2)))
weights = ridge.coef_ / max(ridge.coef_.sum(), 1e-9)
logger.info("ridge OOF rmse = %.4f weights=%s", rmse_stack,
            {{k: float(round(w, 3)) for k, w in zip(results.keys(), weights)}})
stack_test = ridge.predict(np.column_stack([r["test"] for r in results.values()]))

if rmse_avg <= rmse_stack:
    final_test = test_avg
    final_oof = oof_avg
    final_rmse = rmse_avg
    logger.info("Final: simple average")
else:
    final_test = stack_test
    final_oof = stack_oof
    final_rmse = rmse_stack
    logger.info("Final: ridge stack")
logger.info("Final raw OOF rmse: %.4f", final_rmse)

# ---------------------------------------------------------------------------
# 10) Anchor-shrinkage post-process (THE v10 LEVER)
#
# Apply BEFORE EWM smoothing because EWM smooths absolute TVT. We shrink
# the delta first, then reconstruct, then EWM. The order matters: shrink
# operates on (TVT - last_known_TVT).
# ---------------------------------------------------------------------------
shrunk_delta = constant_shrinkage(final_test.astype(np.float64), alpha=SHRINK_ALPHA)
shrunk_delta = hard_cap(shrunk_delta, band=HARD_CAP_BAND)

shrunk_oof = constant_shrinkage(final_oof.astype(np.float64), alpha=SHRINK_ALPHA)
shrunk_oof = hard_cap(shrunk_oof, band=HARD_CAP_BAND)

shrunk_oof_rmse = float(np.sqrt(np.mean((shrunk_oof - train_df["target"].values) ** 2)))
logger.info("Post-shrink OOF rmse: %.4f (alpha=%.2f, band=%.1f ft)",
            shrunk_oof_rmse, SHRINK_ALPHA, HARD_CAP_BAND)

# ---------------------------------------------------------------------------
# 11) Reconstruct absolute TVT and apply EWM(span=4) per well
# ---------------------------------------------------------------------------
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


pre_ewm_tvt = submission["tvt"].copy()
submission = submission.groupby("well", group_keys=False).apply(_apply_ewm)
ewm_change = float(np.mean(np.abs(submission["tvt"].values - pre_ewm_tvt.values)))
logger.info("EWM(span=%.1f) avg |delta| = %.3f ft", EWM_SPAN, ewm_change)

submission_out = submission[["id", "tvt"]].copy()
if submission_out["tvt"].isna().any():
    n_bad = int(submission_out["tvt"].isna().sum())
    logger.warning("NaN tvt in %d rows; backfilling with last_known_tvt", n_bad)
    bad = submission_out["tvt"].isna()
    submission_out.loc[bad, "tvt"] = test_anchor[bad.to_numpy()]

if not np.isfinite(submission_out["tvt"]).all():
    n_bad = int((~np.isfinite(submission_out["tvt"])).sum())
    median_tvt = float(np.median(test_anchor[np.isfinite(test_anchor)]))
    logger.warning("Non-finite tvt in %d rows; replacing with median=%.2f", n_bad, median_tvt)
    bad = ~np.isfinite(submission_out["tvt"])
    submission_out.loc[bad, "tvt"] = median_tvt

submission_out.to_csv(OUTPUT, index=False)
oof_df = pd.DataFrame({{
    "prediction_id": train_df["prediction_id"],
    "well": train_df["well"],
    "row_idx": train_df["row_idx"].astype(np.int32),
    "target": train_df["target"].values,
    "oof_pred": final_oof.astype(np.float64),
    "oof_pred_shrunk": shrunk_oof.astype(np.float64),
    "last_known_tvt": train_df["last_known_tvt"].astype(np.float64),
}})
oof_df.to_csv(OOF_OUT, index=False)

logger.info("Wrote %s (%d rows) and %s", OUTPUT, len(submission_out), OOF_OUT)
print(f"Submission: {{len(submission_out)}} rows, {{submission_out['id'].nunique()}} unique ids")
print("TVT stats:")
print(submission_out["tvt"].describe())
print("Head:")
print(submission_out.head(10))
print("Tail:")
print(submission_out.tail(10))
print(f"Final raw OOF rmse:    {{final_rmse:.4f}}")
print(f"Final shrunk OOF rmse: {{shrunk_oof_rmse:.4f}}")
print(f"Shrinkage alpha={{SHRINK_ALPHA}}, hard-cap band={{HARD_CAP_BAND}} ft")
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
OUT_IPYNB_V10.write_text(json.dumps(nb, indent=1))
OUT_IPYNB_ACTIVE.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT_IPYNB_V10}")
print(f"Wrote {OUT_IPYNB_ACTIVE} (active for kernel-metadata.json)")
