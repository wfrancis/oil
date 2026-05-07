# ROGII Wellbore Geology Prediction - submission notebook v5
# Strategy: last-known TVT_input plus a conservative residual LightGBM.
# Local 5-fold CV improved from constant RMSE ~15.83 mean to ~14.55 mean
# with shrink=0.75 on a 0.8M-row sampled residual model.
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rogii.v5")

SHRINK = 0.75
MAX_TRAIN_ROWS = 1_200_000
RANDOM_SEED = 20260506
FALLBACK_TVT = 11354.51


def discover_data_root() -> str:
    input_root = "/kaggle/input"
    for root, dirs, _files in os.walk(input_root):
        depth = root.replace(input_root, "").count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        if "test" in dirs and "train" in dirs:
            logger.info("Found competition data at %s (depth %d)", root, depth)
            return root
    raise SystemExit("FATAL: could not locate competition test/+train/ directories.")


DATA_ROOT = discover_data_root()
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")


def train_median_tvt(train_dir: str) -> float:
    vals = []
    for path in sorted(glob.glob(os.path.join(train_dir, "*__horizontal_well.csv"))):
        try:
            tvt = pd.read_csv(path, usecols=["TVT"])["TVT"].to_numpy(dtype=np.float64, copy=False)
        except Exception as exc:
            logger.warning("Failed to read train TVT from %s: %s", path, exc)
            continue
        tvt = tvt[np.isfinite(tvt)]
        if tvt.size:
            vals.append(tvt)
    if not vals:
        return FALLBACK_TVT
    return float(np.median(np.concatenate(vals)))


TRAIN_MEDIAN_TVT = train_median_tvt(TRAIN_DIR)
logger.info("Train median TVT fallback: %.2f", TRAIN_MEDIAN_TVT)


def finite_array(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return df[col].to_numpy(dtype=np.float64, copy=False)


def build_eval_features(df: pd.DataFrame):
    md = finite_array(df, "MD")
    x = finite_array(df, "X")
    y = finite_array(df, "Y")
    z = finite_array(df, "Z")
    gr = finite_array(df, "GR")
    tvt_in = finite_array(df, "TVT_input")

    finite = np.isfinite(tvt_in)
    eval_mask = ~finite
    eval_idx = np.flatnonzero(eval_mask)
    n = len(df)
    if eval_idx.size == 0:
        return None
    if not finite.any():
        return {
            "no_anchor": True,
            "eval_idx": eval_idx,
            "last_tvt": TRAIN_MEDIAN_TVT,
            "X": None,
        }

    anchor_i = int(np.flatnonzero(finite)[-1])
    last_tvt = float(tvt_in[anchor_i])
    anchor_md = float(md[anchor_i])
    anchor_x = float(x[anchor_i])
    anchor_y = float(y[anchor_i])
    anchor_z = float(z[anchor_i])
    anchor_gr = float(gr[anchor_i])

    finite_idx = np.flatnonzero(finite)
    tail = finite_idx[-min(300, finite_idx.size):]
    if tail.size >= 2:
        mm = md[tail]
        tt = tvt_in[tail]
        dm = mm - np.nanmean(mm)
        den = float(np.nansum(dm * dm))
        slope = float(np.nansum((tt - np.nanmean(tt)) * dm) / den) if den > 1e-9 else 0.0
        slope = float(np.clip(slope, -0.005, 0.005))
        tvt_std = float(np.nanstd(tt))
        tvt_range = float(np.nanmax(tt) - np.nanmin(tt))
    else:
        slope = 0.0
        tvt_std = 0.0
        tvt_range = 0.0

    gr_mean = float(np.nanmean(gr)) if np.isfinite(gr).any() else 0.0
    gr_std = max(float(np.nanstd(gr)) if np.isfinite(gr).any() else 1.0, 1.0)
    eval_len = max(int(eval_idx[-1]) - anchor_i, 1)

    rows = []
    for i in eval_idx:
        row_delta = int(i) - anchor_i
        md_delta = float(md[i] - anchor_md)
        frac = row_delta / eval_len
        rows.append(
            [
                md_delta,
                row_delta,
                frac,
                float(x[i] - anchor_x),
                float(y[i] - anchor_y),
                float(z[i] - anchor_z),
                float(gr[i]),
                float(gr[i] - anchor_gr),
                float((gr[i] - gr_mean) / gr_std),
                last_tvt,
                anchor_md,
                anchor_x,
                anchor_y,
                anchor_z,
                anchor_gr,
                slope,
                tvt_std,
                tvt_range,
                n - anchor_i,
                float(md[i]),
                float(x[i]),
                float(y[i]),
                float(z[i]),
            ]
        )

    X = np.asarray(rows, dtype=np.float32)
    return {
        "no_anchor": False,
        "eval_idx": eval_idx,
        "last_tvt": last_tvt,
        "X": X,
    }


def train_residual_model(train_dir: str):
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        logger.warning("LightGBM unavailable (%s); using constant baseline.", exc)
        return None

    X_blocks = []
    y_blocks = []
    n_wells = 0
    for path in sorted(glob.glob(os.path.join(train_dir, "*__horizontal_well.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to read train horizontal %s: %s", path, exc)
            continue
        if "TVT" not in df.columns:
            continue
        built = build_eval_features(df)
        if built is None or built["no_anchor"] or built["X"] is None:
            continue
        tvt = df["TVT"].to_numpy(dtype=np.float64, copy=False)
        target = tvt[built["eval_idx"]] - float(built["last_tvt"])
        good = np.isfinite(target)
        if not good.any():
            continue
        X_blocks.append(built["X"][good])
        y_blocks.append(target[good].astype(np.float32, copy=False))
        n_wells += 1

    if not X_blocks:
        logger.warning("No residual training rows; using constant baseline.")
        return None

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    logger.info("Residual training matrix: X=%s y=%s wells=%d", X.shape, y.shape, n_wells)

    if X.shape[0] > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(RANDOM_SEED)
        take = rng.choice(X.shape[0], MAX_TRAIN_ROWS, replace=False)
        X = X[take]
        y = y[take]
        logger.info("Sampled residual training rows to %d.", X.shape[0])

    model = LGBMRegressor(
        objective="regression",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
    )
    try:
        model.fit(X, y)
    except Exception as exc:
        logger.warning("Residual model fit failed (%s); using constant baseline.", exc)
        return None
    return model


MODEL = train_residual_model(TRAIN_DIR)
logger.info("Residual model: %s", "enabled" if MODEL is not None else "disabled")


all_ids = []
all_tvts = []
for h_path in sorted(glob.glob(os.path.join(TEST_DIR, "*__horizontal_well.csv"))):
    wellname = Path(h_path).name.replace("__horizontal_well.csv", "")
    try:
        df = pd.read_csv(h_path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", h_path, exc)
        continue
    if "TVT_input" not in df.columns:
        logger.error("Well %s: no TVT_input column", wellname)
        continue

    built = build_eval_features(df)
    if built is None:
        logger.info("Well %s: no eval rows", wellname)
        continue

    eval_idx = built["eval_idx"]
    last_tvt = float(built["last_tvt"])
    if MODEL is None or built["X"] is None:
        pred_eval = np.full(eval_idx.size, last_tvt, dtype=np.float64)
    else:
        try:
            residual = MODEL.predict(built["X"])
            pred_eval = last_tvt + SHRINK * np.asarray(residual, dtype=np.float64)
        except Exception as exc:
            logger.warning("Residual predict failed for %s (%s); using constant.", wellname, exc)
            pred_eval = np.full(eval_idx.size, last_tvt, dtype=np.float64)

    bad = ~np.isfinite(pred_eval)
    if bad.any():
        pred_eval = pred_eval.copy()
        pred_eval[bad] = last_tvt

    for i, tvt in zip(eval_idx, pred_eval):
        all_ids.append(f"{wellname}_{int(i)}")
        all_tvts.append(float(tvt))

submission = pd.DataFrame({"id": all_ids, "tvt": all_tvts})
if submission["id"].duplicated().any():
    logger.error("Duplicate ids: %d", int(submission["id"].duplicated().sum()))
if submission["tvt"].isna().any():
    logger.error("NaN tvt: %d; median-patching", int(submission["tvt"].isna().sum()))
    submission["tvt"] = submission["tvt"].fillna(TRAIN_MEDIAN_TVT)
inf_mask = ~np.isfinite(submission["tvt"].to_numpy(dtype=np.float64, copy=False))
if inf_mask.any():
    submission.loc[inf_mask, "tvt"] = TRAIN_MEDIAN_TVT

OUTPUT_PATH = "/kaggle/working/submission.csv"
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)
logger.info("Wrote %s: %d rows", OUTPUT_PATH, len(submission))

print(f"Submission: {len(submission)} rows, {submission['id'].nunique()} unique ids")
print("TVT stats:")
print(submission["tvt"].describe())
print("Head:")
print(submission.head(10))
print("Tail:")
print(submission.tail(10))
