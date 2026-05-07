"""Assemble the 3 src/ modules into a single Kaggle notebook cell.

Embeds alignment.py, geology.py, inference.py as base64-encoded strings,
then writes the cell body to notebook/kaggle_cell.py — that file is what
gets pasted into the Kaggle notebook editor.
"""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "notebook" / "kaggle_cell.py"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


alignment_b64 = b64(SRC / "alignment.py")
geology_b64 = b64(SRC / "geology.py")
inference_b64 = b64(SRC / "inference.py")


CELL = f'''# ROGII Wellbore Geology Prediction — submission notebook v2
# Auto-discovers the competition data path under /kaggle/input/.

import os
import sys
import base64
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rogii.notebook")

# --- Write the three modules to /kaggle/working/rogii_src/ ---
SRC_DIR = "/kaggle/working/rogii_src"
os.makedirs(SRC_DIR, exist_ok=True)

_MODULES = {{
    "alignment.py": "{alignment_b64}",
    "geology.py": "{geology_b64}",
    "inference.py": "{inference_b64}",
}}

for _name, _payload in _MODULES.items():
    _path = os.path.join(SRC_DIR, _name)
    with open(_path, "wb") as _f:
        _f.write(base64.b64decode(_payload))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Discover the competition data path under /kaggle/input/ ---
INPUT_ROOT = "/kaggle/input"
logger.info("Listing /kaggle/input/ ...")
if not os.path.isdir(INPUT_ROOT):
    raise SystemExit(f"FATAL: {{INPUT_ROOT}} does not exist on this kernel.")

input_entries = sorted(os.listdir(INPUT_ROOT))
logger.info("/kaggle/input/ contains: %s", input_entries)

# Walk up to 3 levels deep to find a directory that contains test/ and train/ subdirs.
# Real Kaggle layouts seen so far:
#   /kaggle/input/<comp-slug>/...                       (1 level)
#   /kaggle/input/competitions/<comp-slug>/test|train   (2 levels)
DATA_ROOT = None
for root, dirs, files in os.walk(INPUT_ROOT):
    depth = root.replace(INPUT_ROOT, "").count(os.sep)
    if depth > 3:
        # Don't recurse forever.
        dirs[:] = []
        continue
    if "test" in dirs and "train" in dirs:
        DATA_ROOT = root
        logger.info("  *** found competition data at %s (depth %d)", DATA_ROOT, depth)
        break
    if depth <= 2:
        logger.info("  %s -> dirs=%s", root, sorted(dirs)[:10])

if DATA_ROOT is None:
    raise SystemExit("FATAL: could not locate competition test/ + train/ directories.")

TEST_DIR = os.path.join(DATA_ROOT, "test") + "/"
TRAIN_DIR = os.path.join(DATA_ROOT, "train") + "/"
logger.info("TEST_DIR = %s", TEST_DIR)
logger.info("TRAIN_DIR = %s", TRAIN_DIR)

# Spot-check the directory contents.
test_files = sorted(os.listdir(TEST_DIR))[:6]
logger.info("First test files: %s", test_files)
n_horiz = sum(1 for f in os.listdir(TEST_DIR) if f.endswith("__horizontal_well.csv"))
n_typew = sum(1 for f in os.listdir(TEST_DIR) if f.endswith("__typewell.csv"))
logger.info("Test dir: %d horizontal wells, %d typewells", n_horiz, n_typew)

# --- Imports + Numba warm-up ---
import numpy as np
import pandas as pd
import polars as pl

from alignment import predict_well_dtw, dtw_align_gr
from geology import (
    fit_formation_gr_model,
    constrain_tvt_predictions,
    regional_dip_prior,
    FORMATION_ORDER,
)
from inference import predict_well, build_submission, rts_smooth

logger.info("Imports OK. FORMATION_ORDER=%s", FORMATION_ORDER)

try:
    _h = np.linspace(50.0, 150.0, 200, dtype=np.float64)
    _t = np.linspace(50.0, 150.0, 200, dtype=np.float64)
    _t_tvt = np.linspace(0.0, 100.0, 200, dtype=np.float64)
    _h_known = np.full(200, np.nan, dtype=np.float64)
    _h_known[0] = 0.0
    _ = dtw_align_gr(_h, _h_known, _t, _t_tvt, band_pct=0.15)
    logger.info("Numba DTW kernel warmed up.")
except Exception as exc:
    logger.warning("DTW warm-up failed (non-fatal): %s", exc)

# --- Run inference and write submission.csv ---
OUTPUT_PATH = "/kaggle/working/submission.csv"

df = build_submission(
    test_dir=TEST_DIR,
    output_path=OUTPUT_PATH,
    n_jobs=1,
    smoother="rts",
)

print(f"Submission: {{len(df)}} rows, {{df['id'].nunique() if len(df) else 0}} unique ids")
if len(df) > 0:
    print("TVT stats:")
    print(df['tvt'].describe())
    print("Head:")
    print(df.head(10))
    print("Tail:")
    print(df.tail(10))
else:
    print("WARNING: empty submission — investigate input path / well file naming.")
'''

OUT.write_text(CELL)
print(f"Wrote {OUT} ({len(CELL):,} chars)")
