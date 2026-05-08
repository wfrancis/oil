"""Tiny kernel: predicts last_known_TVT_input for every eval row.
Equivalent to v4 (LB 15.883) but runs in ~1 minute on Kaggle CPU.
Used to verify Kaggle's scoring path is working at all.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_IPYNB = ROOT / "notebook_anchor_baseline" / "submission.ipynb"
OUT_META = ROOT / "notebook_anchor_baseline" / "kernel-metadata.json"


CELL = '''# Anchor-only baseline: predicts last_known_TVT_input for every hidden row.
# Used to verify Kaggle scoring path is functional.

import os, glob
from pathlib import Path
import numpy as np
import pandas as pd

# Locate competition data
INPUT_ROOT = Path("/kaggle/input")
DATA_ROOT = None
for root, dirs, _files in os.walk(INPUT_ROOT):
    depth = root.count("/") - str(INPUT_ROOT).count("/")
    if depth > 3:
        dirs[:] = []
        continue
    if "test" in dirs and "train" in dirs:
        DATA_ROOT = root
        break
if DATA_ROOT is None:
    raise SystemExit("FATAL: no test/+train/ found")

TEST_DIR = Path(DATA_ROOT) / "test"
print(f"DATA_ROOT={DATA_ROOT}")
print(f"test wells: {len(list(TEST_DIR.glob('*__horizontal_well.csv')))}")

rows = []
for h_path in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
    wid = h_path.name.replace("__horizontal_well.csv", "")
    df = pd.read_csv(h_path)
    if "TVT_input" not in df.columns:
        continue
    tvt_in = df["TVT_input"].to_numpy(dtype=np.float64)
    finite = np.isfinite(tvt_in)
    if finite.any():
        last = float(tvt_in[np.flatnonzero(finite)[-1]])
    else:
        last = 11354.51  # train-median fallback
    eval_idx = np.flatnonzero(~finite)
    for i in eval_idx:
        rows.append({"id": f"{wid}_{int(i)}", "tvt": last})

sub = pd.DataFrame(rows)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"submission rows: {len(sub)}")
print(sub.head())
'''

OUT_IPYNB.parent.mkdir(parents=True, exist_ok=True)
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
OUT_IPYNB.write_text(json.dumps(nb, indent=1))

OUT_META.write_text(json.dumps({
    "id": "wbfranci/rogii-anchor-baseline-fast",
    "title": "ROGII anchor-baseline fast",
    "code_file": "submission.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "kernel_sources": [],
}, indent=2))

print(f"wrote {OUT_IPYNB}")
print(f"wrote {OUT_META}")
