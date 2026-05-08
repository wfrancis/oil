"""Build a tiny Kaggle kernel that emits the v9 submission.csv directly.
Bypasses the need to re-run the heavy v9 pipeline (~2.2 hours) when we
already have a successfully-produced submission.csv.

This is for: 'submit something for public scoring NOW'.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = Path("/tmp/v9_output_latest/submission.csv")
OUT_IPYNB = ROOT / "notebook_v9_replay" / "submission.ipynb"
OUT_META = ROOT / "notebook_v9_replay" / "kernel-metadata.json"


csv_bytes = CSV.read_bytes()
gz_bytes = gzip.compress(csv_bytes, compresslevel=9)
b64 = base64.b64encode(gz_bytes).decode("ascii")
print(f"raw csv: {len(csv_bytes):,} bytes")
print(f"gz:      {len(gz_bytes):,} bytes")
print(f"b64:     {len(b64):,} chars")


CELL = f'''# v9 submission replay — emits the precomputed submission.csv
# Source: kernel `wbfranci/rogii-eagle-ford-dtw-rts-v1` v8 output.
# Original local OOF = 11.41, Kaggle OOF = 11.341.

import base64
import gzip
from pathlib import Path

OUT = Path("/kaggle/working/submission.csv")

PAYLOAD = "{b64}"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes(gzip.decompress(base64.b64decode(PAYLOAD)))

# Sanity check
import pandas as pd
df = pd.read_csv(OUT)
print(f"Submission rows: {{len(df)}}")
print(f"id duplicates: {{df['id'].duplicated().sum()}}")
print(f"tvt range:     {{df['tvt'].min():.2f}}  {{df['tvt'].max():.2f}}")
print("head:")
print(df.head())
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
    "id": "wbfranci/rogii-v9-replay-fast",
    "title": "ROGII v9 replay (fast)",
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

print(f"Wrote {OUT_IPYNB}")
print(f"Wrote {OUT_META}")
