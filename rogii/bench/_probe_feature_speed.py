"""Sanity probe: how fast can we build features for 10 wells without MLP?

If this takes >60s on M1 Pro, we know feature building is the bottleneck and
need to cut the well count further.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from feature_builder import FormationPlaneKNN, RowKNN, build_dataset  # noqa: E402

train_dir = ROOT / "data" / "competition" / "train"
paths_all = sorted(train_dir.glob("*__horizontal_well.csv"))

# Use the SAME deterministic 150-well sample as the runner so we know if any
# of the first 10 are pathologically large.
rng = np.random.default_rng(42)
idx = rng.choice(len(paths_all), size=150, replace=False)
paths = [paths_all[i] for i in sorted(idx)][:10]
print(f"Probe wells: {[p.stem.split('__')[0] for p in paths]}", flush=True)

t0 = time.perf_counter()
plane = FormationPlaneKNN.fit(paths)
print(f"  plane fit: {time.perf_counter() - t0:.2f}s, n={len(plane.df)}", flush=True)

t0 = time.perf_counter()
row = RowKNN.fit(paths)
print(f"  row fit: {time.perf_counter() - t0:.2f}s, n={len(row.targets):,}", flush=True)

t0 = time.perf_counter()
df = build_dataset(
    paths, plane, row, is_train=True, mlp_imputer=None,
    primary_formation="ANCC", enable_beam=False, label="probe", progress_every=1,
)
print(f"  build: {time.perf_counter() - t0:.2f}s, shape={df.shape}", flush=True)

# Per-well row counts so we can see if any well has 20k+ rows
import pandas as pd
counts = df.groupby("well").size()
print(
    f"  per-well rows: median={counts.median():.0f}  p90={counts.quantile(0.9):.0f}  "
    f"max={counts.max()}", flush=True,
)
