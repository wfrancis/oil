"""Neural ANCC(X, Y) surface model.

Hypothesis (from STRATEGY_RESET): konbu17's per-well plane fit and row-level
KNN are *local* — they fail in spatial regions with sparse training neighbors,
producing the catastrophic well-RMSE outliers we see in v8 OOF (max 56 ft).
A small MLP with sinusoidal positional encoding (NeRF-style) on (X, Y) learns
a *global smooth surface* that should generalize better in those sparse regions
while still matching local features via the high-frequency PE bands.

The load-bearing identity:
    TVT = -Z + ANCC + b_well   (intra-well sigma 0.0065 ft, exact)

So predicting ANCC(X, Y) at a held-out well's (X, Y), then plugging into the
closed-form TVT formula with the well's median b_well from its visible prefix,
is sufficient to recover TVT with the same fidelity as ANCC.

Design (per the brief, no tuning saga):
  1. (X, Y) normalized to [-1, 1] over the train extent.
  2. Sinusoidal positional encoding: gamma(p) = [sin(2^k * pi * p), cos(...)]
     for k = 0..L-1, applied to X and Y separately. Output dim = 4 * L per coord.
     Plus the raw (X, Y) feature concatenated.
  3. MLP: 4 layers x 256, SiLU, NeRF-style skip from input to layer 2.
  4. Adam, lr=1e-3, cosine decay, batch 4096, 500k rows / epoch, MPS backend.
  5. Squared loss on ANCC. Multi-output variant predicts all 6 formation tops.

All training is per-fold (train fold rows only). Inference: model.predict_xy
on the held-out (X, Y) -> ANCC -> (-Z + ANCC + b_prefix_median) -> TVT.

This module is self-contained — no pandas, polars + torch only. The 5M (X, Y,
formation) tuples fit comfortably in 32GB.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_rows(
    train_dir: Path,
    formations: Sequence[str] = FORMATIONS,
    paths: Iterable[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all (X, Y, formations[, well]) rows from training CSVs.

    Returns
    -------
    xy : (N, 2) float64
    targets : (N, F) float32   F = len(formations)
    wids : (N,) object str    well ID per row
    """
    if paths is None:
        paths = sorted(Path(train_dir).glob("*__horizontal_well.csv"))
    cols = ["X", "Y", *formations]
    xy_blocks: list[np.ndarray] = []
    f_blocks: list[np.ndarray] = []
    wid_blocks: list[np.ndarray] = []
    skipped = 0
    for p in paths:
        wid = p.stem.replace("__horizontal_well", "")
        try:
            # Force ANCC float (some wells store it as Utf8 with all-null);
            # polars read_csv schema_overrides handles either.
            df = pl.read_csv(p, columns=cols, infer_schema_length=10_000)
        except Exception:
            skipped += 1
            continue
        # Coerce all formations to Float64 if they came back as Utf8.
        for c in formations:
            if df[c].dtype == pl.Utf8:
                df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        df = df.drop_nulls(subset=cols)
        if len(df) == 0:
            skipped += 1
            continue
        xy_blocks.append(df.select(["X", "Y"]).to_numpy().astype(np.float64))
        f_blocks.append(df.select(list(formations)).to_numpy().astype(np.float32))
        wid_blocks.append(np.full(len(df), wid, dtype=object))
    if not xy_blocks:
        raise RuntimeError(f"No training rows loaded from {train_dir}")
    xy = np.concatenate(xy_blocks)
    targets = np.concatenate(f_blocks)
    wids = np.concatenate(wid_blocks)
    return xy, targets, wids


# ---------------------------------------------------------------------------
# Neural model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """NeRF-style sinusoidal positional encoding on each coordinate.

    p assumed normalized to roughly [-1, 1]. Output dim = 4*L (cos/sin x 2 coords).
    """

    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        # 2^k * pi for k = 0 .. L-1
        freqs = (2.0 ** torch.arange(num_freqs)) * math.pi
        self.register_buffer("freqs", freqs.to(torch.float32))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (B, 2)
        if self.num_freqs == 0:
            return xy
        scaled = xy.unsqueeze(-1) * self.freqs   # (B, 2, L)
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        # interleave to (B, 4 * L)
        encoded = torch.cat([sin, cos], dim=-1).flatten(start_dim=1)
        return torch.cat([xy, encoded], dim=-1)  # raw + PE


class NerfMLP(nn.Module):
    """4 hidden layers x 256, SiLU, with skip from input to layer 2."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden + in_dim, hidden)   # skip
        self.fc4 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))
        h = F.silu(self.fc3(torch.cat([h, x], dim=-1)))
        h = F.silu(self.fc4(h))
        return self.head(h)


@dataclass
class TrainConfig:
    num_freqs: int = 8
    hidden: int = 256
    out_dim: int = 1
    rows_per_epoch: int = 500_000
    batch_size: int = 4096
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    val_frac: float = 0.0  # no internal val: external GroupKFold owns val.


class AnccNet:
    """Wraps the model + train extent normalizer + train loop.

    out_dim==1 -> ANCC only. out_dim==len(FORMATIONS) -> all-formations head.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.pe = PositionalEncoding(cfg.num_freqs)
        in_dim = 2 + (4 * cfg.num_freqs if cfg.num_freqs > 0 else 0)
        self.mlp = NerfMLP(in_dim, cfg.hidden, cfg.out_dim)
        self.device = torch.device(cfg.device)
        self.pe.to(self.device)
        self.mlp.to(self.device)
        # train-extent normalizer (set in fit)
        self.x_mid = 0.0
        self.x_scl = 1.0
        self.y_mid = 0.0
        self.y_scl = 1.0
        # target normalizer (mean / std per output dim, set in fit)
        self.t_mean = np.zeros(cfg.out_dim, dtype=np.float32)
        self.t_std = np.ones(cfg.out_dim, dtype=np.float32)

    # -- normalizers --------------------------------------------------------

    def _fit_norm(self, xy: np.ndarray, targets: np.ndarray) -> None:
        x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
        y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
        self.x_mid = 0.5 * (x_min + x_max)
        self.x_scl = max(0.5 * (x_max - x_min), 1.0)
        self.y_mid = 0.5 * (y_min + y_max)
        self.y_scl = max(0.5 * (y_max - y_min), 1.0)
        self.t_mean = targets.mean(axis=0).astype(np.float32)
        # Avoid div-by-zero on degenerate cases.
        self.t_std = np.maximum(targets.std(axis=0), 1.0).astype(np.float32)

    def _norm_xy(self, xy: np.ndarray) -> np.ndarray:
        out = np.empty_like(xy, dtype=np.float32)
        out[:, 0] = (xy[:, 0] - self.x_mid) / self.x_scl
        out[:, 1] = (xy[:, 1] - self.y_mid) / self.y_scl
        return out

    # -- training ----------------------------------------------------------

    def fit(self, xy_train: np.ndarray, t_train: np.ndarray, *, verbose: bool = False) -> dict:
        cfg = self.cfg
        if t_train.ndim == 1:
            t_train = t_train.reshape(-1, 1)
        assert t_train.shape[1] == cfg.out_dim, (t_train.shape, cfg.out_dim)
        self._fit_norm(xy_train, t_train)
        xy_n = self._norm_xy(xy_train)
        t_n = ((t_train - self.t_mean) / self.t_std).astype(np.float32)

        device = self.device
        xy_t = torch.from_numpy(xy_n).to(device)
        t_t = torch.from_numpy(t_n).to(device)
        N = xy_t.shape[0]

        opt = torch.optim.Adam(self.mlp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # Cosine decay over total iterations (across all epochs).
        rows_per_epoch = min(cfg.rows_per_epoch, N)
        steps_per_epoch = max(rows_per_epoch // cfg.batch_size, 1)
        total_steps = cfg.epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

        rng = np.random.default_rng(cfg.seed)
        epoch_loss: list[float] = []
        t_start = time.perf_counter()

        self.mlp.train()
        for ep in range(cfg.epochs):
            # Sample rows_per_epoch random row indices for this epoch.
            sel = torch.from_numpy(
                rng.choice(N, rows_per_epoch, replace=False).astype(np.int64)
            ).to(device)
            xy_ep = xy_t[sel]
            t_ep = t_t[sel]
            # Shuffle within epoch is implicit by sampling.
            n_loss = 0.0
            for s in range(0, rows_per_epoch, cfg.batch_size):
                xb = xy_ep[s:s + cfg.batch_size]
                yb = t_ep[s:s + cfg.batch_size]
                feats = self.pe(xb)
                pred = self.mlp(feats)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                n_loss += loss.item() * xb.shape[0]
            avg = n_loss / rows_per_epoch
            epoch_loss.append(avg)
            if verbose:
                print(f"  ep {ep:02d}  loss(norm)={avg:.5f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        elapsed = time.perf_counter() - t_start
        return {"epoch_loss": epoch_loss, "fit_time_s": elapsed}

    @torch.no_grad()
    def predict(self, xy: np.ndarray, *, batch_size: int = 200_000) -> np.ndarray:
        """Predict targets at xy. Returns (N, out_dim) float32 in original units."""
        self.mlp.eval()
        xy_n = self._norm_xy(xy)
        xy_t = torch.from_numpy(xy_n).to(self.device)
        outs: list[np.ndarray] = []
        for s in range(0, xy_t.shape[0], batch_size):
            feats = self.pe(xy_t[s:s + batch_size])
            pred = self.mlp(feats).cpu().numpy()
            outs.append(pred)
        out = np.concatenate(outs, axis=0)
        out = out * self.t_std[None, :] + self.t_mean[None, :]
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Closed-form b_well from prefix
# ---------------------------------------------------------------------------

def fit_b_prefix_median(
    prefix_tvt_input: np.ndarray, prefix_z: np.ndarray, prefix_ancc_pred: np.ndarray
) -> float:
    """Median per-row b = TVT_input + Z - ANCC_pred over the visible prefix."""
    res = prefix_tvt_input + prefix_z - prefix_ancc_pred
    res = res[np.isfinite(res)]
    return float(np.median(res)) if res.size else 0.0
