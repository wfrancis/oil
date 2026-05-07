"""Sequence Transformer over the lateral-well MD axis with cross-attention
to the typewell (TVT, GR) sequence.

Per row token = the GBM feature vector at that MD step. The encoder applies
causal self-attention over the lateral so eval rows only see past lateral
context, plus cross-attention with the *full* typewell sequence (the typewell
is fully observed and not part of what we predict, so it's safe to attend to
it bidirectionally).

The regression head predicts (TVT - last_known_TVT) per row, matching v8/v9
target convention so we can compare apples-to-apples.

Designed for M1 Pro / MPS:
    - d_model=128, n_heads=4, 4 layers (small enough that attention on ~6k
      lateral tokens is workable; chunked into 1024 to keep peak memory
      bounded).
    - LayerNorm + GELU + standard PreNorm style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    n_features: int           # input feature dim per lateral row
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_lateral_chunk: int = 1024   # chunk lateral seq to bound attn memory
    typewell_max_len: int = 1024    # truncate / linearly resample typewell


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class SinusoidalPE(nn.Module):
    """Standard 1D sinusoidal PE (additive). Good enough for prototype."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TypewellEncoder(nn.Module):
    """Project (TVT, GR) sequence into d_model space."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2, d_model)
        self.pe = SinusoidalPE(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tw: torch.Tensor) -> torch.Tensor:
        # tw: (B, M, 2)
        h = self.proj(tw)
        h = self.pe(h)
        return self.norm(h)


class CrossAttnBlock(nn.Module):
    """Pre-norm self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_cross_q = nn.LayerNorm(d_model)
        self.ln_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,             # (B, N, d)
        kv: torch.Tensor,            # (B, M, d)
        attn_mask: torch.Tensor | None = None,
        kv_key_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self attention with causal mask
        h = self.ln_self(x)
        a, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        # Cross attention to typewell
        q = self.ln_cross_q(x)
        kv2 = self.ln_cross_kv(kv)
        c, _ = self.cross_attn(q, kv2, kv2, key_padding_mask=kv_key_padding,
                               need_weights=False)
        x = x + c
        x = x + self.ffn(self.ln_ffn(x))
        return x


class SequenceTransformer(nn.Module):
    """End-to-end model. Inputs:

        feats: (B, N, F)  per-row feature matrix (lateral)
        tw:    (B, M, 2)  typewell (TVT, GR)
        tw_mask: (B, M)   True where padded, optional
    Output:
        delta_pred: (B, N) predicted (TVT - last_known_TVT)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.input_pe = SinusoidalPE(cfg.d_model, max_len=cfg.max_lateral_chunk + 64)
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.tw_encoder = TypewellEncoder(cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                CrossAttnBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    @staticmethod
    def _causal_mask(n: int, device) -> torch.Tensor:
        # nn.MultiheadAttention expects True = mask out / -inf addition; we
        # use an additive float mask for compatibility with MPS (which
        # supports float masks but is finicky about boolean ones).
        m = torch.full((n, n), float("-inf"), device=device)
        m = torch.triu(m, diagonal=1)
        return m

    def forward(
        self,
        feats: torch.Tensor,
        tw: torch.Tensor,
        tw_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = feats.shape
        h = self.input_proj(feats)
        h = self.input_pe(h)
        h = self.input_norm(h)

        kv = self.tw_encoder(tw)
        attn_mask = self._causal_mask(N, h.device)

        for blk in self.blocks:
            h = blk(h, kv, attn_mask=attn_mask, kv_key_padding=tw_mask)

        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers for batching variable-length lateral sequences
# ---------------------------------------------------------------------------


def resample_typewell(tw_tvt: np.ndarray, tw_gr: np.ndarray, max_len: int) -> np.ndarray:
    """Linearly resample (TVT, GR) onto fixed-length grid for batching.

    Truncating would lose physically distant typewell context; resampling
    preserves global shape at the cost of slight detail.
    """
    n = len(tw_tvt)
    if n <= max_len:
        # Pad
        out = np.zeros((max_len, 2), dtype=np.float32)
        out[:n, 0] = tw_tvt
        out[:n, 1] = tw_gr
        mask = np.zeros(max_len, dtype=bool)
        mask[n:] = True
        return out, mask
    idx = np.linspace(0, n - 1, max_len)
    i0 = np.clip(np.floor(idx).astype(np.int64), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    f = (idx - i0).astype(np.float32)
    tvt_r = tw_tvt[i0] * (1 - f) + tw_tvt[i1] * f
    gr_r = tw_gr[i0] * (1 - f) + tw_gr[i1] * f
    out = np.stack([tvt_r, gr_r], axis=1).astype(np.float32)
    mask = np.zeros(max_len, dtype=bool)
    return out, mask


def normalize_typewell(tw: np.ndarray, ref_tvt: float) -> np.ndarray:
    """Anchor TVT to last_known_TVT and zscore GR per-well."""
    out = tw.copy()
    out[:, 0] = out[:, 0] - ref_tvt
    gr = out[:, 1]
    gr_mu = float(np.mean(gr))
    gr_sd = float(np.std(gr) + 1e-6)
    out[:, 1] = (gr - gr_mu) / gr_sd
    return out
