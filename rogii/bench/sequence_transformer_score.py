"""5-fold OOF benchmark for the Sequence Transformer prototype.

Steps:
  1. Build the v9-style feature matrix (no beam — fast) for a subset of wells.
  2. GroupKFold(5) over wells; for each fold, train a small Transformer on
     train wells, evaluate on held-out wells.
  3. Report OOF RMSE, per-well stats, wall time.

This is a PROTOTYPE — we do not tune. We measure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from feature_builder import (   # noqa: E402
    FormationPlaneKNN,
    RowKNN,
    build_dataset,
)
from sequence_transformer import (   # noqa: E402
    SequenceTransformer,
    TransformerConfig,
    resample_typewell,
    normalize_typewell,
)


# ---------------------------------------------------------------------------
# Feature construction (cached on disk by subset signature)
# ---------------------------------------------------------------------------


def build_features_cached(
    paths: list[Path],
    cache_path: Path,
) -> pl.DataFrame:
    """Build features for the given subset, caching the result."""
    if cache_path.exists():
        print(f"[cache hit] loading features from {cache_path}", flush=True)
        return pl.read_parquet(cache_path)
    print(f"[cache miss] building features for {len(paths)} wells", flush=True)
    t0 = time.perf_counter()
    plane = FormationPlaneKNN.fit(paths)
    print(f"   plane fit: {time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    row = RowKNN.fit(paths)
    print(f"   row KNN fit: {time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    train_df_pd = build_dataset(
        paths,
        plane,
        row,
        is_train=True,
        primary_formation="ANCC",
        enable_beam=False,
        label="seq",
        progress_every=25,
    )
    print(
        f"   build_dataset: {time.perf_counter() - t0:.1f}s, shape={train_df_pd.shape}",
        flush=True,
    )
    df = pl.from_pandas(train_df_pd)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"   cached -> {cache_path}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Per-well dataset assembly
# ---------------------------------------------------------------------------


def load_typewells(wells: list[str], train_dir: Path) -> dict[str, np.ndarray]:
    """Load typewell (TVT, GR) for each well, return dict of (M, 2) arrays."""
    out: dict[str, np.ndarray] = {}
    for w in wells:
        p = train_dir / f"{w}__typewell.csv"
        try:
            df = pl.read_csv(p, columns=["TVT", "GR"]).drop_nulls()
            out[w] = df.to_numpy().astype(np.float32)
        except Exception:
            out[w] = np.zeros((0, 2), dtype=np.float32)
    return out


def build_well_arrays(
    feats: pl.DataFrame,
    feature_cols: list[str],
    typewells: dict[str, np.ndarray],
    typewell_max_len: int,
) -> dict[str, dict]:
    """Group features by well, normalize, build per-well tensors.

    Returns dict mapping wid -> {feats, target, last_known_tvt, tw, tw_mask}.
    """
    out: dict[str, dict] = {}
    grouped = feats.partition_by("well", maintain_order=True, as_dict=True)
    for well_key, sub in grouped.items():
        wid = well_key[0] if isinstance(well_key, tuple) else well_key
        # Order by row_idx to enforce sequence order
        sub = sub.sort("row_idx")
        x = sub.select(feature_cols).to_numpy().astype(np.float32)
        y = sub.get_column("target").to_numpy().astype(np.float32)
        last_known = float(sub.get_column("last_known_tvt").to_numpy()[0])
        tw = typewells.get(wid, np.zeros((0, 2), dtype=np.float32))
        if len(tw) == 0:
            tw_normed = np.zeros((typewell_max_len, 2), dtype=np.float32)
            tw_mask = np.ones(typewell_max_len, dtype=bool)
        else:
            tw_n = normalize_typewell(tw, ref_tvt=last_known)
            tw_normed, tw_mask = resample_typewell(
                tw_n[:, 0], tw_n[:, 1], typewell_max_len
            )
        out[wid] = {
            "x": x,
            "y": y,
            "last_known": last_known,
            "tw": tw_normed,
            "tw_mask": tw_mask,
        }
    return out


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


def fit_scaler(arrays: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std from concatenated train rows."""
    parts = [a["x"] for a in arrays.values()]
    cat = np.concatenate(parts, axis=0)
    mu = cat.mean(axis=0).astype(np.float32)
    sd = cat.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd


def apply_scaler(arrays: dict[str, dict], mu: np.ndarray, sd: np.ndarray) -> None:
    for v in arrays.values():
        v["x"] = (v["x"] - mu) / sd


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def chunked_iter(seq_len: int, chunk: int):
    """Yield (start, end) chunks covering [0, seq_len)."""
    for s in range(0, seq_len, chunk):
        yield s, min(s + chunk, seq_len)


def forward_well_eval(
    model: SequenceTransformer,
    arr: dict,
    device: torch.device,
    chunk: int,
    stride: int = 1,
) -> torch.Tensor:
    """Run model over a single well, chunked. Returns (N,) preds.

    If stride > 1, evaluate every `stride`-th row and linearly upsample.
    """
    x_full = torch.from_numpy(arr["x"]).to(device)
    tw = torch.from_numpy(arr["tw"]).unsqueeze(0).to(device)
    tw_mask = torch.from_numpy(arr["tw_mask"]).unsqueeze(0).to(device)
    N = x_full.shape[0]
    if stride > 1:
        idx = torch.arange(0, N, stride, device=device)
        x = x_full[idx].unsqueeze(0)
    else:
        x = x_full.unsqueeze(0)
    Nq = x.shape[1]
    preds = []
    for s, e in chunked_iter(Nq, chunk):
        out = model(x[:, s:e], tw, tw_mask)
        preds.append(out.squeeze(0))
    p = torch.cat(preds, dim=0)
    if stride > 1:
        # Upsample linearly back to N
        idx_full = torch.arange(N, device=device, dtype=torch.float32)
        p_full = torch.empty(N, device=device, dtype=p.dtype)
        # simple linear interp: scatter known, interpolate gaps
        known_pos = idx.float()
        # Use torch.searchsorted on known_pos
        pos_in_known = torch.searchsorted(known_pos, idx_full)
        pos_in_known = torch.clamp(pos_in_known, 0, len(known_pos) - 1)
        left = torch.clamp(pos_in_known - 1, 0, len(known_pos) - 1)
        right = pos_in_known
        wleft = (known_pos[right] - idx_full).float()
        wright = (idx_full - known_pos[left]).float()
        denom = (known_pos[right] - known_pos[left]).clamp(min=1.0)
        w_l = wleft / denom
        w_r = wright / denom
        # When right == left, default to p[left]
        same = (right == left)
        p_full = w_l * p[left] + w_r * p[right]
        p_full = torch.where(same, p[left], p_full)
        return p_full
    return p


def loss_well_train(
    model: SequenceTransformer,
    arr: dict,
    device: torch.device,
    chunk: int,
    train_chunk_rows: int,
) -> torch.Tensor:
    """Sample one chunk_rows-long random window from this well.

    For training only — random window keeps causal-mask semantics intact
    (we still attend over `chunk_rows` of past lateral context starting
    at the random start), and dramatically cuts wall clock vs. the full
    well in every minibatch.
    """
    x_full = torch.from_numpy(arr["x"]).to(device)
    y_full = torch.from_numpy(arr["y"]).to(device)
    tw = torch.from_numpy(arr["tw"]).unsqueeze(0).to(device)
    tw_mask = torch.from_numpy(arr["tw_mask"]).unsqueeze(0).to(device)
    N = x_full.shape[0]
    if N <= train_chunk_rows:
        x = x_full.unsqueeze(0)
        y = y_full.unsqueeze(0)
    else:
        s = int(torch.randint(0, N - train_chunk_rows + 1, (1,)).item())
        e = s + train_chunk_rows
        x = x_full[s:e].unsqueeze(0)
        y = y_full[s:e].unsqueeze(0)
    pred = model(x, tw, tw_mask)
    return torch.mean((pred - y) ** 2)


def train_fold(
    train_arrays: dict[str, dict],
    val_arrays: dict[str, dict],
    n_features: int,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    chunk: int,
    train_chunk_rows: int,
    eval_stride: int,
    seed: int,
    verbose: bool = True,
) -> tuple[SequenceTransformer, dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = TransformerConfig(n_features=n_features, max_lateral_chunk=chunk)
    model = SequenceTransformer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_keys = list(train_arrays.keys())
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"      model params: {n_params/1e6:.2f}M", flush=True)

    best_val = float("inf")
    best_oof: dict[str, np.ndarray] = {}
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + ep)
        order = rng.permutation(len(train_keys))
        ep_loss_sum = 0.0
        ep_count = 0
        t0 = time.perf_counter()
        for i in order:
            wid = train_keys[i]
            arr = train_arrays[wid]
            opt.zero_grad(set_to_none=True)
            loss = loss_well_train(
                model, arr, device, chunk, train_chunk_rows
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss_sum += float(loss.item())
            ep_count += 1
        train_rmse = math.sqrt(ep_loss_sum / max(ep_count, 1))

        # Quick val pass (stride for speed)
        model.eval()
        v_loss = 0.0
        v_rows = 0
        oof_this_ep: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for wid, arr in val_arrays.items():
                pred = forward_well_eval(
                    model, arr, device, chunk, stride=eval_stride
                )
                tgt = torch.from_numpy(arr["y"]).to(device)
                v_loss += float(torch.sum((pred - tgt) ** 2).item())
                v_rows += len(arr["y"])
                oof_this_ep[wid] = pred.detach().cpu().numpy().astype(np.float32)
        val_rmse = math.sqrt(v_loss / max(v_rows, 1))
        if val_rmse < best_val:
            best_val = val_rmse
            best_oof = oof_this_ep
        if verbose:
            print(
                f"      ep {ep+1:2d}/{epochs}  train_rmse(window)={train_rmse:7.3f}  "
                f"val_rmse={val_rmse:7.3f}  best={best_val:7.3f}  "
                f"{time.perf_counter()-t0:5.1f}s",
                flush=True,
            )

    return model, best_oof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-wells", type=int, default=200)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--train-chunk-rows", type=int, default=1024,
                   help="Random window size used for each well per epoch.")
    p.add_argument("--eval-stride", type=int, default=4,
                   help="Stride for OOF eval (predictions interp'd back).")
    p.add_argument("--tw-max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--cache", default="/tmp/seq_tx_features.parquet")
    args = p.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f">> device = {device}", flush=True)

    train_dir = ROOT / "data" / "competition" / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(len(paths))
    rng.shuffle(all_idx)
    sub = [paths[i] for i in all_idx[: args.n_wells]]
    print(f">> using {len(sub)} of {len(paths)} wells (seed={args.seed})", flush=True)

    cache_path = Path(args.cache)
    sig = f".n{args.n_wells}.s{args.seed}"
    if not str(cache_path).endswith(sig + ".parquet"):
        cache_path = cache_path.with_suffix(sig + ".parquet")

    t_overall = time.perf_counter()
    feats = build_features_cached(sub, cache_path)
    feature_cols = [
        c for c in feats.columns
        if c not in {"well", "prediction_id", "target", "row_idx",
                     "last_known_tvt", "known_len", "hidden_len"}
    ]
    print(f">> #features = {len(feature_cols)}, total rows = {len(feats):,}",
          flush=True)

    # Drop non-finite rows (replace NaN/inf in features)
    feats = feats.with_columns(
        [pl.col(c).cast(pl.Float32).fill_nan(0.0).fill_null(0.0) for c in feature_cols]
    )
    # Replace inf
    feats = feats.with_columns(
        [
            pl.when(pl.col(c).is_infinite()).then(0.0).otherwise(pl.col(c)).alias(c)
            for c in feature_cols
        ]
    )

    wells = feats.get_column("well").unique(maintain_order=True).to_list()
    typewells = load_typewells(wells, train_dir)
    print(f">> loaded {len(typewells)} typewells", flush=True)

    arrays_all = build_well_arrays(feats, feature_cols, typewells, args.tw_max_len)
    print(f">> built {len(arrays_all)} per-well tensor sets", flush=True)

    # GroupKFold over wells
    well_arr = np.array(list(arrays_all.keys()))
    dummy_target = np.arange(len(well_arr))
    gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(gkf.split(well_arr, dummy_target, groups=well_arr))

    oof_per_well: dict[str, np.ndarray] = {}
    target_per_well: dict[str, np.ndarray] = {}
    last_known_per_well: dict[str, float] = {}
    fold_rmses = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        tr_wells = well_arr[tr_idx].tolist()
        va_wells = well_arr[va_idx].tolist()
        print(
            f"\n>> fold {fold+1}/{args.folds}  train_wells={len(tr_wells)}  "
            f"val_wells={len(va_wells)}", flush=True,
        )
        train_arrays = {w: {**arrays_all[w], "x": arrays_all[w]["x"].copy()}
                        for w in tr_wells}
        val_arrays = {w: {**arrays_all[w], "x": arrays_all[w]["x"].copy()}
                      for w in va_wells}
        mu, sd = fit_scaler(train_arrays)
        apply_scaler(train_arrays, mu, sd)
        apply_scaler(val_arrays, mu, sd)

        t_fold = time.perf_counter()
        _, fold_oof = train_fold(
            train_arrays, val_arrays,
            n_features=len(feature_cols),
            epochs=args.epochs, lr=args.lr,
            device=device, chunk=args.chunk,
            train_chunk_rows=args.train_chunk_rows,
            eval_stride=args.eval_stride,
            seed=args.seed + fold,
        )
        print(f"   fold wall: {time.perf_counter()-t_fold:.1f}s", flush=True)

        for w, pred in fold_oof.items():
            oof_per_well[w] = pred
            target_per_well[w] = arrays_all[w]["y"]
            last_known_per_well[w] = arrays_all[w]["last_known"]

        # Per-fold RMSE
        flat_pred = np.concatenate([fold_oof[w] for w in va_wells])
        flat_tgt = np.concatenate([arrays_all[w]["y"] for w in va_wells])
        rmse = float(np.sqrt(np.mean((flat_pred - flat_tgt) ** 2)))
        fold_rmses.append(rmse)
        print(f"   fold {fold+1} RMSE={rmse:.4f}", flush=True)

    # Global OOF
    flat_pred = np.concatenate([oof_per_well[w] for w in well_arr])
    flat_tgt = np.concatenate([target_per_well[w] for w in well_arr])
    overall = float(np.sqrt(np.mean((flat_pred - flat_tgt) ** 2)))

    # Per-well RMSE
    per_well_rmse = {
        w: float(np.sqrt(np.mean((oof_per_well[w] - target_per_well[w]) ** 2)))
        for w in well_arr
    }
    rmse_arr = np.array(list(per_well_rmse.values()))
    summary = {
        "n_wells": len(well_arr),
        "n_features": len(feature_cols),
        "epochs": args.epochs,
        "fold_rmses": fold_rmses,
        "overall_rmse": overall,
        "well_rmse_median": float(np.median(rmse_arr)),
        "well_rmse_mean": float(np.mean(rmse_arr)),
        "well_rmse_p90": float(np.quantile(rmse_arr, 0.9)),
        "well_rmse_max": float(np.max(rmse_arr)),
        "wall_time_s": time.perf_counter() - t_overall,
    }
    print("\n==== Sequence Transformer prototype OOF ====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    out_path = Path("/tmp/seq_tx_oof.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
