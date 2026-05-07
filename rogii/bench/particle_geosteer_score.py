"""Local scorer for the particle-filter geosteering predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from particle_geosteer import (
    ParticleGeosteerConfig,
    particle_filter_well,
)
from formation_stack import (
    FormationStackPredictor,
    load_train_horizontals,
    FORMATION_COLS,
)

DEFAULT_TRAIN_DIR = ROOT / "data" / "competition" / "train"
NULL_VALUES = ["", "NA", "NaN", "nan", "null"]


def _stable_score(well: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{well}".encode(), digest_size=8).digest(),
        "big",
    )


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=2000,
                       null_values=NULL_VALUES, truncate_ragged_lines=True)


def cmd_hold(args: argparse.Namespace) -> None:
    train_dir = Path(args.train_dir)
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    wells = [p.name.replace("__horizontal_well.csv", "") for p in paths]

    held = sorted(wells, key=lambda w: _stable_score(w, args.seed))[:args.n]
    held_set = set(held)

    print(f">> Building spatial prior from {len(wells) - len(held)} non-held wells ...")
    t0 = time.perf_counter()
    train_for_fit = load_train_horizontals(train_dir, formations=FORMATION_COLS)
    fold_train = {k: v for k, v in train_for_fit.items() if k not in held_set}
    pred = FormationStackPredictor(
        train_wells=fold_train, formations=FORMATION_COLS,
        k_row=20, k_plane=10, b_method="median", primary_formation="ANCC",
    ).fit()
    fit_s = time.perf_counter() - t0
    print(f"   spatial fit_s={fit_s:.1f}")

    cfg = ParticleGeosteerConfig(
        n_particles=args.n_particles,
        sigma_step=args.sigma_step,
        obs_sigma=args.obs_sigma,
        prior_sigma=args.prior_sigma,
        use_prior=bool(args.use_prior),
        sigma_init=args.sigma_init,
        sigma_step_jump=args.sigma_step_jump,
        p_jump=args.p_jump,
        seed=args.pf_seed,
    )
    print(f">> Particle-filter config: {cfg}")

    rows: list[dict] = []
    pf_only_rows: list[dict] = []
    formula_only_rows: list[dict] = []
    blend_rows: list[dict] = []
    pred_t0 = time.perf_counter()

    for wid in held:
        h_path = train_dir / f"{wid}__horizontal_well.csv"
        t_path = train_dir / f"{wid}__typewell.csv"
        if not h_path.exists() or not t_path.exists():
            continue
        h = _read_csv(h_path)
        t = _read_csv(t_path)
        # Cast for safety
        for c in ("MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"):
            if c in h.columns:
                h = h.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        for c in ("TVT", "GR"):
            if c in t.columns:
                t = t.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        if "TVT" not in h.columns or "TVT_input" not in h.columns:
            continue

        tvt = h["TVT"].to_numpy().astype(np.float64)
        tvt_in = h["TVT_input"].to_numpy().astype(np.float64)
        eval_mask = ~np.isfinite(tvt_in) & np.isfinite(tvt)
        eval_idx = np.flatnonzero(eval_mask)
        if eval_idx.size == 0:
            continue
        truth = tvt[eval_idx]

        # 1) Spatial prior (v8 formula on row KNN, primary=ANCC)
        feats = pred.features_for_well(h, well_id=wid)
        formula = feats["tvt_formula_row"]
        formula_pred = formula.copy()
        # Replace NaN with last_known_tvt
        finite_in = np.isfinite(tvt_in)
        if finite_in.any():
            anchor_tvt = float(tvt_in[np.flatnonzero(finite_in)[-1]])
            formula_pred = np.where(np.isfinite(formula_pred), formula_pred, anchor_tvt)
        # Pin prefix
        formula_pred = np.where(finite_in, tvt_in, formula_pred)
        formula_err = formula_pred[eval_idx] - truth
        formula_only_rows.append({"well": wid, "err": formula_err})

        # 2) Particle filter only (no spatial prior)
        cfg_no_prior = ParticleGeosteerConfig(**{**cfg.__dict__, "use_prior": False})
        pf_no = particle_filter_well(h, t, config=cfg_no_prior, spatial_prior=None)
        pf_no_pred = np.where(finite_in, tvt_in, pf_no["tvt"])
        bad = ~np.isfinite(pf_no_pred)
        if bad.any() and finite_in.any():
            pf_no_pred = np.where(bad, anchor_tvt, pf_no_pred)
        pf_no_err = pf_no_pred[eval_idx] - truth
        pf_only_rows.append({"well": wid, "err": pf_no_err})

        # 3) Particle filter WITH spatial prior (the proper combo)
        pf_with = particle_filter_well(h, t, config=cfg, spatial_prior=formula)
        pf_with_pred = np.where(finite_in, tvt_in, pf_with["tvt"])
        bad = ~np.isfinite(pf_with_pred)
        if bad.any() and finite_in.any():
            pf_with_pred = np.where(bad, anchor_tvt, pf_with_pred)
        pf_with_err = pf_with_pred[eval_idx] - truth
        rows.append({"well": wid, "err": pf_with_err})

        # 4) Blend formula and PF (alpha sweep would happen at model level)
        alpha = float(args.blend_alpha)
        blend = alpha * pf_with_pred + (1 - alpha) * formula_pred
        blend = np.where(finite_in, tvt_in, blend)
        blend_err = blend[eval_idx] - truth
        blend_rows.append({"well": wid, "err": blend_err})

    pred_s = time.perf_counter() - pred_t0
    print(f">> Predicted {len(rows)} wells in {pred_s:.1f}s "
          f"({pred_s / max(1, len(rows)):.2f}s/well)")

    def summarize(label, items):
        if not items:
            print(f"  {label}: empty")
            return
        err = np.concatenate([r["err"] for r in items])
        rmse = float(np.sqrt(np.mean(err * err)))
        bias = float(np.mean(err))
        well_rmse = np.array([float(np.sqrt(np.mean(r["err"] ** 2))) for r in items])
        print(f"  {label}:  rmse={rmse:.3f}  bias={bias:+.3f}  "
              f"median_well={np.median(well_rmse):.2f}  "
              f"max_well={well_rmse.max():.2f}  wells={len(items)}")

    print("\n=== Summary ===")
    summarize("formula only (v8 spatial)", formula_only_rows)
    summarize(f"PF (no prior, sigma_step={cfg.sigma_step}, obs_sigma={cfg.obs_sigma})", pf_only_rows)
    summarize(f"PF + spatial prior", rows)
    summarize(f"blend alpha={args.blend_alpha:.2f}", blend_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hold", help="Hold N wells; PF score.")
    p.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-particles", type=int, default=4000)
    p.add_argument("--sigma-init", type=float, default=8.0)
    p.add_argument("--sigma-step", type=float, default=0.6)
    p.add_argument("--sigma-step-jump", type=float, default=30.0)
    p.add_argument("--p-jump", type=float, default=0.002)
    p.add_argument("--obs-sigma", type=float, default=12.0)
    p.add_argument("--prior-sigma", type=float, default=30.0)
    p.add_argument("--use-prior", type=int, default=1)
    p.add_argument("--pf-seed", type=int, default=42)
    p.add_argument("--blend-alpha", type=float, default=0.5)
    p.set_defaults(func=cmd_hold)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
