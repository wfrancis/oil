"""Smoke test + equivalence check for triple_signal_pf.

What this verifies:

1. **Equivalence to the source notebook's PFs.** We pull the PF code out of
   the notebook by re-implementing it inline as ``_source_*`` functions
   (verbatim copies of cells 5/6) and run them against our ported versions
   on a single well, with the same RNG seed. Asserts max-abs delta is 0.

2. **Parallel driver correctness.** Sequential single-well calls vs. the
   ``run_pfs_for_wells`` driver on the same wells must agree to within
   float32 rounding (the driver casts to float32 in the result dict).

3. **Wall-time benchmark.** 30 wells, sequential vs. 8-worker parallel.
   Print extrapolation to 773 wells.

Run:
    python bench/triple_signal_pf_smoke.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd  # only inside this test for pandas-parity reference
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triple_signal_pf import (  # noqa: E402
    run_pf_z_velocity,
    run_pf_ancc,
    run_pfs_for_wells,
    load_wells_from_dir,
    PF_N_PARTICLES,
    ANCC_N_PARTICLES,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# Verbatim copies of the source-notebook constants and PF functions, used as
# the parity oracle. DO NOT EDIT to match our port — these are the spec.
# (Source: research/public_kernels/triple-signal-beam-search-dual-pf-lightgbm.ipynb)
# ---------------------------------------------------------------------------

PF_MOMENTUM_ALPHA = 0.993
PF_Z_SIGMA_FLOOR = 0.005
PF_Z_SIGMA_SCALE = 2.0
PF_VELOCITY_NOISE_STD = 0.005
PF_POSITION_NOISE_STD = 0.01
PF_INIT_VELOCITY_STD = 0.02
PF_GR_SIGMA_MIN = 10.0
PF_GR_SIGMA_MAX = 60.0
PF_GR_SIGMA_DEFAULT = 30.0
PF_INIT_SPREAD_STD = 0.5
PF_RESAMPLE_THRESHOLD = 0.5
PF_ROUGHENING_STD_POS = 0.2
PF_ROUGHENING_STD_VEL = 0.003
PF_GR_ROLLING_WINDOW = 5
PF_GR_ROLLING_WEIGHT = 0.3

ANCC_ALPHA = 0.998
ANCC_RATE_NOISE_STD = 0.002
ANCC_POS_NOISE_STD = 0.005
ANCC_INIT_RATE_STD = 0.01
ANCC_INIT_SPREAD_STD = 0.3
ANCC_ROUGHENING_STD_POS = 0.1
ANCC_ROUGHENING_STD_RATE = 0.001


def _source_pf_calibrate_gr_sigma(hw, tw_tvt, tw_gr):
    from scipy.interpolate import interp1d
    known = hw[hw['TVT_input'].notna()]
    known_gr = known[known['GR'].notna()]
    if len(known_gr) < 20:
        return PF_GR_SIGMA_DEFAULT
    tw_func = interp1d(tw_tvt, tw_gr, bounds_error=False,
                       fill_value=(tw_gr[0], tw_gr[-1]))
    expected = tw_func(known_gr['TVT_input'].values)
    residuals = known_gr['GR'].values - expected
    return np.clip(np.std(residuals), PF_GR_SIGMA_MIN, PF_GR_SIGMA_MAX)


def _source_pf_estimate_init_velocity(hw):
    known = hw[hw['TVT_input'].notna()]
    if len(known) < 10:
        return 0.0
    tail = known.tail(20)
    if len(tail) < 5:
        return 0.0
    dtvt = np.diff(tail['TVT_input'].values)
    dmd = np.diff(tail['MD'].values)
    mask = dmd > 0
    if mask.sum() < 3:
        return 0.0
    return np.median(dtvt[mask] / dmd[mask])


def _source_pf_learn_z_beta(hw):
    known = hw[hw['TVT_input'].notna()]
    if len(known) < 30:
        return -1.0, 0.0, 0.1
    dz = np.diff(known['Z'].values)
    dtvt = np.diff(known['TVT_input'].values)
    dmd = np.diff(known['MD'].values)
    mask = dmd > 0
    if mask.sum() < 10:
        return -1.0, 0.0, 0.1
    vz = dz[mask] / dmd[mask]
    vt = dtvt[mask] / dmd[mask]
    A = np.column_stack([vz, np.ones_like(vz)])
    coef, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
    residuals = vt - (coef[0] * vz + coef[1])
    sigma = max(np.std(residuals), 0.001)
    return coef[0], coef[1], sigma


def _source_run_pf_z_velocity(hw, tw_tvt, tw_gr, n_particles=PF_N_PARTICLES):
    from scipy.interpolate import interp1d
    tw_func_point = interp1d(tw_tvt, tw_gr, bounds_error=False,
                             fill_value=(tw_gr[0], tw_gr[-1]))
    tw_smooth_gr = pd.Series(tw_gr).rolling(
        PF_GR_ROLLING_WINDOW, center=True, min_periods=1).mean().values
    tw_func_smooth = interp1d(tw_tvt, tw_smooth_gr, bounds_error=False,
                              fill_value=(tw_smooth_gr[0], tw_smooth_gr[-1]))
    tvt_min, tvt_max = tw_tvt.min(), tw_tvt.max()
    gr_sigma = _source_pf_calibrate_gr_sigma(hw, tw_tvt, tw_gr)
    beta, intercept, z_sigma = _source_pf_learn_z_beta(hw)
    known = hw[hw['TVT_input'].notna()]
    evalz = hw[hw['TVT_input'].isna()]
    if len(evalz) == 0:
        return np.array([]), np.array([])
    hw_gr_smooth = hw['GR'].rolling(
        PF_GR_ROLLING_WINDOW, center=True, min_periods=1).mean()
    last_tvt = known['TVT_input'].iloc[-1]
    positions = last_tvt + np.random.normal(0, PF_INIT_SPREAD_STD, n_particles)
    init_v = _source_pf_estimate_init_velocity(hw)
    velocities = init_v + np.random.normal(0, PF_INIT_VELOCITY_STD, n_particles)
    weights = np.ones(n_particles) / n_particles
    eval_indices = evalz.index.tolist()
    md_vals = evalz['MD'].values
    gr_vals = evalz['GR'].values
    z_vals = evalz['Z'].values
    prev_md = known['MD'].iloc[-1]
    prev_z = known['Z'].iloc[-1]
    pred_tvts = np.empty(len(evalz))
    pred_stds = np.empty(len(evalz))
    for i, idx in enumerate(eval_indices):
        d_md = md_vals[i] - prev_md
        if d_md <= 0:
            d_md = 1.0
        dz_dmd = (z_vals[i] - prev_z) / d_md
        v_expected = beta * dz_dmd + intercept
        velocities = (PF_MOMENTUM_ALPHA * velocities
                      + np.random.normal(0, PF_VELOCITY_NOISE_STD, n_particles))
        positions = (positions + velocities * d_md
                     + np.random.normal(0, PF_POSITION_NOISE_STD, n_particles))
        positions = np.clip(positions, tvt_min - 50, tvt_max + 50)
        if not np.isnan(gr_vals[i]):
            gr_smooth = hw_gr_smooth.iloc[hw.index.get_loc(idx)]
            expected_point = tw_func_point(positions)
            diff_point = gr_vals[i] - expected_point
            lik_point = np.exp(-0.5 * (diff_point / gr_sigma) ** 2)
            if not np.isnan(gr_smooth):
                expected_smooth = tw_func_smooth(positions)
                diff_smooth = gr_smooth - expected_smooth
                lik_smooth = np.exp(-0.5 * (diff_smooth / (gr_sigma * 1.5)) ** 2)
                likelihood = ((1 - PF_GR_ROLLING_WEIGHT) * lik_point
                              + PF_GR_ROLLING_WEIGHT * lik_smooth)
            else:
                likelihood = lik_point
            likelihood = np.maximum(likelihood, 1e-300)
            weights = weights * likelihood
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights[:] = 1.0 / n_particles
        z_sig = max(z_sigma * PF_Z_SIGMA_SCALE, PF_Z_SIGMA_FLOOR)
        diff_v = velocities - v_expected
        lik_z = np.exp(-0.5 * (diff_v / z_sig) ** 2)
        lik_z = np.maximum(lik_z, 1e-300)
        weights = weights * lik_z
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights[:] = 1.0 / n_particles
        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            cum = np.cumsum(weights)
            pos_resample = (np.arange(n_particles) + np.random.uniform()) / n_particles
            indices = np.searchsorted(cum, pos_resample)
            positions = positions[indices]
            velocities = velocities[indices]
            weights[:] = 1.0 / n_particles
            positions += np.random.normal(0, PF_ROUGHENING_STD_POS, n_particles)
            velocities += np.random.normal(0, PF_ROUGHENING_STD_VEL, n_particles)
        pred_tvts[i] = np.average(positions, weights=weights)
        pred_stds[i] = np.sqrt(
            np.average((positions - pred_tvts[i]) ** 2, weights=weights))
        prev_md = md_vals[i]
        prev_z = z_vals[i]
    return pred_tvts, pred_stds


def _source_ancc_estimate_init_rate(hw):
    known = hw[hw['TVT_input'].notna()]
    if len(known) < 10:
        return 0.0
    tail = known.tail(30)
    dtvt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dmd = np.diff(tail['MD'].values)
    dancc = dtvt + dz
    mask = dmd > 0
    if mask.sum() < 3:
        return 0.0
    return np.median(dancc[mask] / dmd[mask])


def _source_run_pf_ancc(hw, tw_tvt, tw_gr, n_particles=ANCC_N_PARTICLES):
    tvt_min, tvt_max = tw_tvt.min(), tw_tvt.max()
    gr_sigma = _source_pf_calibrate_gr_sigma(hw, tw_tvt, tw_gr)
    known = hw[hw['TVT_input'].notna()]
    evalz = hw[hw['TVT_input'].isna()]
    if len(evalz) == 0:
        return np.array([]), np.array([])
    last_state = known['TVT_input'].iloc[-1] + known['Z'].iloc[-1]
    init_rate = _source_ancc_estimate_init_rate(hw)
    pos = last_state + np.random.normal(0, ANCC_INIT_SPREAD_STD, n_particles)
    rate = init_rate + np.random.normal(0, ANCC_INIT_RATE_STD, n_particles)
    w = np.ones(n_particles) / n_particles
    md_vals = evalz['MD'].values
    z_vals = evalz['Z'].values
    gr_vals = evalz['GR'].values
    prev_md = known['MD'].iloc[-1]
    pred_tvts = np.empty(len(evalz))
    pred_stds = np.empty(len(evalz))
    for i in range(len(evalz)):
        d_md = md_vals[i] - prev_md
        if d_md <= 0:
            d_md = 1.0
        rate = ANCC_ALPHA * rate + np.random.normal(0, ANCC_RATE_NOISE_STD, n_particles)
        pos = pos + rate * d_md + np.random.normal(0, ANCC_POS_NOISE_STD, n_particles)
        tvt_est = pos - z_vals[i]
        tvt_clipped = np.clip(tvt_est, tvt_min - 50, tvt_max + 50)
        pos = tvt_clipped + z_vals[i]
        if not np.isnan(gr_vals[i]):
            expected_gr = np.interp(tvt_clipped, tw_tvt, tw_gr)
            diff = gr_vals[i] - expected_gr
            lik = np.exp(-0.5 * (diff / gr_sigma) ** 2)
            lik = np.maximum(lik, 1e-300)
            w *= lik
            w_sum = w.sum()
            if w_sum > 0:
                w /= w_sum
            else:
                w[:] = 1.0 / n_particles
        n_eff = 1.0 / np.sum(w ** 2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            cum = np.cumsum(w)
            u = (np.arange(n_particles) + np.random.uniform()) / n_particles
            idx = np.searchsorted(cum, u)
            pos = pos[idx]
            rate = rate[idx]
            w[:] = 1.0 / n_particles
            pos += np.random.normal(0, ANCC_ROUGHENING_STD_POS, n_particles)
            rate += np.random.normal(0, ANCC_ROUGHENING_STD_RATE, n_particles)
        tvt_weighted = np.average(pos - z_vals[i], weights=w)
        pred_tvts[i] = tvt_weighted
        pred_stds[i] = np.sqrt(
            np.average((pos - z_vals[i] - tvt_weighted) ** 2, weights=w))
        prev_md = md_vals[i]
    return pred_tvts, pred_stds


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def _pick_wells(train_dir: Path, n: int = 30) -> list[str]:
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    wids = [p.name.split("__", 1)[0] for p in paths]
    return wids[:n]


def _load_pandas(train_dir: Path, wid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the same well as the source notebook would."""
    hw = pd.read_csv(train_dir / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(train_dir / f"{wid}__typewell.csv")
    return hw, tw


def parity_check(train_dir: Path, wid: str) -> None:
    """Run port vs. source on the same well, identical seed; require == ."""
    hw_pd, tw_pd = _load_pandas(train_dir, wid)
    tw_tvt = tw_pd["TVT"].to_numpy(dtype=np.float64)
    tw_gr = tw_pd["GR"].to_numpy(dtype=np.float64)

    # Source path
    np.random.seed(RANDOM_STATE)
    src_z_pred, src_z_std = _source_run_pf_z_velocity(hw_pd, tw_tvt, tw_gr)
    np.random.seed(RANDOM_STATE)
    src_a_pred, src_a_std = _source_run_pf_ancc(hw_pd, tw_tvt, tw_gr)

    # Ported path: feed via polars (the production caller will use polars)
    hw_pl = pl.from_pandas(hw_pd)
    np.random.seed(RANDOM_STATE)
    port_z_pred, port_z_std = run_pf_z_velocity(hw_pl, tw_tvt, tw_gr)
    np.random.seed(RANDOM_STATE)
    port_a_pred, port_a_std = run_pf_ancc(hw_pl, tw_tvt, tw_gr)

    print(f"\n[{wid}] eval rows: {len(src_z_pred)}")
    for name, src, port in [
        ("z_pred", src_z_pred, port_z_pred),
        ("z_std", src_z_std, port_z_std),
        ("ancc_pred", src_a_pred, port_a_pred),
        ("ancc_std", src_a_std, port_a_std),
    ]:
        if len(src) == 0:
            print(f"  {name}: empty (skip)")
            continue
        d = np.abs(src - port)
        print(
            f"  {name}: max|delta|={d.max():.3e} "
            f"mean|delta|={d.mean():.3e} "
            f"src_range=[{src.min():.2f}, {src.max():.2f}]"
        )
        assert np.allclose(src, port, rtol=0, atol=1e-10), (
            f"{wid}/{name} parity violated; max|delta|={d.max()}"
        )


def driver_parity_check(
    well_dfs: dict[str, pl.DataFrame],
    typewell_dfs: dict[str, pl.DataFrame],
) -> None:
    """run_pfs_for_wells driver vs. sequential single-well calls."""
    seq: dict[str, dict] = {}
    for wid, hw_pl in well_dfs.items():
        tw_pl = typewell_dfs.get(wid)
        if tw_pl is None or tw_pl.height < 2:
            continue
        tw_tvt = tw_pl["TVT"].to_numpy().astype(np.float64)
        tw_gr = tw_pl["GR"].to_numpy().astype(np.float64)
        np.random.seed(RANDOM_STATE)
        z_pred, z_std = run_pf_z_velocity(hw_pl, tw_tvt, tw_gr)
        np.random.seed(RANDOM_STATE)
        a_pred, a_std = run_pf_ancc(hw_pl, tw_tvt, tw_gr)
        seq[wid] = {
            "pf_z_pred": z_pred.astype(np.float32),
            "pf_z_std": z_std.astype(np.float32),
            "pf_ancc_pred": a_pred.astype(np.float32),
            "pf_ancc_std": a_std.astype(np.float32),
        }

    par = run_pfs_for_wells(well_dfs, typewell_dfs, n_workers=4)

    n_checked = 0
    for wid, ref in seq.items():
        got = par.get(wid)
        if got is None:
            raise AssertionError(f"driver missing well {wid}")
        for k in ("pf_z_pred", "pf_z_std", "pf_ancc_pred", "pf_ancc_std"):
            a, b = ref[k], got[k]
            if a.size == 0:
                continue
            d = np.abs(a - b).max() if a.size else 0.0
            assert np.allclose(a, b, rtol=0, atol=1e-6), (
                f"{wid}/{k} driver-parity violated; max|delta|={d}"
            )
        n_checked += 1
    print(f"\ndriver parity OK on {n_checked} wells")


def benchmark(
    well_dfs: dict[str, pl.DataFrame],
    typewell_dfs: dict[str, pl.DataFrame],
    n_workers_parallel: int = 8,
) -> None:
    n = len(well_dfs)
    print(f"\n--- benchmark on {n} wells ---")

    t0 = time.perf_counter()
    _ = run_pfs_for_wells(well_dfs, typewell_dfs, n_workers=1)
    t_seq = time.perf_counter() - t0
    print(f"sequential (n=1): {t_seq:.2f}s   ({t_seq/max(n,1)*1000:.1f} ms/well)")

    t0 = time.perf_counter()
    _ = run_pfs_for_wells(well_dfs, typewell_dfs, n_workers=n_workers_parallel)
    t_par = time.perf_counter() - t0
    speedup = t_seq / max(t_par, 1e-6)
    print(f"parallel (n={n_workers_parallel}): {t_par:.2f}s   speedup={speedup:.2f}x")

    # Extrapolation. We expect ~linear scaling vs n_wells; assume the same
    # speedup holds for the full 773-well set.
    eta_seq_773 = t_seq / max(n, 1) * 773
    eta_par_773 = t_par / max(n, 1) * 773
    print(
        f"extrapolated to 773 wells: "
        f"sequential={eta_seq_773:.0f}s ({eta_seq_773/60:.1f} min)  "
        f"parallel={eta_par_773:.0f}s ({eta_par_773/60:.1f} min)"
    )


def main() -> int:
    train_dir = ROOT / "data" / "competition" / "train"
    if not train_dir.exists():
        print(f"missing data dir: {train_dir}", file=sys.stderr)
        return 1

    wids = _pick_wells(train_dir, n=30)
    print(f"picked {len(wids)} wells from {train_dir}")

    # Parity vs. source notebook on a single well.
    parity_check(train_dir, wids[0])

    # Bulk-load 30 wells via polars and run driver parity + benchmark.
    hw_dfs, tw_dfs = load_wells_from_dir(train_dir, well_ids=wids)
    print(f"loaded {len(hw_dfs)} hw + {len(tw_dfs)} tw via polars")

    driver_parity_check(hw_dfs, tw_dfs)
    benchmark(hw_dfs, tw_dfs, n_workers_parallel=8)

    print("\nOK")
    return 0


if __name__ == "__main__":
    # Force fork on macOS for Pool(); spawn would re-import at top-level.
    try:
        mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass
    raise SystemExit(main())
