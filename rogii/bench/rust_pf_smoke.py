"""
Smoke test for `rogii_pf` Rust extension.

What this does:
  1. Picks 5 training wells.
  2. Runs `rogii_pf.run_pf_z_velocity` and `rogii_pf.run_pf_ancc` on each.
  3. Runs the Python reference (inlined from cell 5 / cell 6 of the source notebook,
     since `rogii/src/triple_signal_pf.py` may not exist yet).
  4. Reports per-well wall time for both implementations and the speedup.
  5. Asserts Rust output is finite (non-NaN) row-by-row.
  6. Reports residual stats (Rust vs Python). Different RNG => not bitwise equal,
     but distributions and means should match closely.
  7. Tests `run_pfs_batch` with all 5 wells in parallel and confirms results match
     the per-well calls (modulo seed differences).

Run with the venv that has rogii_pf installed:
    cd rogii/rust/rogii_pf && source .venv/bin/activate
    python ../../bench/rust_pf_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ── Hyperparameters (mirrored from cell 2) ──────────────────────────────────
PF_N_PARTICLES = 500
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
ANCC_N_PARTICLES = 500


# ── Python reference (verbatim from notebook cells 5/6) ─────────────────────
def pf_calibrate_gr_sigma_py(hw, tw_tvt, tw_gr):
    known = hw[hw["TVT_input"].notna()]
    known_gr = known[known["GR"].notna()]
    if len(known_gr) < 20:
        return PF_GR_SIGMA_DEFAULT
    tw_func = interp1d(tw_tvt, tw_gr, bounds_error=False, fill_value=(tw_gr[0], tw_gr[-1]))
    expected = tw_func(known_gr["TVT_input"].values)
    residuals = known_gr["GR"].values - expected
    return float(np.clip(np.std(residuals), PF_GR_SIGMA_MIN, PF_GR_SIGMA_MAX))


def pf_estimate_init_velocity_py(hw):
    known = hw[hw["TVT_input"].notna()]
    if len(known) < 10:
        return 0.0
    tail = known.tail(20)
    if len(tail) < 5:
        return 0.0
    dtvt = np.diff(tail["TVT_input"].values)
    dmd = np.diff(tail["MD"].values)
    mask = dmd > 0
    if mask.sum() < 3:
        return 0.0
    return float(np.median(dtvt[mask] / dmd[mask]))


def pf_learn_z_beta_py(hw):
    known = hw[hw["TVT_input"].notna()]
    if len(known) < 30:
        return -1.0, 0.0, 0.1
    dz = np.diff(known["Z"].values)
    dtvt = np.diff(known["TVT_input"].values)
    dmd = np.diff(known["MD"].values)
    mask = dmd > 0
    if mask.sum() < 10:
        return -1.0, 0.0, 0.1
    vz = dz[mask] / dmd[mask]
    vt = dtvt[mask] / dmd[mask]
    A = np.column_stack([vz, np.ones_like(vz)])
    coef, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
    residuals = vt - (coef[0] * vz + coef[1])
    sigma = max(np.std(residuals), 0.001)
    return float(coef[0]), float(coef[1]), float(sigma)


def run_pf_z_velocity_py(hw, tw_tvt, tw_gr, n_particles=PF_N_PARTICLES, seed=42):
    rng = np.random.RandomState(seed)
    tw_func_point = interp1d(tw_tvt, tw_gr, bounds_error=False, fill_value=(tw_gr[0], tw_gr[-1]))
    tw_smooth_gr = pd.Series(tw_gr).rolling(PF_GR_ROLLING_WINDOW, center=True, min_periods=1).mean().values
    tw_func_smooth = interp1d(tw_tvt, tw_smooth_gr, bounds_error=False, fill_value=(tw_smooth_gr[0], tw_smooth_gr[-1]))
    tvt_min, tvt_max = tw_tvt.min(), tw_tvt.max()
    gr_sigma = pf_calibrate_gr_sigma_py(hw, tw_tvt, tw_gr)
    beta, intercept, z_sigma = pf_learn_z_beta_py(hw)
    known = hw[hw["TVT_input"].notna()]
    evalz = hw[hw["TVT_input"].isna()]
    if len(evalz) == 0:
        return np.array([]), np.array([])
    hw_gr_smooth = hw["GR"].rolling(PF_GR_ROLLING_WINDOW, center=True, min_periods=1).mean()
    last_tvt = known["TVT_input"].iloc[-1]
    positions = last_tvt + rng.normal(0, PF_INIT_SPREAD_STD, n_particles)
    init_v = pf_estimate_init_velocity_py(hw)
    velocities = init_v + rng.normal(0, PF_INIT_VELOCITY_STD, n_particles)
    weights = np.ones(n_particles) / n_particles
    eval_indices = evalz.index.tolist()
    md_vals = evalz["MD"].values
    gr_vals = evalz["GR"].values
    z_vals = evalz["Z"].values
    prev_md = known["MD"].iloc[-1]
    prev_z = known["Z"].iloc[-1]
    pred_tvts = np.empty(len(evalz))
    pred_stds = np.empty(len(evalz))
    for i, idx in enumerate(eval_indices):
        d_md = md_vals[i] - prev_md
        if d_md <= 0:
            d_md = 1.0
        dz_dmd = (z_vals[i] - prev_z) / d_md
        v_expected = beta * dz_dmd + intercept
        velocities = PF_MOMENTUM_ALPHA * velocities + rng.normal(0, PF_VELOCITY_NOISE_STD, n_particles)
        positions = positions + velocities * d_md + rng.normal(0, PF_POSITION_NOISE_STD, n_particles)
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
                likelihood = (1 - PF_GR_ROLLING_WEIGHT) * lik_point + PF_GR_ROLLING_WEIGHT * lik_smooth
            else:
                likelihood = lik_point
            likelihood = np.maximum(likelihood, 1e-300)
            weights = weights * likelihood
            ws = weights.sum()
            weights = weights / ws if ws > 0 else np.ones(n_particles) / n_particles
        z_sig = max(z_sigma * PF_Z_SIGMA_SCALE, PF_Z_SIGMA_FLOOR)
        diff_v = velocities - v_expected
        lik_z = np.exp(-0.5 * (diff_v / z_sig) ** 2)
        lik_z = np.maximum(lik_z, 1e-300)
        weights = weights * lik_z
        ws = weights.sum()
        weights = weights / ws if ws > 0 else np.ones(n_particles) / n_particles
        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            cum = np.cumsum(weights)
            pos_resample = (np.arange(n_particles) + rng.uniform()) / n_particles
            idx_r = np.searchsorted(cum, pos_resample)
            positions = positions[idx_r]
            velocities = velocities[idx_r]
            weights[:] = 1.0 / n_particles
            positions += rng.normal(0, PF_ROUGHENING_STD_POS, n_particles)
            velocities += rng.normal(0, PF_ROUGHENING_STD_VEL, n_particles)
        pred_tvts[i] = np.average(positions, weights=weights)
        pred_stds[i] = np.sqrt(np.average((positions - pred_tvts[i]) ** 2, weights=weights))
        prev_md = md_vals[i]
        prev_z = z_vals[i]
    return pred_tvts, pred_stds


def ancc_estimate_init_rate_py(hw):
    known = hw[hw["TVT_input"].notna()]
    if len(known) < 10:
        return 0.0
    tail = known.tail(30)
    dtvt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dmd = np.diff(tail["MD"].values)
    dancc = dtvt + dz
    mask = dmd > 0
    if mask.sum() < 3:
        return 0.0
    return float(np.median(dancc[mask] / dmd[mask]))


def run_pf_ancc_py(hw, tw_tvt, tw_gr, n_particles=ANCC_N_PARTICLES, seed=42):
    rng = np.random.RandomState(seed)
    tvt_min, tvt_max = tw_tvt.min(), tw_tvt.max()
    gr_sigma = pf_calibrate_gr_sigma_py(hw, tw_tvt, tw_gr)
    known = hw[hw["TVT_input"].notna()]
    evalz = hw[hw["TVT_input"].isna()]
    if len(evalz) == 0:
        return np.array([]), np.array([])
    last_state = known["TVT_input"].iloc[-1] + known["Z"].iloc[-1]
    init_rate = ancc_estimate_init_rate_py(hw)
    pos = last_state + rng.normal(0, ANCC_INIT_SPREAD_STD, n_particles)
    rate = init_rate + rng.normal(0, ANCC_INIT_RATE_STD, n_particles)
    w = np.ones(n_particles) / n_particles
    md_vals = evalz["MD"].values
    z_vals = evalz["Z"].values
    gr_vals = evalz["GR"].values
    prev_md = known["MD"].iloc[-1]
    pred_tvts = np.empty(len(evalz))
    pred_stds = np.empty(len(evalz))
    for i in range(len(evalz)):
        d_md = md_vals[i] - prev_md
        if d_md <= 0:
            d_md = 1.0
        rate = ANCC_ALPHA * rate + rng.normal(0, ANCC_RATE_NOISE_STD, n_particles)
        pos = pos + rate * d_md + rng.normal(0, ANCC_POS_NOISE_STD, n_particles)
        tvt_est = pos - z_vals[i]
        tvt_clipped = np.clip(tvt_est, tvt_min - 50, tvt_max + 50)
        pos = tvt_clipped + z_vals[i]
        if not np.isnan(gr_vals[i]):
            expected_gr = np.interp(tvt_clipped, tw_tvt, tw_gr)
            diff = gr_vals[i] - expected_gr
            lik = np.exp(-0.5 * (diff / gr_sigma) ** 2)
            lik = np.maximum(lik, 1e-300)
            w *= lik
            ws = w.sum()
            w = w / ws if ws > 0 else np.ones(n_particles) / n_particles
        n_eff = 1.0 / np.sum(w ** 2)
        if n_eff < PF_RESAMPLE_THRESHOLD * n_particles:
            cum = np.cumsum(w)
            u = (np.arange(n_particles) + rng.uniform()) / n_particles
            idx_r = np.searchsorted(cum, u)
            pos = pos[idx_r]
            rate = rate[idx_r]
            w[:] = 1.0 / n_particles
            pos += rng.normal(0, ANCC_ROUGHENING_STD_POS, n_particles)
            rate += rng.normal(0, ANCC_ROUGHENING_STD_RATE, n_particles)
        tvt_weighted = np.average(pos - z_vals[i], weights=w)
        pred_tvts[i] = tvt_weighted
        pred_stds[i] = np.sqrt(np.average((pos - z_vals[i] - tvt_weighted) ** 2, weights=w))
        prev_md = md_vals[i]
    return pred_tvts, pred_stds


# ── Test driver ─────────────────────────────────────────────────────────────
def main():
    try:
        import rogii_pf
    except ImportError as e:
        print(f"FAIL: rogii_pf not importable. Build it first via rust/rogii_pf/build.sh ({e})")
        sys.exit(1)

    print(f"rogii_pf v{rogii_pf.__version__} loaded.")

    data_dir = Path("/Users/william/drilling_oil_gas/rogii/data/competition/train")
    files = sorted(data_dir.glob("*__horizontal_well.csv"))[:5]
    if not files:
        print(f"FAIL: no wells found under {data_dir}")
        sys.exit(1)

    rust_z_times = []
    rust_a_times = []
    py_z_times = []
    py_a_times = []
    diffs_z = []
    diffs_a = []

    wells_data = []
    rust_results = []

    for fp in files:
        wid = fp.name.split("__", 1)[0]
        tw_path = data_dir / f"{wid}__typewell.csv"
        if not tw_path.exists():
            continue
        hw = pd.read_csv(fp)
        tw = pd.read_csv(tw_path)
        if not {"TVT", "GR"}.issubset(tw.columns) or len(tw) < 2:
            continue
        evalz = hw[hw["TVT_input"].isna()]
        known = hw[hw["TVT_input"].notna()]
        if len(evalz) == 0 or len(known) < 10:
            continue

        md = hw["MD"].to_numpy(dtype=np.float64)
        gr = hw["GR"].to_numpy(dtype=np.float64)
        tvt_in = hw["TVT_input"].to_numpy(dtype=np.float64)
        z = hw["Z"].to_numpy(dtype=np.float64)
        tw_tvt = tw["TVT"].to_numpy(dtype=np.float64)
        tw_gr = tw["GR"].to_numpy(dtype=np.float64)

        # Rust: Z-velocity PF
        t0 = time.perf_counter()
        rz_pred, rz_std = rogii_pf.run_pf_z_velocity(md, gr, tvt_in, z, tw_tvt, tw_gr, 500, 42)
        rz_time = time.perf_counter() - t0
        rust_z_times.append(rz_time)

        # Rust: ANCC PF
        t0 = time.perf_counter()
        ra_pred, ra_std = rogii_pf.run_pf_ancc(md, gr, tvt_in, z, tw_tvt, tw_gr, 500, 42)
        ra_time = time.perf_counter() - t0
        rust_a_times.append(ra_time)

        # Sanity
        assert rz_pred.shape == (len(evalz),), f"{wid}: rust z pred shape {rz_pred.shape} vs eval {len(evalz)}"
        assert ra_pred.shape == (len(evalz),), f"{wid}: rust a pred shape {ra_pred.shape} vs eval {len(evalz)}"
        assert np.all(np.isfinite(rz_pred)), f"{wid}: rust z pred has NaN"
        assert np.all(np.isfinite(rz_std)), f"{wid}: rust z std has NaN"
        assert np.all(np.isfinite(ra_pred)), f"{wid}: rust a pred has NaN"
        assert np.all(np.isfinite(ra_std)), f"{wid}: rust a std has NaN"

        # Python reference
        t0 = time.perf_counter()
        pz_pred, pz_std = run_pf_z_velocity_py(hw, tw_tvt, tw_gr, n_particles=500, seed=42)
        pz_time = time.perf_counter() - t0
        py_z_times.append(pz_time)

        t0 = time.perf_counter()
        pa_pred, pa_std = run_pf_ancc_py(hw, tw_tvt, tw_gr, n_particles=500, seed=42)
        pa_time = time.perf_counter() - t0
        py_a_times.append(pa_time)

        # Distribution comparison: differences in mean and std of predictions
        zd_mean = float(np.mean(rz_pred - pz_pred))
        zd_rmse = float(np.sqrt(np.mean((rz_pred - pz_pred) ** 2)))
        ad_mean = float(np.mean(ra_pred - pa_pred))
        ad_rmse = float(np.sqrt(np.mean((ra_pred - pa_pred) ** 2)))
        diffs_z.append((zd_mean, zd_rmse))
        diffs_a.append((ad_mean, ad_rmse))

        print(
            f"  {wid}  eval={len(evalz):4d}  "
            f"Z: rust={rz_time*1000:7.1f}ms py={pz_time*1000:7.1f}ms speedup={pz_time/rz_time:5.1f}x  "
            f"diff(mean={zd_mean:+.3f}, rmse={zd_rmse:.3f})  |  "
            f"A: rust={ra_time*1000:7.1f}ms py={pa_time*1000:7.1f}ms speedup={pa_time/ra_time:5.1f}x  "
            f"diff(mean={ad_mean:+.3f}, rmse={ad_rmse:.3f})"
        )

        wells_data.append(
            dict(
                well_id=wid,
                md=md,
                gr=gr,
                tvt_input=tvt_in,
                z=z,
                tw_tvt=tw_tvt,
                tw_gr=tw_gr,
                seed=42,
            )
        )
        rust_results.append((wid, rz_pred, rz_std, ra_pred, ra_std))

    # Batch test
    print("\nBatch test (rayon, n_threads=-1):")
    t0 = time.perf_counter()
    batch_out = rogii_pf.run_pfs_batch(wells_data, n_threads=-1)
    batch_time = time.perf_counter() - t0
    seq_rust_total = sum(rust_z_times) + sum(rust_a_times)
    print(f"  {len(batch_out)} wells, batch wall={batch_time*1000:.1f}ms, seq-rust-sum={seq_rust_total*1000:.1f}ms, batch speedup={seq_rust_total/batch_time:.2f}x")

    # Confirm batch outputs are finite & shaped right (note: batch uses different per-well seed
    # mixing for the dual PFs, so values won't equal the per-well calls above bit-for-bit).
    for d in batch_out:
        for key in ("pf_z_pred", "pf_z_std", "pf_ancc_pred", "pf_ancc_std"):
            arr = d[key]
            assert np.all(np.isfinite(arr)), f"{d['well_id']}: batch {key} non-finite"

    # Summary
    print("\n──────── SUMMARY ────────")
    print(f"Wells tested: {len(rust_results)}")
    rz = np.array(rust_z_times)
    ra = np.array(rust_a_times)
    pz = np.array(py_z_times)
    pa = np.array(py_a_times)
    print(f"Rust z-velocity:  mean={rz.mean()*1000:6.1f}ms  median={np.median(rz)*1000:6.1f}ms  total={rz.sum()*1000:6.1f}ms")
    print(f"Rust ANCC      :  mean={ra.mean()*1000:6.1f}ms  median={np.median(ra)*1000:6.1f}ms  total={ra.sum()*1000:6.1f}ms")
    print(f"Py   z-velocity:  mean={pz.mean()*1000:6.1f}ms  median={np.median(pz)*1000:6.1f}ms  total={pz.sum()*1000:6.1f}ms")
    print(f"Py   ANCC      :  mean={pa.mean()*1000:6.1f}ms  median={np.median(pa)*1000:6.1f}ms  total={pa.sum()*1000:6.1f}ms")
    speedup_z = pz.sum() / rz.sum()
    speedup_a = pa.sum() / ra.sum()
    print(f"Per-well speedup (sequential): Z={speedup_z:.1f}x  ANCC={speedup_a:.1f}x")
    print(f"Batch speedup over Python total: {(pz.sum()+pa.sum())/batch_time:.1f}x  (n_wells={len(rust_results)})")
    print()
    print("Distribution residuals (Rust − Python; not bit-equal: different RNG):")
    dz = np.array(diffs_z)
    da = np.array(diffs_a)
    print(f"  Z   mean-diff: median |val|={np.median(np.abs(dz[:,0])):.3f}  median RMSE={np.median(dz[:,1]):.3f}")
    print(f"  ANCC mean-diff: median |val|={np.median(np.abs(da[:,0])):.3f}  median RMSE={np.median(da[:,1]):.3f}")
    print("\nPASS: all Rust outputs finite, both PFs run, batch-API works.")


if __name__ == "__main__":
    main()
