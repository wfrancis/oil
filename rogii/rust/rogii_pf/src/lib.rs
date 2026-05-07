//! Rogii particle filters — Rust port of cells 5 (TVT z-velocity PF) and 6 (ANCC PF)
//! from `triple-signal-beam-search-dual-pf-lightgbm.ipynb`.
//!
//! The PF inner loop is sequential per well (each step depends on the previous);
//! parallelism is *across* wells via Rayon. Each well gets its own seeded RNG so
//! Rust→Rust runs are bitwise reproducible. Numerical agreement with the Python
//! reference is approximate (different RNG stream).
//!
//! Hyperparameters are mirrored exactly from cell 2 of the source notebook.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

// ── Hyperparameters: TVT z-velocity PF ───────────────────────────────────────
const PF_N_PARTICLES_DEFAULT: usize = 500;
const PF_MOMENTUM_ALPHA: f64 = 0.993;
const PF_Z_SIGMA_FLOOR: f64 = 0.005;
const PF_Z_SIGMA_SCALE: f64 = 2.0;
const PF_VELOCITY_NOISE_STD: f64 = 0.005;
const PF_POSITION_NOISE_STD: f64 = 0.01;
const PF_INIT_VELOCITY_STD: f64 = 0.02;
const PF_GR_SIGMA_MIN: f64 = 10.0;
const PF_GR_SIGMA_MAX: f64 = 60.0;
const PF_GR_SIGMA_DEFAULT: f64 = 30.0;
const PF_INIT_SPREAD_STD: f64 = 0.5;
const PF_RESAMPLE_THRESHOLD: f64 = 0.5;
const PF_ROUGHENING_STD_POS: f64 = 0.2;
const PF_ROUGHENING_STD_VEL: f64 = 0.003;
const PF_GR_ROLLING_WINDOW: usize = 5;
const PF_GR_ROLLING_WEIGHT: f64 = 0.3;

// ── Hyperparameters: ANCC PF ─────────────────────────────────────────────────
const ANCC_ALPHA: f64 = 0.998;
const ANCC_RATE_NOISE_STD: f64 = 0.002;
const ANCC_POS_NOISE_STD: f64 = 0.005;
const ANCC_INIT_RATE_STD: f64 = 0.01;
const ANCC_INIT_SPREAD_STD: f64 = 0.3;
const ANCC_ROUGHENING_STD_POS: f64 = 0.1;
const ANCC_ROUGHENING_STD_RATE: f64 = 0.001;
const ANCC_N_PARTICLES_DEFAULT: usize = 500;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Linear interpolation on (xp, fp). xp must be sorted ascending. Edge-clamped
/// (matches scipy `interp1d(..., bounds_error=False, fill_value=(fp[0], fp[-1]))`
/// and `np.interp` default).
#[inline]
fn lerp_clamped(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 || x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // binary search for insertion point
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let x0 = xp[lo];
    let x1 = xp[hi];
    if x1 == x0 {
        return fp[lo];
    }
    let t = (x - x0) / (x1 - x0);
    fp[lo] + t * (fp[hi] - fp[lo])
}

/// Centered rolling mean with min_periods=1 (matches pandas `rolling(window, center=True, min_periods=1).mean()`).
fn rolling_mean_center_min1(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 || window == 0 {
        return values.to_vec();
    }
    let half = window / 2;
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let mut s = 0.0;
        let mut c = 0usize;
        for j in lo..hi {
            let v = values[j];
            if v.is_finite() {
                s += v;
                c += 1;
            }
        }
        out[i] = if c > 0 { s / c as f64 } else { f64::NAN };
    }
    out
}

/// Linear-interpolate-then-fill missing values to match pandas
/// `Series.interpolate(limit_direction='both')`. Endpoints get nearest finite value.
/// (Unused right now — kept for future helpers that need pandas-compatible interpolation.)
#[allow(dead_code)]
fn pandas_interpolate_both(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = values.to_vec();
    if n == 0 {
        return out;
    }
    // Find finite anchors
    let mut anchors: Vec<usize> = Vec::with_capacity(n);
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() {
            anchors.push(i);
        }
    }
    if anchors.is_empty() {
        return out;
    }
    // Fill leading NaNs with first anchor
    let first = anchors[0];
    for i in 0..first {
        out[i] = values[first];
    }
    // Fill trailing NaNs with last anchor
    let last = *anchors.last().unwrap();
    for i in (last + 1)..n {
        out[i] = values[last];
    }
    // Linear interpolate between consecutive anchors
    for w in anchors.windows(2) {
        let a = w[0];
        let b = w[1];
        if b == a + 1 {
            continue;
        }
        let va = values[a];
        let vb = values[b];
        for i in (a + 1)..b {
            let t = (i - a) as f64 / (b - a) as f64;
            out[i] = va + t * (vb - va);
        }
    }
    out
}

/// Standard deviation (population, ddof=0) — matches `np.std`.
fn np_std_pop(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let mean = xs.iter().copied().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

/// Median of a slice (uses sort, returns NaN for empty).
fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

/// Linear least-squares for `y = a*x + b` (matches `np.polyfit(x,y,1)` ordering: returns (slope, intercept)).
/// (Unused right now — Python pf code goes through `lstsq_2f` directly.)
#[allow(dead_code)]
fn polyfit1(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let nx = n as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let denom = nx * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return (0.0, sy / nx);
    }
    let slope = (nx * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / nx;
    (slope, intercept)
}

/// Two-feature least-squares: solves `y ≈ c0*v0 + c1*v1` (here v1 is all-ones, so c1 is intercept).
/// Returns (c0, c1, sigma) where sigma is residual std.
fn lstsq_2f(v0: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = v0.len();
    if n < 2 {
        return (-1.0, 0.0, 0.1);
    }
    // Normal equations for [v0, 1]^T [v0, 1]
    let nx = n as f64;
    let s00: f64 = v0.iter().map(|x| x * x).sum();
    let s01: f64 = v0.iter().sum();
    let s11: f64 = nx;
    let r0: f64 = v0.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let r1: f64 = y.iter().sum();
    // Inverse of 2x2 [[s00, s01],[s01, s11]]
    let det = s00 * s11 - s01 * s01;
    if det.abs() < 1e-30 {
        return (-1.0, 0.0, 0.1);
    }
    let c0 = (s11 * r0 - s01 * r1) / det;
    let c1 = (-s01 * r0 + s00 * r1) / det;
    // Residual std (population)
    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        residuals.push(y[i] - (c0 * v0[i] + c1));
    }
    let sigma = np_std_pop(&residuals).max(0.001);
    (c0, c1, sigma)
}

// ─── Calibration helpers (per-well one-shot) ─────────────────────────────────

/// `pf_calibrate_gr_sigma` from Python: residual std between known.GR and the
/// typewell GR(TVT_input). Clipped to [PF_GR_SIGMA_MIN, PF_GR_SIGMA_MAX]. If
/// fewer than 20 known points with finite GR, returns PF_GR_SIGMA_DEFAULT.
fn calibrate_gr_sigma(
    tvt_input: &[f64],
    gr: &[f64],
    tw_tvt: &[f64],
    tw_gr: &[f64],
) -> f64 {
    // count known with TVT_input AND GR finite
    let n = tvt_input.len();
    let mut residuals: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if tvt_input[i].is_finite() && gr[i].is_finite() {
            let expected = lerp_clamped(tvt_input[i], tw_tvt, tw_gr);
            residuals.push(gr[i] - expected);
        }
    }
    if residuals.len() < 20 {
        return PF_GR_SIGMA_DEFAULT;
    }
    np_std_pop(&residuals).clamp(PF_GR_SIGMA_MIN, PF_GR_SIGMA_MAX)
}

/// `pf_estimate_init_velocity`: median of (dTVT_input / dMD) over the last 20 known points.
fn estimate_init_velocity(md: &[f64], tvt_input: &[f64]) -> f64 {
    // Collect known sequence
    let mut known_md: Vec<f64> = Vec::new();
    let mut known_tvt: Vec<f64> = Vec::new();
    for i in 0..md.len() {
        if tvt_input[i].is_finite() {
            known_md.push(md[i]);
            known_tvt.push(tvt_input[i]);
        }
    }
    let n = known_md.len();
    if n < 10 {
        return 0.0;
    }
    // tail(20)
    let tail_start = n.saturating_sub(20);
    let tm = &known_md[tail_start..];
    let tt = &known_tvt[tail_start..];
    if tm.len() < 5 {
        return 0.0;
    }
    let mut ratios: Vec<f64> = Vec::new();
    for i in 1..tm.len() {
        let dmd = tm[i] - tm[i - 1];
        if dmd > 0.0 {
            ratios.push((tt[i] - tt[i - 1]) / dmd);
        }
    }
    if ratios.len() < 3 {
        return 0.0;
    }
    median(&mut ratios)
}

/// `pf_learn_z_beta`: lstsq for `vt = beta*vz + intercept`, returns (beta, intercept, sigma).
fn learn_z_beta(md: &[f64], tvt_input: &[f64], z: &[f64]) -> (f64, f64, f64) {
    let mut km: Vec<f64> = Vec::new();
    let mut kt: Vec<f64> = Vec::new();
    let mut kz: Vec<f64> = Vec::new();
    for i in 0..md.len() {
        if tvt_input[i].is_finite() {
            km.push(md[i]);
            kt.push(tvt_input[i]);
            kz.push(z[i]);
        }
    }
    let n = km.len();
    if n < 30 {
        return (-1.0, 0.0, 0.1);
    }
    let mut vz = Vec::with_capacity(n - 1);
    let mut vt = Vec::with_capacity(n - 1);
    for i in 1..n {
        let dmd = km[i] - km[i - 1];
        if dmd > 0.0 {
            vz.push((kz[i] - kz[i - 1]) / dmd);
            vt.push((kt[i] - kt[i - 1]) / dmd);
        }
    }
    if vz.len() < 10 {
        return (-1.0, 0.0, 0.1);
    }
    lstsq_2f(&vz, &vt)
}

/// ANCC variant of `estimate_init_velocity`: median of d(TVT+Z)/dMD over last 30 known points.
fn ancc_estimate_init_rate(md: &[f64], tvt_input: &[f64], z: &[f64]) -> f64 {
    let mut km: Vec<f64> = Vec::new();
    let mut kt: Vec<f64> = Vec::new();
    let mut kz: Vec<f64> = Vec::new();
    for i in 0..md.len() {
        if tvt_input[i].is_finite() {
            km.push(md[i]);
            kt.push(tvt_input[i]);
            kz.push(z[i]);
        }
    }
    let n = km.len();
    if n < 10 {
        return 0.0;
    }
    let tail_start = n.saturating_sub(30);
    let tm = &km[tail_start..];
    let tt = &kt[tail_start..];
    let tz = &kz[tail_start..];
    let mut ratios: Vec<f64> = Vec::new();
    for i in 1..tm.len() {
        let dmd = tm[i] - tm[i - 1];
        if dmd > 0.0 {
            let dancc = (tt[i] - tt[i - 1]) + (tz[i] - tz[i - 1]);
            ratios.push(dancc / dmd);
        }
    }
    if ratios.len() < 3 {
        return 0.0;
    }
    median(&mut ratios)
}

// ─── Resampling (systematic) ─────────────────────────────────────────────────

/// Systematic resampling indices: cum=cumsum(w); pos=(arange(N)+u)/N; idx=searchsorted(cum,pos).
fn systematic_resample(weights: &[f64], u: f64) -> Vec<usize> {
    let n = weights.len();
    let mut cum = vec![0.0f64; n];
    let mut s = 0.0;
    for i in 0..n {
        s += weights[i];
        cum[i] = s;
    }
    // searchsorted(cum, pos) — left
    let mut idx = vec![0usize; n];
    let mut j = 0usize;
    for i in 0..n {
        let pos = (i as f64 + u) / n as f64;
        while j < n && cum[j] < pos {
            j += 1;
        }
        idx[i] = j.min(n - 1);
    }
    idx
}

// ─── PF core: TVT z-velocity ─────────────────────────────────────────────────

/// Core implementation. `gr_eval[i]` and `gr_smooth_eval[i]` are GR and rolling-5-mean GR
/// at the i-th eval row (row index = `eval_idx[i]`). `md_eval`, `z_eval` likewise.
struct PfTvtIn<'a> {
    md_eval: &'a [f64],
    gr_eval: &'a [f64],
    z_eval: &'a [f64],
    gr_smooth_eval: &'a [f64],
    last_known_md: f64,
    last_known_tvt: f64,
    last_known_z: f64,
    tw_tvt: &'a [f64],
    tw_gr: &'a [f64],
    tw_gr_smooth: &'a [f64],
    tvt_min: f64,
    tvt_max: f64,
    gr_sigma: f64,
    beta: f64,
    intercept: f64,
    z_sigma: f64,
    init_velocity: f64,
}

fn pf_tvt_run(input: &PfTvtIn, n_particles: usize, rng: &mut Pcg64Mcg) -> (Vec<f64>, Vec<f64>) {
    let n_eval = input.md_eval.len();
    if n_eval == 0 {
        return (Vec::new(), Vec::new());
    }
    let n = n_particles;
    let nf = n as f64;

    let normal_pos_init = Normal::new(0.0, PF_INIT_SPREAD_STD).unwrap();
    let normal_vel_init = Normal::new(0.0, PF_INIT_VELOCITY_STD).unwrap();
    let normal_vel_noise = Normal::new(0.0, PF_VELOCITY_NOISE_STD).unwrap();
    let normal_pos_noise = Normal::new(0.0, PF_POSITION_NOISE_STD).unwrap();
    let normal_rough_pos = Normal::new(0.0, PF_ROUGHENING_STD_POS).unwrap();
    let normal_rough_vel = Normal::new(0.0, PF_ROUGHENING_STD_VEL).unwrap();
    let unif01 = Uniform::new(0.0f64, 1.0f64);

    let mut positions: Vec<f64> = (0..n)
        .map(|_| input.last_known_tvt + normal_pos_init.sample(rng))
        .collect();
    let mut velocities: Vec<f64> = (0..n)
        .map(|_| input.init_velocity + normal_vel_init.sample(rng))
        .collect();
    let mut weights = vec![1.0 / nf; n];

    let mut pred_tvts = vec![0.0f64; n_eval];
    let mut pred_stds = vec![0.0f64; n_eval];

    let mut prev_md = input.last_known_md;
    let mut prev_z = input.last_known_z;

    let z_sig = (input.z_sigma * PF_Z_SIGMA_SCALE).max(PF_Z_SIGMA_FLOOR);

    for i in 0..n_eval {
        let mut d_md = input.md_eval[i] - prev_md;
        if d_md <= 0.0 {
            d_md = 1.0;
        }
        let dz_dmd = (input.z_eval[i] - prev_z) / d_md;
        let v_expected = input.beta * dz_dmd + input.intercept;

        // velocities = α*velocities + N(0, vel_noise)
        for k in 0..n {
            velocities[k] = PF_MOMENTUM_ALPHA * velocities[k] + normal_vel_noise.sample(rng);
        }
        // positions = positions + velocities*d_md + N(0, pos_noise)
        for k in 0..n {
            positions[k] += velocities[k] * d_md + normal_pos_noise.sample(rng);
        }
        // clip positions
        let lo = input.tvt_min - 50.0;
        let hi = input.tvt_max + 50.0;
        for k in 0..n {
            if positions[k] < lo {
                positions[k] = lo;
            } else if positions[k] > hi {
                positions[k] = hi;
            }
        }

        // GR likelihood
        let gv = input.gr_eval[i];
        if gv.is_finite() {
            let gs = input.gr_smooth_eval[i];
            let inv_sigma = 1.0 / input.gr_sigma;
            let inv_sigma_smooth = 1.0 / (input.gr_sigma * 1.5);
            if gs.is_finite() {
                let blend_pt = 1.0 - PF_GR_ROLLING_WEIGHT;
                let blend_sm = PF_GR_ROLLING_WEIGHT;
                let mut wsum = 0.0;
                for k in 0..n {
                    let exp_pt = lerp_clamped(positions[k], input.tw_tvt, input.tw_gr);
                    let exp_sm = lerp_clamped(positions[k], input.tw_tvt, input.tw_gr_smooth);
                    let dpt = (gv - exp_pt) * inv_sigma;
                    let dsm = (gs - exp_sm) * inv_sigma_smooth;
                    let lik_pt = (-0.5 * dpt * dpt).exp();
                    let lik_sm = (-0.5 * dsm * dsm).exp();
                    let lik = blend_pt * lik_pt + blend_sm * lik_sm;
                    let lik = lik.max(1e-300);
                    weights[k] *= lik;
                    wsum += weights[k];
                }
                if wsum > 0.0 {
                    let inv = 1.0 / wsum;
                    for w in weights.iter_mut() {
                        *w *= inv;
                    }
                } else {
                    for w in weights.iter_mut() {
                        *w = 1.0 / nf;
                    }
                }
            } else {
                let mut wsum = 0.0;
                for k in 0..n {
                    let exp_pt = lerp_clamped(positions[k], input.tw_tvt, input.tw_gr);
                    let dpt = (gv - exp_pt) * inv_sigma;
                    let lik = (-0.5 * dpt * dpt).exp().max(1e-300);
                    weights[k] *= lik;
                    wsum += weights[k];
                }
                if wsum > 0.0 {
                    let inv = 1.0 / wsum;
                    for w in weights.iter_mut() {
                        *w *= inv;
                    }
                } else {
                    for w in weights.iter_mut() {
                        *w = 1.0 / nf;
                    }
                }
            }
        }

        // Z (velocity) likelihood — always applied
        {
            let inv_z = 1.0 / z_sig;
            let mut wsum = 0.0;
            for k in 0..n {
                let dv = (velocities[k] - v_expected) * inv_z;
                let lik = (-0.5 * dv * dv).exp().max(1e-300);
                weights[k] *= lik;
                wsum += weights[k];
            }
            if wsum > 0.0 {
                let inv = 1.0 / wsum;
                for w in weights.iter_mut() {
                    *w *= inv;
                }
            } else {
                for w in weights.iter_mut() {
                    *w = 1.0 / nf;
                }
            }
        }

        // ESS / resample
        let mut ess_denom = 0.0;
        for &w in weights.iter() {
            ess_denom += w * w;
        }
        let n_eff = if ess_denom > 0.0 { 1.0 / ess_denom } else { nf };
        if n_eff < PF_RESAMPLE_THRESHOLD * nf {
            let u: f64 = rng.sample(unif01);
            let idx = systematic_resample(&weights, u);
            let new_pos: Vec<f64> = idx.iter().map(|&j| positions[j]).collect();
            let new_vel: Vec<f64> = idx.iter().map(|&j| velocities[j]).collect();
            positions = new_pos;
            velocities = new_vel;
            for w in weights.iter_mut() {
                *w = 1.0 / nf;
            }
            for k in 0..n {
                positions[k] += normal_rough_pos.sample(rng);
                velocities[k] += normal_rough_vel.sample(rng);
            }
        }

        // Weighted mean & std
        let mut mean = 0.0;
        for k in 0..n {
            mean += positions[k] * weights[k];
        }
        let mut var = 0.0;
        for k in 0..n {
            let d = positions[k] - mean;
            var += d * d * weights[k];
        }
        pred_tvts[i] = mean;
        pred_stds[i] = var.max(0.0).sqrt();

        prev_md = input.md_eval[i];
        prev_z = input.z_eval[i];
    }

    (pred_tvts, pred_stds)
}

// ─── PF core: ANCC ───────────────────────────────────────────────────────────

struct PfAnccIn<'a> {
    md_eval: &'a [f64],
    gr_eval: &'a [f64],
    z_eval: &'a [f64],
    last_known_md: f64,
    last_known_tvt: f64,
    last_known_z: f64,
    tw_tvt: &'a [f64],
    tw_gr: &'a [f64],
    tvt_min: f64,
    tvt_max: f64,
    gr_sigma: f64,
    init_rate: f64,
}

fn pf_ancc_run(input: &PfAnccIn, n_particles: usize, rng: &mut Pcg64Mcg) -> (Vec<f64>, Vec<f64>) {
    let n_eval = input.md_eval.len();
    if n_eval == 0 {
        return (Vec::new(), Vec::new());
    }
    let n = n_particles;
    let nf = n as f64;

    let normal_pos_init = Normal::new(0.0, ANCC_INIT_SPREAD_STD).unwrap();
    let normal_rate_init = Normal::new(0.0, ANCC_INIT_RATE_STD).unwrap();
    let normal_rate_noise = Normal::new(0.0, ANCC_RATE_NOISE_STD).unwrap();
    let normal_pos_noise = Normal::new(0.0, ANCC_POS_NOISE_STD).unwrap();
    let normal_rough_pos = Normal::new(0.0, ANCC_ROUGHENING_STD_POS).unwrap();
    let normal_rough_rate = Normal::new(0.0, ANCC_ROUGHENING_STD_RATE).unwrap();
    let unif01 = Uniform::new(0.0f64, 1.0f64);

    let last_state = input.last_known_tvt + input.last_known_z;
    let mut pos: Vec<f64> = (0..n).map(|_| last_state + normal_pos_init.sample(rng)).collect();
    let mut rate: Vec<f64> = (0..n)
        .map(|_| input.init_rate + normal_rate_init.sample(rng))
        .collect();
    let mut w = vec![1.0 / nf; n];

    let mut pred_tvts = vec![0.0f64; n_eval];
    let mut pred_stds = vec![0.0f64; n_eval];

    let mut prev_md = input.last_known_md;

    for i in 0..n_eval {
        let mut d_md = input.md_eval[i] - prev_md;
        if d_md <= 0.0 {
            d_md = 1.0;
        }
        // rate update
        for k in 0..n {
            rate[k] = ANCC_ALPHA * rate[k] + normal_rate_noise.sample(rng);
        }
        // pos update
        for k in 0..n {
            pos[k] += rate[k] * d_md + normal_pos_noise.sample(rng);
        }
        // clip via tvt_est
        let z_i = input.z_eval[i];
        let lo = input.tvt_min - 50.0;
        let hi = input.tvt_max + 50.0;
        for k in 0..n {
            let mut tvt_est = pos[k] - z_i;
            if tvt_est < lo {
                tvt_est = lo;
            } else if tvt_est > hi {
                tvt_est = hi;
            }
            pos[k] = tvt_est + z_i;
        }
        // GR likelihood
        let gv = input.gr_eval[i];
        if gv.is_finite() {
            let inv_sigma = 1.0 / input.gr_sigma;
            let mut wsum = 0.0;
            for k in 0..n {
                let tvt_clipped = pos[k] - z_i;
                let exp_gr = lerp_clamped(tvt_clipped, input.tw_tvt, input.tw_gr);
                let d = (gv - exp_gr) * inv_sigma;
                let lik = (-0.5 * d * d).exp().max(1e-300);
                w[k] *= lik;
                wsum += w[k];
            }
            if wsum > 0.0 {
                let inv = 1.0 / wsum;
                for ww in w.iter_mut() {
                    *ww *= inv;
                }
            } else {
                for ww in w.iter_mut() {
                    *ww = 1.0 / nf;
                }
            }
        }
        // ESS / resample
        let mut ess_denom = 0.0;
        for &wk in w.iter() {
            ess_denom += wk * wk;
        }
        let n_eff = if ess_denom > 0.0 { 1.0 / ess_denom } else { nf };
        if n_eff < PF_RESAMPLE_THRESHOLD * nf {
            let u: f64 = rng.sample(unif01);
            let idx = systematic_resample(&w, u);
            let new_pos: Vec<f64> = idx.iter().map(|&j| pos[j]).collect();
            let new_rate: Vec<f64> = idx.iter().map(|&j| rate[j]).collect();
            pos = new_pos;
            rate = new_rate;
            for ww in w.iter_mut() {
                *ww = 1.0 / nf;
            }
            for k in 0..n {
                pos[k] += normal_rough_pos.sample(rng);
                rate[k] += normal_rough_rate.sample(rng);
            }
        }
        // Weighted mean & std of (pos - z)
        let mut mean = 0.0;
        for k in 0..n {
            mean += (pos[k] - z_i) * w[k];
        }
        let mut var = 0.0;
        for k in 0..n {
            let d = (pos[k] - z_i) - mean;
            var += d * d * w[k];
        }
        pred_tvts[i] = mean;
        pred_stds[i] = var.max(0.0).sqrt();

        prev_md = input.md_eval[i];
    }

    (pred_tvts, pred_stds)
}

// ─── Slice prep (shared by both PFs) ─────────────────────────────────────────

struct WellSlice {
    md_eval: Vec<f64>,
    gr_eval: Vec<f64>,
    z_eval: Vec<f64>,
    gr_smooth_eval: Vec<f64>,
    last_known_md: f64,
    last_known_tvt: f64,
    last_known_z: f64,
    tvt_min: f64,
    tvt_max: f64,
    gr_sigma: f64,
    beta: f64,
    intercept: f64,
    z_sigma: f64,
    init_velocity: f64,
    init_rate: f64,
    tw_gr_smooth: Vec<f64>,
}

fn prepare_well_slice(
    md: &[f64],
    gr: &[f64],
    tvt_input: &[f64],
    z: &[f64],
    tw_tvt: &[f64],
    tw_gr: &[f64],
) -> Option<WellSlice> {
    if md.len() != gr.len() || md.len() != tvt_input.len() || md.len() != z.len() {
        return None;
    }
    if tw_tvt.len() != tw_gr.len() || tw_tvt.len() < 2 {
        return None;
    }
    let n_rows = md.len();
    if n_rows == 0 {
        return None;
    }

    // Eval mask = TVT_input is NaN (or non-finite)
    let mut last_known_idx: Option<usize> = None;
    let mut eval_idx: Vec<usize> = Vec::new();
    for i in 0..n_rows {
        if tvt_input[i].is_finite() {
            last_known_idx = Some(i);
        } else {
            eval_idx.push(i);
        }
    }
    if eval_idx.is_empty() {
        return None;
    }
    let lki = last_known_idx?;

    // hw_gr_smooth = rolling mean of GR over all rows, window 5, center=True, min_periods=1
    let hw_gr_smooth = rolling_mean_center_min1(gr, PF_GR_ROLLING_WINDOW);
    // tw_gr_smooth analogous for typewell GR
    let tw_gr_smooth = rolling_mean_center_min1(tw_gr, PF_GR_ROLLING_WINDOW);

    let md_eval: Vec<f64> = eval_idx.iter().map(|&i| md[i]).collect();
    let gr_eval: Vec<f64> = eval_idx.iter().map(|&i| gr[i]).collect();
    let z_eval: Vec<f64> = eval_idx.iter().map(|&i| z[i]).collect();
    let gr_smooth_eval: Vec<f64> = eval_idx.iter().map(|&i| hw_gr_smooth[i]).collect();

    let tvt_min = tw_tvt.iter().copied().fold(f64::INFINITY, f64::min);
    let tvt_max = tw_tvt.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let gr_sigma = calibrate_gr_sigma(tvt_input, gr, tw_tvt, tw_gr);
    let init_velocity = estimate_init_velocity(md, tvt_input);
    let (beta, intercept, z_sigma) = learn_z_beta(md, tvt_input, z);
    let init_rate = ancc_estimate_init_rate(md, tvt_input, z);

    Some(WellSlice {
        md_eval,
        gr_eval,
        z_eval,
        gr_smooth_eval,
        last_known_md: md[lki],
        last_known_tvt: tvt_input[lki],
        last_known_z: z[lki],
        tvt_min,
        tvt_max,
        gr_sigma,
        beta,
        intercept,
        z_sigma,
        init_velocity,
        init_rate,
        tw_gr_smooth,
    })
}

// ─── Top-level: single-well PF entry points ──────────────────────────────────

fn extract_views(
    md: &PyReadonlyArray1<f64>,
    gr: &PyReadonlyArray1<f64>,
    tvt_input: &PyReadonlyArray1<f64>,
    z: &PyReadonlyArray1<f64>,
    tw_tvt: &PyReadonlyArray1<f64>,
    tw_gr: &PyReadonlyArray1<f64>,
) -> PyResult<(
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    Ok((
        md.as_slice()
            .map_err(|e| PyValueError::new_err(format!("md not contiguous: {e}")))?
            .to_vec(),
        gr.as_slice()
            .map_err(|e| PyValueError::new_err(format!("gr not contiguous: {e}")))?
            .to_vec(),
        tvt_input
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("tvt_input not contiguous: {e}")))?
            .to_vec(),
        z.as_slice()
            .map_err(|e| PyValueError::new_err(format!("z not contiguous: {e}")))?
            .to_vec(),
        tw_tvt
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("tw_tvt not contiguous: {e}")))?
            .to_vec(),
        tw_gr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("tw_gr not contiguous: {e}")))?
            .to_vec(),
    ))
}

#[pyfunction]
#[pyo3(signature = (md, gr, tvt_input, z, tw_tvt, tw_gr, n_particles=PF_N_PARTICLES_DEFAULT, seed=42))]
fn run_pf_z_velocity<'py>(
    py: Python<'py>,
    md: PyReadonlyArray1<f64>,
    gr: PyReadonlyArray1<f64>,
    tvt_input: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    tw_tvt: PyReadonlyArray1<f64>,
    tw_gr: PyReadonlyArray1<f64>,
    n_particles: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (md_v, gr_v, tvt_v, z_v, tw_tvt_v, tw_gr_v) =
        extract_views(&md, &gr, &tvt_input, &z, &tw_tvt, &tw_gr)?;
    let result = py.allow_threads(|| {
        let slice = prepare_well_slice(&md_v, &gr_v, &tvt_v, &z_v, &tw_tvt_v, &tw_gr_v);
        match slice {
            None => (Vec::<f64>::new(), Vec::<f64>::new()),
            Some(s) => {
                let mut rng = Pcg64Mcg::seed_from_u64(seed);
                let input = PfTvtIn {
                    md_eval: &s.md_eval,
                    gr_eval: &s.gr_eval,
                    z_eval: &s.z_eval,
                    gr_smooth_eval: &s.gr_smooth_eval,
                    last_known_md: s.last_known_md,
                    last_known_tvt: s.last_known_tvt,
                    last_known_z: s.last_known_z,
                    tw_tvt: &tw_tvt_v,
                    tw_gr: &tw_gr_v,
                    tw_gr_smooth: &s.tw_gr_smooth,
                    tvt_min: s.tvt_min,
                    tvt_max: s.tvt_max,
                    gr_sigma: s.gr_sigma,
                    beta: s.beta,
                    intercept: s.intercept,
                    z_sigma: s.z_sigma,
                    init_velocity: s.init_velocity,
                };
                pf_tvt_run(&input, n_particles, &mut rng)
            }
        }
    });
    let (pred, std) = result;
    Ok((
        Array1::from(pred).into_pyarray_bound(py),
        Array1::from(std).into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (md, gr, tvt_input, z, tw_tvt, tw_gr, n_particles=ANCC_N_PARTICLES_DEFAULT, seed=42))]
fn run_pf_ancc<'py>(
    py: Python<'py>,
    md: PyReadonlyArray1<f64>,
    gr: PyReadonlyArray1<f64>,
    tvt_input: PyReadonlyArray1<f64>,
    z: PyReadonlyArray1<f64>,
    tw_tvt: PyReadonlyArray1<f64>,
    tw_gr: PyReadonlyArray1<f64>,
    n_particles: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (md_v, gr_v, tvt_v, z_v, tw_tvt_v, tw_gr_v) =
        extract_views(&md, &gr, &tvt_input, &z, &tw_tvt, &tw_gr)?;
    let result = py.allow_threads(|| {
        let slice = prepare_well_slice(&md_v, &gr_v, &tvt_v, &z_v, &tw_tvt_v, &tw_gr_v);
        match slice {
            None => (Vec::<f64>::new(), Vec::<f64>::new()),
            Some(s) => {
                let mut rng = Pcg64Mcg::seed_from_u64(seed);
                let input = PfAnccIn {
                    md_eval: &s.md_eval,
                    gr_eval: &s.gr_eval,
                    z_eval: &s.z_eval,
                    last_known_md: s.last_known_md,
                    last_known_tvt: s.last_known_tvt,
                    last_known_z: s.last_known_z,
                    tw_tvt: &tw_tvt_v,
                    tw_gr: &tw_gr_v,
                    tvt_min: s.tvt_min,
                    tvt_max: s.tvt_max,
                    gr_sigma: s.gr_sigma,
                    init_rate: s.init_rate,
                };
                pf_ancc_run(&input, n_particles, &mut rng)
            }
        }
    });
    let (pred, std) = result;
    Ok((
        Array1::from(pred).into_pyarray_bound(py),
        Array1::from(std).into_pyarray_bound(py),
    ))
}

// ─── Top-level: batch over wells (Rayon-parallel) ────────────────────────────

#[derive(Clone)]
struct WellInput {
    well_id: String,
    md: Vec<f64>,
    gr: Vec<f64>,
    tvt_input: Vec<f64>,
    z: Vec<f64>,
    tw_tvt: Vec<f64>,
    tw_gr: Vec<f64>,
    n_particles_z: usize,
    n_particles_ancc: usize,
    seed: u64,
}

fn pyarray_to_vec(arr: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Accept numpy arrays of f64 or anything coercible via numpy
    let pa: PyReadonlyArray1<f64> = arr.extract()?;
    Ok(pa.as_slice()
        .map_err(|e| PyValueError::new_err(format!("array not contiguous: {e}")))?
        .to_vec())
}

fn dict_get_required<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key '{key}' in well dict")))
}

#[pyfunction]
#[pyo3(signature = (wells_data, n_threads=-1))]
fn run_pfs_batch<'py>(
    py: Python<'py>,
    wells_data: Bound<'py, PyList>,
    n_threads: i32,
) -> PyResult<Bound<'py, PyList>> {
    // Parse all wells into Rust-owned vectors (must drop GIL safely first).
    let mut inputs: Vec<WellInput> = Vec::with_capacity(wells_data.len());
    for item in wells_data.iter() {
        let d: Bound<PyDict> = item.downcast_into()?;
        let well_id: String = dict_get_required(&d, "well_id")?.extract()?;
        let md = pyarray_to_vec(&dict_get_required(&d, "md")?)?;
        let gr = pyarray_to_vec(&dict_get_required(&d, "gr")?)?;
        let tvt_input = pyarray_to_vec(&dict_get_required(&d, "tvt_input")?)?;
        let z = pyarray_to_vec(&dict_get_required(&d, "z")?)?;
        let tw_tvt = pyarray_to_vec(&dict_get_required(&d, "tw_tvt")?)?;
        let tw_gr = pyarray_to_vec(&dict_get_required(&d, "tw_gr")?)?;
        let n_particles_z: usize = match d.get_item("n_particles_z")? {
            Some(o) => o.extract()?,
            None => PF_N_PARTICLES_DEFAULT,
        };
        let n_particles_ancc: usize = match d.get_item("n_particles_ancc")? {
            Some(o) => o.extract()?,
            None => ANCC_N_PARTICLES_DEFAULT,
        };
        let seed: u64 = match d.get_item("seed")? {
            Some(o) => o.extract()?,
            None => 42,
        };
        inputs.push(WellInput {
            well_id,
            md,
            gr,
            tvt_input,
            z,
            tw_tvt,
            tw_gr,
            n_particles_z,
            n_particles_ancc,
            seed,
        });
    }

    // Build per-well thread pool config. n_threads = -1 means "use Rayon default".
    let pool_opt = if n_threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads as usize)
                .build()
                .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    let results: Vec<(String, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = py.allow_threads(|| {
        let work = || {
            inputs
                .par_iter()
                .map(|w| {
                    // Per-well RNG: combine global seed + a hash of well_id for distinctness.
                    let mut hasher: u64 = 0x9E3779B97F4A7C15u64.wrapping_mul(w.seed.wrapping_add(1));
                    for b in w.well_id.as_bytes() {
                        hasher ^= *b as u64;
                        hasher = hasher.wrapping_mul(0x100000001B3);
                    }
                    let z_seed = w.seed ^ hasher;
                    let ancc_seed = w.seed ^ hasher.rotate_left(17) ^ 0xA5A5A5A5A5A5A5A5;
                    let slice = prepare_well_slice(
                        &w.md, &w.gr, &w.tvt_input, &w.z, &w.tw_tvt, &w.tw_gr,
                    );
                    let (z_pred, z_std, ancc_pred, ancc_std) = match slice {
                        None => (Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()),
                        Some(s) => {
                            let mut rng_z = Pcg64Mcg::seed_from_u64(z_seed);
                            let z_in = PfTvtIn {
                                md_eval: &s.md_eval,
                                gr_eval: &s.gr_eval,
                                z_eval: &s.z_eval,
                                gr_smooth_eval: &s.gr_smooth_eval,
                                last_known_md: s.last_known_md,
                                last_known_tvt: s.last_known_tvt,
                                last_known_z: s.last_known_z,
                                tw_tvt: &w.tw_tvt,
                                tw_gr: &w.tw_gr,
                                tw_gr_smooth: &s.tw_gr_smooth,
                                tvt_min: s.tvt_min,
                                tvt_max: s.tvt_max,
                                gr_sigma: s.gr_sigma,
                                beta: s.beta,
                                intercept: s.intercept,
                                z_sigma: s.z_sigma,
                                init_velocity: s.init_velocity,
                            };
                            let (zp, zs) = pf_tvt_run(&z_in, w.n_particles_z, &mut rng_z);
                            let mut rng_a = Pcg64Mcg::seed_from_u64(ancc_seed);
                            let a_in = PfAnccIn {
                                md_eval: &s.md_eval,
                                gr_eval: &s.gr_eval,
                                z_eval: &s.z_eval,
                                last_known_md: s.last_known_md,
                                last_known_tvt: s.last_known_tvt,
                                last_known_z: s.last_known_z,
                                tw_tvt: &w.tw_tvt,
                                tw_gr: &w.tw_gr,
                                tvt_min: s.tvt_min,
                                tvt_max: s.tvt_max,
                                gr_sigma: s.gr_sigma,
                                init_rate: s.init_rate,
                            };
                            let (ap, as_) = pf_ancc_run(&a_in, w.n_particles_ancc, &mut rng_a);
                            (zp, zs, ap, as_)
                        }
                    };
                    (w.well_id.clone(), z_pred, z_std, ancc_pred, ancc_std)
                })
                .collect()
        };
        match &pool_opt {
            Some(pool) => pool.install(work),
            None => work(),
        }
    });

    // Build Python result list.
    let out = PyList::empty_bound(py);
    for (wid, zp, zs, ap, as_) in results {
        let d = PyDict::new_bound(py);
        d.set_item("well_id", wid)?;
        d.set_item("pf_z_pred", Array1::from(zp).into_pyarray_bound(py))?;
        d.set_item("pf_z_std", Array1::from(zs).into_pyarray_bound(py))?;
        d.set_item("pf_ancc_pred", Array1::from(ap).into_pyarray_bound(py))?;
        d.set_item("pf_ancc_std", Array1::from(as_).into_pyarray_bound(py))?;
        out.append(d)?;
    }
    Ok(out)
}

// ─── Module init ─────────────────────────────────────────────────────────────

#[pymodule]
fn rogii_pf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_pf_z_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(run_pf_ancc, m)?)?;
    m.add_function(wrap_pyfunction!(run_pfs_batch, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

