#!/usr/bin/env python3
"""
v9 catastrophic-tail diagnosis.

For wells with TVT RMSE > 60 ft in MLP+PE-L8-multi OOF, identify failure modes
across 7 hypotheses. Emit numeric tables + per-well categorization.

Failure-mode codes:
  ISO  - spatial isolation (far from train neighbors in (X,Y))
  MOT  - high eval-region motion (eval TVT range > 30 ft)
  FLT  - eval-region fault/jump (max |dTVT/dMD| spike > threshold)
  ANC  - bad anchor (TVT_input variance over last 100 prefix rows > thr)
  REG  - operator regime change (eval TVT outside prefix TVT range)
  TWM  - typewell mismatch (low |Pearson| of horizontal vs typewell GR at matched TVT)
  COV  - ANCC coverage gap (any null ANCC in well, or > thr fraction)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

TRAIN_DIR = Path('/Users/william/drilling_oil_gas/rogii/data/competition/train')
RESULTS = Path('/Users/william/drilling_oil_gas/rogii/bench/neural_ancc_results.json')
OUT = Path('/Users/william/drilling_oil_gas/rogii/bench/v9_outlier_diagnostics.json')

CAT_TVT_THRESHOLD = 60.0
TOP_K = 20  # diagnose top-K worst (covers all 11 > 60 ft + margin)

# Hypothesis thresholds (calibrated from population stats below)
ISO_DIST_FT = 8000.0       # nearest-neighbor centroid distance > this = spatially isolated
MOT_TVT_RANGE = 30.0       # eval TVT range > 30 ft = high motion
FLT_DTVT_PER_MD = 0.5      # max |dTVT/dMD| step > this in eval = fault-like jump
ANC_VAR_LAST_100 = 5.0     # var of last 100 prefix TVT_input > this = bad anchor
REG_FRACTION_OOR = 0.10    # eval fraction with TVT outside [prefix_min, prefix_max] > 10% = regime drift
TWM_CORR = 0.30            # |Pearson(GR_horiz, GR_typewell_matched)| < 0.30 in prefix = mismatch
COV_NULL_FRAC = 0.05       # ANCC null fraction > 5% = coverage gap


def load_results() -> list[dict]:
    with open(RESULTS) as f:
        d = json.load(f)
    return d['variants']['mlp_pe_l8_multi']['per_well']


def well_summary(wid: str) -> dict | None:
    """Compact summary for a well: (X,Y) centroid, prefix/eval TVT stats, ANCC null frac.
    Used for nearest-neighbor distance and for the bad wells themselves."""
    p = TRAIN_DIR / f'{wid}__horizontal_well.csv'
    if not p.exists():
        return None
    df = pl.read_csv(p)
    n = df.height
    if n == 0:
        return None
    tvt = df['TVT'].to_numpy()
    tvti = df['TVT_input'].to_numpy()
    md = df['MD'].to_numpy()
    x = df['X'].to_numpy()
    y = df['Y'].to_numpy()
    ancc = df['ANCC'].to_numpy()
    gr = df['GR'].to_numpy()

    # Prefix: TVT_input not null (== TVT). Eval: TVT_input null.
    prefix_mask = ~np.isnan(tvti)
    eval_mask = np.isnan(tvti)
    n_prefix = int(prefix_mask.sum())
    n_eval = int(eval_mask.sum())

    summ: dict = {
        'wid': wid,
        'n': int(n),
        'n_prefix': n_prefix,
        'n_eval': n_eval,
        'x_mean': float(np.nanmean(x)),
        'y_mean': float(np.nanmean(y)),
        'ancc_null_frac': float(np.isnan(ancc).mean()),
        'gr_null_frac': float(np.isnan(gr).mean()),
    }

    # Prefix TVT stats
    if n_prefix > 0:
        tvt_pre = tvt[prefix_mask]
        summ['prefix_tvt_min'] = float(np.nanmin(tvt_pre))
        summ['prefix_tvt_max'] = float(np.nanmax(tvt_pre))
        summ['prefix_tvt_mean'] = float(np.nanmean(tvt_pre))
        summ['prefix_tvt_std'] = float(np.nanstd(tvt_pre))
        # Last 100 prefix TVT_input rows (assuming row-order = MD-ordered)
        # Use indices of prefix rows
        idx_prefix = np.where(prefix_mask)[0]
        last100 = tvt[idx_prefix[-min(100, len(idx_prefix)):]]
        summ['anchor_var_last100'] = float(np.nanvar(last100))
        summ['anchor_range_last100'] = float(np.nanmax(last100) - np.nanmin(last100))
        summ['anchor_last_value'] = float(tvt[idx_prefix[-1]])
    else:
        for k in ('prefix_tvt_min', 'prefix_tvt_max', 'prefix_tvt_mean',
                  'prefix_tvt_std', 'anchor_var_last100',
                  'anchor_range_last100', 'anchor_last_value'):
            summ[k] = float('nan')

    # Eval TVT stats
    if n_eval > 0:
        tvt_ev = tvt[eval_mask]
        md_ev = md[eval_mask]
        summ['eval_tvt_min'] = float(np.nanmin(tvt_ev))
        summ['eval_tvt_max'] = float(np.nanmax(tvt_ev))
        summ['eval_tvt_range'] = float(np.nanmax(tvt_ev) - np.nanmin(tvt_ev))
        summ['eval_tvt_mean'] = float(np.nanmean(tvt_ev))
        summ['eval_tvt_std'] = float(np.nanstd(tvt_ev))
        # Fault detection: dTVT/dMD step
        # Use sorted-by-MD ordering; assume already in MD order
        order = np.argsort(md_ev)
        tvt_sorted = tvt_ev[order]
        md_sorted = md_ev[order]
        d_tvt = np.diff(tvt_sorted)
        d_md = np.diff(md_sorted)
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = np.where(d_md > 0, np.abs(d_tvt) / d_md, 0.0)
        summ['eval_max_abs_dtvt_per_dmd'] = float(np.nanmax(slope)) if slope.size else 0.0
        summ['eval_max_abs_dtvt_step'] = float(np.nanmax(np.abs(d_tvt))) if d_tvt.size else 0.0
        # Regime drift: fraction of eval TVT outside prefix TVT range
        if n_prefix > 0:
            lo, hi = summ['prefix_tvt_min'], summ['prefix_tvt_max']
            oor = ((tvt_ev < lo) | (tvt_ev > hi))
            summ['eval_regime_oor_frac'] = float(np.nanmean(oor))
            # how far outside on the worst row
            below = lo - tvt_ev
            above = tvt_ev - hi
            worst_oor = max(float(np.nanmax(below)) if (below > 0).any() else 0.0,
                            float(np.nanmax(above)) if (above > 0).any() else 0.0)
            summ['eval_regime_oor_max_ft'] = worst_oor
        else:
            summ['eval_regime_oor_frac'] = float('nan')
            summ['eval_regime_oor_max_ft'] = float('nan')
    else:
        for k in ('eval_tvt_min', 'eval_tvt_max', 'eval_tvt_range', 'eval_tvt_mean',
                  'eval_tvt_std', 'eval_max_abs_dtvt_per_dmd', 'eval_max_abs_dtvt_step',
                  'eval_regime_oor_frac', 'eval_regime_oor_max_ft'):
            summ[k] = float('nan')

    return summ


def typewell_corr(wid: str, prefix_idx: np.ndarray, df_horiz: pl.DataFrame) -> tuple[float, int]:
    """Pearson correlation of horizontal-prefix GR vs typewell GR sampled at matched TVT.
    Returns (corr, n_matched)."""
    p_tw = TRAIN_DIR / f'{wid}__typewell.csv'
    if not p_tw.exists():
        return float('nan'), 0
    tw = pl.read_csv(p_tw)
    tw_tvt = tw['TVT'].to_numpy()
    tw_gr = tw['GR'].to_numpy()
    # Filter typewell to non-null GR
    m = ~np.isnan(tw_gr)
    tw_tvt = tw_tvt[m]
    tw_gr = tw_gr[m]
    if tw_tvt.size < 10:
        return float('nan'), 0
    # Sort
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    # Horizontal prefix
    h_tvt = df_horiz['TVT'].to_numpy()[prefix_idx]
    h_gr = df_horiz['GR'].to_numpy()[prefix_idx]
    m2 = ~np.isnan(h_gr) & ~np.isnan(h_tvt)
    h_tvt = h_tvt[m2]
    h_gr = h_gr[m2]
    if h_gr.size < 10:
        return float('nan'), 0
    # Interpolate typewell GR at horizontal prefix TVT
    # Limit to overlapping TVT range
    in_range = (h_tvt >= tw_tvt[0]) & (h_tvt <= tw_tvt[-1])
    if in_range.sum() < 10:
        return float('nan'), int(in_range.sum())
    gr_tw_at_h = np.interp(h_tvt[in_range], tw_tvt, tw_gr)
    gr_h = h_gr[in_range]
    if np.std(gr_tw_at_h) < 1e-6 or np.std(gr_h) < 1e-6:
        return 0.0, int(in_range.sum())
    corr = float(np.corrcoef(gr_h, gr_tw_at_h)[0, 1])
    return corr, int(in_range.sum())


def main() -> None:
    print('Loading per-well OOF results...')
    pw = load_results()
    by_wid = {r['well']: r for r in pw}
    sorted_pw = sorted(pw, key=lambda r: r['tvt_rmse'], reverse=True)

    print(f'Total wells: {len(pw)}, > {CAT_TVT_THRESHOLD} ft: '
          f'{sum(1 for r in pw if r["tvt_rmse"] > CAT_TVT_THRESHOLD)}')

    # 1) Build summary index for ALL wells (needed for nearest-neighbor distance)
    # Cache to JSON to avoid recompute on rerun
    cache_path = Path('/Users/william/drilling_oil_gas/rogii/bench/_well_summaries_cache.json')
    if cache_path.exists():
        print(f'Using cached summaries: {cache_path}')
        with open(cache_path) as f:
            all_summaries = json.load(f)
    else:
        print('Building well summaries for all 765 wells...')
        t0 = time.time()
        all_summaries = {}
        for i, r in enumerate(pw):
            wid = r['well']
            s = well_summary(wid)
            if s is None:
                continue
            all_summaries[wid] = s
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{len(pw)}  elapsed={time.time()-t0:.1f}s')
        with open(cache_path, 'w') as f:
            json.dump(all_summaries, f)
        print(f'Done in {time.time()-t0:.1f}s. Cached to {cache_path}')

    # 2) Population stats for context
    xs = np.array([s['x_mean'] for s in all_summaries.values() if not math.isnan(s['x_mean'])])
    ys = np.array([s['y_mean'] for s in all_summaries.values() if not math.isnan(s['y_mean'])])
    eval_ranges = np.array([s.get('eval_tvt_range', np.nan) for s in all_summaries.values()])
    print(f'Population: X range {xs.min():.0f}–{xs.max():.0f}, '
          f'Y range {ys.min():.0f}–{ys.max():.0f}')
    p = np.nanpercentile(eval_ranges, [50, 75, 90, 95, 99])
    print(f'Population eval_tvt_range pcts (50/75/90/95/99): '
          f'{p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f}/{p[3]:.1f}/{p[4]:.1f}')

    # 3) Diagnose top-K worst
    diag_rows = []
    targets = sorted_pw[:TOP_K]
    print(f'\nDiagnosing top-{TOP_K} worst wells...')
    for i, r in enumerate(targets):
        wid = r['well']
        s = all_summaries.get(wid)
        if s is None:
            continue

        # Spatial isolation: nearest train neighbor distance
        # Compute distance from this (X,Y) to ALL OTHER (X,Y)
        x0, y0 = s['x_mean'], s['y_mean']
        nn_dist = float('nan')
        nn_dists_top5 = []
        if not math.isnan(x0) and not math.isnan(y0):
            dists = []
            for o_wid, os in all_summaries.items():
                if o_wid == wid:
                    continue
                if math.isnan(os['x_mean']) or math.isnan(os['y_mean']):
                    continue
                dx = os['x_mean'] - x0
                dy = os['y_mean'] - y0
                dists.append(math.sqrt(dx*dx + dy*dy))
            dists.sort()
            nn_dist = dists[0] if dists else float('nan')
            nn_dists_top5 = dists[:5]

        # Typewell correlation requires loading horizontal CSV
        df = pl.read_csv(TRAIN_DIR / f'{wid}__horizontal_well.csv')
        tvti = df['TVT_input'].to_numpy()
        prefix_idx = np.where(~np.isnan(tvti))[0]
        tw_corr, tw_n = typewell_corr(wid, prefix_idx, df)

        # Build feature dict
        d = {
            'wid': wid,
            'rank': i + 1,
            'tvt_rmse': r['tvt_rmse'],
            'ancc_rmse': r['ancc_rmse'],
            'rows_eval': r['rows'],
            'b_prefix': r['b_prefix'],
            'n_prefix': s['n_prefix'],
            'n_eval': s['n_eval'],
            'x_mean': s['x_mean'],
            'y_mean': s['y_mean'],
            'nn_dist_ft': nn_dist,
            'nn_dist_top5_avg': float(np.mean(nn_dists_top5)) if nn_dists_top5 else float('nan'),
            'eval_tvt_range': s['eval_tvt_range'],
            'eval_max_abs_dtvt_step': s['eval_max_abs_dtvt_step'],
            'eval_max_abs_dtvt_per_dmd': s['eval_max_abs_dtvt_per_dmd'],
            'anchor_var_last100': s['anchor_var_last100'],
            'anchor_range_last100': s['anchor_range_last100'],
            'eval_regime_oor_frac': s['eval_regime_oor_frac'],
            'eval_regime_oor_max_ft': s['eval_regime_oor_max_ft'],
            'tw_gr_corr': tw_corr,
            'tw_match_n': tw_n,
            'ancc_null_frac': s['ancc_null_frac'],
            'gr_null_frac': s['gr_null_frac'],
        }

        # Categorize
        cats = []
        if not math.isnan(d['nn_dist_ft']) and d['nn_dist_ft'] > ISO_DIST_FT:
            cats.append('ISO')
        if not math.isnan(d['eval_tvt_range']) and d['eval_tvt_range'] > MOT_TVT_RANGE:
            cats.append('MOT')
        if not math.isnan(d['eval_max_abs_dtvt_per_dmd']) and \
                d['eval_max_abs_dtvt_per_dmd'] > FLT_DTVT_PER_MD:
            cats.append('FLT')
        if not math.isnan(d['anchor_var_last100']) and \
                d['anchor_var_last100'] > ANC_VAR_LAST_100:
            cats.append('ANC')
        if not math.isnan(d['eval_regime_oor_frac']) and \
                d['eval_regime_oor_frac'] > REG_FRACTION_OOR:
            cats.append('REG')
        if not math.isnan(d['tw_gr_corr']) and abs(d['tw_gr_corr']) < TWM_CORR:
            cats.append('TWM')
        if d['ancc_null_frac'] > COV_NULL_FRAC:
            cats.append('COV')

        d['categories'] = cats
        diag_rows.append(d)

    # 4) Aggregate
    cat_counts: dict[str, int] = {}
    for d in diag_rows:
        for c in d['categories']:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    # 5) Population baselines for the same metrics
    # to gauge how outlier-y the bad wells are vs typical
    pop_baselines = {}
    for k in ['eval_tvt_range', 'eval_max_abs_dtvt_per_dmd', 'eval_max_abs_dtvt_step',
              'anchor_var_last100', 'eval_regime_oor_frac', 'ancc_null_frac', 'gr_null_frac']:
        vals = np.array([s.get(k, np.nan) for s in all_summaries.values()])
        vals = vals[~np.isnan(vals)]
        if vals.size:
            pop_baselines[k] = {
                'p50': float(np.percentile(vals, 50)),
                'p90': float(np.percentile(vals, 90)),
                'p95': float(np.percentile(vals, 95)),
                'p99': float(np.percentile(vals, 99)),
                'max': float(vals.max()),
                'mean': float(vals.mean()),
            }

    # Compute population NN distance
    pop_nn = []
    keys = list(all_summaries.keys())
    coords = np.array([[all_summaries[k]['x_mean'], all_summaries[k]['y_mean']] for k in keys])
    for i in range(len(keys)):
        if math.isnan(coords[i, 0]):
            continue
        d2 = np.sum((coords - coords[i])**2, axis=1)
        d2[i] = np.inf
        pop_nn.append(math.sqrt(d2.min()))
    pop_nn = np.array(pop_nn)
    pop_baselines['nn_dist_ft'] = {
        'p50': float(np.percentile(pop_nn, 50)),
        'p90': float(np.percentile(pop_nn, 90)),
        'p95': float(np.percentile(pop_nn, 95)),
        'p99': float(np.percentile(pop_nn, 99)),
        'max': float(pop_nn.max()),
        'mean': float(pop_nn.mean()),
    }

    out = {
        'thresholds': {
            'CAT_TVT_THRESHOLD': CAT_TVT_THRESHOLD,
            'ISO_DIST_FT': ISO_DIST_FT,
            'MOT_TVT_RANGE': MOT_TVT_RANGE,
            'FLT_DTVT_PER_MD': FLT_DTVT_PER_MD,
            'ANC_VAR_LAST_100': ANC_VAR_LAST_100,
            'REG_FRACTION_OOR': REG_FRACTION_OOR,
            'TWM_CORR': TWM_CORR,
            'COV_NULL_FRAC': COV_NULL_FRAC,
        },
        'top_k': TOP_K,
        'n_above_60ft': sum(1 for r in pw if r['tvt_rmse'] > CAT_TVT_THRESHOLD),
        'population_baselines': pop_baselines,
        'category_counts': cat_counts,
        'per_well': diag_rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {OUT}')
    print(f'\nCategory counts (top-{TOP_K}):')
    for c, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        print(f'  {c}: {n}')


if __name__ == '__main__':
    main()
