#!/usr/bin/env python3
"""Compare forecast models fitted to identical historical observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as ar
import models as ef

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
BASE = ['sin52', 'cos52', 'y_lag1', 'y_lag2']


def columns(model, network):
    cols = BASE.copy()
    if 'wILI' in model or model == 'AR_all':
        cols += ['wili_lag1', 'wili_lag2']
    if 'lab' in model or model == 'AR_all':
        cols += ['rsvpos_lag1', 'rsvpos_lag2'] if network == 'RSV-NET' else ['flupos_lag1', 'flupos_lag2']
    if 'ED' in model or model == 'AR_all':
        cols += ['edrsv_65p_lag1', 'edrsv_65p_lag2'] if network == 'RSV-NET' else ['edflu_65p_lag1', 'edflu_65p_lag2']
    return cols


def train_rows(X, y, blocks, origin, horizon):
    ix = np.arange(max(0, origin - horizon))
    targets = ix + horizon
    ok = (targets < origin) & (blocks[ix] == blocks[targets])
    ok &= np.isfinite(X[ix]).all(axis=1) & np.isfinite(y[targets])
    return ix[ok], targets[ok]


def main():
    settings = ar.SETTINGS
    pred = pd.read_csv(OUT / 'predictions_long.tsv', sep='\t')
    analysis = pd.read_csv(OUT / 'analysis_table.tsv', sep='\t')
    full = pd.read_csv(OUT / 'comparisons.tsv', sep='\t')
    contrasts = full[(full.analysis_set == 'common_modern') & (full.block_len == 4) &
                     (full.inference_method == 'calendar_segment_moving_block_bootstrap')]
    records, summaries, losses, tuning = [], [], [], []
    max_candidate_difference = 0.
    ar.cached_fits()
    for (network, exclusion), group in contrasts.groupby(['outcome', 'exclude_last_weeks']):
        raw = analysis[analysis.surveillance_network == network].sort_values('epiweeks').reset_index(drop=True)
        if exclusion:
            raw = raw.iloc[:-int(exclusion)].copy()
        features = ef.build_features(raw)
        y = features.rate_per_100k.to_numpy(float)
        blocks = features.time_block.to_numpy(int)
        dates = pd.to_datetime(features.week_ending_date)
        matrices, tune_cache = {}, {}

        def matrix(comp, cand):
            key = (comp, cand)
            if key not in matrices:
                cc, ac = columns(comp, network), columns(cand, network)
                assert set(cc) <= set(ac), (comp, cand)
                xx = features[cc].to_numpy(float).copy()
                valid = np.isfinite(features[ac].to_numpy(float)).all(axis=1)
                xx[~valid, :] = np.nan
                matrices[key] = xx
            return matrices[key]

        def tune(comp, cand, X, origin, h):
            # Share cached results only for exactly the same matrix object and cutoff.
            key = (id(X), h, origin)
            if key not in tune_cache:
                tune_cache[key] = ef.tune_ridge_lambda(
                    X, y, time_block=blocks, horizon=h, tune_end_exclusive=origin-h,
                    tune_min_train=settings['tune_min_train'], lam_grid=settings['lam_grid'],
                    tune_last_n=settings['tune_last_n'], base_lam=1.,
                    ridge_preprocess=settings['ridge_preprocess'],
                    nonnegative_handling=settings['nonnegative_handling'])
            lam, n = tune_cache[key]
            return float(lam), int(n)

        for contrast in group.itertuples(index=False):
            comp, cand, h = contrast.comparator_model, contrast.candidate_model, int(contrast.horizon_weeks)
            all_pred = pred[(pred.surveillance_network == network) &
                            (pred.exclude_last_weeks == exclusion) & (pred.horizon_weeks == h) &
                            (pred.method != 'seasonal_naive')]
            common = all_pred.pivot(index=ar.KEYS, columns='method', values='y_pred').dropna().reset_index()
            origins = common.merge(all_pred[all_pred.method == cand], on=ar.KEYS, how='inner').sort_values('origin_idx')
            assert len(origins) == int(contrast.matched_n)
            xc, xa = matrix(comp, cand), matrix(cand, cand)
            # Eligible inner prediction origins and fit rows are the same for both models.
            assert np.array_equal(np.isfinite(xc).all(axis=1), np.isfinite(xa).all(axis=1))
            pair = []
            for origin in origins.itertuples(index=False):
                i = int(origin.origin_idx)
                ix, target_ix = train_rows(xa, y, blocks, i, h)
                ix_comp, target_comp = train_rows(xc, y, blocks, i, h)
                assert np.array_equal(ix, ix_comp) and np.array_equal(target_ix, target_comp)
                assert len(ix) >= 30 and np.all(dates.iloc[target_ix] < dates.iloc[i])
                lc, nc = tune(comp, cand, xc, i, h)
                la, na = tune(cand, cand, xa, i, h)
                assert nc == na, 'Inner validation dates must have equal eligibility'
                pc = ef._fit_one_ridge_X(xc, y, time_block=blocks, train_end_exclusive=i-h,
                    origin_i=i, horizon=h, lam=lc, ridge_preprocess=settings['ridge_preprocess'],
                    nonnegative_handling=settings['nonnegative_handling'])
                pa = ef._fit_one_ridge_X(xa, y, time_block=blocks, train_end_exclusive=i-h,
                    origin_i=i, horizon=h, lam=la, ridge_preprocess=settings['ridge_preprocess'],
                    nonnegative_handling=settings['nonnegative_handling'])
                assert pc is not None and pa is not None
                err = abs(pa - float(origin.y_pred))
                max_candidate_difference = max(max_candidate_difference, err)
                assert err < 1e-10, (network, cand, h, i, err)
                yt = float(origin.y_true)
                assert np.isclose(y[i+h], yt)
                detail = dict(outcome=network, exclude_last_weeks=int(exclusion), horizon_weeks=h,
                    analysis_set='matched_training_common_modern', comparator_model=comp,
                    candidate_model=cand, origin_idx=i,
                    origin_week_ending_date=dates.iloc[i].strftime('%Y-%m-%d'),
                    target_week_ending_date=dates.iloc[i+h].strftime('%Y-%m-%d'),
                    y_true=yt, comparator_pred=float(pc), candidate_pred=float(pa), training_n=len(ix),
                    training_first_target=dates.iloc[target_ix[0]].strftime('%Y-%m-%d'),
                    training_last_target=dates.iloc[target_ix[-1]].strftime('%Y-%m-%d'),
                    comparator_lambda=lc, candidate_lambda=la, comparator_tune_n=nc, candidate_tune_n=na)
                records.append(detail)
                difference = abs(yt-pc)-abs(yt-pa)
                losses.append({**{k: detail[k] for k in ['outcome', 'exclude_last_weeks', 'horizon_weeks',
                    'analysis_set', 'comparator_model', 'candidate_model', 'origin_week_ending_date',
                    'target_week_ending_date']}, 'comparator_absolute_error':abs(yt-pc),
                    'candidate_absolute_error':abs(yt-pa), 'loss_difference':difference})
                pair.append((abs(yt-pc), abs(yt-pa), difference, detail['origin_week_ending_date']))
            paired = pd.DataFrame(pair, columns=['a', 'b', 'd', 'date'])
            base = dict(outcome=network, exclude_last_weeks=int(exclusion), horizon_weeks=h,
                analysis_set='matched_training_common_modern', comparator_model=comp,
                candidate_model=cand, matched_n=len(paired), comparator_loss=paired.a.mean(),
                candidate_loss=paired.b.mean(), loss_difference=paired.d.mean(),
                origin_start=paired.date.min(), origin_end=paired.date.max(),
                target_start=origins.target_week_ending_date.min(), target_end=origins.target_week_ending_date.max())
            for length in [2, 4, 8, 13]:
                seed = ar.stable_seed(20261011, [network, exclusion, h, 'common_modern', comp, cand, length])
                lo, hi = ar.interval(paired.d, paired.date, length, seed)
                summaries.append({**base, 'block_len':length, 'ci_lower':lo, 'ci_upper':hi,
                    'inference_method':'calendar_segment_moving_block_bootstrap', 'bootstrap_reps':1000})
                if length == 4:
                    lo, hi = ar.interval(paired.d, paired.date, length, seed, segment_aware=False)
                    summaries.append({**base, 'block_len':length, 'ci_lower':lo, 'ci_upper':hi,
                        'inference_method':'unsegmented_moving_block_bootstrap_sensitivity', 'bootstrap_reps':1000})
            print(f'{network} exclusion={exclusion} h={h} {comp} -> {cand}: n={len(paired)}', flush=True)

    outputs = {'matched_training_predictions.tsv':records, 'matched_training_comparisons.tsv':summaries,
               'matched_training_paired_losses.tsv':losses}
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(OUT/name, sep='\t', index=False)
    table = pd.DataFrame(records)
    assert np.isfinite(table[['y_true', 'comparator_pred', 'candidate_pred']]).all().all()
    assert table[['comparator_pred', 'candidate_pred']].ge(0).all().all()
    print(f'Computed {len(table)} paired forecasts with matched historical observations.',flush=True)


if __name__ == '__main__':
    main()
