#!/usr/bin/env python3
"""One-week seasonal-form comparisons using fixed common forecast dates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import analysis as ar
import matched_training as mt
import models as ef

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
FORMS = ['annual_harmonic', 'no_calendar_terms', 'periodic_cubic_spline']
CONTRASTS = {
    'RSV-NET': [('AR', 'AR_lab'), ('AR', 'AR_ED'), ('AR_lab', 'AR_lab_ED')],
    'FluSurv-NET': [('AR', 'AR_wILI'), ('AR', 'AR_lab'), ('AR', 'AR_ED'),
                    ('AR_wILI', 'AR_wILI_lab')],
}


def seasonal_features(features, form):
    if form == 'annual_harmonic':
        return features, ['sin52', 'cos52']
    if form == 'no_calendar_terms':
        return features, []
    # Fixed periodic cardinal basis: six unique equally spaced phase nodes.
    # Boundary value repeats the first node; knots do not depend on outcomes.
    phase = ((features.epiweeks.to_numpy(int) % 100 - 1) / 52.) % 1.
    basis = CubicSpline(np.linspace(0., 1., 7),
                        np.vstack([np.eye(6), np.eye(6)[0]]),
                        bc_type='periodic')(phase)
    names = [f'seasonal_spline_{j}' for j in range(6)]
    features = features.copy()
    features[names] = basis
    assert np.isfinite(basis).all() and np.allclose(basis.sum(axis=1), 1.)
    return features, names


def main():
    inputs = ['analysis_table.tsv', 'predictions_long.tsv',
              'matched_training_predictions.tsv', 'comparisons.tsv',
              'matched_training_comparisons.tsv']
    hashes = {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in inputs}
    analysis = pd.read_csv(OUT/inputs[0], sep='\t')
    prior = pd.read_csv(OUT/inputs[1], sep='\t')
    controlled = pd.read_csv(OUT/inputs[2], sep='\t')
    records, summaries = [], []
    reproduction = 0.
    interval_reproduction = 0.
    ar.cached_fits()
    for network, contrasts in CONTRASTS.items():
        for exclusion in [4, 0]:
            raw = analysis[analysis.surveillance_network == network].sort_values('epiweeks').reset_index(drop=True)
            if exclusion:
                raw = raw.iloc[:-exclusion].copy()
            features0 = ef.build_features(raw)
            y = features0.rate_per_100k.to_numpy(float)
            blocks = features0.time_block.to_numpy(int)
            dates = pd.to_datetime(features0.week_ending_date)
            old = prior[(prior.surveillance_network == network) &
                        (prior.exclude_last_weeks == exclusion) & (prior.horizon_weeks == 1) &
                        (prior.method != 'seasonal_naive')]
            common = old.pivot(index=ar.KEYS, columns='method', values='y_pred').dropna().reset_index()[ar.KEYS]
            for form in FORMS:
                features, seasons = seasonal_features(features0, form)
                matrix_cache, tuning_cache = {}, {}

                def matrix(model, candidate, design):
                    key = (model, candidate if design == 'matched_training' else '', design)
                    if key not in matrix_cache:
                        cols = seasons + mt.columns(model, network)[2:]
                        X = features[cols].to_numpy(float).copy()
                        if design == 'matched_training':
                            ac = seasons + mt.columns(candidate, network)[2:]
                            X[~np.isfinite(features[ac].to_numpy(float)).all(axis=1)] = np.nan
                        matrix_cache[key] = X
                    return matrix_cache[key]

                def tune(X, i):
                    key = (id(X), i)
                    if key not in tuning_cache:
                        tuning_cache[key] = ef.tune_ridge_lambda(
                            X, y, time_block=blocks, horizon=1, tune_end_exclusive=i-1,
                            tune_min_train=ar.SETTINGS['tune_min_train'],
                            lam_grid=ar.SETTINGS['lam_grid'], tune_last_n=ar.SETTINGS['tune_last_n'],
                            base_lam=1., ridge_preprocess=ar.SETTINGS['ridge_preprocess'],
                            nonnegative_handling=ar.SETTINGS['nonnegative_handling'])
                    return tuning_cache[key]

                for comp, cand in contrasts:
                    origins = common.merge(old[old.method == cand], on=ar.KEYS, how='inner').sort_values('origin_idx')
                    old_comp = old[old.method == comp].set_index('origin_idx')
                    old_matched = controlled[(controlled.outcome == network) &
                        (controlled.exclude_last_weeks == exclusion) & (controlled.horizon_weeks == 1) &
                        (controlled.comparator_model == comp) & (controlled.candidate_model == cand)].set_index('origin_idx')
                    assert len(origins) == len(old_matched) and set(origins.origin_idx) == set(old_matched.index)
                    for design in ['full_history', 'matched_training']:
                        xc, xa = matrix(comp, cand, design), matrix(cand, cand, design)
                        pairs = []
                        for origin in origins.itertuples(index=False):
                            i = int(origin.origin_idx)
                            ic, tc = mt.train_rows(xc, y, blocks, i, 1)
                            ia, ta = mt.train_rows(xa, y, blocks, i, 1)
                            assert len(ic) >= 30 and len(ia) >= 30
                            assert (dates.iloc[tc] < dates.iloc[i]).all() and (dates.iloc[ta] < dates.iloc[i]).all()
                            lc, nc = tune(xc, i)
                            la, na = tune(xa, i)
                            if design == 'matched_training':
                                assert np.array_equal(ic, ia) and np.array_equal(tc, ta) and nc == na
                            kwargs = dict(time_block=blocks, train_end_exclusive=i-1, origin_i=i,
                                horizon=1, ridge_preprocess=ar.SETTINGS['ridge_preprocess'],
                                nonnegative_handling=ar.SETTINGS['nonnegative_handling'])
                            pc = ef._fit_one_ridge_X(xc, y, lam=float(lc), **kwargs)
                            pa = ef._fit_one_ridge_X(xa, y, lam=float(la), **kwargs)
                            assert pc is not None and pa is not None
                            yt = float(origin.y_true)
                            assert np.isclose(y[i+1], yt) and blocks[i] == blocks[i+1]
                            if form == 'annual_harmonic':
                                expected_c = (old_comp.loc[i].y_pred if design == 'full_history'
                                              else old_matched.loc[i].comparator_pred)
                                expected_a = origin.y_pred
                                error = max(abs(pc-expected_c), abs(pa-expected_a))
                                reproduction = max(reproduction, error)
                                assert error < 1e-10, (network, design, comp, cand, i, error)
                            detail = dict(outcome=network, exclude_last_weeks=exclusion, horizon_weeks=1,
                                seasonal_form=form, training_design=design, comparator_model=comp,
                                candidate_model=cand, origin_idx=i,
                                origin_week_ending_date=dates.iloc[i].strftime('%Y-%m-%d'),
                                target_week_ending_date=dates.iloc[i+1].strftime('%Y-%m-%d'), y_true=yt,
                                comparator_pred=float(pc), candidate_pred=float(pa),
                                comparator_training_n=len(ic), candidate_training_n=len(ia),
                                comparator_lambda=float(lc), candidate_lambda=float(la),
                                comparator_tune_n=int(nc), candidate_tune_n=int(na))
                            records.append(detail)
                            pairs.append((abs(yt-pc), abs(yt-pa), abs(yt-pc)-abs(yt-pa), detail['origin_week_ending_date']))
                        paired = pd.DataFrame(pairs, columns=['a', 'b', 'd', 'date'])
                        base = dict(outcome=network, exclude_last_weeks=exclusion, horizon_weeks=1,
                            seasonal_form=form, training_design=design, comparator_model=comp,
                            candidate_model=cand, matched_n=len(paired), comparator_loss=paired.a.mean(),
                            candidate_loss=paired.b.mean(), loss_difference=paired.d.mean(),
                            target_start=origins.target_week_ending_date.min(), target_end=origins.target_week_ending_date.max())
                        for length in [2, 4, 8, 13]:
                            seed = ar.stable_seed(20261011, [network, exclusion, 1, 'common_modern', comp, cand, length])
                            for segment in ([True, False] if length == 4 else [True]):
                                lo, hi = ar.interval(paired.d, paired.date, length, seed, segment_aware=segment)
                                infer = ('calendar_segment_moving_block_bootstrap' if segment
                                         else 'unsegmented_moving_block_bootstrap_sensitivity')
                                summaries.append({**base, 'block_len':length, 'ci_lower':lo, 'ci_upper':hi,
                                                  'inference_method':infer, 'bootstrap_reps':1000})
                                if form == 'annual_harmonic':
                                    previous = pd.read_csv(OUT/(inputs[3] if design == 'full_history' else inputs[4]), sep='\t')
                                    previous = previous[(previous.outcome == network) &
                                        (previous.exclude_last_weeks == exclusion) & (previous.horizon_weeks == 1) &
                                        (previous.comparator_model == comp) & (previous.candidate_model == cand) &
                                        (previous.block_len == length) & (previous.inference_method == infer)]
                                    if design == 'full_history':previous = previous[previous.analysis_set == 'common_modern']
                                    assert len(previous) == 1
                                    error = max(abs(lo-previous.iloc[0].ci_lower), abs(hi-previous.iloc[0].ci_upper))
                                    interval_reproduction = max(interval_reproduction, error)
                                    assert error < 1e-10
                        print(f'{network} exclusion={exclusion} {form} {design} {comp}->{cand}: n={len(paired)}', flush=True)
    pd.DataFrame(records).to_csv(OUT/'seasonal_sensitivity_predictions.tsv', sep='\t', index=False)
    pd.DataFrame(summaries).to_csv(OUT/'seasonal_sensitivity_comparisons.tsv', sep='\t', index=False)
    assert len(summaries) == 84*5
    assert all(hashlib.sha256((OUT/name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    print(f'Computed {len(records)} paired forecasts under the fixed seasonal alternatives.',flush=True)


if __name__ == '__main__':
    main()
