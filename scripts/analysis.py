"""Forecast accuracy and paired uncertainty from frozen weekly surveillance data."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
import models as ef
from paired_statistics import stable_seed,matched_pair
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"
KEYS = ['origin_epiweek', 'origin_week_ending_date', 'target_epiweek', 'target_week_ending_date']
BASE_MAP = {'ridge_univariate':'AR', 'ridge_with_ed':'AR_ED', 'seasonal_naive':'seasonal_naive'}
FLU_MAP = {**BASE_MAP, 'ridge_with_signals':'AR_wILI', 'ridge_with_signals_plus_flu_pos':'AR_wILI_lab',
           'ridge_with_signals_plus_ed':'AR_wILI_ED', 'ridge_with_signals_plus_flu_pos_plus_ed':'AR_all'}
RSV_MAP = {**BASE_MAP, 'ridge_with_signals':'AR_lab', 'ridge_with_signals_plus_ed':'AR_lab_ED'}
SETTINGS = dict(horizons=[1,2,3,4], season_len=52, min_train=156, eval_last_n=260,
                lam=1., tune_lam=True, tune_per_origin=True, lam_grid=[.01,.1,1.,10.,100.],
                tune_min_train=104, tune_last_n=104, tune_warmup_ed=52,
                ridge_preprocess='standardized_intercept', nonnegative_handling='truncate')


def export(df, name):
    df.to_csv(OUT/name, sep='\t', index=False)

def cached_fits():
    """Memoize identical inner fits; numerical algorithm and tuning remain unchanged."""
    original = ef._fit_one_ridge_X
    matrices, cache = {}, {}
    def wrapped(X,y,**kwargs):
        matrix_key = id(X)
        if matrix_key not in matrices:
            matrices[matrix_key] = (X, hashlib.sha256(X.tobytes()+y.tobytes()+kwargs['time_block'].tobytes()).hexdigest())
        fingerprint = matrices[matrix_key][1]
        key = (fingerprint,)+tuple((k,str(v)) for k,v in kwargs.items() if k!='time_block')
        if key not in cache:
            cache[key] = original(X,y,**kwargs)
        return cache[key]
    ef._fit_one_ridge_X = wrapped

def enforce_origin_safe_fallback():
    """When validation is infeasible, use prespecified 1, never an earlier global tune.

    The fallback penalty is fixed before evaluation. Earlier validation errors
    select the penalty only when sufficient complete observations are available.
    """
    original=ef.tune_ridge_lambda
    def wrapper(*args,**kwargs):
        kwargs['base_lam']=1.0
        return original(*args,**kwargs)
    ef.tune_ridge_lambda=wrapper

def run_model(df, network, exclusion, flags, mapping, horizons=None):
    settings = SETTINGS.copy()
    if horizons is not None:
        settings['horizons'] = horizons
    pred,hp = ef.evaluate_within_series(df, surveillance_network=network,
                    exclude_last_weeks=exclusion, **flags, **settings)
    pred['method'] = pred['method'].map(mapping)
    hp['method'] = hp['method'].map(mapping)
    pred = pred[pred.method.notna()].copy()
    hp = hp[hp.method.notna()].copy()
    for table in [pred,hp]:
        table['surveillance_network']=network
        table['age_group']='65+ yr'
        table['site']='Overall'
        table['exclude_last_weeks']=exclusion
    return pred,hp

def interval(diffs, dates, length, seed, reps=1000, segment_aware=True):
    """Resample moving blocks within each contiguous observed calendar segment.

    Segment-stratified resampling preserves each segment's observation count and
    never treats the last spring observation and first autumn observation as adjacent.
    CIs condition on the observed seasons; they do not estimate between-season variance.
    """
    diffs=np.asarray(diffs,float)
    if len(diffs)<20:
        return np.nan,np.nan
    rng=np.random.default_rng(seed)
    split=np.flatnonzero(np.diff(pd.to_datetime(dates).to_numpy()).astype('timedelta64[D]').astype(int)!=7)+1
    segments=np.split(np.arange(len(diffs)),split) if segment_aware else [np.arange(len(diffs))]
    total=np.zeros(reps)
    for seg in segments:
        n=len(seg); b=min(length,n); count=math.ceil(n/b)
        starts=rng.integers(0,n-b+1,size=(reps,count))
        ix=(starts[:,:,None]+np.arange(b)).reshape(reps,-1)[:,:n]
        total += diffs[seg[ix]].sum(axis=1)
    lo,hi=np.quantile(total/len(diffs),[.025,.975])
    return float(lo),float(hi)

def compare(pred):
    rows=[]; losses=[]; calendar=[]; calibration=[]
    for (network,exclusion,h), group in pred.groupby(['surveillance_network','exclude_last_weeks','horizon_weeks']):
        methods=sorted(set(group.method)-{'seasonal_naive'})
        wide=group[group.method.isin(methods)].pivot(index=KEYS,columns='method',values='y_pred')
        common=wide.dropna().reset_index()[KEYS]
        for setting in ['pairwise','common_modern']:
            sub=group if setting=='pairwise' else group.merge(common,on=KEYS,how='inner')
            pairs=[('AR',m) for m in methods if m!='AR']
            if network=='FluSurv-NET':
                pairs += [('AR_wILI','AR_wILI_lab'),('AR_wILI','AR_wILI_ED'),('AR_wILI_lab','AR_all'),('AR_lab','AR_all')]
            else:
                pairs += [('AR_lab','AR_lab_ED'),('AR_ED','AR_lab_ED')]
            for comp,cand in pairs:
                matched=matched_pair(sub,comparator=comp,candidate=cand)
                if matched.empty: continue
                dates=matched.origin_week_ending_date
                y=matched.y_true.to_numpy(); a=matched.comparator_pred.to_numpy(); b=matched.candidate_pred.to_numpy()
                diff=np.abs(y-a)-np.abs(y-b)
                base=dict(outcome=network,exclude_last_weeks=exclusion,horizon_weeks=h,analysis_set=setting,
                          comparator_model=comp,candidate_model=cand,matched_n=len(y),comparator_loss=np.abs(y-a).mean(),
                          candidate_loss=np.abs(y-b).mean(),loss_difference=diff.mean(),origin_start=dates.min(),origin_end=dates.max(),
                          target_start=matched.target_week_ending_date.min(),target_end=matched.target_week_ending_date.max())
                for block in [2,4,8,13]:
                    seed=stable_seed(20261011,[network,exclusion,h,setting,comp,cand,block])
                    lo,hi=interval(diff,dates,block,seed)
                    rows.append({**base,'block_len':block,'ci_lower':lo,'ci_upper':hi,
                                 'inference_method':'calendar_segment_moving_block_bootstrap','bootstrap_reps':1000})
                    if block==4:
                        u,v=interval(diff,dates,block,seed,segment_aware=False)
                        rows.append({**base,'block_len':block,'ci_lower':u,'ci_upper':v,
                                     'inference_method':'unsegmented_moving_block_bootstrap_sensitivity','bootstrap_reps':1000})
                for i in range(len(y)):
                    losses.append({**{k:base[k] for k in ['outcome','exclude_last_weeks','horizon_weeks','analysis_set','comparator_model','candidate_model']},
                                   'origin_week_ending_date':dates.iloc[i],'target_week_ending_date':matched.target_week_ending_date.iloc[i],
                                   'comparator_absolute_error':abs(y[i]-a[i]),'candidate_absolute_error':abs(y[i]-b[i]),'loss_difference':diff[i]})
            if setting=='common_modern':
                for method, s in sub[sub.method.isin(methods)].groupby('method'):
                    yy=s.y_true.to_numpy(); pp=s.y_pred.to_numpy()
                    var=float(np.var(pp)); slope=float(np.cov(pp,yy,ddof=0)[0,1]/var) if var>0 else np.nan
                    calibration.append(dict(outcome=network,exclude_last_weeks=exclusion,horizon_weeks=h,method=method,matched_n=len(s),
                       MAE=float(np.abs(yy-pp).mean()),RMSE=float(np.sqrt(np.square(yy-pp).mean())),mean_bias=float((pp-yy).mean()),
                       calibration_intercept=float(yy.mean()-slope*pp.mean()),calibration_slope=slope))
                if not common.empty:
                    dd=pd.to_datetime(common.origin_week_ending_date).sort_values()
                    calendar.append(dict(outcome=network,exclude_last_weeks=exclusion,horizon_weeks=h,matched_n=len(common),
                                         n_calendar_segments=int((dd.diff().dt.days.fillna(7)!=7).sum()+1),
                                         origin_start=dd.min().date(),origin_end=dd.max().date(),
                                         seasons=','.join(sorted(set(str(d.year if d.month>=7 else d.year-1)+'-'+str((d.year if d.month>=7 else d.year-1)+1) for d in dd)))))
    export(pd.DataFrame(rows),'comparisons.tsv'); export(pd.DataFrame(losses),'paired_losses.tsv')
    export(pd.DataFrame(calendar),'evaluation_calendar.tsv');export(pd.DataFrame(calibration),'matched_calibration.tsv')
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=True)
    cached_fits()
    enforce_origin_safe_fallback()
    corrected=pd.read_csv(ROOT/'data/weekly_analysis.tsv',sep='\t')
    sources=pd.read_csv(ROOT/'data/surveillance_sources.tsv',sep='\t')
    export(corrected,'analysis_table.tsv')
    export(sources,'source_weekly.tsv')
    predictions=[];hyper=[]
    for network in ['FluSurv-NET','RSV-NET']:
        d=corrected[corrected.surveillance_network==network]
        for exc in [0,4]:
            print(f'Fitting {network}, terminal exclusion={exc}',flush=True)
            flags=dict(include_wili=network=='FluSurv-NET',include_rsvpos=network=='RSV-NET',include_flupos=network=='FluSurv-NET',include_ed=True)
            p,hp=run_model(d,network,exc,flags,FLU_MAP if network=='FluSurv-NET' else RSV_MAP)
            predictions.append(p);hyper.append(hp)
            if network=='FluSurv-NET':
                flags=dict(include_wili=False,include_rsvpos=False,include_flupos=True,include_ed=False)
                p,hp=run_model(d,network,exc,flags,{'ridge_with_signals_plus_flu_pos':'AR_lab'})
                predictions.append(p);hyper.append(hp)
    pred=pd.concat(predictions,ignore_index=True)
    assert not pred.duplicated(['surveillance_network','exclude_last_weeks','horizon_weeks','method']+KEYS).any()
    assert pred.y_pred.ge(0).all() and np.isfinite(pred[['y_true','y_pred']]).all().all()
    delta=(pd.to_datetime(pred.target_week_ending_date)-pd.to_datetime(pred.origin_week_ending_date)).dt.days
    assert (delta==pred.horizon_weeks*7).all()
    export(pred,'predictions_long.tsv')
    export(pd.concat(hyper,ignore_index=True),'hyperparams.tsv')
    compare(pred)
    availability=[]
    for network,d in corrected.groupby('surveillance_network'):
        for signal,col in [('wILI','wili'),('Laboratory RSV','rsv_positivity'),('Laboratory Influenza','flu_positivity'),('ED RSV','ed_rsv_pct_65p'),('ED influenza','ed_influenza_pct_65p')]:
            source=sources[sources.signal==signal];start,end=source.week_ending_date.min(),source.week_ending_date.max()
            within=d.week_ending_date.between(start,end);observed=d[col].notna()
            availability.append(dict(outcome=network,signal=signal,raw_source_start=start,raw_source_end=end,raw_observed_n=len(source),
                outcome_observed_n=len(d),analytic_observed_n=int(observed.sum()),analytic_availability_pct=100*observed.mean(),
                overlap_missing_n=int((within&~observed).sum()),structural_unavailable_n=int((~within).sum()),
                analytic_first_observed=d.loc[observed,'week_ending_date'].min(),analytic_last_observed=d.loc[observed,'week_ending_date'].max()))
    export(pd.DataFrame(availability),'availability.tsv')

if __name__=='__main__':main()
