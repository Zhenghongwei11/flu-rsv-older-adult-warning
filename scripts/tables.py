"""Export source definitions and numerical forecast-comparison tables."""
from pathlib import Path
import csv,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables'
RESULTS=ROOT/'results'
LABEL={'AR':'AR','AR_ED':'AR + ED','AR_wILI':'AR + wILI','AR_lab':'AR + laboratory','AR_wILI_lab':'AR + wILI + laboratory','AR_wILI_ED':'AR + wILI + ED','AR_all':'AR + wILI + laboratory + ED','AR_lab_ED':'AR + laboratory + ED'}
ORDER={'FluSurv-NET':['AR_ED','AR_wILI','AR_lab','AR_wILI_lab','AR_wILI_ED','AR_all'],'RSV-NET':['AR_ED','AR_lab','AR_lab_ED']}
CONDITIONAL={'FluSurv-NET':[('AR_wILI','AR_wILI_lab'),('AR_wILI','AR_wILI_ED'),('AR_wILI_lab','AR_all'),('AR_lab','AR_all')],'RSV-NET':[('AR_lab','AR_lab_ED'),('AR_ED','AR_lab_ED')]}
def number(v):return f'{float(v):.3f}'.replace('-','−')
def ci(r):return number(r.ci_lower)+' to '+number(r.ci_upper)
def improvement(r):return number(r.loss_difference)+' ('+ci(r)+')'
def save(name,rows):
 pd.DataFrame(rows).to_csv(OUT/(name+'.tsv'),sep='\t',index=False)
def select(d,net,exc,h,comp,cand):
 g=d[(d.outcome==net)&(d.exclude_last_weeks==exc)&(d.horizon_weeks==h)&(d.comparator_model==comp)&(d.candidate_model==cand)]
 assert len(g)==1,(net,exc,h,comp,cand)
 return g.iloc[0]
def export_interval_table(data,name,include_lead):
 rows=[]
 for keys,g in data.groupby(['outcome','horizon_weeks','comparator_model','candidate_model'],sort=True):
  net,h,comp,cand=keys
  seg=g[g.inference_method=='calendar_segment_moving_block_bootstrap'].set_index('block_len')
  u=g[g.inference_method=='unsegmented_moving_block_bootstrap_sensitivity'].iloc[0]
  row={'Outcome':net,'Contrast':LABEL[comp]+' → '+LABEL[cand]}
  if include_lead:row['Lead (weeks)']=h
  else:row['Improvement']=number(seg.loc[4].loss_difference)
  row.update({f'Block {b} CI':ci(seg.loc[b]) for b in [2,4,8,13]});row['Unsegmented block 4 CI']=ci(u);rows.append(row)
 save(name,rows)
def main():
 OUT.mkdir(exist_ok=True)
 definitions=json.loads((ROOT/'data/table_definitions.json').read_text())
 for name in ['Table1','TableS2','Publication_timing']:
  with (OUT/(name+'.tsv')).open('w',newline='') as f:csv.writer(f,delimiter='\t',lineterminator='\n').writerows(definitions[name])
 a=pd.read_csv(RESULTS/'availability.tsv',sep='\t');rows=[]
 for net,signals in [('FluSurv-NET',['wILI','Laboratory Influenza','ED influenza']),('RSV-NET',['wILI','Laboratory RSV','ED RSV'])]:
  for signal in signals:
   r=a[(a.outcome==net)&(a.signal==signal)].iloc[0]
   rows.append({'Outcome':net,'Indicator':signal,'Observed source span':r.raw_source_start+'–'+r.raw_source_end,'Observed outcome overlap':r.analytic_first_observed+'–'+r.analytic_last_observed,'Observed/full outcome weeks':f'{r.analytic_observed_n}/{r.outcome_observed_n} ({r.analytic_availability_pct:.1f}%)','Outside source span':int(r.structural_unavailable_n),'Missing/overlap weeks':f'{int(r.overlap_missing_n)}/{int(r.analytic_observed_n+r.overlap_missing_n)}'})
 save('TableS1',rows)
 d=pd.read_csv(RESULTS/'comparisons.tsv',sep='\t');d=d[d.analysis_set=='common_modern'];p=d[(d.block_len==4)&(d.inference_method=='calendar_segment_moving_block_bootstrap')]
 rows=[];main_rows=[]
 for net in ORDER:
  for h in [1,2,3,4]:
   for cand in ORDER[net]:
    r=select(p,net,4,h,'AR',cand)
    rows.append({'Outcome':net,'Lead (weeks)':h,'Candidate':LABEL[cand],'n':r.matched_n,'AR MAE':number(r.comparator_loss),'Candidate MAE':number(r.candidate_loss),'Improvement (95% CI)':improvement(r)})
    if h==1:main_rows.append({'Outcome':net,'Candidate model':LABEL[cand],'Forecast weeks':r.matched_n,'AR MAE':number(r.comparator_loss),'Candidate MAE':number(r.candidate_loss),'Improvement (95% CI)':improvement(r)})
 save('Table2',main_rows);save('TableS3',rows)
 rows=[]
 for net in CONDITIONAL:
  for h in [1,2,3,4]:
   for comp,cand in CONDITIONAL[net]:
    r=select(p,net,4,h,comp,cand);rows.append({'Outcome':net,'Lead (weeks)':h,'Reference':LABEL[comp],'Candidate':LABEL[cand],'n':r.matched_n,'Improvement (95% CI)':improvement(r)})
 save('TableS4',rows)
 # Principal indicator and conditional contrasts for block-length sensitivity.
 rows=[]
 sensitive={'FluSurv-NET':[('AR','AR_ED'),('AR','AR_wILI'),('AR','AR_lab'),('AR_wILI','AR_wILI_lab'),('AR_wILI_lab','AR_all')],'RSV-NET':[('AR','AR_ED'),('AR','AR_lab'),('AR','AR_lab_ED'),('AR_lab','AR_lab_ED')]}
 for net,pairs in sensitive.items():
  for comp,cand in pairs:
   g=d[(d.outcome==net)&(d.exclude_last_weeks==4)&(d.horizon_weeks==1)&(d.comparator_model==comp)&(d.candidate_model==cand)]
   seg=g[g.inference_method=='calendar_segment_moving_block_bootstrap'].set_index('block_len');u=g[g.inference_method=='unsegmented_moving_block_bootstrap_sensitivity'].iloc[0]
   rows.append({'Outcome':net,'Contrast':LABEL[comp]+' → '+LABEL[cand],'Improvement':number(seg.loc[4].loss_difference),**{f'Block {b} CI':ci(seg.loc[b]) for b in [2,4,8,13]},'Unsegmented block 4 CI':ci(u)})
 save('TableS5',rows)
 rows=[]
 for r in p[(p.exclude_last_weeks==0)&(p.horizon_weeks==1)].itertuples(index=False):rows.append({'Outcome':r.outcome,'Reference':LABEL[r.comparator_model],'Candidate':LABEL[r.candidate_model],'n':r.matched_n,'Improvement (95% CI)':improvement(r)})
 save('TableS6',rows)
 cal=pd.read_csv(RESULTS/'matched_calibration.tsv',sep='\t');rows=[]
 for r in cal[(cal.exclude_last_weeks==4)&(cal.horizon_weeks==1)].itertuples(index=False):rows.append({'Outcome':r.outcome,'Model':LABEL[r.method],'n':r.matched_n,'RMSE':number(r.RMSE),'Mean bias':number(r.mean_bias),'Calibration intercept':number(r.calibration_intercept),'Calibration slope':number(r.calibration_slope)})
 save('TableS7',rows)
 m=pd.read_csv(RESULTS/'matched_training_comparisons.tsv',sep='\t');mp=m[(m.block_len==4)&(m.inference_method=='calendar_segment_moving_block_bootstrap')]
 for name,selected in [('TableS8',mp[mp.exclude_last_weeks==4]),('TableS9',mp[(mp.exclude_last_weeks==0)&(mp.horizon_weeks==1)])]:
  rows=[]
  for r in selected.itertuples(index=False):rows.append({'Outcome':r.outcome,'Lead (weeks)':r.horizon_weeks,'Reference':LABEL[r.comparator_model],'Candidate':LABEL[r.candidate_model],'n':r.matched_n,'Reference MAE':number(r.comparator_loss),'Candidate MAE':number(r.candidate_loss),'Improvement (95% CI)':improvement(r)})
  save(name,rows)
 selected=m[(m.exclude_last_weeks==4)&((m.horizon_weeks==1)|((m.outcome=='FluSurv-NET')&(((m.horizon_weeks==2)&(((m.comparator_model=='AR')&(m.candidate_model=='AR_lab'))|((m.comparator_model=='AR_wILI')&(m.candidate_model=='AR_wILI_lab'))))|((m.horizon_weeks==3)&(m.comparator_model=='AR')&(m.candidate_model=='AR_ED')))))]
 export_interval_table(selected,'TableS10',True)
 d=pd.read_csv(RESULTS/'seasonal_sensitivity_comparisons.tsv',sep='\t');p=d[(d.block_len==4)&(d.inference_method=='calendar_segment_moving_block_bootstrap')]
 forms={'annual_harmonic':'Annual harmonic','no_calendar_terms':'No calendar terms','periodic_cubic_spline':'Periodic spline'};designs={'full_history':'Full history','matched_training':'Matched training'}
 for n,exc in [(11,4),(12,0)]:
  rows=[]
  for net in ORDER:
   for form in forms:
    for design in designs:
     g=p[(p.outcome==net)&(p.exclude_last_weeks==exc)&(p.seasonal_form==form)&(p.training_design==design)]
     for r in g.itertuples(index=False):rows.append({'Outcome':net,'Seasonal terms':forms[form],'Training history':designs[design],'Contrast':LABEL[r.comparator_model]+' → '+LABEL[r.candidate_model],'n':r.matched_n,'Improvement (95% CI)':improvement(r)})
  assert len(rows)==42;save('TableS'+str(n),rows)
 print('Exported fourteen numbered tables and source-publication information.')
if __name__=='__main__':main()
