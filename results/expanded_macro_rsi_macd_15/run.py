from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time,importlib.util
import numpy as np,pandas as pd,torch
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('technical',OUT.parent/'rsi_macd_annual_14/run.py')
v=load('validation',OUT.parent/'multi_threshold_annual_14/run.py')
RULES={'gold_vs_ma20':[.03],'copper_vs_ma20':[.03,.08],'a500_vs_ma20':[.05,.08],'us10y_vs_ma20':[.05,.10,.20],'cn10y_vs_ma20':[.03],'brent_vs_ma20':[.04,.10,.15],'dollar_vs_ma20':[.01,.02]}
NEW=['cn10y_vs_ma20','brent_vs_ma20','dollar_vs_ma20'];TECH=m.TECH['Both']
def work(year):
 folder=OUT/str(year);folder.mkdir(exist_ok=True)
 if (folder/'done.json').exists():return year
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time()
 prior=OUT.parent/'multi_threshold_annual_14'/str(year)
 data=pd.read_csv(prior/'features_and_target.csv',parse_dates=['Date']).set_index('Date')
 raw=pd.read_pickle(ROOT/'src/pickle/Dataset.pkl')['X']
 tech=pd.read_csv(OUT.parent/'rsi_macd_annual_14/technical_features.csv',parse_dates=['Date','Indicator_source_date']).set_index('Date').loc[data.index]
 assert (tech.Indicator_source_date.to_numpy()<data.index.to_numpy()).all()
 x=data[m.BASE+m.RETURNS+m.EXTRA].join(tech[TECH]);y=data.Actual;m.y=y;v.y=y;v.x=x
 for new,col in zip(NEW,['cn_10y_rate','brent','Dollar Index']):x[new]=(raw[col]/raw[col].rolling(20,min_periods=20).mean()-1).reindex(x.index)
 assert np.isfinite(x.to_numpy()).all()
 te=x.index[x.index.year==year];signals=[]
 for col,thresholds in RULES.items():
  for threshold in thresholds:
   outside=x[col].abs().ge(threshold);prev=outside.shift(1);change=outside.ne(prev)&prev.notna()
   for day in te[change.loc[te].to_numpy()]:
    i=x.index.get_loc(day)
    if i+1<len(x) and x.index[i+1].year==year:
     signals.append(dict(Trigger_date=day,Effective_date=x.index[i+1],Variable=col,Threshold=threshold,Direction='outward' if outside.loc[day] else 'inward',Previous_deviation=x.iloc[i-1][col],Deviation=x.loc[day,col]))
 sig=pd.DataFrame(signals).sort_values(['Trigger_date','Variable','Threshold']);sig.to_csv(folder/'signals.csv',index=False)
 dates=[te.min()]+sorted(sig.Effective_date.unique());assert len(set(dates))==len(dates)
 oldup=pd.read_csv(prior/'updates.csv',parse_dates=['Effective_date','Trigger_date']);oldup=oldup[oldup.Method=='Features_and_triggers'].set_index('Effective_date')
 assert set(oldup.index)<=set(dates)
 sel=folder/'selections';sel.mkdir(exist_ok=True)
 oldpred=pd.read_csv(OUT.parent/'rsi_macd_annual_14/predictions.csv',parse_dates=['Date']).set_index('Date').loc[te]
 assert np.allclose(oldpred.Actual,y.loc[te])
 parts=[];audits=[]
 for j,date in enumerate(dates):
  date=pd.Timestamp(date);stop=pd.Timestamp(dates[j+1]) if j+1<len(dates) else pd.Timestamp(f'{year+1}-01-01')
  tr=x.index[x.index<date];applied=te[(te>=date)&(te<stop)];assert tr.max()<applied.min()
  trigger=pd.NaT if j==0 else sig.loc[sig.Effective_date==date,'Trigger_date'].iloc[0]
  if j:assert tr.max()==trigger
  if date in oldup.index:
   seed=int(oldup.loc[date,'Seed']);path=prior/f'{date:%Y%m%d}_extra.pt'
   if not path.exists() and year==2016:path=OUT.parent/'multi_threshold_2016_14'/path.name
   saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['seed']==seed and pd.Timestamp(saved['train_end'])==tr.max()
   base=m.restore(saved['stages'][0]);bp=m.pred_saved
   # Reproduce latest baseline from saved stages before fitting the expanded layer.
   meta=x.loc[applied,m.RETURNS+m.EXTRA].copy();meta.insert(0,'base_prediction',bp(base,x.loc[applied,m.BASE]));meta=meta.join(x.loc[applied,TECH])
   baseline=m.restore(torch.load(OUT.parent/f'rsi_macd_annual_14/{year}/{date:%Y%m%d}_Both.pt',map_location='cpu',weights_only=True))
   assert np.allclose(m.pred_saved(baseline,meta),oldpred.loc[applied,'Both'],atol=1e-5,rtol=0)
  else:
   seed=int(v.select_seed(date,sel)['Selected_seed']);base=m.fit(x[m.BASE],tr,seed,60);bp=m.predict
  columns=m.RETURNS+m.EXTRA+TECH+NEW
  trainmeta=x.loc[tr,columns].copy();trainmeta.insert(0,'base_prediction',bp(base,x.loc[tr,m.BASE]))
  testmeta=x.loc[applied,columns].copy();testmeta.insert(0,'base_prediction',bp(base,x.loc[applied,m.BASE]))
  stack=m.fit(trainmeta,tr,seed,60);prediction=m.predict(stack,testmeta)
  v.save_bundle((base,stack),folder/f'{date:%Y%m%d}.pt',seed,60,tr)
  parts.append(pd.DataFrame({'Date':applied,'Actual':y.loc[applied].to_numpy(),'Baseline':oldpred.loc[applied,'Both'].to_numpy(),'Expanded':prediction.to_numpy()}))
  audits.append(dict(Year=year,Effective_date=date,Trigger_date=trigger,Seed=seed,Train_end=tr.max(),Train_N=len(tr),N=len(applied),Stage2_features='|'.join(trainmeta.columns)))
  if (j+1)%20==0:print(year,f'{j+1}/{len(dates)} models',flush=True)
 p=pd.concat(parts,ignore_index=True);assert p.Date.tolist()==te.tolist() and p.Date.is_unique
 p.to_csv(folder/'predictions.csv',index=False);pd.DataFrame(audits).to_csv(folder/'updates.csv',index=False)
 summary=[]
 for method,updates in [('Baseline',len(oldup)-1),('Expanded',len(dates)-1)]:
  summary.append(dict(Year=year,Method=method,N=len(p),MAPE=float(100*((p[method]-p.Actual).abs()/p.Actual.abs()).mean()),Updates=updates))
 pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False)
 (folder/'done.json').write_text(json.dumps(dict(year=year,seconds=time.time()-start,baseline_verified=True)))
 print(year,'DONE',summary,flush=True);return year
if __name__=='__main__':
 cfg=dict(rules=RULES,new_stage2_features=NEW,baseline='RSI+MACD+original expanded triggers (Both)',half_life=60,technical_lag=1,epochs=100,hidden=[16,32],selection='reuse baseline seeds on shared dates; original 10-seed equal-MSE preceding-3-month validation for additional dates',dataset_sha256=hashlib.sha256((ROOT/'src/pickle/Dataset.pkl').read_bytes()).hexdigest())
 assert cfg['dataset_sha256']==json.loads((OUT.parent/'rsi_macd_annual_14/config.json').read_text())['dataset_sha256']
 (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
 with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(work,y) for y in range(2015,2027)]):f.result()
