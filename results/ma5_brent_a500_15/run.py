from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,json,time,hashlib
import numpy as np,pandas as pd,torch
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
OUT=Path(__file__).resolve().parent;PRIOR=OUT.parent/'expanded_macro_rsi_macd_15'
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('technical',OUT.parent/'rsi_macd_annual_14/run.py');v=load('validation',OUT.parent/'multi_threshold_annual_14/run.py')
NEW=['brent_vs_ma5','a500_vs_ma5'];COLS=m.RETURNS+m.EXTRA+m.TECH['Both']+['cn10y_vs_ma20','brent_vs_ma20','dollar_vs_ma20']
def work(year):
 folder=OUT/str(year);folder.mkdir(exist_ok=True)
 if (folder/'done.json').exists():return year
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time()
 data=pd.read_csv(OUT.parent/f'multi_threshold_annual_14/{year}/features_and_target.csv',parse_dates=['Date']).set_index('Date');y=data.Actual;m.y=y
 raw=pd.read_pickle(ROOT/'src/pickle/Dataset.pkl')['X']
 tech=pd.read_csv(OUT.parent/'rsi_macd_annual_14/technical_features.csv',parse_dates=['Date','Indicator_source_date']).set_index('Date').loc[data.index]
 assert (tech.Indicator_source_date.to_numpy()<data.index.to_numpy()).all()
 x=data[m.BASE+m.RETURNS+m.EXTRA].join(tech[m.TECH['Both']])
 for name,col,k in [('cn10y_vs_ma20','cn_10y_rate',20),('brent_vs_ma20','brent',20),('dollar_vs_ma20','Dollar Index',20),('brent_vs_ma5','brent',5),('a500_vs_ma5','A_500',5)]:x[name]=(raw[col]/raw[col].rolling(k,min_periods=k).mean()-1).reindex(x.index)
 assert np.isfinite(x.to_numpy()).all()
 te=x.index[x.index.year==year];old=pd.read_csv(PRIOR/f'{year}/predictions.csv',parse_dates=['Date']).set_index('Date');assert old.index.tolist()==te.tolist() and np.allclose(old.Actual,y.loc[te])
 updates=pd.read_csv(PRIOR/f'{year}/updates.csv',parse_dates=['Effective_date','Trigger_date']);parts=[];audit=[]
 for j,row in enumerate(updates.itertuples(index=False)):
  date=row.Effective_date;stop=updates.iloc[j+1].Effective_date if j+1<len(updates) else pd.Timestamp(f'{year+1}-01-01')
  tr=x.index[x.index<date];applied=te[(te>=date)&(te<stop)];assert tr.max()<applied.min()
  saved=torch.load(PRIOR/f'{year}/{date:%Y%m%d}.pt',map_location='cpu',weights_only=True);seed=int(row.Seed)
  assert seed==saved['seed'] and pd.Timestamp(saved['train_end'])==tr.max() and saved['half_life']==60
  base=m.restore(saved['stages'][0]);previous=m.restore(saved['stages'][1])
  trainmeta=x.loc[tr,COLS].copy();trainmeta.insert(0,'base_prediction',m.pred_saved(base,x.loc[tr,m.BASE]))
  testmeta=x.loc[applied,COLS].copy();testmeta.insert(0,'base_prediction',m.pred_saved(base,x.loc[applied,m.BASE]))
  assert np.allclose(m.pred_saved(previous,testmeta),old.loc[applied,'Expanded'],atol=1e-5,rtol=0)
  trainmeta=trainmeta.join(x.loc[tr,NEW]);testmeta=testmeta.join(x.loc[applied,NEW]);assert trainmeta.shape[1]==17
  stack=m.fit(trainmeta,tr,seed,60);prediction=m.predict(stack,testmeta)
  v.save_bundle((base,stack),folder/f'{date:%Y%m%d}.pt',seed,60,tr)
  parts.append(pd.DataFrame({'Date':applied,'Actual':y.loc[applied].to_numpy(),'Baseline':old.loc[applied,'Expanded'].to_numpy(),'Expanded':prediction.to_numpy()}))
  audit.append(dict(Year=year,Effective_date=date,Trigger_date=row.Trigger_date,Seed=seed,Train_end=tr.max(),N=len(applied),Stage2_features='|'.join(trainmeta.columns)))
 p=pd.concat(parts,ignore_index=True);assert p.Date.tolist()==te.tolist();p.to_csv(folder/'predictions.csv',index=False);pd.DataFrame(audit).to_csv(folder/'updates.csv',index=False)
 summary=[dict(Year=year,Method=method,N=len(p),MAPE=100*((p[method]-p.Actual).abs()/p.Actual.abs()).mean(),Updates=len(updates)-1) for method in ['Baseline','Expanded']]
 pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False);(folder/'done.json').write_text(json.dumps(dict(year=year,baseline_verified=True,seconds=time.time()-start)))
 print(year,'DONE',summary,flush=True)
if __name__=='__main__':
 cfg=json.loads((PRIOR/'config.json').read_text());assert cfg['dataset_sha256']==hashlib.sha256((ROOT/'src/pickle/Dataset.pkl').read_bytes()).hexdigest()
 cfg.update(baseline='expanded_macro_rsi_macd_15 Expanded (15 inputs)',new_stage2_features=NEW,stage2_features=['base_prediction']+COLS+NEW,selection='same update dates and seeds as baseline, frozen first stage; only second stage adds two MA5 deviation inputs')
 (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
 with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(work,y) for y in range(2015,2027)]):f.result()
