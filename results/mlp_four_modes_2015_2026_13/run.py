from pathlib import Path
import os,json,hashlib,random,argparse,time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
SEEDS=[965409,640934,550343,798291,374583,42716,785634,565766,385735,741131]
BASE=['A_500','Dollar Index','cn_10y_rate','us_10y_rate','brent','Copper Futures Price','Gold Futures Price']
RETURNS=['gold_daily','gold_vs_ma5','gold_vs_ma20','copper_daily','copper_vs_ma5','copper_vs_ma20']
EPOCHS=100

def locate(name):
    p=OUT.parent/name
    return p if p.exists() else ROOT/'results'/name

def initialize():
    global x,y
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=pd.read_csv(OUT/'features_and_target.csv',parse_dates=['Date']).set_index('Date')
    x=data[BASE+RETURNS];y=data.Actual

def fit(X,idx,seed,half):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    sx=StandardScaler().fit(X.loc[idx]);sy=StandardScaler().fit(y.loc[idx].to_numpy().reshape(-1,1))
    xt=torch.tensor(sx.transform(X.loc[idx]),dtype=torch.float32)
    yt=torch.tensor(sy.transform(y.loc[idx].to_numpy().reshape(-1,1)),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
    if half:
        w=2.0**(-np.arange(len(idx)-1,-1,-1)/half)
        assert np.isclose(w[-half-1]/w[-1],.5)
        wt=torch.tensor(w/w.mean(),dtype=torch.float32).reshape(-1,1)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    for epoch in range(EPOCHS):
        model.train();opt.zero_grad();err=(model(xt)-yt)**2
        loss=(wt*err).mean() if half else err.mean()
        assert torch.isfinite(loss)
        loss.backward();opt.step()
    model.eval()
    return model,sx,sy

def predict(bundle,X):
    model,sx,sy=bundle
    with torch.no_grad():z=model(torch.tensor(sx.transform(X),dtype=torch.float32)).numpy()
    p=sy.inverse_transform(z).ravel();assert np.isfinite(p).all()
    return pd.Series(p,index=X.index)

def train_stack(idx,seed,half):
    known=x.loc[idx]
    base=fit(known[BASE],idx,seed,half)
    meta=known[RETURNS].copy();meta.insert(0,'base_prediction',predict(base,known[BASE]))
    stack=fit(meta,idx,seed,half)
    return base,stack

def stack_predict(bundle,X):
    base,stack=bundle
    meta=X[RETURNS].copy();meta.insert(0,'base_prediction',predict(base,X[BASE]))
    return predict(stack,meta)

def save_bundle(bundle,path,seed,half,idx):
    packs=[]
    for model,sx,sy in bundle:
        packs.append(dict(state_dict=model.state_dict(),sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist()))
    torch.save(dict(stages=packs,seed=seed,half_life=half,train_start=str(idx.min()),train_end=str(idx.max()),epochs=100),path)

def select_seed(anchor,folder):
    key=anchor.strftime('%Y-%m-%d');path=folder/f'selection_{key}.json'
    tr=x.index[x.index<anchor-pd.DateOffset(months=3)]
    va=x.index[(x.index>=anchor-pd.DateOffset(months=3))&(x.index<anchor)]
    assert tr.max()<va.min() and va.max()<anchor
    if path.exists():return json.loads(path.read_text())
    candidates=[]
    for i,seed in enumerate(SEEDS):
        bundle=train_stack(tr,seed,None)
        pred=stack_predict(bundle,x.loc[va]);err=pred.to_numpy()-y.loc[va].to_numpy()
        candidates.append(dict(Seed=seed,Candidate_order=i,MAPE=float(100*np.mean(np.abs(err)/np.abs(y.loc[va].to_numpy()))),MSE=float(np.mean(err**2))))
    winner=min(candidates,key=lambda r:(r['MAPE'],r['Candidate_order']))
    decision=dict(Anchor=key,Selected_seed=winner['Seed'],Validation_MAPE=winner['MAPE'],Train_end=str(tr.max().date()),Validation_start=str(va.min().date()),Validation_end=str(va.max().date()),Candidates=candidates,Source='new equal-MSE validation selection')
    path.write_text(json.dumps(decision,indent=2));return decision

def year_run(year):
    initialize();folder=OUT/str(year);folder.mkdir(exist_ok=True)
    te=x.index[x.index.year==year]
    if (folder/'done.json').exists():return json.loads((folder/'done.json').read_text())
    start=pd.Timestamp(f'{year}-01-01');end=pd.Timestamp(f'{year+1}-01-01')
    outside=x[['gold_vs_ma20','copper_vs_ma20']].abs().ge(.03)
    prev=outside.shift(1);cross=outside.ne(prev)&prev.notna()
    events=[]
    for day in te:
        if cross.loc[day].any():
            pos=x.index.get_loc(day);effective=x.index[pos+1] if pos+1<len(x) else pd.NaT
            if pd.notna(effective) and effective.year==year:
                events.append(dict(Trigger_date=day,Effective_date=effective,Gold_trigger=bool(cross.loc[day,'gold_vs_ma20']),Copper_trigger=bool(cross.loc[day,'copper_vs_ma20'])))
    monthly=[]
    for month in sorted(te.to_period('M').unique()):
        dates=te[te.to_period('M')==month]
        monthly.append(dict(Anchor=month.start_time,Effective_date=dates.min(),Trigger_date=pd.NaT))
    event=[dict(Anchor=start,Effective_date=te.min(),Trigger_date=pd.NaT)]+[dict(Anchor=r['Effective_date'],Effective_date=r['Effective_date'],Trigger_date=r['Trigger_date']) for r in events]
    schedules={'Monthly_equal':monthly,'Monthly_H60':monthly,'Crossing_H60':event}
    predictions=pd.DataFrame({'Date':te,'Actual':y.loc[te].to_numpy()})
    updates=[];selection_cache={};model_cache={};trained=0;started=time.time()
    # Reuse original 2020 decisions to preserve this chapter's exact controls.
    if year==2020:
        old_month=pd.read_csv(locate('mlp_monthly_validation_12')/'selection.csv')
        old_event=pd.read_csv(locate('mlp_crossing_update_12')/'updates.csv')
        for r in old_month.itertuples(index=False):
            selection_cache[pd.Timestamp(r.Month+'-01')]=dict(Selected_seed=int(r.Selected_seed),Validation_MAPE=r.Validation_MAPE,Source='reused 12 monthly selection')
        for r in old_event.itertuples(index=False):
            key=start if r.Kind=='Initial' else pd.Timestamp(r.Effective_date)
            decision=dict(Selected_seed=int(r.Selected_seed),Validation_MAPE=r.Validation_MAPE,Source='reused 12 crossing selection')
            if key in selection_cache:assert decision['Selected_seed']==selection_cache[key]['Selected_seed']
            selection_cache[key]=decision
        assert [pd.Timestamp(r['Effective_date']) for r in event]==pd.to_datetime(old_event.Effective_date).tolist()
    anchors=sorted(set(r['Anchor'] for rows in schedules.values() for r in rows))
    for i,anchor in enumerate(anchors):
        if anchor not in selection_cache:selection_cache[anchor]=select_seed(anchor,folder)
        if (i+1)%10==0:print(f'{year} selection {i+1}/{len(anchors)} ({time.time()-started:.0f}s)',flush=True)
    for method,rows in schedules.items():
        half=None if method=='Monthly_equal' else 60
        method_frames=[]
        for j,r in enumerate(rows):
            effective=r['Effective_date'];anchor=r['Anchor'];stop=rows[j+1]['Effective_date'] if j+1<len(rows) else end
            idx=x.index[x.index<effective];applied=te[(te>=effective)&(te<stop)]
            decision=selection_cache[anchor];seed=decision['Selected_seed']
            assert idx.max()<applied.min() and len(applied)>0
            if pd.notna(r['Trigger_date']):assert idx.max()==r['Trigger_date']
            key=(str(idx.max()),seed,half)
            if key not in model_cache:
                bundle=train_stack(idx,seed,half);model_cache[key]=bundle;trained+=2
                save_bundle(bundle,folder/f'{method}_{effective.strftime("%Y%m%d")}.pt',seed,half,idx)
            bundle=model_cache[key]
            pred=stack_predict(bundle,x.loc[applied])
            method_frames.append(pd.DataFrame({'Date':applied,method:pred.to_numpy()}))
            updates.append(dict(Year=year,Method=method,Anchor=anchor,Effective_date=effective,Trigger_date=r['Trigger_date'],Seed=seed,Train_start=idx.min(),Train_end=idx.max(),Train_N=len(idx),Applied_start=applied.min(),Applied_end=applied.max(),Applied_N=len(applied),Validation_MAPE=decision['Validation_MAPE'],Selection_source=decision['Source']))
        frame=pd.concat(method_frames,ignore_index=True)
        assert pd.DatetimeIndex(frame.Date).equals(te)
        predictions=predictions.merge(frame,on='Date',validate='one_to_one')
    initial_stop=min(monthly[1]['Effective_date'] if len(monthly)>1 else end,event[1]['Effective_date'] if len(event)>1 else end)
    first=predictions.Date<initial_stop
    assert np.allclose(predictions.loc[first,'Monthly_H60'],predictions.loc[first,'Crossing_H60'],rtol=0,atol=1e-5)
    if year==2020:
        old=pd.read_csv(locate('mlp_crossing_h60_13')/'predictions.csv',parse_dates=['Date'])
        assert pd.DatetimeIndex(old.Date).equals(te)
        for new,prior in [('Monthly_equal','A'),('Monthly_H60','M60'),('Crossing_H60','E60')]:
            assert np.allclose(predictions[new],old[prior],atol=1e-5,rtol=0),(new,'2020 mismatch')
    predictions['Year']=year
    predictions.to_csv(folder/'predictions.csv',index=False)
    pd.DataFrame(updates).to_csv(folder/'updates.csv',index=False)
    pd.DataFrame(events).to_csv(folder/'events.csv',index=False)
    result=dict(Year=year,N=len(te),Start=str(te.min().date()),End=str(te.max().date()),Monthly_models=len(monthly),Crossing_models=len(event),Unique_selection_dates=len(anchors),New_final_stage_trainings=trained,Seconds=time.time()-started)
    (folder/'done.json').write_text(json.dumps(result,indent=2))
    print(f'{year} DONE: {len(monthly)} monthly / {len(event)} crossing models; {time.time()-started:.0f}s',flush=True)
    return result

def prepare():
    dataset=ROOT/'src/pickle/Dataset.pkl';d=pd.read_pickle(dataset)
    raw=d['X'].copy();target=d['Y'].copy();assert raw.index.equals(target.index)
    assert raw.index.is_monotonic_increasing and raw.index.is_unique
    raw=raw.loc[raw.index.year<=2026];data=raw.copy()
    for metal,col in [('gold','Gold Futures Price'),('copper','Copper Futures Price')]:
        data[metal+'_daily']=raw[col]/raw[col].shift(1)-1
        for k in [5,20]:data[f'{metal}_vs_ma{k}']=raw[col]/raw[col].rolling(k,min_periods=k).mean()-1
    actual=pd.Series(np.asarray(target).reshape(-1),index=target.index).loc[raw.index]
    mask=np.isfinite(data.to_numpy()).all(axis=1)&np.isfinite(actual)
    data=data.loc[mask];data['Actual']=actual.loc[data.index]
    data.to_csv(OUT/'features_and_target.csv',index_label='Date')
    old=locate('mlp_annual_ten_ma_11');cfg=json.loads((old/'config.json').read_text())
    digest=hashlib.sha256(dataset.read_bytes()).hexdigest();assert cfg['dataset_sha256']==digest
    config=dict(years=list(range(2015,2027)),dataset_sha256=digest,epochs_per_stage=100,hidden=[16,32],activation='ReLU',optimizer='Adam',lr=.01,seeds=SEEDS,half_life=60,age_unit='valid records',selection='equal-MSE candidates; preceding three calendar months validation MAPE; fixed seeds and tie order; refit all pre-effective data',annual_initialization='reset each year; shared Jan 1 selection anchor for monthly and crossing',trigger='either metal absolute MA20 deviation crosses 3%, next-record effective, merged same-day signals',old_annual_source=str(old),old_annual_model='Stacking',old_annual_aggregation='mean of ten seed metrics, NOT metrics of ensemble prediction',old_annual_loss='equal MSE',old_annual_seeds=cfg['seeds'],old_annual_metrics_sha256=hashlib.sha256((old/'metrics.csv').read_bytes()).hexdigest(),partial_2026_end=str(data.index.max().date()),torch_version=str(torch.__version__))
    if (OUT/'config.json').exists():
        previous_config=json.loads((OUT/'config.json').read_text())
        assert {k:v for k,v in previous_config.items() if k!='old_annual_source'}=={k:v for k,v in config.items() if k!='old_annual_source'}, 'Configuration changed: use a new results directory before rerunning.'
    (OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);parser.add_argument('--years',nargs='*',type=int);args=parser.parse_args()
    prepare();years=args.years or list(range(2015,2027))
    if args.workers==1:
        for yr in years:year_run(yr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
            jobs={pool.submit(year_run,yr):yr for yr in years}
            for f in as_completed(jobs):f.result()
    print('REQUESTED YEARS COMPLETED',flush=True)
