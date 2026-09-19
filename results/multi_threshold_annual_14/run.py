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
RUN_OUT=Path(__file__).resolve().parent
OUT=RUN_OUT
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


def experiment(year):
    global x,y
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    prior=locate('mlp_four_modes_2015_2026_13')
    frame=pd.read_csv(prior/'features_and_target.csv',parse_dates=['Date']).set_index('Date')
    dataset=ROOT/'src/pickle/Dataset.pkl';d=pd.read_pickle(dataset);raw=d['X']
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==json.loads((prior/'config.json').read_text())['dataset_sha256']
    x=frame[BASE+RETURNS].copy();y=frame.Actual
    extra=['a500_vs_ma20','us10y_vs_ma20']
    for name,col in zip(extra,['A_500','us_10y_rate']):x[name]=(raw[col]/raw[col].rolling(20,min_periods=20).mean()-1).reindex(x.index)
    assert np.isfinite(x.to_numpy()).all()
    te=x.index[x.index.year==year];assert len(te)>0
    old_rules={'gold_vs_ma20':[.03],'copper_vs_ma20':[.03]}
    new_rules={'gold_vs_ma20':[.03],'copper_vs_ma20':[.03,.08],'a500_vs_ma20':[.05,.08],'us10y_vs_ma20':[.10,.20]}
    def schedule(rules,label):
        signals=[]
        for col,thresholds in rules.items():
            for threshold in thresholds:
                outside=x[col].abs().ge(threshold);previous=outside.shift(1)
                changed=outside.ne(previous)&previous.notna()
                for day in te[changed.loc[te].to_numpy()]:
                    i=x.index.get_loc(day);effective=x.index[i+1] if i+1<len(x) else pd.NaT
                    if pd.notna(effective) and effective.year==year:
                        signals.append(dict(Trigger_date=day,Effective_date=effective,Variable=col,Threshold=threshold,Direction='outward' if outside.loc[day] else 'inward',Previous_deviation=float(x.iloc[i-1][col]),Deviation=float(x.loc[day,col])))
        sig=pd.DataFrame(signals).sort_values(['Trigger_date','Variable','Threshold']);sig.to_csv(OUT/f'{label}_signals.csv',index=False)
        dates=sorted(sig.Effective_date.unique())
        rows=[dict(Effective_date=te.min(),Trigger_date=pd.NaT)]+[dict(Effective_date=pd.Timestamp(date),Trigger_date=sig.loc[sig.Effective_date==date,'Trigger_date'].iloc[0]) for date in dates]
        return pd.DataFrame(rows),sig
    old_schedule,old_signals=schedule(old_rules,'original');new_schedule,new_signals=schedule(new_rules,'expanded')
    old_updates=pd.read_csv(prior/str(year)/'updates.csv',parse_dates=['Effective_date']);old_updates=old_updates[old_updates.Method=='Crossing_H60']
    assert old_schedule.Effective_date.tolist()==old_updates.Effective_date.tolist()
    assert set(old_schedule.Effective_date)<=set(new_schedule.Effective_date)
    selection_dir=OUT/'selections';selection_dir.mkdir(exist_ok=True)
    decisions={};new_selections=0
    for date in new_schedule.Effective_date:
        anchor=pd.Timestamp(f'{year}-01-01') if date==te.min() else date
        cache=prior/str(year)/f'selection_{anchor:%Y-%m-%d}.json'
        matched=old_updates[old_updates.Effective_date==date]
        if len(matched):
            r=matched.iloc[0];decision=dict(Selected_seed=int(r.Seed),Validation_MAPE=float(r.Validation_MAPE),Source='reused baseline update seed')
            (selection_dir/cache.name).write_text(json.dumps(decision,indent=2))
        elif cache.exists():
            decision=json.loads(cache.read_text());(selection_dir/cache.name).write_text(json.dumps(decision,indent=2))
        else:
            decision=select_seed(anchor,selection_dir);new_selections+=1
        decisions[date]=decision
    print(f'{year}: Original updates {len(old_schedule)-1}; expanded {len(new_schedule)-1}; new selection dates {new_selections}',flush=True)
    for r in old_updates.itertuples():assert decisions[r.Effective_date]['Selected_seed']==r.Seed
    # Keep baseline validation-selected seed identical across paired input variants.
    bundles={};fitrows=[]
    for j,date in enumerate(new_schedule.Effective_date):
        idx=x.index[x.index<date];seed=decisions[date]['Selected_seed'];known=x.loc[idx]
        assert idx.max()<date
        if j:assert idx.max()==new_schedule.iloc[j].Trigger_date
        base=fit(known[BASE],idx,seed,60)
        meta=known[RETURNS].copy();meta.insert(0,'base_prediction',predict(base,known[BASE]))
        for expanded in [False,True]:
            X=meta.join(known[extra]) if expanded else meta
            stack=fit(X,idx,seed,60)
            bundles[(date,expanded)]=(base,stack)
            path=OUT/f'{date:%Y%m%d}_{"extra" if expanded else "original"}.pt'
            save_bundle((base,stack),path,seed,60,idx)
            fitrows.append(dict(Effective_date=date,Extra_features=expanded,Seed=seed,Train_start=idx.min(),Train_end=idx.max(),Train_N=len(idx),Stage2_features='|'.join(X.columns)))
        if (j+1)%25==0:print(f'{year}: Refit {j+1}/{len(new_schedule)} dates',flush=True)
    methods={'Original':(old_schedule,False),'Features_only':(old_schedule,True),'Triggers_only':(new_schedule,False),'Features_and_triggers':(new_schedule,True)}
    predictions=pd.DataFrame({'Date':te,'Actual':y.loc[te].to_numpy()});updates=[]
    for method,(sched,expanded) in methods.items():
        parts=[]
        for j,row in enumerate(sched.itertuples(index=False)):
            date=row.Effective_date;stop=sched.iloc[j+1].Effective_date if j+1<len(sched) else pd.Timestamp(f'{year+1}-01-01')
            applied=te[(te>=date)&(te<stop)];base,stack=bundles[(date,expanded)]
            X=x.loc[applied,RETURNS].copy();X.insert(0,'base_prediction',predict(base,x.loc[applied,BASE]))
            if expanded:X=X.join(x.loc[applied,extra])
            p=predict(stack,X)
            parts.append(pd.DataFrame({'Date':applied,method:p.to_numpy()}))
            updates.append(dict(Method=method,Effective_date=date,Trigger_date=row.Trigger_date,Seed=decisions[date]['Selected_seed'],N=len(applied),Applied_end=applied.max()))
        p=pd.concat(parts);assert pd.DatetimeIndex(p.Date).equals(te)
        predictions=predictions.merge(p,on='Date',validate='one_to_one')
    oldpred=pd.read_csv(prior/str(year)/'predictions.csv',parse_dates=['Date'])
    assert oldpred.Date.tolist()==predictions.Date.tolist()
    assert np.allclose(predictions.Original,oldpred.Crossing_H60,atol=1e-5,rtol=0),'Original replication failed'
    predictions['Month']=predictions.Date.dt.strftime('%Y-%m');monthly=[];summary=[]
    for method,(sched,expanded) in methods.items():
        err=predictions[method]-predictions.Actual
        summary.append(dict(Method=method,MAPE=100*(err.abs()/predictions.Actual.abs()).mean(),MSE=(err**2).mean(),MAE=err.abs().mean(),Updates=len(sched)-1,Initializations=1,Models=len(sched)))
        for month,g in predictions.groupby('Month'):
            e=g[method]-g.Actual;monthly.append(dict(Month=month,Method=method,N=len(g),MAPE=100*(e.abs()/g.Actual.abs()).mean(),MSE=(e**2).mean()))
    summary=pd.DataFrame(summary);monthly=pd.DataFrame(monthly)
    summary['MAPE_difference_vs_original_pp']=summary.MAPE-summary.iloc[0].MAPE
    summary['MSE_change_vs_original_pct']=100*(summary.MSE/summary.iloc[0].MSE-1)
    for name,f in [('predictions',predictions),('summary',summary),('monthly_metrics',monthly),('updates',pd.DataFrame(updates)),('training_audit',pd.DataFrame(fitrows))]:f.to_csv(OUT/f'{name}.csv',index=False)
    x.join(y.rename('Actual')).to_csv(OUT/'features_and_target.csv',index_label='Date')
    config=dict(year=year,half_life=60,half_life_unit='valid records',rules=new_rules,trigger='absolute MA20 deviation changes side of threshold; union across variables/thresholds; next record effective',original_rules=old_rules,extra_stage2_features=extra,stage1_features=BASE,original_stage2_features=['base_prediction']+RETURNS,selection='baseline-feature equal-MSE validation candidates; preceding 3 calendar months; same selected seed for paired feature variants; existing selection reused',seed_candidates=SEEDS,epochs_per_stage=100,hidden=[16,32],lr=.01,optimizer='Adam',scaler='unweighted StandardScaler',rate_threshold_unit='relative change in yield, not percentage points',dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),original_reproduced=True,N=len(te),new_selection_dates=new_selections,additional_update_days=len(new_schedule)-len(old_schedule))
    (OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print(summary.to_string(index=False),flush=True)



def year_run(year):
    global OUT
    OUT=RUN_OUT/str(year);OUT.mkdir(exist_ok=True)
    if (OUT/'config.json').exists() and (OUT/'summary.csv').exists():
        print(f'{year}: reuse completed results',flush=True);return year
    experiment(year)
    print(f'{year}: COMPLETED',flush=True)
    return year

if __name__=='__main__':
    import shutil
    old2016=RUN_OUT.parent/'multi_threshold_2016_14'
    if not old2016.exists():old2016=ROOT/'results/multi_threshold_2016_14'
    target=RUN_OUT/'2016';target.mkdir(exist_ok=True)
    for pattern in ['*.csv','config.json']:
        for file in old2016.glob(pattern):shutil.copy2(file,target/file.name)
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(year_run,year) for year in range(2015,2027)]
        for f in as_completed(futures):f.result()
    print('ALL YEARS COMPLETE',flush=True)
