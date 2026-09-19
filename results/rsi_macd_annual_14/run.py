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
RUN_OUT=Path(os.environ.get('MLP14_RSI_MACD_OUT',str(Path(__file__).resolve().parent)))
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


EXTRA=['a500_vs_ma20','us10y_vs_ma20']
TECH={'RSI':['RSI14_lag1'],'MACD':['DIFF_lag1','DEA_lag1'],'Both':['RSI14_lag1','DIFF_lag1','DEA_lag1']}

def source(name):
    p=RUN_OUT.parent/name
    return p if p.exists() else ROOT/'results'/name

def indicators(price):
    delta=price.diff().to_numpy();up=np.maximum(delta,0);down=np.maximum(-delta,0)
    n=14;gain=np.full(len(price),np.nan);loss=gain.copy()
    gain[n]=np.mean(up[1:n+1]);loss[n]=np.mean(down[1:n+1])
    for i in range(n+1,len(price)):
        gain[i]=(13*gain[i-1]+up[i])/14;loss[i]=(13*loss[i-1]+down[i])/14
    total=gain+loss
    rsi=np.divide(100*gain,total,out=np.full(len(price),50.0),where=total!=0)
    rsi[:n]=np.nan
    fast=price.ewm(span=12,adjust=False).mean();slow=price.ewm(span=26,adjust=False).mean();diff=fast-slow;dea=diff.ewm(span=9,adjust=False).mean()
    return pd.DataFrame({'RSI14':rsi,'DIFF':diff,'DEA':dea},index=price.index)

def restore(pack):
    state=pack['state_dict'];dim=state['0.weight'].shape[1]
    model=nn.Sequential(nn.Linear(dim,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1));model.load_state_dict(state);model.eval()
    scalers=[]
    for prefix in ['sx','sy']:
        s=StandardScaler();s.mean_=np.array(pack[prefix+'_mean']);s.scale_=np.array(pack[prefix+'_scale']);s.var_=s.scale_**2;s.n_features_in_=len(s.mean_);scalers.append(s)
    return model,*scalers

def pred_saved(bundle,X):
    model,sx,sy=bundle
    with torch.no_grad():z=model(torch.tensor(sx.transform(X.to_numpy()),dtype=torch.float32)).numpy()
    return pd.Series(sy.inverse_transform(z).ravel(),index=X.index)

def prepare():
    d=pd.read_pickle(ROOT/'src/pickle/Dataset.pkl');price=pd.Series(np.asarray(d['Y']).reshape(-1),index=d['Y'].index)
    assert price.index.is_unique and price.index.is_monotonic_increasing and np.isfinite(price).all()
    tech=indicators(price).shift(1);tech.columns=[c+'_lag1' for c in tech.columns]
    # Perturbing target-day and future prices must not change that day's lagged inputs.
    for day in ['2015-01-01','2016-06-01','2020-07-01','2026-07-23']:
        if pd.Timestamp(day) not in price.index:continue
        at=price.index.get_loc(day);changed=price.copy();changed.iloc[at:]*=1.37
        check=indicators(changed).shift(1)
        assert np.allclose(check.iloc[at],tech.iloc[at],atol=1e-12,rtol=0)
    # Edge cases verify Wilder's convention and recursive EMA initialization.
    assert np.allclose(indicators(pd.Series(np.arange(1.,61))).RSI14.dropna(),100)
    assert np.allclose(indicators(pd.Series(np.arange(60.,0.,-1))).RSI14.dropna(),0)
    assert np.allclose(indicators(pd.Series(np.ones(60))).RSI14.dropna(),50)
    previous_dates=pd.Series(price.index,index=price.index).shift(1)
    valid=tech.dropna().index;assert (previous_dates.loc[valid].to_numpy()<valid.to_numpy()).all()
    tech['Indicator_source_date']=previous_dates
    tech.to_csv(RUN_OUT/'technical_features.csv',index_label='Date')
    cfg=dict(years=list(range(2015,2027)),baseline='Features_and_triggers',half_life=60,epochs_per_stage=100,stage1='reuse frozen baseline stage1 checkpoints',stage2_original_inputs=['base_prediction']+RETURNS+EXTRA,technical_inputs=TECH,rsi='Wilder 14: initial average of first 14 changes, then alpha 1/14; zero gain and loss => 50',macd='DIFF=EMA12-EMA26; DEA=EMA9(DIFF); adjust=False seeded with first price',lag='1 raw Dataset record; computed over full history before slicing',target='same-date stock price; lagged technical indicators use prior observed price only',seeds_and_updates='exactly baseline; no reranking',standardization='fit on pre-effective training only; unweighted StandardScaler',hidden=[16,32],lr=.01,dataset_sha256=hashlib.sha256((ROOT/'src/pickle/Dataset.pkl').read_bytes()).hexdigest(),leakage_perturbation_check=True)
    (RUN_OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))

def run_year(year):
    global OUT,x,y
    OUT=RUN_OUT/str(year);OUT.mkdir(exist_ok=True)
    if (OUT/'done.json').exists():return year
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    prior=source('multi_threshold_annual_14')/str(year)
    oldcfg=json.loads((prior/'config.json').read_text());cfg=json.loads((RUN_OUT/'config.json').read_text());assert oldcfg['dataset_sha256']==cfg['dataset_sha256']
    data=pd.read_csv(prior/'features_and_target.csv',parse_dates=['Date']).set_index('Date')
    y=data.Actual;x=data[BASE+RETURNS+EXTRA].copy()
    tech=pd.read_csv(RUN_OUT/'technical_features.csv',parse_dates=['Date','Indicator_source_date']).set_index('Date').loc[x.index]
    cols=['RSI14_lag1','DIFF_lag1','DEA_lag1'];assert np.isfinite(tech[cols]).all().all()
    assert (tech.Indicator_source_date.to_numpy()<tech.index.to_numpy()).all()
    x=x.join(tech[cols]);te=x.index[x.index.year==year]
    old=pd.read_csv(prior/'predictions.csv',parse_dates=['Date']);assert old.Date.tolist()==te.tolist()
    updates=pd.read_csv(prior/'updates.csv',parse_dates=['Effective_date','Trigger_date']);updates=updates[updates.Method=='Features_and_triggers'].sort_values('Effective_date')
    predframes=[];audits=[];started=time.time()
    for j,row in enumerate(updates.itertuples(index=False)):
        date=row.Effective_date;stop=updates.iloc[j+1].Effective_date if j+1<len(updates) else pd.Timestamp(f'{year+1}-01-01')
        tr=x.index[x.index<date];applied=te[(te>=date)&(te<stop)];seed=int(row.Seed)
        assert tr.max()<applied.min() and len(applied)>0
        checkpoint=prior/f'{date:%Y%m%d}_extra.pt'
        if not checkpoint.exists() and year==2016:checkpoint=source('multi_threshold_2016_14')/checkpoint.name
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['seed']==seed and pd.Timestamp(saved['train_end'])==tr.max() and saved['half_life']==60
        base=restore(saved['stages'][0]);baseline=restore(saved['stages'][1])
        known=x.loc[tr];trainmeta=known[RETURNS+EXTRA].copy();trainmeta.insert(0,'base_prediction',pred_saved(base,known[BASE]))
        testmeta=x.loc[applied,RETURNS+EXTRA].copy();testmeta.insert(0,'base_prediction',pred_saved(base,x.loc[applied,BASE]))
        baseline_pred=pred_saved(baseline,testmeta)
        expected=old.set_index('Date').loc[applied,'Features_and_triggers']
        assert np.allclose(baseline_pred,expected,atol=1e-5,rtol=0),'Checkpoint baseline replication failed'
        result=pd.DataFrame({'Date':applied,'Actual':y.loc[applied].to_numpy(),'Baseline':expected.to_numpy()})
        for method,add in TECH.items():
            model=fit(trainmeta.join(known[add]),tr,seed,60)
            prediction=predict(model,testmeta.join(x.loc[applied,add]));result[method]=prediction.to_numpy()
            net,sx,sy=model
            torch.save(dict(state_dict=net.state_dict(),features=list(trainmeta.columns)+add,seed=seed,half_life=60,epochs=100,train_end=str(tr.max()),sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist()),OUT/f'{date:%Y%m%d}_{method}.pt')
            audits.append(dict(Method=method,Effective_date=date,Seed=seed,Train_start=tr.min(),Train_end=tr.max(),Train_N=len(tr),Test_start=applied.min(),Test_end=applied.max(),N=len(applied),Feature_N=len(trainmeta.columns)+len(add)))
        predframes.append(result)
    p=pd.concat(predframes,ignore_index=True);assert p.Date.tolist()==te.tolist() and p.Date.is_unique
    s=[]
    for method in ['Baseline']+list(TECH):
        err=p[method]-p.Actual;s.append(dict(Year=year,Method=method,N=len(p),MAPE=100*(err.abs()/p.Actual.abs()).mean(),MSE=(err**2).mean(),MAE=err.abs().mean(),Updates=len(updates)-1,Models=len(updates),Start=str(te.min().date()),End=str(te.max().date())))
    p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(s).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'audit.csv',index=False)
    (OUT/'done.json').write_text(json.dumps(dict(year=year,N=len(te),baseline_checkpoint_verified=True,models=len(updates),seconds=time.time()-started),indent=2))
    print(year,'DONE',pd.DataFrame(s)[['Method','MAPE']].round(4).to_dict('records'),flush=True)
    return year

if __name__=='__main__':
    prepare()
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(run_year,year) for year in range(2015,2027)]
        for f in as_completed(futures):f.result()
    print('ALL YEARS COMPLETE',flush=True)
