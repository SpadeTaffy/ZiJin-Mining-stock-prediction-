from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(src,as_version=4); nb.write(n,'outputs/mlp_seed_validation_2020_11/11_before_seed_selection.ipynb')
Path('outputs/mlp_seed_validation_2020_11/start_cell.txt').write_text(str(len(n.cells)))
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 2020年前三个月验证选种子：单阶段DirectMA、100轮、五个种子

模型：原7变量＋金铜相对5/20期滚动均价涨幅，共11项输入；单阶段MLP，隐藏层16→32、ReLU、Adam学习率0.01、全批量、100轮。种子预先固定42、7、123、2024、2025；均价包含当日、窗口按原始记录计数。目标仍为原数据同日股价，不是次日预测。

1. 训练：2019-12-31及以前，保留上一实验共同剔除最初20条记录的口径；标准化仅拟合训练集。
2. 先展示五个模型2020年的预测，灰色区域为验证期。全年的展示只供事后说明，不参与程序选种子。
3. 验证：2020-01-01至2020-03-31，以验证MAPE最低者胜出；精确并列时按预先给定种子顺序。保存选择记录后，再计算后续测试误差。
4. 测试：2020-04-01至2020-12-31。锁定选中的模型，**不把验证集加入训练、不重新训练、不根据4—12月结果改选**。其它种子测试结果只作事后对照。

这是验证“选初始化”的实验，不能找到对所有未来时期最优的种子。由于此前已反复分析2020年，本节属于历史回放，而非全新盲测。没有月份内滚动重训练；“滚动”指输入的均价指标。
''')
code('''from pathlib import Path
import os,json,hashlib,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display,Image
ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/pickle/Dataset.pkl').exists()),Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-'))
OUT=Path(os.environ.get('MLP11_SEEDVAL_OUT',str(ROOT/'results/mlp_seed_validation_2020_11'))); OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,7,123,2024,2025]; EPOCHS=100; LR=0.01
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
dataset=ROOT/'src/pickle/Dataset.pkl'; d=pd.read_pickle(dataset)
raw=d['X'].copy(); target=d['Y'].copy(); assert raw.index.equals(target.index)
assert raw.index.is_unique and raw.index.is_monotonic_increasing
raw=raw.loc[raw.index.year<=2020]; BASE=list(raw.columns); assert len(BASE)==7
y=pd.Series(np.asarray(target).reshape(-1),index=target.index,name='Actual').loc[raw.index]
x=raw.copy(); DIRECT=[]
for metal,col in [('gold','Gold Futures Price'),('copper','Copper Futures Price')]:
    for k in [5,20]:
        name=f'{metal}_ma_deviation{k}'; DIRECT.append(name)
        x[name]=raw[col]/raw[col].rolling(k,min_periods=k).mean()-1
COLS=BASE+DIRECT
valid=np.isfinite(x.to_numpy()).all(axis=1)&np.isfinite(y.to_numpy())&(x.index>=raw.index[20])
x=x.loc[valid]; y=y.loc[x.index]
tr=x.index[x.index<'2020-01-01']; va=x.index[(x.index>='2020-01-01')&(x.index<'2020-04-01')]; te=x.index[(x.index>='2020-04-01')&(x.index<'2021-01-01')]
whole=x.index[x.index.year==2020]
assert len(tr)==2034 and len(va)+len(te)==271 and tr.max()<va.min() and va.max()<te.min() and (y>0).all()
splits=pd.DataFrame([dict(Split=s,N=len(i),Start=str(i.min().date()),End=str(i.max().date())) for s,i in [('Train',tr),('Validation',va),('Test',te)]])
splits.to_csv(OUT/'splits.csv',index=False); display(splits)
config=dict(seeds=SEEDS,epochs=EPOCHS,hidden=[16,32],features=COLS,activation='ReLU',optimizer='Adam',lr=LR,batch='full',train_loss='standardized target MSE',selection='minimum Jan-Mar 2020 MAPE; ties by seed list order',refit=False,target='same-date original price',ma_includes_current=True,common_warmup_records=20,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),torch_version=str(torch.__version__))
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
x.join(y).to_csv(OUT/'features_and_target.csv',index_label='Date')
sx=StandardScaler().fit(x.loc[tr,COLS]); sy=StandardScaler().fit(y.loc[tr].to_numpy().reshape(-1,1))
xt=torch.tensor(sx.transform(x.loc[tr,COLS]),dtype=torch.float32); yt=torch.tensor(sy.transform(y.loc[tr].to_numpy().reshape(-1,1)),dtype=torch.float32)
''')
code('''history=[]; preds={}; models={}
for seed in SEEDS:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=nn.Sequential(nn.Linear(len(COLS),16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,EPOCHS+1):
        model.train(); optimizer.zero_grad(); loss=((model(xt)-yt)**2).mean(); assert torch.isfinite(loss)
        loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad(): z=model(xt).numpy()
        p=sy.inverse_transform(z).ravel(); mse=np.mean((p-y.loc[tr].to_numpy())**2)
        history.append(dict(Seed=seed,Epoch=epoch,Train_MSE=mse,Train_MSE_standardized=mse/float(sy.scale_[0]**2)))
    with torch.no_grad(): z=model(torch.tensor(sx.transform(x[COLS]),dtype=torch.float32)).numpy()
    preds[seed]=pd.Series(sy.inverse_transform(z).ravel(),index=x.index); models[seed]=model
    assert np.isfinite(preds[seed]).all()
    torch.save(dict(state_dict=model.state_dict(),seed=seed,epochs=EPOCHS,features=COLS,sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist()),OUT/f'DirectMA_seed{seed}.pt')
pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
predictions=pd.concat([pd.DataFrame(dict(Date=x.index,Seed=seed,Actual=y.to_numpy(),Predicted=preds[seed].to_numpy(),Split=np.where(x.index<'2020-01-01','Train',np.where(x.index<'2020-04-01','Validation','Test')))) for seed in SEEDS],ignore_index=True)
predictions.to_csv(OUT/'predictions.csv',index=False)
# 复现此前2020年DirectMA，按float32往返精度核对。
prior=OUT.parent/'mlp_direct_definition_11/predictions.csv'
if prior.exists():
    old=pd.read_csv(prior,parse_dates=['Date']); old=old[(old.Year==2020)&(old.Model=='DirectMA')&old.Seed.isin(SEEDS)]
    check=old.merge(predictions,on=['Date','Seed'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==len(predictions)
    assert np.array_equal(check.Predicted_old.to_numpy(dtype=np.float32),check.Predicted_new.to_numpy(dtype=np.float32))
    print('复核通过：五个种子的预测与此前2020年DirectMA实验一致。')
# 按请求先展示五条全年曲线，此时尚未用后续测试误差选种子。
colors=dict(zip(SEEDS,['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']))
fig,axes=plt.subplots(5,1,figsize=(13,14),sharex=True,constrained_layout=True)
for ax,seed in zip(axes,SEEDS):
    ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2020-04-01'),color='#999999',alpha=.15,label='Validation: Jan-Mar')
    ax.axvline(pd.Timestamp('2020-04-01'),color='gray',ls='--')
    ax.plot(whole,y.loc[whole],color='black',lw=1.5,label='Actual')
    ax.plot(whole,preds[seed].loc[whole],color=colors[seed],lw=1.3,label=f'Seed {seed}')
    ax.set(title=f'DirectMA | Seed {seed} | 100 epochs',ylabel='Price (dataset units)'); ax.grid(alpha=.2); ax.legend(loc='upper left',ncol=3,fontsize=8)
fig.savefig(OUT/'five_seed_predictions.png',dpi=140);plt.close(fig);display(Image(filename=str(OUT/'five_seed_predictions.png')))
''')
md('''## 只用第一季度选择并锁定种子

依次计算各候选在2020年1—3月的MAPE，按低到高排序，选取第一名。这个步骤不读取4—12月误差；保存选择记录后再评估后续表现。
''')
code('''def metrics_at(seed,idx):
    a=y.loc[idx].to_numpy(); p=preds[seed].loc[idx].to_numpy(); err=p-a
    return dict(N=len(idx),MAPE=100*np.mean(np.abs(err)/np.abs(a)),MSE=np.mean(err**2))
validation=pd.DataFrame([dict(Seed=seed,Candidate_order=i,**metrics_at(seed,va)) for i,seed in enumerate(SEEDS)])
validation=validation.sort_values(['MAPE','Candidate_order']).reset_index(drop=True)
validation.insert(0,'Validation_rank',np.arange(1,len(validation)+1))
selected_seed=int(validation.iloc[0].Seed)
validation['Selected']=validation.Seed.eq(selected_seed)
validation.to_csv(OUT/'validation_ranking.csv',index=False)
selection=dict(selected_seed=selected_seed,criterion='Jan-Mar 2020 MAPE only',validation_mape=float(validation.iloc[0].MAPE),validation_start=str(va.min().date()),validation_end=str(va.max().date()),candidate_order=SEEDS,epochs=EPOCHS,refit=False)
(OUT/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
print(f'按验证MAPE锁定 seed {selected_seed}，不重新训练。'); display(validation[['Validation_rank','Seed','MAPE','Selected']].round(4))
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
bars=ax.bar(validation.Seed.astype(str),validation.MAPE,color=['#009E73' if s==selected_seed else '#a6b4c3' for s in validation.Seed])
ax.bar_label(bars,fmt='%.3f%%',padding=4);ax.margins(y=.2)
ax.set(title=f'Validation only: Jan-Mar 2020 | Selected seed {selected_seed}',xlabel='Seed (ranked by validation MAPE)',ylabel='Validation MAPE (%)')
fig.savefig(OUT/'validation_selection.png',dpi=160);plt.close(fig);display(Image(filename=str(OUT/'validation_selection.png')))
''')
md('''## 锁定后评估2020年4—12月

下表展示五个候选的后续表现，用于回顾验证排名是否延续。所选种子保持不变；后续测试期表现最好的种子只是事后对照，不能据此改选。
''')
code('''locked=json.loads((OUT/'selection.json').read_text()); assert selected_seed==locked['selected_seed']
metrics=pd.DataFrame([dict(Seed=seed,Split=split,**metrics_at(seed,idx)) for seed in SEEDS for split,idx in [('Train',tr),('Validation',va),('Test',te)]])
metrics.to_csv(OUT/'metrics.csv',index=False)
comparison=metrics.pivot(index='Seed',columns='Split',values='MAPE').reindex(SEEDS)
comparison['Validation_rank']=comparison.Validation.rank(method='first').astype(int)
comparison['Test_rank']=comparison.Test.rank(method='min').astype(int)
comparison['Selected']=comparison.index==selected_seed
comparison.to_csv(OUT/'comparison.csv')
display(comparison.round(4))
selected_test=float(comparison.loc[selected_seed,'Test']); test_mean=float(comparison.Test.mean())
result=dict(selected_seed=selected_seed,selected_validation_mape=float(comparison.loc[selected_seed,'Validation']),selected_test_mape=selected_test,selected_test_rank=int(comparison.loc[selected_seed,'Test_rank']),five_seed_mean_test_mape=test_mean,five_seed_test_sd=float(comparison.Test.std()),selected_minus_seed_mean_pp=selected_test-test_mean,refit=False)
(OUT/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(f'验证选中seed {selected_seed}；4—12月MAPE={selected_test:.4f}%，五种子MAPE均值={test_mean:.4f}%。')
fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
for ax,idx,title in [(axes[0],va,'Validation: Jan-Mar'),(axes[1],te,'Held-out period: Apr-Dec')]:
    for seed in SEEDS:
        if seed!=selected_seed:ax.plot(idx,preds[seed].loc[idx],color='gray',alpha=.3,lw=1)
    ax.plot(idx,y.loc[idx],color='black',lw=1.7,label='Actual')
    ax.plot(idx,preds[selected_seed].loc[idx],color='#009E73',lw=1.8,label=f'Selected seed {selected_seed}')
    ax.set(title=title,ylabel='Price (dataset units)');ax.legend();ax.grid(alpha=.2)
fig.savefig(OUT/'selected_prediction.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'selected_prediction.png')))
fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
comparison[['Validation','Test']].plot.bar(ax=ax,rot=0,color=['#0072B2','#D55E00'])
ax.set(title=f'Validation vs subsequent test | Locked seed {selected_seed}',ylabel='MAPE (%)');ax.margins(y=.2)
for c in ax.containers:ax.bar_label(c,fmt='%.2f',padding=3)
fig.savefig(OUT/'validation_vs_test.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'validation_vs_test.png')))
''')
nb.validate(n);nb.write(n,'outputs/11_改变模型构建.ipynb')
