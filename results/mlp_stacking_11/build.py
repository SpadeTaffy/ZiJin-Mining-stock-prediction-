from pathlib import Path
import nbformat as nb
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
p=ROOT/'notebooks/04_采用神经网络模型/11_改变模型构建.ipynb'
n=nb.read(p,as_version=4)
nb.write(n,'outputs/mlp_stacking_11/11_before_stacking.ipynb')
cells=[]
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))
md('''# 两阶段堆叠模型与单阶段基准：2020年测试，三个种子

本节是实际执行方案；上方其他模型组合保留为历史设想，本次只比较两组。

- **Base**：原7个宏观/价格变量 → 16 → ReLU → 32 → ReLU → 预测股价。
- **Stacking**：冻结上述Base，将其预测股价与金、铜各3个涨幅合并 → 16 → ReLU → 32 → ReLU → 最终股价。第二阶段直接预测股价，不是预测残差。
- 金、铜分别构造 `P(t)/P(t-1)-1`、`P(t)/MA5(t)-1`、`P(t)/MA20(t)-1`，滚动均值包含当日。日指原数据相邻记录，窗口为5/20条记录。
- **按用户要求不用折外预测**：第二阶段用第一阶段对同一训练集的样本内预测，充分利用可用训练数据；第一阶段冻结，两个阶段各训练100轮。
- 训练集为2020年以前，测试集仅2020年。共同剔除涨幅窗口不足或缺失的记录，两个阶段和基准使用相同有效日期。2021年以后不参与本次实验。
- 种子42、7、123；ReLU、Adam、学习率0.01、全批量、标准化目标MSE训练，按原股价尺度计算MAPE（百分数，越低越好）。所有标准化器仅拟合训练集。
- “单阶段”基准本身也有16和32两个隐藏层。堆叠第一阶段直接复用对应种子的基准，保证比较一致。

沿用原数据**同日股价**目标：这不是次日预测。样本内堆叠可能使训练误差偏乐观，结论以2020年测试误差为准，不使用测试集选择轮数或种子。
''')
code('''from pathlib import Path
import os, json, hashlib, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT=next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/pickle/Dataset.pkl').exists()), Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-'))
OUT=Path(os.environ.get('MLP11_OUT', str(ROOT/'results/mlp_stacking_11')))
OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,7,123]; EPOCHS=100; LR=0.01
 torch_placeholder=0
'''.replace(' torch_placeholder=0', '''torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
dataset=ROOT/'src/pickle/Dataset.pkl'
d=pd.read_pickle(dataset)
raw=d['X'].copy(); target=d['Y'].copy()
assert raw.index.equals(target.index)
assert raw.index.is_unique and raw.index.is_monotonic_increasing
BASE=list(raw.columns); assert len(BASE)==7
# 截止2020年，先计算纯向后窗口，再分割，测试年初可使用2019年历史。
raw=raw.loc[raw.index.year<=2020]
y=pd.Series(np.asarray(target).reshape(-1),index=target.index,name='Actual').loc[raw.index]
x=raw.copy(); RETURNS=[]
for metal,col in [('gold','Gold Futures Price'),('copper','Copper Futures Price')]:
    name=f'{metal}_daily'; x[name]=raw[col]/raw[col].shift(1)-1; RETURNS.append(name)
    for k in [5,20]:
        name=f'{metal}_vs_ma{k}'
        x[name]=raw[col]/raw[col].rolling(k,min_periods=k).mean()-1
        RETURNS.append(name)
valid=np.isfinite(x.to_numpy()).all(axis=1)&np.isfinite(y.to_numpy())
x=x.loc[valid]; y=y.loc[x.index]
tr=x.index[x.index.year<2020]; te=x.index[x.index.year==2020]
assert len(tr)>0 and len(te)>0 and tr.max()<te.min() and (y>0).all()
splits=pd.DataFrame([dict(Split=s,N=len(idx),Start=str(idx.min().date()),End=str(idx.max().date())) for s,idx in [('Train',tr),('Test',te)]])
splits.to_csv(OUT/'splits.csv',index=False)
x.join(y).to_csv(OUT/'features_and_target.csv',index_label='Date')
config=dict(seeds=SEEDS,epochs_per_stage=EPOCHS,hidden=[16,32],activation='ReLU',optimizer='Adam',lr=LR,batch='full',train_loss='standardized target MSE',metric='MAPE percent',test_year=2020,base_features=BASE,stage2_features=['base_prediction']+RETURNS,stage2_training='in-sample predictions, no OOF',ma_includes_current=True,target='same-date original price',dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),torch_version=str(torch.__version__))
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
display(splits)
display(pd.DataFrame({'Stage2 input':config['stage2_features']}))'''))
code('''history=[]
def fit_model(X,Y,seed,label):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sx=StandardScaler().fit(X); sy=StandardScaler().fit(np.asarray(Y).reshape(-1,1))
    xt=torch.tensor(sx.transform(X),dtype=torch.float32)
    yt=torch.tensor(sy.transform(np.asarray(Y).reshape(-1,1)),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,EPOCHS+1):
        model.train(); opt.zero_grad()
        loss=((model(xt)-yt)**2).mean(); assert torch.isfinite(loss)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): mse=((model(xt)-yt)**2).mean().item()
        history.append(dict(Seed=seed,Stage=label,Epoch=epoch,Train_MSE=mse))
    return model,sx,sy

def predict(bundle,X):
    model,sx,sy=bundle
    model.eval()
    with torch.no_grad(): z=model(torch.tensor(sx.transform(X),dtype=torch.float32)).numpy()
    p=sy.inverse_transform(z).ravel(); assert np.isfinite(p).all()
    return p

def save_model(bundle,seed,label,features):
    model,sx,sy=bundle
    torch.save(dict(state_dict=model.state_dict(),features=features,seed=seed,epochs=EPOCHS,
                    sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),
                    sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist(),
                    train_start=str(tr.min()),train_end=str(tr.max())),OUT/f'{label}_seed{seed}.pt')

rows=[]; frames=[]
for seed in SEEDS:
    base=fit_model(x.loc[tr,BASE],y.loc[tr],seed,'Base')
    bp=pd.Series(predict(base,x[BASE]),index=x.index)
    # 用户指定：全训练集样本内预测；不做折外划分、不反向更新第一阶段。
    meta=x[RETURNS].copy(); meta.insert(0,'base_prediction',bp)
    stack=fit_model(meta.loc[tr],y.loc[tr],seed,'Stage2')
    sp=pd.Series(predict(stack,meta),index=x.index)
    save_model(base,seed,'Base',BASE); save_model(stack,seed,'Stage2',list(meta.columns))
    meta.to_csv(OUT/f'stage2_inputs_seed{seed}.csv',index_label='Date')
    for model,pred in [('Base',bp),('Stacking',sp)]:
        for split,idx in [('Train',tr),('Test',te)]:
            actual=y.loc[idx].to_numpy(); p=pred.loc[idx].to_numpy()
            ape=100*np.abs(p-actual)/np.abs(actual)
            rows.append(dict(Seed=seed,Model=model,Split=split,N=len(idx),MAPE=ape.mean()))
            frames.append(pd.DataFrame(dict(Date=idx,Seed=seed,Model=model,Split=split,Actual=actual,Predicted=p,APE=ape)))
    print(f'Seed {seed}: two stages completed, each {EPOCHS} epochs')
metrics=pd.DataFrame(rows); predictions=pd.concat(frames,ignore_index=True)
history=pd.DataFrame(history)
metrics.to_csv(OUT/'metrics.csv',index=False)
predictions.to_csv(OUT/'predictions.csv',index=False)
history.to_csv(OUT/'training_history.csv',index=False)
test=metrics[metrics.Split=='Test']
comparison=test.pivot(index='Seed',columns='Model',values='MAPE').reindex(SEEDS)
comparison['Stacking_minus_Base_pp']=comparison.Stacking-comparison.Base
summary=test.groupby('Model').MAPE.agg(['mean','std','min','max']).reindex(['Base','Stacking'])
comparison.to_csv(OUT/'seed_comparison.csv'); summary.to_csv(OUT/'summary.csv')
assert len(metrics)==12 and len(history)==3*2*100
assert predictions.groupby(['Seed','Model','Split']).size().nunique()==2
print('2020 Test MAPE (%); negative paired difference means stacking improves')
display(comparison.round(4)); display(summary.round(4))
print('Train MAPE (%), based on in-sample predictions:')
display(metrics[metrics.Split=='Train'].pivot(index='Seed',columns='Model',values='MAPE').reindex(SEEDS).round(4))
''')
md('''## 预测曲线与训练过程

每个种子单独画出2020年预测；汇总值是三个种子的MAPE均值与样本标准差，并非先平均预测再算MAPE。配对差为堆叠MAPE减基准MAPE，负数表示堆叠改善。''')
code('''fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
for ax,seed in zip(axes,SEEDS):
    ax.plot(te,y.loc[te],color='black',lw=1.5,label='Actual')
    for name,color in [('Base','#0072B2'),('Stacking','#D55E00')]:
        g=predictions[(predictions.Seed==seed)&(predictions.Model==name)&(predictions.Split=='Test')]
        ax.plot(g.Date,g.Predicted,color=color,lw=1.2,label=f'{name}: MAPE {comparison.loc[seed,name]:.2f}%')
    ax.set(title=f'2020 test | Seed {seed}',ylabel='Price (dataset units)'); ax.legend(); ax.grid(alpha=.2)
fig.savefig(OUT/'predictions_2020.png',dpi=150); plt.close(fig)
display(Image(filename=str(OUT/'predictions_2020.png')))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
comparison[['Base','Stacking']].plot.bar(ax=axes[0],rot=0,color=['#0072B2','#D55E00'])
axes[0].set(ylabel='Test MAPE (%)',title='2020 | 100 epochs per stage')
for c in axes[0].containers: axes[0].bar_label(c,fmt='%.2f',padding=3)
axes[0].margins(y=.15)
for (seed,stage),g in history.groupby(['Seed','Stage']):
    axes[1].plot(g.Epoch,g.Train_MSE,label=f'{stage} / {seed}')
axes[1].set(xlabel='Epoch',ylabel='Standardized target MSE',yscale='log',title='Training convergence')
axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
fig.savefig(OUT/'comparison.png',dpi=150); plt.close(fig)
display(Image(filename=str(OUT/'comparison.png')))
monthly=predictions[predictions.Split=='Test'].copy()
monthly['Month']=monthly.Date.dt.strftime('%Y-%m')
monthly=monthly.groupby(['Month','Seed','Model']).APE.mean().unstack('Model')
monthly.to_csv(OUT/'monthly_mape.csv')
mean_delta=comparison.Stacking_minus_Base_pp.mean()
wins=int((comparison.Stacking_minus_Base_pp<0).sum())
print(f'堆叠模型在{wins}/3个种子中测试MAPE更低；平均配对差为{mean_delta:+.4f}个百分点。')
print('单个测试年、三个种子只能说明本次设定下的结果，不能据此证明其他年份也更好。')
''')
n.cells.extend(cells)
n.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nb.validate(n)
nb.write(n,'outputs/11_改变模型构建.ipynb')
