from pathlib import Path
import nbformat as nb
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
original=ROOT/'notebooks/04_采用神经网络模型/10_再退回原始模型,确认近日涨幅的作用.ipynb'
n=nb.read(original,as_version=4)
cells=[]
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))
md('''# 基准模型与金铜涨幅模型：2020年对比

按此前实验口径，“2019年前”采用**截至2019年12月31日（含2019年）**的历史数据训练，2020年全部记录测试。比较：

- Base_100：原7个水平变量，不含金铜涨幅，100轮。
- Base_400：相同基准模型，沿同一训练路径继续至400轮。
- Full_100：原7变量＋金铜5/20条记录累计涨幅，100轮。

三组统一为ReLU、16/32隐藏层、线性输出、Adam学习率0.01、全批量、seed42，输入和目标仅用训练集拟合StandardScaler。三组使用相同日期，统一去掉计算20期涨幅所需的前20条记录。因输入维度不同，两类模型不能视为完全相同的初始化权重；本次是单随机种子的探索性对比。

沿用原Dataset.pkl的**同日目标股价**，并非下一日收益率预测。涨幅按相邻数据记录计算，不保证是A股交易日；这里不重新修改原数据对齐口径。本次不根据2020结果选择训练轮数。
''')
code('''from pathlib import Path
import os, json, hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display, Image
ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/pickle/Dataset.pkl').exists()),Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-'))
OUT=Path(os.environ.get('MLP10_OUT',str(ROOT/'results/mlp_base_vs_returns_10')))
OUT.mkdir(parents=True,exist_ok=True)
SEED=42; LR=0.01
torch.set_num_threads(1)
dataset=ROOT/'src/pickle/Dataset.pkl'
d=pd.read_pickle(dataset); raw=d['X'].copy(); target=d['Y'].copy()
assert raw.index.equals(target.index) and raw.shape[1]==7
assert raw.index.is_unique and raw.index.is_monotonic_increasing
BASE=list(raw.columns); x=raw.copy()
for metal,col in [('gold','Gold Futures Price'),('copper','Copper Futures Price')]:
    for k in [5,20]: x[f'{metal}_{k}']=raw[col]/raw[col].shift(k)-1
x=x.dropna(); FULL=list(x.columns)
y=pd.Series(np.asarray(target).reshape(-1),index=raw.index,name='Actual').loc[x.index]
tr=x.index[x.index.year<2020]; te=x.index[x.index.year==2020]
assert len(tr)>0 and len(te)>0 and tr.max()<te.min()
assert np.isfinite(x.to_numpy()).all() and np.isfinite(y).all() and (y>0).all()
splits=pd.DataFrame([{'Split':s,'N':len(idx),'Start':str(idx.min().date()),'End':str(idx.max().date())} for s,idx in [('Train',tr),('Test',te)]])
splits.to_csv(OUT/'splits.csv',index=False)
config={'seed':SEED,'lr':LR,'hidden':[16,32],'activation':'ReLU','batch':'full','target':'original same-date price','base_features':BASE,'full_features':FULL,'train_end':str(tr.max()),'test_year':2020,'common_samples':True,'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'torch_version':str(torch.__version__)}
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
display(splits); display(pd.DataFrame({'Base features':BASE}))
''')
code('''sy=StandardScaler().fit(y.loc[tr].to_numpy().reshape(-1,1))
yt=torch.tensor(sy.transform(y.loc[tr].to_numpy().reshape(-1,1)),dtype=torch.float32)
metric_rows=[]; frames=[]; histories=[]
for family,cols,checkpoints in [('Base',BASE,[100,400]),('Full',FULL,[100])]:
    sx=StandardScaler().fit(x.loc[tr,cols])
    xt=torch.tensor(sx.transform(x.loc[tr,cols]),dtype=torch.float32)
    torch.manual_seed(SEED)
    model=nn.Sequential(nn.Linear(len(cols),16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,max(checkpoints)+1):
        model.train(); optimizer.zero_grad()
        loss=((model(xt)-yt)**2).mean(); assert torch.isfinite(loss)
        loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad(): mse=((model(xt)-yt)**2).mean().item()
        histories.append({'Family':family,'Epoch':epoch,'Train_MSE_standardized':mse})
        if epoch not in checkpoints: continue
        name=f'{family}_{epoch}'
        torch.save({'state_dict':model.state_dict(),'features':cols,'epoch':epoch,'seed':SEED,
                    'train_start':str(tr.min()),'train_end':str(tr.max()),
                    'sx_mean':sx.mean_.tolist(),'sx_scale':sx.scale_.tolist(),
                    'sy_mean':sy.mean_.tolist(),'sy_scale':sy.scale_.tolist()},OUT/f'{name}.pt')
        for split,idx in [('Train',tr),('Test',te)]:
            with torch.no_grad():
                z=model(torch.tensor(sx.transform(x.loc[idx,cols]),dtype=torch.float32)).numpy()
            pred=sy.inverse_transform(z).ravel(); actual=y.loc[idx].to_numpy()
            assert np.isfinite(pred).all()
            err=pred-actual
            frames.append(pd.DataFrame({'Date':idx,'Model':name,'Split':split,'Actual':actual,'Predicted':pred,'APE':100*np.abs(err)/actual,'PE':100*err/actual}))
            metric_rows.append({'Model':name,'Split':split,'N':len(idx),'MAPE':100*np.mean(np.abs(err)/actual),'MAE':np.abs(err).mean(),'RMSE':np.sqrt(np.mean(err**2)),'MPE':100*np.mean(err/actual)})
metrics=pd.DataFrame(metric_rows); predictions=pd.concat(frames,ignore_index=True); history=pd.DataFrame(histories)
metrics.to_csv(OUT/'metrics.csv',index=False); predictions.to_csv(OUT/'predictions.csv',index=False); history.to_csv(OUT/'training_history.csv',index=False)
order=['Base_100','Base_400','Full_100']
comparison=metrics.pivot(index='Model',columns='Split',values='MAPE').loc[order]
comparison.columns=[f'{c}_MAPE_percent' for c in comparison.columns]
display(comparison.round(4)); display(metrics[metrics.Split=='Test'].set_index('Model').loc[order].round(4))
# 同数据、同模型的Full_100应复现08中的已存结果；该检查不用于训练。
old_path=ROOT/'results/mlp_epochs100_400_annual_08/predictions.csv'
old_config=ROOT/'results/mlp_epochs100_400_annual_08/config.json'
if old_path.exists() and old_config.exists():
    assert json.loads(old_config.read_text())['dataset_sha256']==config['dataset_sha256']
    old=pd.read_csv(old_path,parse_dates=['Date'])
    old=old[(old.Year==2020)&(old.Split=='Test')&(old.Epoch==100)].sort_values('Date')
    current=predictions[(predictions.Split=='Test')&(predictions.Model=='Full_100')].sort_values('Date')
    assert pd.DatetimeIndex(old.Date).equals(pd.DatetimeIndex(current.Date))
    assert np.allclose(old.Actual,current.Actual)
    delta=float(np.max(np.abs(old.Predicted.to_numpy()-current.Predicted.to_numpy())))
    assert delta<1e-4, f'与原100轮结果不一致: {delta}'
    print(f'已复核：Full_100复现08结果，预测最大绝对差={delta:.8g}')
''')
md('''## 2020年预测曲线、误差与训练收敛

黑线为原目标股价。MAPE为平均绝对百分比误差，越小越好；MPE保留符号，负值表示平均低估。下面同时保留静态图和可缩放、悬停的交互图。''')
code('''test=predictions[predictions.Split=='Test'].copy()
colors={'Base_100':'#0072B2','Base_400':'#D55E00','Full_100':'#009E73'}
fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
axes[0].plot(te,y.loc[te],color='black',lw=1.7,label='Actual')
for name in order:
    g=test[test.Model==name]
    axes[0].plot(g.Date,g.Predicted,color=colors[name],lw=1.3,label=name)
    axes[1].plot(g.Date,g.PE,color=colors[name],lw=1.1,label=name)
axes[0].set(title='2020 | Train through 2019 | Base vs metal returns',ylabel='Target price (dataset units)')
axes[1].set(ylabel='Signed prediction error (%)',xlabel='Date'); axes[1].axhline(0,color='gray',ls='--')
for ax in axes: ax.legend(); ax.grid(alpha=.2)
fig.savefig(OUT/'test_comparison.png',dpi=150); plt.close(fig); display(Image(filename=str(OUT/'test_comparison.png')))
fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
comparison[['Train_MAPE_percent','Test_MAPE_percent']].plot.bar(ax=axes[0],rot=0,color=['#7f8c8d','#0072B2'])
axes[0].set(title='Train / test MAPE',ylabel='MAPE (%)',xlabel='Model'); axes[0].margins(y=.2)
for c in axes[0].containers: axes[0].bar_label(c,fmt='%.2f',padding=3)
for family,g in history.groupby('Family'): axes[1].plot(g.Epoch,g.Train_MSE_standardized,label=family)
axes[1].set(title='Training convergence',xlabel='Epoch',ylabel='Standardized target MSE',yscale='log'); axes[1].legend(); axes[1].grid(alpha=.2)
fig.savefig(OUT/'metrics_convergence.png',dpi=150); plt.close(fig); display(Image(filename=str(OUT/'metrics_convergence.png')))
actual=pd.DataFrame({'Date':te,'Model':'Actual','Price':y.loc[te].to_numpy()})
lines=pd.concat([actual,test[['Date','Model','Predicted']].rename(columns={'Predicted':'Price'})],ignore_index=True)
fig=px.line(lines,x='Date',y='Price',color='Model',render_mode='svg',color_discrete_map={'Actual':'black',**colors},title='2020: Actual / Base 100 / Base 400 / Full 100',height=570)
fig.update_layout(hovermode='x unified',xaxis_rangeslider_visible=True)
fig.write_html(OUT/'test_comparison.html',include_plotlyjs=True)
display(fig)
monthly=test.assign(Month=test.Date.dt.strftime('%Y-%m')).groupby(['Month','Model']).APE.mean().unstack()[order]
monthly.to_csv(OUT/'monthly_mape.csv');display(monthly.round(3))
''')
md('''## 如何解释本次对比

Base_100与Base_400比较的是同一训练路径上的训练时长。Base_100与Full_100比较的是有无金铜5/20期涨幅的特征方案。只凭一次年度、单种子的误差不能证明涨幅变量必然有用或无用，也不能推导经济因果。

开头保留的实验设想属于待检验假设：单个金价变量的条件响应为负，不等于黄金实际经济作用为负；横盘时涨幅变量为零不代表模型整体预测必须不变。此次先检验删除涨幅后预测曲线和误差如何变化。''')
n.cells.extend(cells)
nb.write(n,Path('outputs')/original.name)
print(Path('outputs')/original.name)
