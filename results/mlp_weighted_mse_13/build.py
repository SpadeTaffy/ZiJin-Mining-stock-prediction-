from pathlib import Path
import nbformat as nb
out=Path('outputs/mlp_weighted_mse_13')
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/13_加权MSE的训练.ipynb')
n=nb.read(src,4);nb.write(n,out/'13_before.ipynb')
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md(r'''# A 等权训练 vs B 加权 MSE（半衰期 60）

A：沿用12的逐月重训方案；复用前三个月验证所选的种子，把预测月之前全部可用数据纳入训练。B：同一训练集、同一种子、相同标准化器拟合方式及超参数，只更换训练损失。两阶段都采用对应损失，第一阶段冻结后以样本内预测训练第二阶段。

半衰期按 **60条有效日度记录**，不是60个日历日；数据一年有271条记录，因此也不直接称60个交易日。以每次训练集最新一条记录为年龄0，向前递增：

$$w_i=2^{-a_i/60},\qquad L_B=\frac{\sum_i w_i(\hat y_i-y_i)^2}{\sum_i w_i},\qquad L_A=\frac1N\sum_i(\hat y_i-y_i)^2.$$

损失在标准化目标上计算；归一化权重均值为1，避免仅因损失尺度变化改变优化强度。标准化仍等权拟合，避免同时更改预处理。保持16→32隐藏层、ReLU、Adam学习率0.01、全批量、两阶段各100轮，每月重新初始化。

为单独比较损失，不为B另选种子。复用A种子有利于配对控制，但本实验不能代表B独立调参后的最佳表现。以2020年相同日期普通MSE、MAPE比较；报告月度与全年结果，不以测试误差反向选择半衰期。

输入包含同日市场数据，这是历史同日股价拟合的样本外实验；2020年已被多次查看，不是新的盲测。
''')
setup=(out/'setup_source.txt').read_text()
setup=setup[:setup.index("PRIOR=OUT.parent")]
setup=setup.replace('MLP12_REFIT_OUT','MLP13_OUT').replace('results/mlp_monthly_refit_12','results/mlp_weighted_mse_13')
setup+='''
HALF_LIFE=60
PRIOR=OUT.parent/'mlp_monthly_refit_12'
if not (PRIOR/'reused_selection.csv').exists(): PRIOR=ROOT/'results/mlp_monthly_refit_12'
selection=pd.read_csv(PRIOR/'reused_selection.csv')
previous=pd.read_csv(PRIOR/'stitched_comparison.csv',parse_dates=['Date'])
prior_config=json.loads((PRIOR/'config.json').read_text())
assert prior_config['dataset_sha256']==config['dataset_sha256']
config.update(half_life=HALF_LIFE,age_unit='valid observations',weight_normalization='sum(weights)',weighted_stages=['Base','Stacking'],scaler='unweighted StandardScaler',refit_after_selection=True,selection='reuse A seeds; no B reranking',selection_sha256=hashlib.sha256((PRIOR/'reused_selection.csv').read_bytes()).hexdigest())
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
selection.to_csv(OUT/'reused_selection.csv',index=False)
'''
code(setup)
helpers=(out/'helpers_source.txt').read_text()
helpers=helpers.replace("    opt=torch.optim.Adam", "    ages=np.arange(len(train_idx)-1,-1,-1)\n    weights=np.ones(len(train_idx)) if METHOD=='A' else 2.0**(-ages/HALF_LIFE)\n    wt=torch.tensor(weights/weights.mean(),dtype=torch.float32).reshape(-1,1)\n    opt=torch.optim.Adam")
helpers=helpers.replace('loss=((model(xt)-yt)**2).mean()','loss=(wt*(model(xt)-yt)**2).mean()')
helpers=helpers.replace('dict(Month=month,Seed=seed,Stage=stage,Epoch=epoch,Train_MSE=mse,','dict(Method=METHOD,Month=month,Seed=seed,Stage=stage,Epoch=epoch,Weighted_MSE=float(np.average((p-y.loc[train_idx].to_numpy())**2,weights=weights)),Train_MSE=mse,')
helpers=helpers.replace("OUT/f'{month}_{stage}_seed{seed}.pt'","OUT/f'{METHOD}_{month}_{stage}_seed{seed}.pt'")
code(helpers)
code('''frames=[];splits=[];weight_rows=[]
for row in selection.itertuples(index=False):
    month=row.Month;seed=int(row.Selected_seed);start=pd.Timestamp(month+'-01');end=start+pd.DateOffset(months=1)
    tr=x.index[x.index<start];te=x.index[(x.index>=start)&(x.index<end)]
    assert tr.max()<te.min()
    ages=np.arange(len(tr)-1,-1,-1);w=2.0**(-ages/HALF_LIFE)
    assert np.isclose(w[-61]/w[-1],.5)
    weight_rows.append(dict(Month=month,N=len(tr),Effective_sample_size=w.sum()**2/(w*w).sum(),Recent60_weight_share=w[-60:].sum()/w.sum()))
    for split,idx in [('Train',tr),('Test',te)]:splits.append(dict(Month=month,Split=split,N=len(idx),Start=str(idx.min().date()),End=str(idx.max().date())))
    result=pd.DataFrame(dict(Date=te,Month=month,Seed=seed,Actual=y.loc[te].to_numpy()))
    for METHOD in ['A','B']:
        known=x.loc[tr]
        base=fit_stage(known[BASE],tr,seed,'Base',month)
        meta=known[RETURNS].copy();meta.insert(0,'base_prediction',predict(base,known[BASE]))
        stack=fit_stage(meta,tr,seed,'Stacking',month)
        result[METHOD]=stack_predict(base,stack,x.loc[te]).to_numpy()
    frames.append(result)
    print(month,'completed',flush=True)
comparison=pd.concat(frames,ignore_index=True)
assert comparison.Date.is_unique and pd.DatetimeIndex(comparison.Date).equals(whole)
assert np.allclose(comparison.A,previous.set_index('Date').loc[comparison.Date,'Refit_prediction'],rtol=0,atol=1e-6), 'A must reproduce prior monthly refit'
rows=[]
for month,g in comparison.groupby('Month'):
    r=dict(Month=month,N=len(g))
    for method in ['A','B']:
        err=g[method]-g.Actual
        r[method+'_MAPE']=100*(err.abs()/g.Actual.abs()).mean();r[method+'_MSE']=(err**2).mean()
    r['B_minus_A_MAPE_pp']=r['B_MAPE']-r['A_MAPE'];r['B_minus_A_MSE']=r['B_MSE']-r['A_MSE'];rows.append(r)
monthly=pd.DataFrame(rows)
summary=pd.DataFrame([dict(Method=method,Annual_MSE=((comparison[method]-comparison.Actual)**2).mean(),Annual_MAPE=100*((comparison[method]-comparison.Actual).abs()/comparison.Actual.abs()).mean(),Monthly_equal_MAPE=monthly[method+'_MAPE'].mean()) for method in ['A','B']])
assert len(history)==4800
for name,frame in [('predictions',comparison),('monthly_comparison',monthly),('summary',summary),('splits',pd.DataFrame(splits)),('weight_diagnostics',pd.DataFrame(weight_rows)),('training_history',pd.DataFrame(history))]:frame.to_csv(OUT/f'{name}.csv',index=False)
display(summary.round(5));display(monthly.round(5));display(pd.DataFrame(weight_rows).round(4))
print('B improves monthly MAPE:',int((monthly.B_minus_A_MAPE_pp<0).sum()),'/12')
print('A reproduction verified against 12; train/test chronology and half-life verified.')
''')
code('''fig,axes=plt.subplots(2,1,figsize=(14,9),constrained_layout=True)
axes[0].plot(comparison.Date,comparison.Actual,color='black',label='Actual',lw=1.7)
for method,color in [('A','#0072B2'),('B','#D55E00')]:
    axes[0].plot(comparison.Date,comparison[method],label=method+(' equal MSE' if method=='A' else ' weighted MSE, half-life 60'),color=color,lw=1.3)
axes[0].set(title='Monthly refit: equal vs recency-weighted training | 2020',ylabel='Price');axes[0].legend();axes[0].grid(alpha=.2)
p=np.arange(12)
axes[1].bar(p-.18,monthly.A_MAPE,width=.36,label='A equal MSE',color='#0072B2');axes[1].bar(p+.18,monthly.B_MAPE,width=.36,label='B weighted MSE',color='#D55E00')
axes[1].set(xticks=p,xticklabels=monthly.Month.str[-2:],ylabel='Test MAPE (%)',xlabel='2020 month');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'comparison.png')))
''')
nb.write(n,'outputs/13_加权MSE的训练.ipynb')
