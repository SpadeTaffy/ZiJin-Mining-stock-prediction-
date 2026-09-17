from pathlib import Path
import nbformat as nb
p=Path('outputs/13_加权MSE的训练.ipynb');n=nb.read(p,4)
out=Path('outputs/mlp_half_life_comparison_13');nb.write(n,out/'13_before_half_life.ipynb')
(out/'start_cell.txt').write_text(str(len(n.cells)))
codes=[c.source for c in n.cells if c.cell_type=='code']
n.cells.append(nb.v4.new_markdown_cell('''# 扩展比较：半衰期30、60、120、240与等权A

延续上一节控制条件，新增30、120、240条有效日度记录的半衰期。复用A和60的已验证预测；每月仍使用同一既定种子、月前全部历史、两阶段各100轮、等权标准化，仅改变两阶段损失的时间权重。B不单独重新选择种子。

统一用同一271条2020年测试记录计算普通MSE与MAPE，同时报告12个月等权MAPE、相对A改善月份数、最差月份MAPE。不同方案的加权训练损失不可直接横向作为优劣依据。2020年结果仅用于探索，不能据此证明最优半衰期能泛化到未参与选择的年份。
'''))
setup=codes[0].replace('MLP13_OUT','MLP13_GRID_OUT').replace('results/mlp_weighted_mse_13','results/mlp_half_life_comparison_13')
setup+='''
config.update(half_lives=[30,60,120,240],reused_methods=['A','H60'])
config.pop('half_life',None)
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
CACHED=OUT.parent/'mlp_weighted_mse_13'
if not (CACHED/'predictions.csv').exists(): CACHED=ROOT/'results/mlp_weighted_mse_13'
cached_config=json.loads((CACHED/'config.json').read_text())
assert cached_config['dataset_sha256']==config['dataset_sha256']
assert cached_config['selection_sha256']==config['selection_sha256']
comparison=pd.read_csv(CACHED/'predictions.csv',parse_dates=['Date']).rename(columns={'B':'H60'})
assert pd.DatetimeIndex(comparison.Date).equals(whole)
assert np.allclose(comparison.Actual,y.loc[whole])
config['cached_predictions_sha256']=hashlib.sha256((CACHED/'predictions.csv').read_bytes()).hexdigest()
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
'''
n.cells.append(nb.v4.new_code_cell(setup))
n.cells.append(nb.v4.new_code_cell(codes[1]))
n.cells.append(nb.v4.new_code_cell('''splits=[];weight_rows=[]
for HALF_LIFE in [30,120,240]:
    METHOD=f'H{HALF_LIFE}';frames=[]
    for row in selection.itertuples(index=False):
        month=row.Month;seed=int(row.Selected_seed);start=pd.Timestamp(month+'-01');end=start+pd.DateOffset(months=1)
        tr=x.index[x.index<start];te=x.index[(x.index>=start)&(x.index<end)]
        assert tr.max()<te.min()
        ages=np.arange(len(tr)-1,-1,-1);w=2.0**(-ages/HALF_LIFE)
        assert np.isclose(w[-HALF_LIFE-1]/w[-1],.5)
        weight_rows.append(dict(Method=METHOD,Month=month,N=len(tr),Effective_sample_size=w.sum()**2/(w*w).sum(),Recent60_weight_share=w[-60:].sum()/w.sum()))
        for split,idx in [('Train',tr),('Test',te)]:splits.append(dict(Method=METHOD,Month=month,Split=split,N=len(idx),Start=str(idx.min().date()),End=str(idx.max().date())))
        known=x.loc[tr];base=fit_stage(known[BASE],tr,seed,'Base',month)
        meta=known[RETURNS].copy();meta.insert(0,'base_prediction',predict(base,known[BASE]))
        stack=fit_stage(meta,tr,seed,'Stacking',month)
        frames.append(pd.DataFrame({'Date':te,METHOD:stack_predict(base,stack,x.loc[te]).to_numpy()}))
    pred=pd.concat(frames,ignore_index=True)
    comparison=comparison.merge(pred,on='Date',validate='one_to_one')
    print(METHOD,'12 months completed',flush=True)
assert len(history)==7200 and len(comparison)==271
METHODS=['A','H30','H60','H120','H240']
rows=[]
for month,g in comparison.groupby('Month'):
    for method in METHODS:
        err=g[method]-g.Actual
        rows.append(dict(Month=month,Method=method,N=len(g),MAPE=100*(err.abs()/g.Actual.abs()).mean(),MSE=(err**2).mean()))
monthly=pd.DataFrame(rows)
base_month=monthly[monthly.Method=='A'].set_index('Month')
monthly['MAPE_minus_A_pp']=monthly.MAPE-monthly.Month.map(base_month.MAPE)
monthly['MSE_minus_A']=monthly.MSE-monthly.Month.map(base_month.MSE)
rows=[]
for method in METHODS:
    g=monthly[monthly.Method==method];err=comparison[method]-comparison.Actual;worst=g.loc[g.MAPE.idxmax()]
    rows.append(dict(Method=method,Annual_MSE=(err**2).mean(),Annual_MAPE=100*(err.abs()/comparison.Actual.abs()).mean(),Monthly_equal_MAPE=g.MAPE.mean(),Months_MAPE_better_than_A=int((g.MAPE_minus_A_pp<0).sum()),Months_MSE_better_than_A=int((g.MSE_minus_A<0).sum()),Worst_month=worst.Month,Worst_month_MAPE=worst.MAPE))
summary=pd.DataFrame(rows)
summary['MSE_reduction_vs_A_pct']=100*(1-summary.Annual_MSE/summary.iloc[0].Annual_MSE)
summary['MAPE_reduction_vs_A_pp']=summary.iloc[0].Annual_MAPE-summary.Annual_MAPE
for method,old in [('A','A'),('H60','B')]:
    cached_summary=pd.read_csv(CACHED/'summary.csv').set_index('Method')
    assert np.isclose(summary.set_index('Method').loc[method,'Annual_MAPE'],cached_summary.loc[old,'Annual_MAPE'])
for name,frame in [('predictions',comparison),('monthly_comparison',monthly),('summary',summary),('splits',pd.DataFrame(splits)),('weight_diagnostics',pd.DataFrame(weight_rows)),('training_history',pd.DataFrame(history))]:frame.to_csv(OUT/f'{name}.csv',index=False)
display(summary.round(5));display(monthly.pivot(index='Month',columns='Method',values='MAPE')[METHODS].round(4))
print('Verified: identical test dates and seeds; strict train/test chronology; half-life ratios; cached A and H60 metrics; 7200 new training records.')
'''))
n.cells.append(nb.v4.new_code_cell('''colors={'A':'#444444','H30':'#CC79A7','H60':'#D55E00','H120':'#0072B2','H240':'#009E73'}
fig,axes=plt.subplots(3,1,figsize=(14,13),constrained_layout=True)
axes[0].plot(comparison.Date,comparison.Actual,color='black',lw=2,label='Actual')
for method in METHODS:axes[0].plot(comparison.Date,comparison[method],color=colors[method],lw=1.1,label=method,alpha=.85)
axes[0].set(title='2020 monthly refit | equal MSE and half-lives 30 / 60 / 120 / 240',ylabel='Price');axes[0].legend(ncol=6)
p=np.arange(12)
for j,method in enumerate(METHODS):
    g=monthly[monthly.Method==method]
    axes[1].bar(p+(j-2)*.16,g.MAPE,width=.16,color=colors[method],label=method)
axes[1].set(xticks=p,xticklabels=g.Month.str[-2:],ylabel='Monthly test MAPE (%)',xlabel='2020 month');axes[1].legend(ncol=5)
axes[2].bar(summary.Method,summary.Annual_MAPE,color=[colors[m] for m in METHODS])
for i,v in enumerate(summary.Annual_MAPE):axes[2].text(i,v+.08,f'{v:.3f}%',ha='center')
axes[2].set(ylabel='Annual test MAPE (%)',ylim=(0,summary.Annual_MAPE.max()*1.16))
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'comparison.png')))
'''))
nb.validate(n);nb.write(n,p)
s=Path('outputs/mlp_weighted_mse_13/execute.py').read_text().replace('MLP13_OUT','MLP13_GRID_OUT').replace('outputs/mlp_weighted_mse_13','outputs/mlp_half_life_comparison_13').replace('n.cells[0:]','n.cells[int(Path("outputs/mlp_half_life_comparison_13/start_cell.txt").read_text()):]')
(out/'execute.py').write_text(s)
