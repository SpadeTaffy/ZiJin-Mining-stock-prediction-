from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/12_减少预测的时间,扩大验证集的范围.ipynb')
n=nb.read(src,as_version=4);nb.write(n,'outputs/mlp_monthly_refit_12/12_before_refit.ipynb')
Path('outputs/mlp_monthly_refit_12/start_cell.txt').write_text(str(len(n.cells)))
setup=next(c.source for c in n.cells if c.cell_type=='code' and "MLP12_OUT" in c.source)
train=next(c.source for c in n.cells if c.cell_type=='code' and 'def fit_stage' in c.source)
helpers=train[:train.index('splitrows=[];')]
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 固定上次逐月种子，纳入验证数据后重训对比

本节不重新选择种子：直接读取上一实验每月selection.csv中的种子。每个测试月M，把原训练与前三个月验证数据合并为M月之前的全部可用数据，重新拟合标准化器，并用既定种子从头初始化、训练两个阶段各100轮。不是在旧权重上继续加训100轮。

保留原模型结构、特征定义、样本起点、训练超参数、同日目标与测试日期。第二阶段仍用第一阶段样本内预测，不用折外预测。每月测试数据绝不进入重训或标准化。

对比：A上次选好种子直接预测、不重训；B同样种子纳入验证数据后从头重训。只重训每月选中的一个候选，共12个堆叠模型，不重跑10候选验证选择。复用原选择记录，不根据本次测试结果更换种子。

该对比同时反映新增训练数据及其引起的标准化/优化路径变化。最新三个月用于学习是否有益，由相同日期的样本外误差检验。仍是使用同日输入的历史回测，不是月初提前预测全部月内输入未知的股价。
''')
setup=setup.replace('MLP12_OUT','MLP12_REFIT_OUT').replace('results/mlp_monthly_validation_12','results/mlp_monthly_refit_12')
# Load prior outputs either sibling result directory or actual repository fallback.
setup+='''
PRIOR=OUT.parent/'mlp_monthly_validation_12'
if not (PRIOR/'selection.csv').exists():PRIOR=ROOT/'results/mlp_monthly_validation_12'
prior_config=json.loads((PRIOR/'config.json').read_text())
assert prior_config['dataset_sha256']==config['dataset_sha256'] and prior_config['seeds']==SEEDS
selection=pd.read_csv(PRIOR/'selection.csv')
original=pd.read_csv(PRIOR/'stitched_predictions.csv',parse_dates=['Date'])
assert len(selection)==12 and selection.Month.is_unique and len(original)==len(whole)
assert pd.DatetimeIndex(original.Date).equals(whole)
assert np.allclose(original.Actual,y.loc[whole])
config.update(refit_after_selection=True,seed_selection='reuse previous monthly selection.csv without reranking',refit='fresh initialization with selected seed; 100 epochs per stage',selection_sha256=hashlib.sha256((PRIOR/'selection.csv').read_bytes()).hexdigest())
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
selection.to_csv(OUT/'reused_selection.csv',index=False)
display(selection[['Month','Selected_seed','Validation_MAPE']])
'''
code(setup)
code(helpers+'''frames=[];splits=[];train_metrics=[]
for row in selection.itertuples(index=False):
    month=row.Month;seed=int(row.Selected_seed);start=pd.Timestamp(month+'-01');end=start+pd.DateOffset(months=1)
    tr=x.index[x.index<start];te=x.index[(x.index>=start)&(x.index<end)]
    assert tr.max()<te.min() and str(te.min().date())==row.Test_start and str(te.max().date())==row.Test_end
    assert tr.max()<=pd.Timestamp(row.Validation_end) and tr.min()==x.index.min()
    known=x.loc[tr]
    base=fit_stage(known[BASE],tr,seed,'Base',month)
    bp=predict(base,known[BASE]);meta=known[RETURNS].copy();meta.insert(0,'base_prediction',bp)
    stack=fit_stage(meta,tr,seed,'Stacking',month)
    pred=stack_predict(base,stack,x.loc[te]);train_pred=predict(stack,meta)
    train_metrics.append(dict(Month=month,Seed=seed,**score(train_pred,tr)))
    frames.append(pd.DataFrame(dict(Date=te,Month=month,Seed=seed,Actual=y.loc[te].to_numpy(),Refit_prediction=pred.to_numpy())))
    for split,idx in [('Refit_train',tr),('Test',te)]:splits.append(dict(Month=month,Split=split,N=len(idx),Start=str(idx.min().date()),End=str(idx.max().date())))
    print(f'{month}: reused seed {seed}; refit {len(tr)} records; test {len(te)} records',flush=True)
refit=pd.concat(frames,ignore_index=True).sort_values('Date')
assert refit.Date.is_unique and pd.DatetimeIndex(refit.Date).equals(whole) and len(history)==2400
comparison=refit.merge(original[['Date','Month','Seed','Actual','Predicted']].rename(columns={'Predicted':'No_refit_prediction'}),on=['Date','Month','Seed'],suffixes=('','_old'),validate='one_to_one')
assert len(comparison)==len(whole) and np.allclose(comparison.Actual,comparison.Actual_old)
comparison=comparison.drop(columns='Actual_old')
comparison['Refit_APE']=100*np.abs(comparison.Refit_prediction-comparison.Actual)/comparison.Actual.abs()
comparison['No_refit_APE']=100*np.abs(comparison.No_refit_prediction-comparison.Actual)/comparison.Actual.abs()
monthly=comparison.groupby('Month').agg(Selected_seed=('Seed','first'),N=('Date','size'),No_refit_MAPE=('No_refit_APE','mean'),Refit_MAPE=('Refit_APE','mean'))
monthly['Refit_minus_no_refit_pp']=monthly.Refit_MAPE-monthly.No_refit_MAPE
monthly['Improved']=monthly.Refit_minus_no_refit_pp<0
summary=pd.DataFrame([dict(Method=label,Annual_daily_MAPE=comparison[ape].mean(),Monthly_equal_MAPE=monthly[col].mean(),Annual_MSE=np.mean((comparison[predcol]-comparison.Actual)**2)) for label,ape,col,predcol in [('No refit (previous selected models)','No_refit_APE','No_refit_MAPE','No_refit_prediction'),('Refit training plus validation','Refit_APE','Refit_MAPE','Refit_prediction')]])
assert np.isclose(summary.iloc[1].Annual_daily_MAPE,np.average(monthly.Refit_MAPE,weights=monthly.N))
old_monthly=pd.read_csv(PRIOR/'monthly_results.csv').set_index('Month')
assert np.allclose(monthly.No_refit_MAPE,old_monthly.Test_MAPE,atol=1e-5)
comparison.to_csv(OUT/'stitched_comparison.csv',index=False);monthly.to_csv(OUT/'monthly_comparison.csv');summary.to_csv(OUT/'summary.csv',index=False)
pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False);pd.DataFrame(train_metrics).to_csv(OUT/'refit_train_metrics.csv',index=False);pd.DataFrame(splits).to_csv(OUT/'splits.csv',index=False)
print('Monthly paired results (negative difference is improvement):');display(monthly.round(4))
print('Annual summary:');display(summary.round(4))
print(f'Refit improved {int(monthly.Improved.sum())}/12 months')
''')
code('''fig,axes=plt.subplots(2,1,figsize=(14,9),constrained_layout=True)
axes[0].plot(comparison.Date,comparison.Actual,color='black',lw=1.8,label='Actual')
axes[0].plot(comparison.Date,comparison.No_refit_prediction,color='#0072B2',lw=1.3,label='Monthly selected, no refit')
axes[0].plot(comparison.Date,comparison.Refit_prediction,color='#D55E00',lw=1.5,label='Same seed, refit including validation')
for d in pd.date_range('2020-02-01','2020-12-01',freq='MS'):axes[0].axvline(d,color='gray',alpha=.15)
axes[0].set(title='2020 monthly stitched comparison | Stacking | 100 epochs per stage',ylabel='Price (dataset units)');axes[0].legend();axes[0].grid(alpha=.2)
pos=np.arange(12)
axes[1].bar(pos-.18,monthly.No_refit_MAPE,width=.35,color='#0072B2',label='No refit')
axes[1].bar(pos+.18,monthly.Refit_MAPE,width=.35,color='#D55E00',label='Refit')
axes[1].set(xticks=pos,xticklabels=[m[-2:] for m in monthly.index],xlabel='2020 month',ylabel='Test MAPE (%)',title='Monthly paired error, same selected seed');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
fig.savefig(OUT/'refit_comparison.png',dpi=160);plt.close(fig);display(Image(filename=str(OUT/'refit_comparison.png')))
fig,axes=plt.subplots(4,3,figsize=(15,12),constrained_layout=True)
for ax,(month,g) in zip(axes.flat,comparison.groupby('Month')):
    ax.plot(g.Date,g.Actual,color='black',lw=1.5,label='Actual');ax.plot(g.Date,g.No_refit_prediction,color='#0072B2',lw=1.2,label='No refit');ax.plot(g.Date,g.Refit_prediction,color='#D55E00',lw=1.2,label='Refit')
    r=monthly.loc[month]
    ax.set(title=f'{month} | {r.No_refit_MAPE:.2f}% -> {r.Refit_MAPE:.2f}%');ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=30,labelsize=7)
axes[0,0].legend(fontsize=8);fig.savefig(OUT/'monthly_panels.png',dpi=140);plt.close(fig);display(Image(filename=str(OUT/'monthly_panels.png')))
''')
nb.validate(n);nb.write(n,'outputs/12_减少预测的时间,扩大验证集的范围.ipynb')
