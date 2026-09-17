from pathlib import Path
import nbformat as nb,hashlib
out=Path('outputs/mlp_crossing_h60_13')
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/13_加权MSE的训练.ipynb')
b=src.read_bytes();(out/'13_before.ipynb').write_bytes(b);(out/'original.sha256').write_text(hashlib.sha256(b).hexdigest());n=nb.reads(b.decode(),4)
(out/'start_cell.txt').write_text(str(len(n.cells)))
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 半衰期60：金铜MA20±3%双向穿越更新 vs 每月更新

沿用12的双向触发规则：金或铜任一的 `abs(P/MA20-1)` 在 `<3%` 与 `>=3%` 两种状态之间转换即触发；同日信号合并，无冷却期。持续超出不重复触发；从+4%直接到-4%仍属区间外，不额外推断盘中穿越。触发当天用旧模型，下条有效记录才使用新模型，训练标签最多到触发日。

两种更新方式均采用半衰期60条有效日度记录、两阶段加权MSE、等权标准化、16→32隐藏层、ReLU、Adam 0.01、全批量各100轮，从头重训。主对比为：M60 每月更新，E60 3%双向穿越更新。

**种子控制**：为延续本章月度60的口径，M60复用上一节结果；E60复用12中每次事件根据此前三个月验证MAPE选出的种子，仅将最终重训损失改为半衰期60，不重新运行候选选择。两种策略都沿用各自旧的等权候选选择记录，因此这是固定选种规则下的更新策略比较，不是同一种子贯穿全年的纯频率实验，也不是加权候选重新调参后的比较。

补充保留A月度等权、E等权双向触发的历史结果，便于区分损失变化和更新策略变化。年初初始化单列，事件次数和月度次数按相同口径报告。仍为2020年同日输入的历史开发实验。
''')
setup=Path('/private/tmp/cross13_1.txt').read_text().replace('MLP12_CROSSING_OUT','MLP13_CROSS60_OUT').replace('results/mlp_crossing_update_12','results/mlp_crossing_h60_13')
setup+='''
HALF_LIFE=60;METHOD='E60'
PRIOR=OUT.parent/'mlp_crossing_update_12'
if not (PRIOR/'updates.csv').exists():PRIOR=ROOT/'results/mlp_crossing_update_12'
MONTHLY=OUT.parent/'mlp_weighted_mse_13'
if not (MONTHLY/'predictions.csv').exists():MONTHLY=ROOT/'results/mlp_weighted_mse_13'
old_updates=pd.read_csv(PRIOR/'updates.csv',parse_dates=['Effective_date'])
prior_cfg=json.loads((PRIOR/'config.json').read_text());month_cfg=json.loads((MONTHLY/'config.json').read_text())
assert prior_cfg['dataset_sha256']==month_cfg['dataset_sha256']==config['dataset_sha256']
assert old_updates.Update_ID.tolist()==schedule.Update_ID.tolist()
assert old_updates.Effective_date.tolist()==schedule.Effective_date.tolist()
config.update(half_life=60,age_unit='valid observations',weight_normalization='sum(weights)',weighted_stages=['Base','Stacking'],scaler='unweighted StandardScaler',selection='reuse historical equal-MSE validation-selected event seeds; no reranking',source_updates_sha256=hashlib.sha256((PRIOR/'updates.csv').read_bytes()).hexdigest())
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
old_updates.to_csv(OUT/'reused_selection.csv',index=False)
'''
code(setup)
base=nb.read('outputs/mlp_half_life_comparison_13/13_before_half_life.ipynb',4)
helpers=next(c.source for c in base.cells if c.cell_type=='code' and 'def fit_stage' in c.source)
code(helpers)
code('''frames=[];splitrows=[];updates=[];jumps=[];previous_bundle=None
for j,row in enumerate(schedule.itertuples(index=False)):
    tag=row.Update_ID;effective=pd.Timestamp(row.Effective_date)
    stop=pd.Timestamp(schedule.iloc[j+1].Effective_date) if j+1<len(schedule) else pd.Timestamp('2021-01-01')
    tr=x.index[x.index<effective];applied=x.index[(x.index>=effective)&(x.index<stop)]
    prior=old_updates.loc[old_updates.Update_ID==tag].iloc[0];seed=int(prior.Selected_seed)
    assert tr.max()<applied.min() and pd.Timestamp(prior.Refit_train_end)==tr.max()
    if row.Kind=='Triggered':assert tr.max()==row.Trigger_date and x.index[x.index.get_loc(row.Trigger_date)+1]==effective
    known=x.loc[tr];base=fit_stage(known[BASE],tr,seed,'Base',tag)
    meta=known[RETURNS].copy();meta.insert(0,'base_prediction',predict(base,known[BASE]))
    stack=fit_stage(meta,tr,seed,'Stacking',tag)
    pred=stack_predict(base,stack,x.loc[applied])
    if previous_bundle is not None:
        same=x.loc[[effective]]
        old_p=float(stack_predict(*previous_bundle,same).iloc[0]);new_p=float(pred.iloc[0])
        jumps.append(dict(Update_ID=tag,Date=effective,Old_model_prediction=old_p,New_model_prediction=new_p,Switch_change=new_p-old_p,Abs_switch_change_pct=100*abs(new_p-old_p)/abs(old_p)))
    previous_bundle=(base,stack)
    frames.append(pd.DataFrame(dict(Date=applied,Update_ID=tag,Seed=seed,Actual=y.loc[applied].to_numpy(),E60=pred.to_numpy())))
    updates.append(dict(Update_ID=tag,Trigger_date=row.Trigger_date,Effective_date=effective,Seed=seed,Train_end=tr.max(),Train_N=len(tr),Applied_N=len(applied)))
    for split,idx in [('Train',tr),('Test',applied)]:splitrows.append(dict(Update_ID=tag,Split=split,N=len(idx),Start=idx.min(),End=idx.max()))
    if j%10==0:print(tag,'completed',flush=True)
comparison=pd.concat(frames,ignore_index=True)
assert pd.DatetimeIndex(comparison.Date).equals(whole) and comparison.Date.is_unique
assert len(history)==len(schedule)*200
cached=pd.read_csv(MONTHLY/'predictions.csv',parse_dates=['Date']).rename(columns={'B':'M60'})
old_event=pd.read_csv(PRIOR/'stitched_predictions.csv',parse_dates=['Date'])
assert pd.DatetimeIndex(cached.Date).equals(whole) and pd.DatetimeIndex(old_event.Date).equals(whole)
assert np.allclose(comparison.Actual,cached.Actual) and np.allclose(comparison.Actual,old_event.Actual)
comparison['A']=cached.A.to_numpy();comparison['M60']=cached.M60.to_numpy();comparison['E_equal']=old_event.Predicted.to_numpy()
initial=comparison.Update_ID.eq('initial')
assert np.allclose(comparison.loc[initial,'E60'],comparison.loc[initial,'M60'],rtol=0,atol=1e-5)
comparison['Month']=comparison.Date.dt.strftime('%Y-%m')
METHODS=['A','E_equal','M60','E60'];rows=[]
for month,g in comparison.groupby('Month'):
    r=dict(Month=month,N=len(g))
    for m in METHODS:
        err=g[m]-g.Actual;r[m+'_MAPE']=100*(err.abs()/g.Actual.abs()).mean();r[m+'_MSE']=(err**2).mean()
    r['E60_minus_M60_MAPE_pp']=r['E60_MAPE']-r['M60_MAPE'];r['E60_minus_M60_MSE']=r['E60_MSE']-r['M60_MSE'];rows.append(r)
monthly=pd.DataFrame(rows)
summary=pd.DataFrame([dict(Method=m,Annual_MAPE=100*((comparison[m]-comparison.Actual).abs()/comparison.Actual.abs()).mean(),Annual_MSE=((comparison[m]-comparison.Actual)**2).mean(),Monthly_equal_MAPE=monthly[m+'_MAPE'].mean(),Worst_month_MAPE=monthly[m+'_MAPE'].max(),Updates_excluding_initial=11 if m in ['A','M60'] else len(schedule)-1,Initializations=1) for m in METHODS])
counts=dict(Triggered_updates=len(schedule)-1,Initializations=1,Deployed_models=len(schedule),New_stage_trainings=len(schedule)*2,Epoch_records=len(history),Months_E60_MAPE_better_than_M60=int((monthly.E60_minus_M60_MAPE_pp<0).sum()))
for name,frame in [('predictions',comparison),('monthly_comparison',monthly),('summary',summary),('updates',pd.DataFrame(updates)),('splits',pd.DataFrame(splitrows)),('training_history',pd.DataFrame(history)),('event_switch_diagnostics',pd.DataFrame(jumps))]:frame.to_csv(OUT/f'{name}.csv',index=False)
(OUT/'update_counts.json').write_text(json.dumps(counts,indent=2))
display(summary.round(5));display(monthly[['Month','M60_MAPE','E60_MAPE','E60_minus_M60_MAPE_pp']].round(4));print(counts)
print('Verified initial predictions reproduce M60; event schedule matches 12; all training ends before application; same 271 test records.')
''')
code('''fig,axes=plt.subplots(3,1,figsize=(14,12),constrained_layout=True)
axes[0].plot(comparison.Date,comparison.Actual,color='black',lw=1.8,label='Actual')
for m,color in [('M60','#0072B2'),('E60','#009E73')]:axes[0].plot(comparison.Date,comparison[m],color=color,lw=1.3,label=m)
for dt in schedule.Effective_date.iloc[1:]:axes[0].axvline(dt,color='#009E73',alpha=.15,lw=.7)
axes[0].set(title='Half-life 60 | monthly vs MA20 +/-3% crossing updates | 2020',ylabel='Price');axes[0].legend()
for metal,col,color in [('Gold','gold_vs_ma20','#D59400'),('Copper','copper_vs_ma20','#0072B2')]:axes[1].plot(whole,x.loc[whole,col]*100,color=color,label=metal)
for v in [-3,3]:axes[1].axhline(v,color='gray',ls='--')
axes[1].set(ylabel='MA20 deviation (%)',title='Either metal crosses into or out of the band: update next record');axes[1].legend()
pos=np.arange(12)
axes[2].bar(pos-.18,monthly.M60_MAPE,width=.36,color='#0072B2',label='M60 monthly')
axes[2].bar(pos+.18,monthly.E60_MAPE,width=.36,color='#009E73',label='E60 crossing')
axes[2].set(xticks=pos,xticklabels=monthly.Month.str[-2:],ylabel='Test MAPE (%)',xlabel='2020 month');axes[2].legend()
for ax in axes:ax.grid(alpha=.2)
fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'comparison.png')))
''')
nb.validate(n);nb.write(n,'outputs/13_加权MSE的训练.ipynb')
s=Path('outputs/mlp_half_life_comparison_13/execute.py').read_text().replace('MLP13_GRID_OUT','MLP13_CROSS60_OUT').replace('mlp_half_life_comparison_13','mlp_crossing_h60_13');(out/'execute.py').write_text(s)
