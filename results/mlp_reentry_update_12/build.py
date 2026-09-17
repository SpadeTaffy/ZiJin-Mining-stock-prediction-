from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/12_减少预测的时间,扩大验证集的范围.ipynb')
n=nb.read(src,as_version=4);nb.write(n,'outputs/mlp_reentry_update_12/12_before_reentry.ipynb')
Path('outputs/mlp_reentry_update_12/start_cell.txt').write_text(str(len(n.cells)))
setup=next(c.source for c in n.cells if c.cell_type=='code' and 'MLP12_OUT' in c.source)
train=next(c.source for c in n.cells if c.cell_type=='code' and 'splitrows=[];valrows=[]' in c.source)
helpers=train[:train.index('splitrows=[];')].replace('Month=month','Update_ID=month')
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 金铜MA20偏离回归正常后触发更新：次日选种子并重训

沿用2020年、既定10个新种子、两阶段Stacking、每阶段100轮和此前特征。**触发改为回归正常**：对金、铜分别观察 `abs(P/MA20−1)`；达到或超过3%为偏离状态，后来首次回到严格小于3%时触发。一直正常不触发，一直超标不反复触发；若再次超标后又回归，可再次触发。两金属独立判断，同一日触发合并为一次更新，即使另一金属尚未正常也更新。

在触发日收盘后确认信号，在**下一条有效数据记录开始时**执行更新并应用新模型；本数据含非交易日记录，不能将“下一条记录”直接称作下一A股交易日。触发日仍由旧模型预测。没有冷却期或阈值缓冲带。

更新流程：以生效日之前最近3个日历月为验证期，更早全部有效数据为候选训练期；十个种子各训练两阶段100轮，选验证MAPE最低者；合并训练与验证，再用所选种子从头训练两阶段各100轮。所有标准化仅拟合对应训练区间，新模型从生效日开始使用，直到下次更新。测试当天真实股价不得进入当天选种子或训练。

年初2020-01-01按同样流程建立初始模型，单独计数；年度触发次数、实际更新次数、金/铜各自次数均记录。如果年末触发后数据中没有下一条2020记录，则记录为待执行、不计入年内实际更新。

对照上一节“每月验证选种子＋合并验证数据重训”。两个方案使用同样种子与训练设置，区别在更新时点和相应验证窗口。

仍沿用同日金铜等输入，因此是收盘输入条件下的历史样本外回测，而非在当天开始时已知当天股价预测；更新训练本身只用生效日前的数据。该年曾参与开发，结论不能当作全新盲测。
''')
setup=setup.replace('MLP12_OUT','MLP12_REENTRY_OUT').replace('results/mlp_monthly_validation_12','results/mlp_reentry_update_12')
setup+='''
THRESHOLD=.03
config.update(trigger='individual metal abs(MA20 deviation) re-enters below 3% after at least 3%; same-date events merged',threshold=THRESHOLD,effective='next available record',refit_after_selection=True,validation_window='three calendar months immediately before effective date',cooldown=None)
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
# 在完整历史计算状态，使2020年初状态继承自2019年；只执行2020年触发。
deviations=x[['gold_vs_ma20','copper_vs_ma20']]
outside=deviations.abs().ge(THRESHOLD)
reentry=outside.shift(1,fill_value=False)&~outside
trigger_dates=reentry.index[(reentry.any(axis=1))&(reentry.index.year==2020)]
eventrows=[]
for day in trigger_dates:
    pos=x.index.get_loc(day);nextday=x.index[pos+1] if pos+1<len(x) else pd.NaT
    executable=pd.notna(nextday) and nextday.year==2020
    eventrows.append(dict(Trigger_date=day,Effective_date=nextday,Gold_trigger=bool(reentry.loc[day,'gold_vs_ma20']),Copper_trigger=bool(reentry.loc[day,'copper_vs_ma20']),Gold_deviation_percent=100*deviations.loc[day,'gold_vs_ma20'],Copper_deviation_percent=100*deviations.loc[day,'copper_vs_ma20'],Executable_in_2020=executable))
events=pd.DataFrame(eventrows)
events.to_csv(OUT/'trigger_events.csv',index=False)
schedule=[dict(Update_ID='initial',Kind='Initial',Trigger_date=pd.NaT,Effective_date=whole.min())]
for i,r in enumerate(events[events.Executable_in_2020].itertuples(index=False),1):
    schedule.append(dict(Update_ID=f'update_{i:02d}',Kind='Triggered',Trigger_date=r.Trigger_date,Effective_date=r.Effective_date))
schedule=pd.DataFrame(schedule).sort_values('Effective_date').reset_index(drop=True)
assert schedule.Effective_date.is_unique
print(f'触发日 {len(events)} 次；年内计划更新 {len(schedule)-1} 次；另有年初初始化1次。')
display(events)
'''
code(setup)
code(helpers+'''all_rankings=[];updates=[];frames=[];splitrows=[]
for j,row in enumerate(schedule.itertuples(index=False)):
    tag=row.Update_ID;effective=pd.Timestamp(row.Effective_date)
    stop=pd.Timestamp(schedule.iloc[j+1].Effective_date) if j+1<len(schedule) else pd.Timestamp('2021-01-01')
    val_start=effective-pd.DateOffset(months=VALIDATION_MONTHS)
    tr=x.index[x.index<val_start];va=x.index[(x.index>=val_start)&(x.index<effective)];refit_idx=x.index[x.index<effective]
    applied=x.index[(x.index>=effective)&(x.index<stop)]
    assert len(tr)>0 and len(va)>0 and len(applied)>0 and tr.max()<va.min() and va.max()<effective
    assert refit_idx.max()<effective and tr.union(va).equals(refit_idx)
    if row.Kind=='Triggered':
        assert refit_idx.max()==row.Trigger_date
        assert x.index[x.index.get_loc(row.Trigger_date)+1]==effective
    known=x.loc[refit_idx];candidates=[]
    for i,seed in enumerate(SEEDS):
        base=fit_stage(known[BASE],tr,seed,'CandidateBase',tag)
        bp=predict(base,known[BASE]);meta=known[RETURNS].copy();meta.insert(0,'base_prediction',bp)
        stack=fit_stage(meta,tr,seed,'CandidateStacking',tag)
        vp=predict(stack,meta)
        candidates.append(dict(Update_ID=tag,Seed=seed,Candidate_order=i,**score(vp,va)))
    ranking=pd.DataFrame(candidates).sort_values(['MAPE','Candidate_order']).reset_index(drop=True)
    ranking['Rank']=np.arange(1,11);chosen=int(ranking.iloc[0].Seed);ranking['Selected']=ranking.Seed.eq(chosen)
    all_rankings.append(ranking)
    decision=dict(Update_ID=tag,Kind=row.Kind,Trigger_date=None if pd.isna(row.Trigger_date) else str(row.Trigger_date.date()),Effective_date=str(effective.date()),Selected_seed=chosen,Validation_MAPE=float(ranking.iloc[0].MAPE),Candidate_train_end=str(tr.max().date()),Validation_start=str(va.min().date()),Validation_end=str(va.max().date()),Refit_train_end=str(refit_idx.max().date()),Refit_N=len(refit_idx))
    # 固定选择记录后，从头在训练+验证上重训；不读生效日真实股价。
    (OUT/f'{tag}_selection.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    final_base=fit_stage(known[BASE],refit_idx,chosen,'RefitBase',tag)
    bp=predict(final_base,known[BASE]);meta=known[RETURNS].copy();meta.insert(0,'base_prediction',bp)
    final_stack=fit_stage(meta,refit_idx,chosen,'RefitStacking',tag)
    pred=stack_predict(final_base,final_stack,x.loc[applied])
    updates.append(dict(**decision,Applied_end=str(applied.max().date()),Applied_N=len(applied),Applied_MAPE=score(pred,applied)['MAPE']))
    frames.append(pd.DataFrame(dict(Date=applied,Update_ID=tag,Seed=chosen,Actual=y.loc[applied].to_numpy(),Predicted=pred.to_numpy())))
    for split,idx in [('Candidate_train',tr),('Validation',va),('Refit_train',refit_idx),('Applied_test',applied)]:splitrows.append(dict(Update_ID=tag,Split=split,N=len(idx),Start=str(idx.min().date()),End=str(idx.max().date())))
    print(f'{tag}: effective {effective.date()}, seed {chosen}, validation {ranking.iloc[0].MAPE:.3f}%, train through {refit_idx.max().date()}',flush=True)

stitched=pd.concat(frames,ignore_index=True).sort_values('Date')
assert stitched.Date.is_unique and pd.DatetimeIndex(stitched.Date).equals(whole)
assert len(history)==len(schedule)*(len(SEEDS)+1)*2*EPOCHS
stitched['APE']=100*np.abs(stitched.Predicted-stitched.Actual)/stitched.Actual.abs();stitched['Month']=stitched.Date.dt.strftime('%Y-%m')
monthly=stitched.groupby('Month').agg(N=('Date','size'),Event_MAPE=('APE','mean'))
PRIOR=OUT.parent/'mlp_monthly_refit_12'
if not (PRIOR/'stitched_comparison.csv').exists():PRIOR=ROOT/'results/mlp_monthly_refit_12'
old=pd.read_csv(PRIOR/'stitched_comparison.csv',parse_dates=['Date'])
check=stitched.merge(old[['Date','Actual','Refit_prediction']],on='Date',suffixes=('','_old'),validate='one_to_one')
assert len(check)==len(whole) and np.allclose(check.Actual,check.Actual_old)
stitched['Monthly_refit_prediction']=check.Refit_prediction.to_numpy()
stitched['Monthly_refit_APE']=100*np.abs(stitched.Monthly_refit_prediction-stitched.Actual)/stitched.Actual.abs()
monthly['Monthly_refit_MAPE']=stitched.groupby('Month').Monthly_refit_APE.mean()
monthly['Event_minus_monthly_pp']=monthly.Event_MAPE-monthly.Monthly_refit_MAPE
# 初始化与原1月合并验证重训使用同一规则，首个更新前应复现其预测。
first=stitched.Update_ID=='initial';assert np.allclose(stitched.loc[first,'Predicted'],stitched.loc[first,'Monthly_refit_prediction'],atol=1e-5,rtol=0)
counts=dict(Gold_reentries=int(events.Gold_trigger.sum()),Copper_reentries=int(events.Copper_trigger.sum()),Merged_trigger_days=len(events),Triggered_updates=int((schedule.Kind=='Triggered').sum()),Pending_after_2020=int((~events.Executable_in_2020).sum()),Initializations=1,Total_deployed_models=len(schedule),Candidate_models_trained=len(schedule)*len(SEEDS),Refit_models_trained=len(schedule),Total_stage_trainings=len(schedule)*(len(SEEDS)+1)*2)
summary=pd.DataFrame([dict(Method='Return-to-normal updates',Annual_daily_MAPE=stitched.APE.mean(),Monthly_equal_MAPE=monthly.Event_MAPE.mean(),Annual_MSE=np.mean((stitched.Predicted-stitched.Actual)**2),Updates_excluding_initial=counts['Triggered_updates']),dict(Method='Monthly selected seed + refit',Annual_daily_MAPE=stitched.Monthly_refit_APE.mean(),Monthly_equal_MAPE=monthly.Monthly_refit_MAPE.mean(),Annual_MSE=np.mean((stitched.Monthly_refit_prediction-stitched.Actual)**2),Updates_excluding_initial=11)])
assert np.isclose(stitched.APE.mean(),np.average(monthly.Event_MAPE,weights=monthly.N))
for name,frame in [('stitched_predictions',stitched),('updates',pd.DataFrame(updates)),('validation_ranking',pd.concat(all_rankings,ignore_index=True)),('splits',pd.DataFrame(splitrows)),('summary',summary),('training_history',pd.DataFrame(history))]:frame.to_csv(OUT/f'{name}.csv',index=False)
monthly.to_csv(OUT/'monthly_comparison.csv');(OUT/'update_counts.json').write_text(json.dumps(counts,indent=2))
print('更新次数（初始化单列）：',counts);display(pd.DataFrame(updates));display(monthly.round(4));display(summary.round(4))
''')
code('''fig,axes=plt.subplots(3,1,figsize=(14,11),constrained_layout=True)
axes[0].plot(stitched.Date,stitched.Actual,color='black',lw=1.8,label='Actual')
axes[0].plot(stitched.Date,stitched.Predicted,color='#009E73',lw=1.3,label='Re-entry triggered + refit')
axes[0].plot(stitched.Date,stitched.Monthly_refit_prediction,color='#0072B2',lw=1.2,alpha=.7,label='Monthly update + refit')
for dt in schedule.loc[schedule.Kind=='Triggered','Effective_date']:axes[0].axvline(dt,color='#009E73',lw=.6,alpha=.2)
axes[0].set(title=f'2020 | Return-to-normal triggered updates: {counts["Triggered_updates"]} (+1 initialization)',ylabel='Price (dataset units)');axes[0].legend()
for metal,col,color in [('Gold','gold_vs_ma20','#D59400'),('Copper','copper_vs_ma20','#0072B2')]:
    axes[1].plot(whole,100*x.loc[whole,col],color=color,label=metal+' MA20 deviation')
axes[1].axhline(3,color='gray',ls='--');axes[1].axhline(-3,color='gray',ls='--');axes[1].axhline(0,color='gray',lw=.6)
for r in events.itertuples():
    if r.Gold_trigger:axes[1].scatter(r.Trigger_date,r.Gold_deviation_percent,color='#D59400',marker='v',s=35,zorder=5)
    if r.Copper_trigger:axes[1].scatter(r.Trigger_date,r.Copper_deviation_percent,color='#0072B2',marker='v',s=35,zorder=5)
axes[1].set(title='Return below absolute 3% triggers next-record update | triangles: trigger dates',ylabel='Deviation (%)');axes[1].legend()
monthly[['Event_MAPE','Monthly_refit_MAPE']].plot.bar(ax=axes[2],color=['#009E73','#0072B2'],rot=35)
axes[2].set(title='Monthly test MAPE',ylabel='MAPE (%)',xlabel='Month')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(OUT/'reentry_comparison.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'reentry_comparison.png')))
''')
nb.validate(n);nb.write(n,'outputs/12_减少预测的时间,扩大验证集的范围.ipynb')
