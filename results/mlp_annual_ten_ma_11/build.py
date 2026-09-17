from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(src,as_version=4); nb.write(n,'outputs/mlp_annual_ten_ma_11/11_before_annual.ipynb')
Path('outputs/mlp_annual_ten_ma_11/start_cell.txt').write_text(str(len(n.cells)))
setup=next(c.source for c in reversed(n.cells) if c.cell_type=='code' and 'MLP11_TEN_MA_OUT' in c.source)
train=next(c.source for c in reversed(n.cells) if c.cell_type=='code' and 'def fit3' in c.source)
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 2015—2026逐年滚动测试：十种子、100轮、三个模型

每个测试年Y仅使用Y年以前全部有效数据训练，Y年测试，不在测试年内更新模型。种子固定为42、7、123、2024、2025、0、1、21、100、999；每个网络100轮，堆叠两个阶段各100轮，不用折外预测。三个模型使用相同日期，所有标准化仅拟合对应年的训练集。

模型定义沿用上一节：Base原7变量；DirectReturns原7变量加金铜相对5/20期滚动均价涨幅；Stacking为Base预测加金铜昨日涨幅及相对5/20期均价涨幅。均价包含当日，按原始记录窗口计算，目标为同日股价。共同剔除最初20条记录，保持2020年的实验可复现。

报告每年10种子平均MAPE及样本标准差，年度等权平均与按测试样本数加权平均分别列示。2026仅数据已有日期，不完整年份单列，同时报告2015—2025完整年份汇总。

“突变”仅作描述性检查：列出相邻年份平均MAPE变化，绝对变化≥5个百分点标记为明显年度跳变，另外列出相邻年份变化最大的项目。这不是统计变点检验，也不是经济因果判断。不用这些测试结果进行早停或筛选种子。
''')
setup=setup.replace('MLP11_TEN_MA_OUT','MLP11_ANNUAL_MA_OUT').replace('results/mlp_ten_seeds_ma_11','results/mlp_annual_ten_ma_11').replace('EPOCHS=200','EPOCHS=100')
setup=setup.replace('raw=raw.loc[raw.index.year<=2020]','raw=raw.loc[raw.index.year<=2026]')
setup=setup.replace("display(splits)", "# 此处2020切分仅为初始化；实际按下面逐年循环保存全部切分。")
code(setup)
train=train.replace("config['epoch_budgets_per_stage']=[100,200]", "config['epoch_budgets_per_stage']=[100]\nconfig['test_years']=list(range(2015,2027))\nconfig.pop('test_year',None)\nconfig['split']='expanding train before test year; no OOF'\nconfig['jump_threshold_pp']=5")
train=train.replace('dict(Seed=seed,Budget=EPOCHS','dict(Year=YEAR,Seed=seed,Budget=EPOCHS')
train=train.replace('dict(Date=idx,Seed=seed,Budget=EPOCHS','dict(Date=idx,Year=YEAR,Seed=seed,Budget=EPOCHS')
train=train.replace("f'{name}_{EPOCHS}_seed{seed}.pt'", "f'{YEAR}_{name}_{EPOCHS}_seed{seed}.pt'")
train=train.replace('rows=[]; frames=[]','rows=[]; frames=[]; splitrows=[]')
train=train.replace('for EPOCHS in [100,200]:', '''EPOCHS=100
for YEAR in range(2015,2027):
    tr=x.index[x.index.year<YEAR]; te=x.index[x.index.year==YEAR]
    assert len(tr)>0 and len(te)>0 and tr.max()<te.min()
    for split,idx in [('Train',tr),('Test',te)]:
        splitrows.append(dict(Year=YEAR,Split=split,N=len(idx),Start=str(idx.min().date()),End=str(idx.max().date())))''')
# Evaluate only current training and test rows, excluding subsequent years entirely.
train=train.replace('base=fit3', 'current=x.loc[x.index.year<=YEAR]\n        base=fit3')
train=train.replace('bp=pred3(base,x[BASE])','bp=pred3(base,current[BASE])')
train=train.replace('meta=x[RETURNS].copy()', 'meta=current[RETURNS].copy()')
train=train.replace("('Base',base,x[BASE])", "('Base',base,current[BASE])").replace("('DirectReturns',direct,x[FULL])", "('DirectReturns',direct,current[FULL])")
train=train.replace("print(f'{EPOCHS} epochs / seed {seed}: completed',flush=True)", "if seed==SEEDS[-1]: print(f'{YEAR}: ten seeds / three models completed',flush=True)")
code(train)
code('''annual_metrics=pd.DataFrame(rows); annual_history=pd.DataFrame(H); annual_predictions=pd.concat(frames,ignore_index=True)
assert len(annual_metrics)==720 and len(annual_history)==36000
assert np.isfinite(annual_metrics[['MAPE','MSE']]).all().all()
annual_metrics.to_csv(OUT/'metrics.csv',index=False); annual_history.to_csv(OUT/'training_history.csv',index=False)
annual_predictions.to_csv(OUT/'predictions.csv',index=False); pd.DataFrame(splitrows).to_csv(OUT/'splits.csv',index=False)
order=['Base','Stacking','DirectReturns']
test=annual_metrics[annual_metrics.Split=='Test']
annual_mean=test.groupby(['Year','Model']).MAPE.mean().unstack().reindex(columns=order)
annual_std=test.groupby(['Year','Model']).MAPE.std().unstack().reindex(columns=order)
train_mean=annual_metrics[annual_metrics.Split=='Train'].groupby(['Year','Model']).MAPE.mean().unstack().reindex(columns=order)
mse_mean=annual_metrics[annual_metrics.Split=='Train'].groupby(['Year','Model']).MSE.mean().unstack().reindex(columns=order)
for name,table in [('annual_test_mean',annual_mean),('annual_test_std',annual_std),('annual_train_mean',train_mean),('annual_train_mse',mse_mean)]:table.to_csv(OUT/f'{name}.csv')
yoy=annual_mean.diff(); yoy.to_csv(OUT/'year_over_year_change_pp.csv')
jumps=yoy.stack().rename('Change_pp').reset_index(); jumps['Abs_change_pp']=jumps.Change_pp.abs(); jumps['Flag_ge_5pp']=jumps.Abs_change_pp>=5
jumps=jumps.sort_values('Abs_change_pp',ascending=False); jumps.to_csv(OUT/'annual_jumps.csv',index=False)
summaryrows=[]
for period,selection in [('2015-2025',test[test.Year<=2025]),('2015-2026_partial',test)]:
    for model,g in selection.groupby('Model'):
        seedmean=g.groupby('Seed').MAPE.mean()
        summaryrows.append(dict(Period=period,Model=model,Annual_equal_mean_MAPE=seedmean.mean(),Across_seed_SD=seedmean.std(),Sample_weighted_MAPE=np.average(g.MAPE,weights=g.N)))
annual_summary=pd.DataFrame(summaryrows); annual_summary.to_csv(OUT/'summary.csv',index=False)
paired=test.pivot(index=['Year','Seed'],columns='Model',values='MAPE')
paired['Stacking_minus_Base']=paired.Stacking-paired.Base
paired['DirectReturns_minus_Base']=paired.DirectReturns-paired.Base
paired.to_csv(OUT/'paired_seed_differences.csv')
relative=paired.groupby('Year')[['Stacking_minus_Base','DirectReturns_minus_Base']].mean()
relative.to_csv(OUT/'annual_difference_vs_base.csv')
winrows=[]
for year,g in paired.groupby('Year'):
    winrows.append(dict(Year=year,Stacking_seeds_better=int((g.Stacking_minus_Base<0).sum()),DirectReturns_seeds_better=int((g.DirectReturns_minus_Base<0).sum())))
wins=pd.DataFrame(winrows); wins.to_csv(OUT/'annual_seed_wins.csv',index=False)
# 复核2020年100轮全种子全模型预测与上节一致。
prior=OUT.parent/'mlp_ten_seeds_ma_11/predictions.csv'
if prior.exists():
    old=pd.read_csv(prior,parse_dates=['Date']); old=old[old.Budget==100]
    new=annual_predictions[annual_predictions.Year==2020]
    check=old.merge(new,on=['Date','Seed','Model','Split','Budget'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==len(old)==len(new)
    assert np.allclose(check.Predicted_old,check.Predicted_new,rtol=0,atol=1e-6)
    print('复核通过：2020年100轮、十种子、三模型预测与上节一致。')
print('年度测试MAPE均值：'); display(annual_mean.round(3))
print('年度种子标准差：'); display(annual_std.round(3))
print('跨年度汇总：'); display(annual_summary.round(4))
print('年度跳变（>=5个百分点）：'); display(jumps[jumps.Flag_ge_5pp].round(3))
print('相对基准平均差、逐年改善种子数：'); display(relative.round(3)); display(wins)
''')
code('''colors={'Base':'#0072B2','Stacking':'#D55E00','DirectReturns':'#009E73'}
fig,axes=plt.subplots(2,1,figsize=(13,9),constrained_layout=True)
for model in order:
    axes[0].errorbar(annual_mean.index,annual_mean[model],yerr=annual_std[model],marker='o',capsize=3,color=colors[model],label=model)
    axes[1].plot(train_mean.index,train_mean[model],marker='o',color=colors[model],label=model)
axes[0].set(title='Annual test MAPE | 10-seed mean ± sample SD | 100 epochs',ylabel='Test MAPE (%)')
axes[1].set(title='Annual training MAPE | 10-seed mean',ylabel='Train MAPE (%)',xlabel='Test year')
for ax in axes:
    ax.set_xticks(annual_mean.index); ax.set_xticklabels([str(y) if y!=2026 else '2026*' for y in annual_mean.index]); ax.grid(alpha=.2); ax.legend()
fig.savefig(OUT/'annual_mape.png',dpi=160); plt.close(fig); display(Image(filename=str(OUT/'annual_mape.png')))
fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
for model in order:
    axes[0].plot(yoy.index,yoy[model],marker='o',color=colors[model],label=model)
for model in ['Stacking','DirectReturns']:
    axes[1].plot(annual_mean.index,annual_mean[model]-annual_mean.Base,marker='o',color=colors[model],label=model+' - Base')
axes[0].set(title='Change in mean test MAPE from previous year',ylabel='Percentage points'); axes[0].axhline(5,color='gray',ls=':',lw=1); axes[0].axhline(-5,color='gray',ls=':',lw=1)
axes[1].set(title='Difference versus Base | Negative means improvement',ylabel='Percentage points',xlabel='Test year')
for ax in axes:
    ax.axhline(0,color='black',lw=.7); ax.set_xticks(annual_mean.index); ax.set_xticklabels([str(y) if y!=2026 else '2026*' for y in annual_mean.index]); ax.grid(alpha=.2); ax.legend()
fig.savefig(OUT/'annual_changes.png',dpi=160); plt.close(fig); display(Image(filename=str(OUT/'annual_changes.png')))
print('* 2026 is partial; see splits.csv for exact data end date.')
''')
nb.validate(n); nb.write(n,'outputs/11_改变模型构建.ipynb')
