from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(src,as_version=4); nb.write(n,'outputs/mlp_direct_definition_11/11_before_definition_comparison.ipynb')
Path('outputs/mlp_direct_definition_11/start_cell.txt').write_text(str(len(n.cells)))
setup=next(c.source for c in reversed(n.cells) if c.cell_type=='code' and 'MLP11_ANNUAL_MA_OUT' in c.source)
oldtrain=next(c.source for c in reversed(n.cells) if c.cell_type=='code' and 'def fit3' in c.source)
helpers=oldtrain[oldtrain.index('H=[]'):oldtrain.index('rows=[]; frames=[]; splitrows=[]')]
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# DirectReturns两种定义：2015—2026年、十种子、100轮

只比较两个单阶段模型，均为原7项输入加金铜4项变化指标，总输入维度11，隐藏层16→32、ReLU、Adam学习率0.01、全批量、每个模型100轮。

- **DirectMA**：金/铜各自 `P(t)/MA5(t)−1`、`P(t)/MA20(t)−1`；均价包含当日。
- **DirectLag**：金/铜各自 `P(t)/P(t−5)−1`、`P(t)/P(t−20)−1`；与之前06的定义一致。

5/20均按原数据相邻记录计数。特征顺序均为原7项、金5、金20、铜5、铜20；同一个种子在两组中使用相同的初始网络权重，以配对比较特征定义。种子固定42、7、123、2024、2025、0、1、21、100、999。

每个测试年仅使用此前历史训练，各年两模型的训练/测试日期完全一致；标准化器只拟合当年训练集，目标仍为同日股价。沿用共同剔除最初20条记录的口径。2026为部分年份，不设验证集，不根据测试误差选择种子或训练轮数。

按年报告十种子平均MAPE、样本标准差及配对差（DirectLag减DirectMA，负数表示滞后变化率更好）。总体同时报告2015—2025完整年份与含2026部分年份的年度等权平均、样本数加权平均。保留全部训练MSE与训练MAPE。
''')
setup=setup.replace('MLP11_ANNUAL_MA_OUT','MLP11_DEFINITION_OUT').replace('results/mlp_annual_ten_ma_11','results/mlp_direct_definition_11')
setup=setup.replace('FULL=BASE+DIRECT', '''LAG=[]
for metal,col in [('gold','Gold Futures Price'),('copper','Copper Futures Price')]:
    for k in [5,20]:
        name=f'{metal}_lag_return{k}'
        x[name]=raw[col]/raw[col].shift(k)-1
        LAG.append(name)
FULL=BASE+DIRECT
GROUPS={'DirectMA':FULL,'DirectLag':BASE+LAG}''')
setup=setup.replace("display(pd.DataFrame({'Stage2 input':config['stage2_features']}))",'display(pd.DataFrame(GROUPS))')
code(setup)
code('''config.pop('test_year',None); config.pop('stage2_features',None); config.pop('stage2_training',None)
config.update(test_years=list(range(2015,2027)),epochs=100,groups=GROUPS,
    definitions={'DirectMA':'P(t)/MA_k(t)-1; includes current record','DirectLag':'P(t)/P(t-k)-1'},
    windows=[5,20],split='expanding train before test year',paired_initialization=True)
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
'''+helpers+'''rows=[]; frames=[]; splitrows=[]
EPOCHS=100
for YEAR in range(2015,2027):
    tr=x.index[x.index.year<YEAR]; te=x.index[x.index.year==YEAR]
    assert len(tr)>0 and len(te)>0 and tr.max()<te.min()
    for split,idx in [('Train',tr),('Test',te)]:
        splitrows.append(dict(Year=YEAR,Split=split,N=len(idx),Start=str(idx.min().date()),End=str(idx.max().date())))
    current=x.loc[x.index.year<=YEAR]
    for seed in SEEDS:
        for name,cols in GROUPS.items():
            bundle=fit3(x.loc[tr,cols],y.loc[tr],seed,name)
            pred=pred3(bundle,current[cols]); model,sx,sy=bundle
            torch.save(dict(state_dict=model.state_dict(),features=cols,seed=seed,epochs=EPOCHS,year=YEAR,
                sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist()),OUT/f'{YEAR}_{name}_seed{seed}.pt')
            for split,idx in [('Train',tr),('Test',te)]:
                a=y.loc[idx].to_numpy(); p=pred.loc[idx].to_numpy(); mse=np.mean((p-a)**2)
                rows.append(dict(Year=YEAR,Seed=seed,Budget=EPOCHS,Model=name,Split=split,N=len(idx),MAPE=100*np.mean(np.abs(p-a)/np.abs(a)),MSE=mse,MSE_standardized=mse/float(sy.scale_[0]**2)))
                frames.append(pd.DataFrame(dict(Date=idx,Year=YEAR,Seed=seed,Budget=EPOCHS,Model=name,Split=split,Actual=a,Predicted=p)))
    print(f'{YEAR}: ten seeds / both definitions completed',flush=True)
''')
code('''metrics_def=pd.DataFrame(rows); history_def=pd.DataFrame(H); predictions_def=pd.concat(frames,ignore_index=True)
assert len(metrics_def)==480 and len(history_def)==24000
assert np.isfinite(metrics_def[['MSE','MAPE']]).all().all()
metrics_def.to_csv(OUT/'metrics.csv',index=False); history_def.to_csv(OUT/'training_history.csv',index=False)
predictions_def.to_csv(OUT/'predictions.csv',index=False); pd.DataFrame(splitrows).to_csv(OUT/'splits.csv',index=False)
order=['DirectMA','DirectLag']
test=metrics_def[metrics_def.Split=='Test']
annual_mean=test.groupby(['Year','Model']).MAPE.mean().unstack().reindex(columns=order)
annual_std=test.groupby(['Year','Model']).MAPE.std().unstack().reindex(columns=order)
annual_train=metrics_def[metrics_def.Split=='Train'].groupby(['Year','Model'])[['MAPE','MSE']].mean()
annual_mean.to_csv(OUT/'annual_test_mean.csv'); annual_std.to_csv(OUT/'annual_test_std.csv'); annual_train.to_csv(OUT/'annual_train.csv')
paired=test.pivot(index=['Year','Seed'],columns='Model',values='MAPE')
paired['Lag_minus_MA_pp']=paired.DirectLag-paired.DirectMA
paired.to_csv(OUT/'paired_differences.csv')
paired_summary=paired.groupby('Year').Lag_minus_MA_pp.agg(['mean','std','min','max'])
paired_summary['Lag_better_seeds']=paired.Lag_minus_MA_pp.lt(0).groupby('Year').sum()
paired_summary.to_csv(OUT/'annual_paired_summary.csv')
summaryrows=[]
for period,selection in [('2015-2025',test[test.Year<=2025]),('2015-2026_partial',test)]:
    for model,g in selection.groupby('Model'):
        seedmean=g.groupby('Seed').MAPE.mean()
        summaryrows.append(dict(Period=period,Model=model,Annual_equal_mean_MAPE=seedmean.mean(),Across_seed_SD=seedmean.std(),Sample_weighted_MAPE=np.average(g.MAPE,weights=g.N)))
summary_def=pd.DataFrame(summaryrows); summary_def.to_csv(OUT/'summary.csv',index=False)
annual_mean.diff().to_csv(OUT/'year_over_year_change_pp.csv')
# 相对均价模型应复现上一节所有年度与种子的逐日预测。
prior=OUT.parent/'mlp_annual_ten_ma_11/predictions.csv'
if prior.exists():
    old=pd.read_csv(prior,parse_dates=['Date']); old=old[old.Model=='DirectReturns'].copy(); old['Model']='DirectMA'
    new=predictions_def[predictions_def.Model=='DirectMA']
    check=old.merge(new,on=['Date','Year','Seed','Model','Split','Budget'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==len(old)==len(new)
    assert np.array_equal(check.Predicted_old.to_numpy(dtype=np.float32),check.Predicted_new.to_numpy(dtype=np.float32))
    print('复核通过：DirectMA在全部12年、十种子的预测均与上一节一致。')
# 原2020十种子滞后变化率也应复现（100轮）。
prior_lag=OUT.parent/'mlp_ten_seeds_11/metrics.csv'
if prior_lag.exists():
    old=pd.read_csv(prior_lag); old=old[(old.Model=='DirectReturns')&(old.Budget==100)]
    new=metrics_def[(metrics_def.Year==2020)&(metrics_def.Model=='DirectLag')]
    check=old.merge(new,on=['Seed','Split'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==20 and np.allclose(check.MAPE_old,check.MAPE_new,rtol=0,atol=1e-6)
    print('复核通过：DirectLag的2020年100轮结果复现此前十种子实验。')
print('年度平均MAPE：'); display(annual_mean.round(4))
print('年度种子标准差：'); display(annual_std.round(4))
print('配对差（滞后−均价）及改善种子数：'); display(paired_summary.round(4))
print('总体汇总：'); display(summary_def.round(4))
''')
code('''colors={'DirectMA':'#0072B2','DirectLag':'#D55E00'}
fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
for model in order:
    axes[0].errorbar(annual_mean.index,annual_mean[model],yerr=annual_std[model],marker='o',capsize=4,color=colors[model],label=model)
axes[0].set(title='Direct inputs: MA deviation vs lag return | 100 epochs | 10 seeds',ylabel='Test MAPE (%)'); axes[0].legend()
axes[1].bar(paired_summary.index,paired_summary['mean'],color=np.where(paired_summary['mean']<0,'#009E73','#D55E00'))
axes[1].axhline(0,color='black',lw=.8)
axes[1].set(title='DirectLag minus DirectMA | Negative means lag returns improve',ylabel='MAPE difference (percentage points)',xlabel='Test year')
for ax in axes:
    ax.set_xticks(annual_mean.index); ax.set_xticklabels([str(y) if y!=2026 else '2026*' for y in annual_mean.index]); ax.grid(alpha=.2)
fig.savefig(OUT/'definition_comparison.png',dpi=160); plt.close(fig); display(Image(filename=str(OUT/'definition_comparison.png')))
fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
for model in order:
    g=annual_train.xs(model,level='Model')
    axes[0].plot(g.index,g.MAPE,marker='o',color=colors[model],label=model)
    axes[1].plot(g.index,g.MSE,marker='o',color=colors[model],label=model)
axes[0].set(title='Mean training MAPE',ylabel='MAPE (%)'); axes[1].set(title='Mean training MSE (original price scale)',ylabel='MSE',yscale='log')
for ax in axes: ax.grid(alpha=.2); ax.legend(); ax.set_xlabel('Test year')
fig.savefig(OUT/'training_comparison.png',dpi=160); plt.close(fig); display(Image(filename=str(OUT/'training_comparison.png')))
print('* 2026 is partial through the source data end date.')
''')
nb.validate(n); nb.write(n,'outputs/11_改变模型构建.ipynb')
