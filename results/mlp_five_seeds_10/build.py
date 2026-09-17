from pathlib import Path
import nbformat as nb,hashlib
r=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-');p=next(r.glob('notebooks/**/10_*.ipynb'));n=nb.read(p,4)
out=Path('outputs/mlp_five_seeds_10');out.joinpath('original.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest());out.joinpath('start_cell.txt').write_text(str(len(n.cells)))
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 五个随机种子：2015—2026逐年平均比较

固定种子 **42、7、123、2024、2025**，在原三个种子上增加两个，不根据测试表现挑选种子。继续比较Base_100、Base_400、Full_100；每个测试年使用此前全部共同样本训练，每年、每种子重新初始化。Base的100/400轮来自同一条训练路径。

其余设置沿用前节：ReLU、16/32隐藏层、Adam 0.01、全批量；标准化仅拟合对应训练集；统一去掉前20条记录，目标仍是原始同日股价。

**先对每个模型、每年5次独立训练的MAPE取平均，再对年份等权平均。不是先平均预测价格，也不是集成模型的误差。**同时保留每个种子的结果、种子标准差、成对差异和年度胜率。2026截至7月23日，单列2015—2025完整年度平均。五种子用于衡量初始化敏感性，不是五份独立市场数据，误差棒不是置信区间。本次仅新增静态图，不使用PX。
''')
s=n.cells[12].source.replace('import plotly.express as px\n','').replace('MLP10_ANNUAL_OUT','MLP10_FIVE_OUT').replace('mlp_annual_three_models_10','mlp_five_seeds_10')
s=s.replace('YEARS=list(range(2015,2027))','YEARS=list(range(2015,2027))\nSEEDS=[42,7,123,2024,2025]')
s=s.replace("'seed':SEED","'seeds':SEEDS")
code(s)
s=n.cells[13].source
s=s[:s.index('# Each model')]
a=s.index('    sy=');b=s.index("    print(f'{year} complete")
s=s[:a]+'    for seed in SEEDS:\n'+''.join('    '+line+'\n' for line in s[a:b].splitlines())+s[b:]
s=s.replace('torch.manual_seed(SEED)','torch.manual_seed(seed)').replace("'seed':SEED","'seed':seed")
s=s.replace("{'Year':year,'Family'","{'Year':year,'Seed':seed,'Family'").replace("{'Year':year,'Date'","{'Year':year,'Seed':seed,'Date'").replace("{'Year':year,'Model'","{'Year':year,'Seed':seed,'Model'")
s=s.replace("f'{year}_{name}.pt'","f'{year}_{name}_seed{seed}.pt'")
s+='''
assert len(metrics)==12*5*3*2
assert len(list(OUT.glob('*.pt')))==180
prior=pd.read_csv(ROOT/'results/mlp_annual_three_models_10/metrics.csv')
current=metrics[metrics.Seed==42]
check=prior.merge(current,on=['Year','Model','Split'],suffixes=('_old','_new'),validate='one_to_one')
assert len(check)==72 and np.allclose(check.MAPE_old,check.MAPE_new,atol=1e-6)
print('seed42全部年份三组模型均复现上一节结果。')
# Compare with the saved 06 three-seed experiment, if available.
prior06=ROOT/'results/mlp_basis_06/metrics.csv'
if not prior06.exists():prior06=Path('outputs/mlp_basis_06/metrics.csv')
if prior06.exists():
    old=pd.read_csv(prior06);old=old[old.Model.isin(['A_base','C_returns'])].copy()
    old['Model']=old.Model.map({'A_base':'Base_100','C_returns':'Full_100'})
    check=old.merge(metrics,on=['Year','Seed','Model','Split'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==72 and np.allclose(check.MAPE_old,check.MAPE_new,atol=1e-6)
    print('06中2021—2026的三个种子A/C结果全部复现。')
'''
code(s)
code('''test=metrics[metrics.Split=='Test'].copy()
annual=test.groupby(['Year','Model']).MAPE.mean().unstack()[order]
seed_std=test.groupby(['Year','Model']).MAPE.std().unstack()[order]
annual.to_csv(OUT/'annual_mean_mape.csv');seed_std.to_csv(OUT/'annual_seed_std.csv')
seed_means=[]; summary=[]
for period,years in [('2015-2026 (2026 partial)',YEARS),('2015-2025 complete',list(range(2015,2026)))]:
    a=annual.loc[years]
    for model in order:
        t=test[test.Year.isin(years)&(test.Model==model)]
        summary.append({'Period':period,'Model':model,'Mean_MAPE':a[model].mean(),'Median_annual_MAPE':a[model].median(),'Across_year_std':a[model].std(),'Mean_within_year_seed_std':seed_std.loc[years,model].mean(),'Worst_annual_mean_MAPE':a[model].max(),'Worst_year':int(a[model].idxmax()),'Winning_years':int((a.idxmin(axis=1)==model).sum())})
        for seed,g in t.groupby('Seed'):
            seed_means.append({'Period':period,'Model':model,'Seed':seed,'Mean_annual_MAPE':g.MAPE.mean()})
summary=pd.DataFrame(summary);seed_means=pd.DataFrame(seed_means)
summary.to_csv(OUT/'summary.csv',index=False);seed_means.to_csv(OUT/'per_seed_means.csv',index=False)
pair=test.pivot(index=['Year','Seed'],columns='Model',values='MAPE')
contrasts=pd.DataFrame({'Full100_minus_Base100_pp':pair.Full_100-pair.Base_100,'Base400_minus_Base100_pp':pair.Base_400-pair.Base_100,'Full100_minus_Base400_pp':pair.Full_100-pair.Base_400})
contrasts.to_csv(OUT/'paired_differences.csv')
paired_annual=contrasts.groupby('Year').agg(['mean','std']);paired_annual.to_csv(OUT/'annual_paired_differences.csv')
print('每年五个种子的平均MAPE（%）：');display(annual.round(4))
print('每年种子间标准差（百分点）：');display(seed_std.round(4))
print('汇总：');display(summary.round(4))
print('每个种子的年度平均：');display(seed_means.pivot(index=['Period','Seed'],columns='Model',values='Mean_annual_MAPE')[order].round(4))
print('成对差异，负值为前者更好：');display(paired_annual.round(4))
''')
code('''colors={'Base_100':'#0072B2','Base_400':'#D55E00','Full_100':'#009E73'}
fig,axes=plt.subplots(2,1,figsize=(15,10),constrained_layout=True)
pos=np.arange(len(YEARS));width=.25
for j,model in enumerate(order):
    bars=axes[0].bar(pos+(j-1)*width,annual[model],width,label=model,color=colors[model],yerr=seed_std[model],capsize=2,error_kw={'elinewidth':.7})
    axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=7)
axes[0].set_xticks(pos,[str(v)+('*' if v==2026 else '') for v in YEARS])
axes[0].set(title='Annual mean MAPE over 5 seeds | error bars: seed SD | *2026 partial',ylabel='MAPE (%)',xlabel='Test year')
axes[0].legend();axes[0].grid(axis='y',alpha=.2)
b=summary.pivot(index='Period',columns='Model',values='Mean_MAPE')[order]
b.plot.bar(ax=axes[1],color=list(colors.values()),rot=0)
for c in axes[1].containers:axes[1].bar_label(c,fmt='%.2f',padding=3)
axes[1].set(title='Equal-year mean of 5-seed MAPE',ylabel='MAPE (%)',xlabel='Period');axes[1].margins(y=.2);axes[1].grid(axis='y',alpha=.2)
fig.savefig(OUT/'five_seed_comparison.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'five_seed_comparison.png')))
fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
for ax,period in zip(axes,['2015-2026 (2026 partial)','2015-2025 complete']):
    z=seed_means[seed_means.Period==period].pivot(index='Seed',columns='Model',values='Mean_annual_MAPE').loc[SEEDS,order]
    z.plot.bar(ax=ax,color=list(colors.values()),rot=0)
    ax.set(title=period,xlabel='Seed',ylabel='Annual mean MAPE (%)');ax.grid(axis='y',alpha=.2)
fig.savefig(OUT/'per_seed_means.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'per_seed_means.png')))
''')
nb.write(n,Path('outputs')/p.name)
