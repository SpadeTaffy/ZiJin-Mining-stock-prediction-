from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(src,as_version=4); nb.write(n,'outputs/mlp_ten_seeds_11/11_before_ten_seeds.ipynb')
Path('outputs/mlp_ten_seeds_11/start_cell.txt').write_text(str(len(n.cells)))
setup=next(c.source for c in reversed(n.cells) if c.cell_type=='code' and "MLP11_THREE_OUT" in c.source)
train=next(c.source for c in reversed(n.cells) if c.cell_type=='code' and 'def fit3' in c.source)
def md(s): n.cells.append(nb.v4.new_markdown_cell(s))
def code(s): n.cells.append(nb.v4.new_code_cell(s))
md('''# 十个种子：100与200轮、三种模型

沿用最近一次三模型实验的共同样本、特征定义、架构和训练设置。种子预先固定为42、7、123、2024、2025、0、1、21、100、999。对每个种子分别运行100和200轮；相同种子、模型的100轮路径应与200轮路径的前100轮一致。

Base：7项原始输入；Stacking：基准预测＋6项金铜涨幅，两个阶段各E轮，样本内训练；DirectReturns：原7项＋金铜5/20期累计收益率，单阶段E轮。训练为2020年前的2034条共同样本，测试为2020年271条，不设验证集、不早停、不按测试结果选种子。

报告训练与测试MAPE的10种子均值、样本标准差、最小和最大值，以及逐种子200轮减100轮的配对差。训练MSE采用原股价尺度。

**收敛诊断**：对每条训练路径，比较倒数第40至21轮与最后20轮的平均MSE，计算相对下降幅度。绝对变化不超过1%标记为“末段近似平台”，下降超过1%标记为“仍在下降”，上升超过1%标记为“末段上升”。这是预先定义的局部诊断，不是数学收敛证明，也不能据此保证泛化。堆叠诊断只针对对应第一阶段冻结后的第二阶段路径。
''')
setup=setup.replace('MLP11_THREE_OUT','MLP11_TEN_OUT').replace('mlp_three_models_200_11','mlp_ten_seeds_11').replace('SEEDS=[42,7,123]','SEEDS=[42,7,123,2024,2025,0,1,21,100,999]')
code(setup)
# Reuse verified fit and evaluation code, retaining separate budgets.
train=train[:train.index('metrics3=pd.DataFrame(rows)')]
train=train.replace("config['direct_features']=FULL", "config['direct_features']=FULL\nconfig['epoch_budgets_per_stage']=[100,200]\nconfig.pop('epochs_per_stage',None)")
train=train.replace('dict(Seed=seed,Model=label,Epoch=epoch','dict(Seed=seed,Budget=EPOCHS,Model=label,Epoch=epoch')
train=train.replace('dict(Seed=seed,Model=name,Split=split','dict(Seed=seed,Budget=EPOCHS,Model=name,Split=split')
train=train.replace('dict(Date=idx,Seed=seed,Model=name','dict(Date=idx,Seed=seed,Budget=EPOCHS,Model=name')
train=train.replace("f'{name}_seed{seed}.pt'","f'{name}_{EPOCHS}_seed{seed}.pt'")
train=train.replace("print(f'Seed {seed}: all three models completed')", "print(f'{EPOCHS} epochs / seed {seed}: completed',flush=True)")
start=train.index('for seed in SEEDS:')
train=train[:start]+'for EPOCHS in [100,200]:\n'+''.join('    '+line+'\n' for line in train[start:].splitlines())
code(train)
code('''metrics10=pd.DataFrame(rows); history10=pd.DataFrame(H); predictions10=pd.concat(frames,ignore_index=True)
assert len(metrics10)==120 and len(history10)==9000
assert np.isfinite(metrics10[['MSE','MAPE']]).all().all()
metrics10.to_csv(OUT/'metrics.csv',index=False); history10.to_csv(OUT/'training_history.csv',index=False); predictions10.to_csv(OUT/'predictions.csv',index=False)
# 同种子Base/DirectReturns，两预算前100轮完全一致；Stacking因第一阶段不同不适用。
for seed in SEEDS:
    for name in ['Base','DirectReturns']:
        a=history10[(history10.Seed==seed)&(history10.Model==name)&(history10.Budget==100)].Train_MSE.to_numpy()
        b=history10[(history10.Seed==seed)&(history10.Model==name)&(history10.Budget==200)&(history10.Epoch<=100)].Train_MSE.to_numpy()
        assert np.allclose(a,b,rtol=0,atol=1e-12)
for r in metrics10[metrics10.Split=='Train'].itertuples():
    h=history10[(history10.Seed==r.Seed)&(history10.Model==r.Model)&(history10.Budget==r.Budget)]
    assert np.isclose(h.iloc[-1].Train_MSE,r.MSE,atol=1e-8)
summary10=metrics10.groupby(['Model','Budget','Split'])[['MAPE','MSE']].agg(['mean','std','min','max'])
summary10.to_csv(OUT/'summary.csv')
paired10=metrics10.pivot(index=['Model','Seed','Split'],columns='Budget',values='MAPE')
paired10['delta_200_minus_100_pp']=paired10[200]-paired10[100]
paired10.to_csv(OUT/'paired_mape.csv')
diagnostics=[]
for (name,seed,budget),g in history10.groupby(['Model','Seed','Budget']):
    g=g.sort_values('Epoch'); prev=g.iloc[-40:-20].Train_MSE.mean(); last=g.iloc[-20:].Train_MSE.mean()
    change=100*(prev-last)/prev
    status='plateau' if abs(change)<=1 else ('decreasing' if change>1 else 'increasing')
    diagnostics.append(dict(Model=name,Seed=seed,Budget=budget,Initial_MSE=g.iloc[0].Train_MSE,Final_MSE=g.iloc[-1].Train_MSE,Previous20_mean=prev,Last20_mean=last,Decrease_percent=change,Status=status))
diagnostics10=pd.DataFrame(diagnostics); diagnostics10.to_csv(OUT/'convergence_diagnostics.csv',index=False)
counts10=diagnostics10.groupby(['Model','Budget','Status']).size().unstack(fill_value=0)
counts10.to_csv(OUT/'convergence_counts.csv')
print('训练与测试MAPE / MSE汇总：'); display(summary10.round(6))
print('末段训练损失状态（decreasing仍下降、plateau近似平台、increasing上升）：'); display(counts10)
print('2020逐种子MAPE：'); display(metrics10[metrics10.Split=='Test'].pivot(index='Seed',columns=['Model','Budget'],values='MAPE').reindex(SEEDS).round(4))
# 200轮的前三个种子应复现前一组三模型实验。
prior=OUT.parent/'mlp_three_models_200_11/metrics.csv'
if prior.exists():
    old=pd.read_csv(prior)
    check=old.merge(metrics10[metrics10.Budget==200],on=['Seed','Model','Split'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==18 and np.allclose(check.MSE_old,check.MSE_new,atol=1e-8)
    print('已验证：200轮前三个种子复现上一实验；Base及DirectReturns的100/200轮前缀训练路径一致。')
''')
code('''order=['Base','Stacking','DirectReturns']; colors={100:'#0072B2',200:'#D55E00'}
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for j,name in enumerate(order):
    for i,budget in enumerate([100,200]):
        ax=axes[i,j]
        for seed in SEEDS:
            g=history10[(history10.Model==name)&(history10.Budget==budget)&(history10.Seed==seed)]
            ax.plot(g.Epoch,g.Train_MSE,lw=1,alpha=.7,label=str(seed))
        ax.set(title=f'{name} | {budget} epochs',xlabel='Epoch',ylabel='Train MSE (original scale)',yscale='log'); ax.grid(alpha=.2)
axes[0,0].legend(ncol=2,fontsize=7)
fig.savefig(OUT/'training_mse_ten_seeds.png',dpi=150); plt.close(fig)
display(Image(filename=str(OUT/'training_mse_ten_seeds.png')))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,split in zip(axes,['Train','Test']):
    for offset,budget in [(-.17,100),(.17,200)]:
        g=metrics10[(metrics10.Split==split)&(metrics10.Budget==budget)].groupby('Model').MAPE.agg(['mean','std']).reindex(order)
        bars=ax.bar(np.arange(3)+offset,g['mean'],width=.32,yerr=g['std'],capsize=4,color=colors[budget],label=f'{budget} epochs')
        ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=8)
    ax.set(xticks=np.arange(3),xticklabels=order,ylabel='MAPE (%)',title=split+' | Mean ± sample SD (10 seeds)'); ax.legend(); ax.margins(y=.25)
fig.savefig(OUT/'mape_comparison.png',dpi=150); plt.close(fig)
display(Image(filename=str(OUT/'mape_comparison.png')))
''')
nb.validate(n); nb.write(n,'outputs/11_改变模型构建.ipynb')
