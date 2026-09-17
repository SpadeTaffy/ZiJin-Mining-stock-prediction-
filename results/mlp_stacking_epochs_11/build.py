from pathlib import Path
import nbformat as nb
p=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(p,as_version=4)
nb.write(n,'outputs/mlp_stacking_epochs_11/11_before_epochs.ipynb')
start=len(n.cells)
Path('outputs/mlp_stacking_epochs_11/start_cell.txt').write_text(str(start))
def md(s): n.cells.append(nb.v4.new_markdown_cell(s))
def code(s): n.cells.append(nb.v4.new_code_cell(s))
md('''# 堆叠模型训练轮数对比：100 / 200 / 400 / 800

保持前一实验的训练/测试日期、原7个基准输入、金铜6项涨幅、16×32隐藏层、ReLU、Adam学习率0.01、全批量、种子42/7/123不变。**不用折外预测**，第二阶段使用第一阶段对全部有效训练样本的样本内预测。

每组轮数E表示：**第一阶段训练E轮，冻结后生成预测，第二阶段从相同种子的初始状态训练E轮**。第一阶段在同一800轮训练路径上保存四个检查点；不同E的第二阶段因输入预测不同，独立初始化训练。每组总计2E轮，不是两个阶段合计E轮。

2020年为固定测试集；报告所有预先指定轮数，不使用测试集早停或挑选种子。第一阶段基准结果也保留，便于判断第二阶段是否改善。以下单元可单独从本节起运行。
''')
setup=next(c.source for c in n.cells if c.cell_type=='code')
setup=setup.replace("'MLP11_OUT'", "'MLP11_EPOCH_OUT'").replace("results/mlp_stacking_11", "results/mlp_stacking_epochs_11")
code(setup)
code('''import copy
BUDGETS=[100,200,400,800]
config['epoch_budgets_per_stage']=BUDGETS
config.pop('epochs_per_stage',None)
config['stage1_path']='shared checkpoints within seed'
config['stage2_path']='fresh fit for each budget using corresponding stage1 predictions'
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
H=[]
def train_checkpoints(X,Y,seed,stage,budget,checkpoints):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sx=StandardScaler().fit(X); sy=StandardScaler().fit(np.asarray(Y).reshape(-1,1))
    xt=torch.tensor(sx.transform(X),dtype=torch.float32)
    yt=torch.tensor(sy.transform(np.asarray(Y).reshape(-1,1)),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
    opt=torch.optim.Adam(model.parameters(),lr=LR); saved={}
    for epoch in range(1,max(checkpoints)+1):
        model.train(); opt.zero_grad(); loss=((model(xt)-yt)**2).mean()
        assert torch.isfinite(loss); loss.backward(); opt.step(); model.eval()
        with torch.no_grad(): mse=((model(xt)-yt)**2).mean().item()
        H.append(dict(Seed=seed,Stage=stage,Budget=budget,Epoch=epoch,Train_MSE=mse))
        if epoch in checkpoints: saved[epoch]=(copy.deepcopy(model),sx,sy)
    return saved

def predict_e(bundle,X):
    model,sx,sy=bundle; model.eval()
    with torch.no_grad(): z=model(torch.tensor(sx.transform(X),dtype=torch.float32)).numpy()
    result=sy.inverse_transform(z).ravel(); assert np.isfinite(result).all()
    return pd.Series(result,index=X.index)

def save_e(bundle,seed,stage,budget,features):
    model,sx,sy=bundle
    torch.save(dict(state_dict=model.state_dict(),features=features,seed=seed,epochs=budget,
        hidden=[16,32],sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),
        sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist(),
        train_start=str(tr.min()),train_end=str(tr.max())),OUT/f'{stage}_{budget}_seed{seed}.pt')

rows=[]; frames=[]
for seed in SEEDS:
    bases=train_checkpoints(x.loc[tr,BASE],y.loc[tr],seed,'Base',800,BUDGETS)
    for budget in BUDGETS:
        base=bases[budget]; bp=predict_e(base,x[BASE])
        meta=x[RETURNS].copy(); meta.insert(0,'base_prediction',bp)
        stack=train_checkpoints(meta.loc[tr],y.loc[tr],seed,'Stage2',budget,[budget])[budget]
        sp=predict_e(stack,meta)
        save_e(base,seed,'Base',budget,BASE); save_e(stack,seed,'Stage2',budget,list(meta.columns))
        for name,pred in [('Base',bp),('Stacking',sp)]:
            for split,idx in [('Train',tr),('Test',te)]:
                actual=y.loc[idx].to_numpy(); p=pred.loc[idx].to_numpy()
                ape=100*np.abs(p-actual)/np.abs(actual)
                rows.append(dict(Seed=seed,Epochs=budget,Model=name,Split=split,N=len(idx),MAPE=ape.mean()))
                frames.append(pd.DataFrame(dict(Date=idx,Seed=seed,Epochs=budget,Model=name,Split=split,Actual=actual,Predicted=p,APE=ape)))
        print(f'Seed {seed}, {budget} epochs per stage completed',flush=True)
metrics_e=pd.DataFrame(rows); predictions_e=pd.concat(frames,ignore_index=True); history_e=pd.DataFrame(H)
metrics_e.to_csv(OUT/'metrics.csv',index=False)
predictions_e.to_csv(OUT/'predictions.csv',index=False)
history_e.to_csv(OUT/'training_history.csv',index=False)
assert len(metrics_e)==3*4*2*2
assert history_e.groupby(['Seed','Stage','Budget']).Epoch.max().tolist().count(800)==6
# 复核100轮与此前实验完全一致（若此前结果存在）。
prior=OUT.parent/'mlp_stacking_11/predictions.csv'
if prior.exists():
    old=pd.read_csv(prior,parse_dates=['Date'])
    new=predictions_e[predictions_e.Epochs==100]
    check=old.merge(new,on=['Date','Seed','Model','Split'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==len(old)==len(new)
    assert np.allclose(check.Actual_old,check.Actual_new,rtol=0,atol=1e-10)
    delta=np.max(np.abs(check.Predicted_old-check.Predicted_new))
    assert delta<1e-4
    print(f'100轮复现检查通过，最大预测差={delta:.8g}')
stack_metrics=metrics_e[metrics_e.Model=='Stacking']
comparison_e=stack_metrics[stack_metrics.Split=='Test'].pivot(index='Seed',columns='Epochs',values='MAPE').reindex(SEEDS)
summary_e=stack_metrics.groupby(['Epochs','Split']).MAPE.agg(['mean','std','min','max'])
paired_e=metrics_e[metrics_e.Split=='Test'].pivot(index=['Epochs','Seed'],columns='Model',values='MAPE')
paired_e['Stacking_minus_Base_pp']=paired_e.Stacking-paired_e.Base
comparison_e.to_csv(OUT/'test_mape_by_seed.csv'); summary_e.to_csv(OUT/'summary.csv'); paired_e.to_csv(OUT/'paired_comparison.csv')
print('堆叠模型2020测试MAPE (%)：'); display(comparison_e.round(4))
print('堆叠模型训练/测试MAPE均值及样本标准差：'); display(summary_e.round(4))
print('同轮数堆叠与第一阶段基准的测试对比：'); display(paired_e.round(4))
''')
code('''colors={100:'#0072B2',200:'#009E73',400:'#D55E00',800:'#CC79A7'}
fig,axes=plt.subplots(3,1,figsize=(13,10),constrained_layout=True)
for ax,seed in zip(axes,SEEDS):
    ax.plot(te,y.loc[te],color='black',lw=1.7,label='Actual')
    for budget in BUDGETS:
        g=predictions_e[(predictions_e.Seed==seed)&(predictions_e.Epochs==budget)&(predictions_e.Model=='Stacking')&(predictions_e.Split=='Test')]
        ax.plot(g.Date,g.Predicted,color=colors[budget],lw=1.1,label=f'{budget}: MAPE {comparison_e.loc[seed,budget]:.2f}%')
    ax.set(title=f'Stacking | 2020 test | Seed {seed}',ylabel='Price (dataset units)'); ax.legend(ncol=3); ax.grid(alpha=.2)
fig.savefig(OUT/'predictions_by_epochs.png',dpi=150); plt.close(fig)
display(Image(filename=str(OUT/'predictions_by_epochs.png')))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
comparison_e.plot.bar(ax=axes[0],rot=0,color=[colors[e] for e in BUDGETS])
axes[0].set(title='Stacking test MAPE by seed',ylabel='MAPE (%)')
for c in axes[0].containers: axes[0].bar_label(c,fmt='%.2f',padding=3,fontsize=8)
axes[0].margins(y=.2)
for split,col in [('Train','#009E73'),('Test','#0072B2')]:
    g=summary_e.xs(split,level='Split').reindex(BUDGETS)
    axes[1].errorbar(range(4),g['mean'],yerr=g['std'],marker='o',capsize=5,color=col,label=split+' mean ± sample SD')
axes[1].set(xticks=range(4),xticklabels=BUDGETS,xlabel='Epochs per stage',ylabel='MAPE (%)',title='Stacking | Three-seed summary')
axes[1].legend(); axes[1].grid(alpha=.2)
fig.savefig(OUT/'epoch_comparison.png',dpi=150); plt.close(fig)
display(Image(filename=str(OUT/'epoch_comparison.png')))
monthly=predictions_e[predictions_e.Split=='Test'].copy()
monthly['Month']=monthly.Date.dt.strftime('%Y-%m')
monthly.groupby(['Month','Seed','Epochs','Model']).APE.mean().to_csv(OUT/'monthly_mape.csv')
''')
nb.validate(n); nb.write(n,'outputs/11_改变模型构建.ipynb')
