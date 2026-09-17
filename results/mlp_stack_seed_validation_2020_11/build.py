from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(src,as_version=4); nb.write(n,'outputs/mlp_stack_seed_validation_2020_11/11_before_stack_selection.ipynb')
Path('outputs/mlp_stack_seed_validation_2020_11/start_cell.txt').write_text(str(len(n.cells)))
start=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and c.source.startswith('# 2020年前三个月验证选种子：'))
oldcells=n.cells[start:]
setup=next(c.source for c in oldcells if c.cell_type=='code' and 'MLP11_SEEDVAL_OUT' in c.source)
selection=next(c.source for c in oldcells if c.cell_type=='code' and 'def metrics_at' in c.source)
evaluation=next(c.source for c in oldcells if c.cell_type=='code' and 'locked=json' in c.source)
plot=next(c.source for c in oldcells if c.cell_type=='code' and 'history=[]; preds={}' in c.source)
plot=plot[plot.index('# 按请求先展示'):].replace('DirectMA | Seed','Stacking | Seed')
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 两阶段堆叠：五种子，2020年第一季度验证选种子

“嵌套模型”沿用此前Stacking定义：第一阶段原7变量→股价，第二阶段第一阶段预测＋金铜昨日涨幅与相对5/20期滚动均价涨幅（7项输入）→最终股价。两个阶段各16→32隐藏层、ReLU、Adam学习率0.01、全批量，各100轮；第一阶段冻结，不用折外预测，第二阶段用样本内预测训练。

种子42、7、123、2024、2025；每个候选的两个阶段均使用对应种子初始化。恢复截图轮数实验的2035条训练记录（2012-06-05至2019-12-31），窗口按原始记录计数且包含当日。标准化只拟合训练集。

先展示五组2020年预测；仅以2020年1—3月的验证MAPE排序选中一个候选，精确并列按给定种子顺序。保存选择记录后评估4—12月，锁定模型，不重新训练、不用测试结果改选。全年的展示与其余候选测试指标仅供事后说明。

本节对应截图中的堆叠模型，而上一节是2034条训练记录的单阶段DirectMA；两节不是严格同样本的模型结构对照。2020年已反复分析过，本次为历史回放，不是全新盲测。
''')
setup=setup.replace('MLP11_SEEDVAL_OUT','MLP11_STACK_SEEDVAL_OUT').replace('results/mlp_seed_validation_2020_11','results/mlp_stack_seed_validation_2020_11')
setup=setup.replace('x=raw.copy(); DIRECT=[]','''x=raw.copy(); DIRECT=[]
for metal,col in [('gold','Gold Futures Price'),('copper','Copper Futures Price')]:
    name=f'{metal}_daily'; DIRECT.append(name); x[name]=raw[col]/raw[col].shift(1)-1''')
# Reorder to screenshot's stage2 sequence: gold daily/5/20, copper daily/5/20.
setup=setup.replace('COLS=BASE+DIRECT', "DIRECT=[f'{metal}_{suffix}' for metal in ['gold','copper'] for suffix in ['daily','ma_deviation5','ma_deviation20']]\nCOLS=BASE")
setup=setup.replace('raw.index[20]','raw.index[19]').replace('len(tr)==2034','len(tr)==2035')
setup=setup.replace("features=COLS", "base_features=BASE,stage2_features=['base_prediction']+DIRECT,architecture='two-stage stacking',epochs_per_stage=100,stage2_training='in-sample predictions; base frozen'")
setup=setup.replace('common_warmup_records=20','common_warmup_records=19')
# Shared precomputed input tensors not needed; each stage fits its own scaler.
setup=setup[:setup.index('sx=StandardScaler().fit')]
code(setup)
code('''history=[]; preds={}
def fit_stage(X,seed,stage):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sx=StandardScaler().fit(X.loc[tr]); sy=StandardScaler().fit(y.loc[tr].to_numpy().reshape(-1,1))
    xt=torch.tensor(sx.transform(X.loc[tr]),dtype=torch.float32)
    yt=torch.tensor(sy.transform(y.loc[tr].to_numpy().reshape(-1,1)),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,EPOCHS+1):
        model.train();optimizer.zero_grad();loss=((model(xt)-yt)**2).mean();assert torch.isfinite(loss)
        loss.backward();optimizer.step();model.eval()
        with torch.no_grad():z=model(xt).numpy()
        p=sy.inverse_transform(z).ravel();mse=float(np.mean((p-y.loc[tr].to_numpy())**2))
        history.append(dict(Seed=seed,Stage=stage,Epoch=epoch,Train_MSE=mse,Train_MSE_standardized=mse/float(sy.scale_[0]**2)))
    with torch.no_grad():z=model(torch.tensor(sx.transform(X),dtype=torch.float32)).numpy()
    pred=pd.Series(sy.inverse_transform(z).ravel(),index=X.index);assert np.isfinite(pred).all()
    torch.save(dict(state_dict=model.state_dict(),features=list(X.columns),seed=seed,epochs=EPOCHS,train_start=str(tr.min()),train_end=str(tr.max()),sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist()),OUT/f'{stage}_seed{seed}.pt')
    return pred
for seed in SEEDS:
    bp=fit_stage(x[BASE],seed,'Base')
    meta=x[DIRECT].copy();meta.insert(0,'base_prediction',bp)
    preds[seed]=fit_stage(meta,seed,'Stacking')
    meta.to_csv(OUT/f'stage2_inputs_seed{seed}.csv',index_label='Date')
pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
assert len(history)==5*2*100
predictions=pd.concat([pd.DataFrame(dict(Date=x.index,Seed=seed,Actual=y.to_numpy(),Predicted=preds[seed].to_numpy(),Split=np.where(x.index<'2020-01-01','Train',np.where(x.index<'2020-04-01','Validation','Test')))) for seed in SEEDS],ignore_index=True)
predictions.to_csv(OUT/'predictions.csv',index=False)
# 与截图实验原有三个种子进行数值复现检查（CSV按float32精度往返）。
prior=OUT.parent/'mlp_stacking_epochs_11/predictions.csv'
if prior.exists():
    old=pd.read_csv(prior,parse_dates=['Date']);old=old[(old.Model=='Stacking')&(old.Epochs==100)]
    new=predictions[predictions.Seed.isin([42,7,123])]
    check=old.merge(new,on=['Date','Seed'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==len(old)==len(new)
    delta=np.max(np.abs(check.Predicted_old-check.Predicted_new))
    assert delta<1e-5
    print(f'截图实验复现通过：42/7/123三个种子、100轮预测最大差{delta:.8g}。')
'''+plot)
md('''## 第一季度验证排名与选择

仅使用第一季度MAPE，先保存选择，再查看4—12月测试结果。''')
code(selection)
md('''## 锁定后评估4—12月

展示所有候选的后续表现只是回顾验证选择是否有效，不据此重新选择。''')
code(evaluation)
nb.validate(n);nb.write(n,'outputs/11_改变模型构建.ipynb')
