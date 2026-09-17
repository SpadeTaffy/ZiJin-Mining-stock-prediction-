from pathlib import Path
import nbformat as nb
p=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(p,as_version=4); nb.write(n,'outputs/mlp_three_models_200_11/11_before_three_models.ipynb')
Path('outputs/mlp_three_models_200_11/start_cell.txt').write_text(str(len(n.cells)))
def md(s): n.cells.append(nb.v4.new_markdown_cell(s))
def code(s): n.cells.append(nb.v4.new_code_cell(s))
md('''# 200轮：基准、堆叠、原始输入直接加入金铜收益率

三个种子42、7、123；训练截至2019年末，2020年测试；16→32隐藏层、ReLU、Adam学习率0.01、全批量，各网络200轮。

- **Base**：原7变量→股价。
- **Stacking**：Base预测＋原有6项金铜指标（各自昨日涨幅、相对5/20期均价涨幅）→股价；两个阶段各200轮，样本内训练，不用折外预测，冻结Base。
- **DirectReturns**：原7变量＋金、铜各5/20期收益率，共11项输入→股价，单阶段200轮。此处收益率沿用10：`P(t)/P(t-k)-1`，不是相对移动均价的偏离。

三组使用完全相同的有效训练/测试日期。新增20期收益率需要20条历史，较上一实验共同少1条训练记录，因此统一重新训练，不能把本次基准数值与上一实验视为完全相同。

所有标准化器只拟合共同训练集。训练损失为标准化目标MSE，另报告还原后的原股价尺度MSE（单位为数据股价单位的平方）。各轮训练MSE均为当轮更新后计算。堆叠训练MSE指第二阶段最终输出；另保留其第一阶段曲线。三组特征集合不完全相同，本实验比较具体方案，不能单独归因于堆叠结构。
''')
setup=next(c.source for c in n.cells if c.cell_type=='code')
setup=setup.replace("'MLP11_OUT'", "'MLP11_THREE_OUT'").replace('results/mlp_stacking_11','results/mlp_three_models_200_11').replace('EPOCHS=100','EPOCHS=200')
# Add return features before common finite filter.
setup=setup.replace('valid=np.isfinite', '''DIRECT=[]
for metal,col in [('gold','Gold Futures Price'),('copper','Copper Futures Price')]:
    for k in [5,20]:
        name=f'{metal}_return{k}'
        x[name]=raw[col]/raw[col].shift(k)-1
        DIRECT.append(name)
FULL=BASE+DIRECT
valid=np.isfinite''')
code(setup)
code('''config['direct_features']=FULL
config['direct_return_formula']='P(t)/P(t-k)-1; k=5,20'
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
H=[]
def fit3(X,Y,seed,label):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sx=StandardScaler().fit(X); sy=StandardScaler().fit(np.asarray(Y).reshape(-1,1))
    xt=torch.tensor(sx.transform(X),dtype=torch.float32)
    yt=torch.tensor(sy.transform(np.asarray(Y).reshape(-1,1)),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(X.shape[1],16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(1,EPOCHS+1):
        model.train(); opt.zero_grad(); loss=((model(xt)-yt)**2).mean()
        assert torch.isfinite(loss); loss.backward(); opt.step(); model.eval()
        with torch.no_grad(): z=model(xt).numpy()
        p=sy.inverse_transform(z).ravel()
        mse=float(np.mean((p-np.asarray(Y))**2))
        H.append(dict(Seed=seed,Model=label,Epoch=epoch,Train_MSE=mse,Train_MSE_standardized=mse/float(sy.scale_[0]**2)))
    return model,sx,sy

def pred3(bundle,X):
    model,sx,sy=bundle; model.eval()
    with torch.no_grad(): z=model(torch.tensor(sx.transform(X),dtype=torch.float32)).numpy()
    result=sy.inverse_transform(z).ravel(); assert np.isfinite(result).all()
    return pd.Series(result,index=X.index)

rows=[]; frames=[]
for seed in SEEDS:
    base=fit3(x.loc[tr,BASE],y.loc[tr],seed,'Base')
    bp=pred3(base,x[BASE])
    meta=x[RETURNS].copy(); meta.insert(0,'base_prediction',bp)
    stack=fit3(meta.loc[tr],y.loc[tr],seed,'Stacking')
    direct=fit3(x.loc[tr,FULL],y.loc[tr],seed,'DirectReturns')
    for name,bundle,X in [('Base',base,x[BASE]),('Stacking',stack,meta),('DirectReturns',direct,x[FULL])]:
        pred=pred3(bundle,X); model,sx,sy=bundle
        torch.save(dict(state_dict=model.state_dict(),features=list(X.columns),seed=seed,epochs=EPOCHS,
            sx_mean=sx.mean_.tolist(),sx_scale=sx.scale_.tolist(),sy_mean=sy.mean_.tolist(),sy_scale=sy.scale_.tolist()),OUT/f'{name}_seed{seed}.pt')
        for split,idx in [('Train',tr),('Test',te)]:
            a=y.loc[idx].to_numpy(); p=pred.loc[idx].to_numpy(); mse=np.mean((p-a)**2)
            rows.append(dict(Seed=seed,Model=name,Split=split,N=len(idx),MAPE=100*np.mean(np.abs(p-a)/np.abs(a)),MSE=mse,MSE_standardized=mse/float(sy.scale_[0]**2)))
            frames.append(pd.DataFrame(dict(Date=idx,Seed=seed,Model=name,Split=split,Actual=a,Predicted=p)))
    print(f'Seed {seed}: all three models completed')
metrics3=pd.DataFrame(rows); history3=pd.DataFrame(H); predictions3=pd.concat(frames,ignore_index=True)
assert len(metrics3)==18 and len(history3)==1800
for row in metrics3[metrics3.Split=='Train'].itertuples():
    end=history3[(history3.Seed==row.Seed)&(history3.Model==row.Model)&(history3.Epoch==200)].iloc[0]
    assert np.isclose(end.Train_MSE,row.MSE,atol=1e-8)
metrics3.to_csv(OUT/'metrics.csv',index=False); history3.to_csv(OUT/'training_history.csv',index=False); predictions3.to_csv(OUT/'predictions.csv',index=False)
summary3=metrics3.groupby(['Model','Split'])[['MAPE','MSE','MSE_standardized']].agg(['mean','std'])
summary3.to_csv(OUT/'summary.csv')
order=['Base','Stacking','DirectReturns']
train_mse3=metrics3[metrics3.Split=='Train'].pivot(index='Seed',columns='Model',values='MSE').reindex(index=SEEDS,columns=order)
train_std3=metrics3[metrics3.Split=='Train'].pivot(index='Seed',columns='Model',values='MSE_standardized').reindex(index=SEEDS,columns=order)
test_mape3=metrics3[metrics3.Split=='Test'].pivot(index='Seed',columns='Model',values='MAPE').reindex(index=SEEDS,columns=order)
print('训练MSE（原股价尺度）：'); display(train_mse3.round(6))
print('训练MSE（标准化目标）：'); display(train_std3.round(6))
print('2020测试MAPE (%)：'); display(test_mape3.round(4))
display(summary3.round(6))
''')
code('''colors={'Base':'#0072B2','Stacking':'#D55E00','DirectReturns':'#009E73'}
fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
for ax,seed in zip(axes,SEEDS):
    for name in order:
        g=history3[(history3.Seed==seed)&(history3.Model==name)]
        ax.plot(g.Epoch,g.Train_MSE,color=colors[name],label=name)
    ax.set(title=f'Seed {seed}',xlabel='Epoch',ylabel='Train MSE (original price scale)',yscale='log'); ax.grid(alpha=.2); ax.legend()
fig.savefig(OUT/'training_mse.png',dpi=160); plt.close(fig)
display(Image(filename=str(OUT/'training_mse.png')))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,table,title,ylabel in [(axes[0],train_mse3,'Train MSE at epoch 200','MSE (original price scale)'),(axes[1],test_mape3,'2020 test MAPE','MAPE (%)')]:
    table.plot.bar(ax=ax,rot=0,color=[colors[m] for m in order])
    ax.set(title=title,ylabel=ylabel); ax.margins(y=.2)
    for c in ax.containers: ax.bar_label(c,fmt='%.3f',padding=3,fontsize=8)
fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
display(Image(filename=str(OUT/'comparison.png')))
''')
nb.validate(n); nb.write(n,'outputs/11_改变模型构建.ipynb')
