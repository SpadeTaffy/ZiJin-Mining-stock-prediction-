from pathlib import Path
import nbformat as nb,hashlib
root=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
p=root/'notebooks/04_采用神经网络模型/10_再退回原始模型,确认近日涨幅的作用.ipynb'
n=nb.read(p,as_version=4)
Path('outputs/mlp_annual_three_models_10/original.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
Path('outputs/mlp_annual_three_models_10/start_cell.txt').write_text(str(len(n.cells)))
def md(s): n.cells.append(nb.v4.new_markdown_cell(s))
def code(s): n.cells.append(nb.v4.new_code_cell(s))
md('''# 2015—2026逐年比较三个模型：跨年度稳定性

保留前面的2020实验。本节固定Base_100、Base_400、Full_100三个配置，每个测试年仅使用此前全部共同样本重新训练，年内不重训。Base的100/400轮来自同一训练路径；Full独立训练100轮。ReLU、16/32隐藏层、Adam 0.01、全批量、seed42保持不变。金铜涨幅指5/20条记录累计涨幅。

每年输入和目标的标准化只拟合当年训练集，三组使用完全相同日期。主要指标是**先算每年测试MAPE，再对年份等权平均**，另外列出全部测试记录合并MAPE、中位数、最差年份和年度胜出次数。2026为部分年度，另报告2015—2025完整年度的平均值。

本节检验跨年份稳定性，仍为单随机种子，并非随机初始化鲁棒性检验。此前已查看过这些历史年份，本轮属于历史回测；目标仍是原数据同日股价，不等于可直接交易的次日预测。''')
# Reuse the verified data preparation, keeping independent execution.
s=n.cells[3].source
s=s.replace("os.environ.get('MLP10_OUT',str(ROOT/'results/mlp_base_vs_returns_10'))","os.environ.get('MLP10_ANNUAL_OUT',str(ROOT/'results/mlp_annual_three_models_10'))")
s=s[:s.index('tr=x.index')]+'''YEARS=list(range(2015,2027))
assert np.isfinite(x.to_numpy()).all() and np.isfinite(y).all() and (y>0).all()
config={'years':YEARS,'models':{'Base':[100,400],'Full':[100]},'base_features':BASE,'full_features':FULL,'seed':SEED,'lr':LR,'hidden':[16,32],'activation':'ReLU','batch':'full','target':'original same-date price','split':'expanding training, fixed model within test year','dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'torch_version':str(torch.__version__)}
(OUT/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
order=['Base_100','Base_400','Full_100']
'''
code(s)
code('''metric_rows=[]; frames=[]; histories=[]; split_rows=[]
for year in YEARS:
    tr=x.index[x.index.year<year]; te=x.index[x.index.year==year]
    assert len(tr)>0 and len(te)>0 and tr.max()<te.min()
    for split,idx in [('Train',tr),('Test',te)]:
        split_rows.append({'Year':year,'Split':split,'N':len(idx),'Start':str(idx.min().date()),'End':str(idx.max().date())})
    sy=StandardScaler().fit(y.loc[tr].to_numpy().reshape(-1,1))
    yt=torch.tensor(sy.transform(y.loc[tr].to_numpy().reshape(-1,1)),dtype=torch.float32)
    for family,cols,checkpoints in [('Base',BASE,[100,400]),('Full',FULL,[100])]:
        sx=StandardScaler().fit(x.loc[tr,cols])
        xt=torch.tensor(sx.transform(x.loc[tr,cols]),dtype=torch.float32)
        torch.manual_seed(SEED)
        model=nn.Sequential(nn.Linear(len(cols),16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        for epoch in range(1,max(checkpoints)+1):
            model.train();optimizer.zero_grad()
            loss=((model(xt)-yt)**2).mean();assert torch.isfinite(loss)
            loss.backward();optimizer.step();model.eval()
            with torch.no_grad(): mse=((model(xt)-yt)**2).mean().item()
            histories.append({'Year':year,'Family':family,'Epoch':epoch,'Train_MSE_standardized':mse})
            if epoch not in checkpoints:continue
            name=f'{family}_{epoch}'
            torch.save({'state_dict':model.state_dict(),'features':cols,'epoch':epoch,'seed':SEED,
                        'train_start':str(tr.min()),'train_end':str(tr.max()),
                        'sx_mean':sx.mean_.tolist(),'sx_scale':sx.scale_.tolist(),
                        'sy_mean':sy.mean_.tolist(),'sy_scale':sy.scale_.tolist()},OUT/f'{year}_{name}.pt')
            for split,idx in [('Train',tr),('Test',te)]:
                with torch.no_grad(): z=model(torch.tensor(sx.transform(x.loc[idx,cols]),dtype=torch.float32)).numpy()
                pred=sy.inverse_transform(z).ravel();actual=y.loc[idx].to_numpy();err=pred-actual
                assert np.isfinite(pred).all()
                frames.append(pd.DataFrame({'Year':year,'Date':idx,'Model':name,'Split':split,'Actual':actual,'Predicted':pred,'APE':100*np.abs(err)/actual,'PE':100*err/actual}))
                metric_rows.append({'Year':year,'Model':name,'Split':split,'N':len(idx),'MAPE':100*np.mean(np.abs(err)/actual),'MAE':np.abs(err).mean(),'RMSE':np.sqrt(np.mean(err**2)),'MPE':100*np.mean(err/actual)})
    print(f'{year} complete: train={len(tr)}, test={len(te)}, test end={te.max().date()}',flush=True)
metrics=pd.DataFrame(metric_rows);predictions=pd.concat(frames,ignore_index=True);history=pd.DataFrame(histories);splits=pd.DataFrame(split_rows)
for name,frame in [('metrics',metrics),('predictions',predictions),('training_history',history),('splits',splits)]:frame.to_csv(OUT/f'{name}.csv',index=False)
# Each model must use identical dates; verify 2020 matches previous experiment.
assert len(metrics)==12*3*2
for year in YEARS:
    groups=[predictions[(predictions.Year==year)&(predictions.Split=='Test')&(predictions.Model==m)] for m in order]
    assert all(np.array_equal(g.Date.to_numpy(),groups[0].Date.to_numpy()) for g in groups)
previous=ROOT/'results/mlp_base_vs_returns_10/predictions.csv'
if previous.exists():
    old=pd.read_csv(previous,parse_dates=['Date'])
    for name in order:
        a=old[(old.Split=='Test')&(old.Model==name)].sort_values('Date')
        b=predictions[(predictions.Year==2020)&(predictions.Split=='Test')&(predictions.Model==name)].sort_values('Date')
        assert np.array_equal(a.Date.to_numpy(),b.Date.to_numpy()) and np.allclose(a.Predicted,b.Predicted,atol=1e-5)
    print('2020三个模型均复现上一节结果。')
# Cross-check all Full_100 annual runs against prior experiment when available.
previous_full=ROOT/'results/mlp_epochs100_400_annual_08/predictions.csv'
if previous_full.exists():
    old=pd.read_csv(previous_full,parse_dates=['Date'])
    for year in YEARS:
        a=old[(old.Year==year)&(old.Split=='Test')&(old.Epoch==100)].sort_values('Date')
        b=predictions[(predictions.Year==year)&(predictions.Split=='Test')&(predictions.Model=='Full_100')].sort_values('Date')
        assert np.array_equal(a.Date.to_numpy(),b.Date.to_numpy()) and np.allclose(a.Predicted,b.Predicted,atol=1e-5)
    print('2015—2026 Full_100全部复现08结果。')
''')
code('''annual=metrics[metrics.Split=='Test'].pivot(index='Year',columns='Model',values='MAPE')[order]
train=metrics[metrics.Split=='Train'].pivot(index='Year',columns='Model',values='MAPE')[order]
annual['Winner']=annual[order].idxmin(axis=1)
annual['Base400_minus_Base100_pp']=annual.Base_400-annual.Base_100
annual['Full100_minus_Base100_pp']=annual.Full_100-annual.Base_100
summary=[]
for label,years in [('2015-2026 (2026 partial)',YEARS),('2015-2025 complete',list(range(2015,2026)))]:
    a=annual.loc[years,order]
    for model in order:
        pooled=predictions[(predictions.Split=='Test')&(predictions.Year.isin(years))&(predictions.Model==model)]
        summary.append({'Period':label,'Model':model,'Mean_annual_MAPE':a[model].mean(),'Pooled_MAPE':pooled.APE.mean(),'Median_annual_MAPE':a[model].median(),'Std_annual_MAPE':a[model].std(),'Worst_MAPE':a[model].max(),'Worst_year':int(a[model].idxmax()),'Winning_years':int((a.idxmin(axis=1)==model).sum()),'Train_mean_MAPE':train.loc[years,model].mean()})
summary=pd.DataFrame(summary)
annual.to_csv(OUT/'annual_test_mape.csv');train.to_csv(OUT/'annual_train_mape.csv');summary.to_csv(OUT/'summary.csv',index=False)
display(splits[splits.Split=='Test']);display(annual.round(4));display(summary.round(4))
''')
code('''colors={'Base_100':'#0072B2','Base_400':'#D55E00','Full_100':'#009E73'}
fig,axes=plt.subplots(2,1,figsize=(14,9),constrained_layout=True)
a=annual[order].copy();a.index=[str(v)+('*' if v==2026 else '') for v in a.index]
a.plot.bar(ax=axes[0],color=list(colors.values()),rot=0)
axes[0].set(title='Annual test MAPE | *2026 partial year',ylabel='MAPE (%)',xlabel='Test year')
for c in axes[0].containers:axes[0].bar_label(c,fmt='%.1f',padding=2,fontsize=8)
b=summary.pivot(index='Period',columns='Model',values='Mean_annual_MAPE')[order]
b.plot.bar(ax=axes[1],color=list(colors.values()),rot=0)
axes[1].set(title='Equal-year mean test MAPE',ylabel='MAPE (%)',xlabel='Period')
for c in axes[1].containers:axes[1].bar_label(c,fmt='%.2f',padding=3)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.margins(y=.16)
fig.savefig(OUT/'annual_mape_comparison.png',dpi=150);plt.close(fig);display(Image(filename=str(OUT/'annual_mape_comparison.png')))
fig,axes=plt.subplots(4,3,figsize=(16,13),constrained_layout=True)
for ax,year in zip(axes.flat,YEARS):
    test=predictions[(predictions.Year==year)&(predictions.Split=='Test')]
    actual=test[test.Model=='Base_100'];ax.plot(actual.Date,actual.Actual,color='black',lw=1.1,label='Actual')
    for name in order:
        g=test[test.Model==name];ax.plot(g.Date,g.Predicted,color=colors[name],lw=.9,label=name)
    ax.set_title(str(year)+(' (partial)' if year==2026 else ''));ax.grid(alpha=.2)
    ax.tick_params(axis='x',labelrotation=30,labelsize=7)
axes.flat[0].legend(fontsize=8)
fig.suptitle('Annual held-out predictions | same-date target price',fontsize=16)
fig.savefig(OUT/'annual_predictions.png',dpi=140);plt.close(fig);display(Image(filename=str(OUT/'annual_predictions.png')))
plot=metrics[metrics.Split=='Test'].copy();plot['Year']=plot.Year.astype(str)
fig=px.bar(plot,x='Year',y='MAPE',color='Model',barmode='group',color_discrete_map=colors,category_orders={'Model':order},title='Annual test MAPE | 2026 partial',height=550)
fig.write_html(OUT/'annual_mape_comparison.html',include_plotlyjs=True);display(fig)
''')
nb.write(n,Path('outputs')/p.name)
