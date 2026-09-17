from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/11_改变模型构建.ipynb')
n=nb.read(src,as_version=4)
nb.write(n,'outputs/mlp_ten_seeds_ma_11/11_before_ma_refactor.ipynb')
start=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and c.source.startswith('# 十个种子：100与200轮、三种模型'))
Path('outputs/mlp_ten_seeds_ma_11/start_cell.txt').write_text(str(start))
# Keep earlier cells verbatim; replace only this final experiment and its conclusion.
for c in n.cells[start:]:
 if c.cell_type=='markdown':
  c.source=c.source.replace('共同样本、特征定义、架构和训练设置','共同样本、架构和训练设置；仅修改DirectReturns的特征定义')
  c.source=c.source.replace('金铜5/20期累计收益率','金铜相对5/20期滚动均价的涨幅（P(t)/MA_k(t)−1，均价包含当日）')
  if c.source.startswith('# 十个种子：'):
   c.source+='\n\n本节按要求重构，旧版已备份。滚动窗口按原数据相邻记录计数；DirectReturns仅加4项相对均价涨幅，不加昨日涨幅。为保持前后可比，仍剔除最初20条历史不足记录，三组共同样本仍为2034条训练、271条测试。旧实验和旧结果目录不覆盖。\n'
 if c.cell_type!='code': continue
 c.source=c.source.replace('MLP11_TEN_OUT','MLP11_TEN_MA_OUT').replace("results/mlp_ten_seeds_11", "results/mlp_ten_seeds_ma_11")
 c.source=c.source.replace("name=f'{metal}_return{k}'", "name=f'{metal}_ma_deviation{k}'")
 c.source=c.source.replace("x[name]=raw[col]/raw[col].shift(k)-1", "x[name]=raw[col]/raw[col].rolling(k,min_periods=k).mean()-1")
 c.source=c.source.replace('x=x.loc[valid]; y=y.loc[x.index]', 'valid = valid & (x.index >= raw.index[20])\nx=x.loc[valid]; y=y.loc[x.index]\nassert len(x.index[x.index.year<2020])==2034\nfor metal in [\'gold\',\'copper\']:\n    for k in [5,20]:\n        assert np.allclose(x[f\'{metal}_ma_deviation{k}\'], x[f\'{metal}_vs_ma{k}\'])')
 c.source=c.source.replace("'P(t)/P(t-k)-1; k=5,20'", "'P(t)/MA_k(t)-1; k=5,20; trailing mean includes current record'")
 if '# 200轮的前三个种子' in c.source:
  c.source=c.source[:c.source.index('# 200轮的前三个种子')]+'''# Base与Stacking不变，核对十种子所有预算和预测；DirectReturns已改定义。
prior=OUT.parent/'mlp_ten_seeds_11/predictions.csv'
if prior.exists():
    old=pd.read_csv(prior,parse_dates=['Date'])
    old=old[old.Model.isin(['Base','Stacking'])]
    new=predictions10[predictions10.Model.isin(['Base','Stacking'])]
    check=old.merge(new,on=['Date','Seed','Budget','Model','Split'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(check)==len(old)==len(new)
    assert np.allclose(check.Predicted_old,check.Predicted_new,rtol=0,atol=1e-6)
    print('复核通过：Base与Stacking所有种子和轮数的预测保持一致；DirectReturns已改为相对滚动均价涨幅。')
'''
 c.outputs=[]; c.execution_count=None
n.cells=[c for i,c in enumerate(n.cells) if not (i>=start and c.cell_type=='markdown' and c.source.startswith('## 十种子实验结论'))]
nb.validate(n); nb.write(n,'outputs/11_改变模型构建.ipynb')
assert n.cells[:start]==nb.read(src,as_version=4).cells[:start]
