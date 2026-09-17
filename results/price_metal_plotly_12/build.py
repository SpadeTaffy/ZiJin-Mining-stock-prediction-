from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/12_减少预测的时间,扩大验证集的范围.ipynb')
n=nb.read(src,as_version=4);nb.write(n,'outputs/price_metal_plotly_12/12_before_price_chart.ipynb')
Path('outputs/price_metal_plotly_12/start_cell.txt').write_text(str(len(n.cells)))
n.cells.append(nb.v4.new_markdown_cell('''# 股价、黄金、铜：统一尺度与滚动涨幅（Plotly Express）

展示2020年。上图三条价格统一换算为**2020年首条记录=100**，因此比较的是累计相对走势，非原始价格单位；金铜5/20期滚动均价用相同基准换算，在图例中点击可显示。均价在全年历史上计算后再截取2020，不在年初重新起算。

下方分别展示黄金、铜的3项模型涨幅：相邻一期P(t)/P(t−1)−1、相对5期均价P(t)/MA5(t)−1、相对20期均价P(t)/MA20(t)−1，均乘100以百分数展示。两个涨幅面板使用完全一致的纵轴范围，方便比较幅度；价格指数和百分比是不同量纲，不共用同一个纵轴。

滚动均价包含当日，窗口按原数据记录计数。图支持拖动缩放、悬停查看、点击图例切换曲线，底部日期滑块联动三个面板；“全部”按钮恢复全年。原数据使用原Dataset的价格口径，不额外改变复权和日期对齐。
'''))
n.cells.append(nb.v4.new_code_cell('''from pathlib import Path
import os,json
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display
ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/pickle/Dataset.pkl').exists()),Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-'))
OUT_PX=Path(os.environ.get('MLP12_PX_OUT',str(ROOT/'results/price_metal_plotly_12')));OUT_PX.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(ROOT/'src/pickle/Dataset.pkl');raw=d['X'].copy();target=d['Y']
assert raw.index.equals(target.index) and raw.index.is_unique and raw.index.is_monotonic_increasing
prices=pd.DataFrame({'股价':np.asarray(target).reshape(-1),'黄金':raw['Gold Futures Price'].to_numpy(),'铜':raw['Copper Futures Price'].to_numpy()},index=raw.index)
ma={k:prices[['黄金','铜']].rolling(k,min_periods=k).mean() for k in [5,20]}
returns={}
for metal in ['黄金','铜']:
    returns[f'{metal} · 昨日涨跌幅']=100*(prices[metal]/prices[metal].shift(1)-1)
    for k in [5,20]:returns[f'{metal} · 相对MA{k}涨幅']=100*(prices[metal]/ma[k][metal]-1)
idx=prices.index[prices.index.year==2020];assert len(idx)==271
base=prices.loc[idx[0]];indexed=100*prices.loc[idx]/base
ret=pd.DataFrame(returns).loc[idx];assert np.isfinite(ret.to_numpy()).all()
panels=['价格指数（年初=100）','黄金涨跌幅（%）','铜涨跌幅（%）']
frames=[]
for col in prices:
    frames.append(pd.DataFrame({'日期':idx,'数值':indexed[col].to_numpy(),'曲线':col,'面板':panels[0]}))
for metal in ['黄金','铜']:
    for k in [5,20]:
        frames.append(pd.DataFrame({'日期':idx,'数值':(100*ma[k].loc[idx,metal]/base[metal]).to_numpy(),'曲线':f'{metal} MA{k}','面板':panels[0]}))
for name,values in ret.items():
    frames.append(pd.DataFrame({'日期':idx,'数值':values.to_numpy(),'曲线':name,'面板':panels[1] if name.startswith('黄金') else panels[2]}))
plotdata=pd.concat(frames,ignore_index=True)
colors={'股价':'#222222','黄金':'#D59400','铜':'#0072B2','黄金 MA5':'#D59400','黄金 MA20':'#D59400','铜 MA5':'#0072B2','铜 MA20':'#0072B2'}
for metal in ['黄金','铜']:
    colors.update({f'{metal} · 昨日涨跌幅':'#9AA0A6',f'{metal} · 相对MA5涨幅':'#009E73',f'{metal} · 相对MA20涨幅':'#CC79A7'})
fig_px=px.line(plotdata,x='日期',y='数值',color='曲线',facet_row='面板',category_orders={'面板':panels,'曲线':list(colors)},color_discrete_map=colors,render_mode='svg',height=1050,title='2020｜股价与金铜价格、滚动均价及涨跌幅')
# px facet_row按类别顺序从上到下排列：yaxis3价格，yaxis2黄金，yaxis铜。
assert next(t for t in fig_px.data if t.name=='股价').yaxis=='y3'
fig_px.update_yaxes(matches=None,showticklabels=True)
limit=float(np.abs(ret.to_numpy()).max()*1.08)
fig_px.update_layout(yaxis=dict(title='涨跌幅 (%)',range=[-limit,limit],ticksuffix='%',zeroline=True,zerolinecolor='#888'),yaxis2=dict(title='涨跌幅 (%)',range=[-limit,limit],matches='y',ticksuffix='%',zeroline=True,zerolinecolor='#888'),yaxis3=dict(title='价格指数',zeroline=False))
for trace in fig_px.data:
    if trace.name.endswith('MA5') and ' · ' not in trace.name:trace.visible='legendonly';trace.line.dash='dash'
    if trace.name.endswith('MA20') and ' · ' not in trace.name:trace.visible='legendonly';trace.line.dash='dot'
    trace.line.width=1.6 if trace.name!='股价' else 2.2
    unit='%' if ' · ' in trace.name else ''
    trace.hovertemplate='%{x|%Y-%m-%d}<br>'+trace.name+': %{y:.2f}'+unit+'<extra></extra>'
fig_px.for_each_annotation(lambda a:a.update(text=a.text.split('=')[1] if a.text.startswith('面板=') else a.text))
# 上述分隔按第一个等号切分，保留“年初=100”文本。
for a,panel in zip(fig_px.layout.annotations,reversed(panels)):a.text=panel
fig_px.update_xaxes(matches='x',showticklabels=True,title_text=None)
fig_px.update_layout(template='plotly_white',hovermode='x unified',font=dict(family='Arial, PingFang SC, Microsoft YaHei, sans-serif'),margin=dict(l=80,r=245,t=85,b=80),legend=dict(title='点击切换；双击单独显示',x=1.05,y=1,font=dict(size=11)),xaxis=dict(title='日期',rangeslider=dict(visible=True,thickness=.065),rangeselector=dict(buttons=[dict(count=1,label='1个月',step='month',stepmode='backward'),dict(count=3,label='3个月',step='month',stepmode='backward'),dict(count=6,label='6个月',step='month',stepmode='backward'),dict(label='全部',step='all')],y=-.24)),xaxis2=dict(rangeslider=dict(visible=False)),xaxis3=dict(rangeslider=dict(visible=False)))
config_px={'displaylogo':False,'scrollZoom':True,'responsive':True,'toImageButtonOptions':{'filename':'2020_price_metals','format':'png','scale':2}}
fig_px.write_html(OUT_PX/'2020_price_metals_interactive.html',include_plotlyjs=True,config=config_px)
fig_px.write_json(OUT_PX/'figure.json')
prices.loc[idx].join(indexed.add_suffix('_年初100')).join(ret).to_csv(OUT_PX/'chart_data.csv',index_label='Date')
plotdata.to_csv(OUT_PX/'plot_long_data.csv',index=False)
(OUT_PX/'config.json').write_text(json.dumps({'year':2020,'price_base_date':str(idx[0].date()),'price_base_values':base.to_dict(),'return_y_range':[-limit,limit],'ma_includes_current':True,'windows':[5,20]},ensure_ascii=False,indent=2))
assert len(fig_px.data)==13 and all(np.isclose(indexed.iloc[0],100))
assert fig_px.layout.yaxis.range==fig_px.layout.yaxis2.range
# 本图六项涨幅须与12中模型输入逐条一致。
prior=ROOT/'results/mlp_monthly_validation_12/features_and_target.csv'
if prior.exists():
    old=pd.read_csv(prior,index_col='Date',parse_dates=True).loc[idx]
    for metal,eng in [('黄金','gold'),('铜','copper')]:
        assert np.allclose(ret[f'{metal} · 昨日涨跌幅'],100*old[f'{eng}_daily'])
        for k in [5,20]:assert np.allclose(ret[f'{metal} · 相对MA{k}涨幅'],100*old[f'{eng}_vs_ma{k}'])
    print('复核通过：六项涨幅与12中的模型输入一致。')
display(fig_px)
print('上面价格基准为2020年首条记录=100；均价线默认隐藏，点击图例可显示。')
'''))
nb.validate(n);nb.write(n,'outputs/12_减少预测的时间,扩大验证集的范围.ipynb')
