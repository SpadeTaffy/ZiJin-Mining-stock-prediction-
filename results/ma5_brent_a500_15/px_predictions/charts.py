from pathlib import Path
import os
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display
RESULTS=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/results/ma5_brent_a500_15')
CHARTS=Path(os.environ.get('MACRO15_PX_OUT',str(RESULTS/'px_predictions')))
CHARTS.mkdir(exist_ok=True,parents=True)
predictions=pd.read_csv(RESULTS/'predictions.csv',parse_dates=['Date'])
assert predictions.Date.is_unique and np.isfinite(predictions[['Actual','Baseline','Expanded']]).all().all()
labels={'Actual':'实际股价','Baseline':'基准：15项输入','Expanded':'新方案：增加布伦特与A500 MA5偏离'}
colors={'实际股价':'#222222',labels['Baseline']:'#4477AA',labels['Expanded']:'#009E73'}
config={'displaylogo':False,'scrollZoom':True,'responsive':True,'toImageButtonOptions':{'format':'png','scale':2}}
for year in range(2024,2027):
    year_data=predictions[predictions.Year==year].sort_values('Date')
    long=year_data[['Date','Actual','Baseline','Expanded']].rename(columns=labels).melt(id_vars='Date',var_name='曲线',value_name='股价')
    baseline_mape=100*((year_data.Baseline-year_data.Actual).abs()/year_data.Actual.abs()).mean()
    expanded_mape=100*((year_data.Expanded-year_data.Actual).abs()/year_data.Actual.abs()).mean()
    note=f'（截至{year_data.Date.max():%m月%d日}）' if year==2026 else ''
    fig=px.line(long,x='Date',y='股价',color='曲线',color_discrete_map=colors,
        title=f'{year}年{note}｜实际股价与两组模型预测<br><sup>MAPE：基准 {baseline_mape:.4f}% ｜ 新方案 {expanded_mape:.4f}%</sup>',
        labels={'Date':'日期'},template='plotly_white',height=600)
    fig.update_traces(line=dict(width=1.8),hovertemplate='%{x|%Y-%m-%d}<br>股价：%{y:.4f}<extra>%{fullData.name}</extra>')
    fig.update_layout(hovermode='x unified',font=dict(family='Arial, PingFang SC, Microsoft YaHei, sans-serif'),
        legend=dict(orientation='h',y=1.10,x=0,title_text=''),margin=dict(l=65,r=35,t=135,b=60))
    fig.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=[dict(count=1,label='1个月',step='month',stepmode='backward'),dict(count=3,label='3个月',step='month',stepmode='backward'),dict(label='全年',step='all')]))
    fig.write_html(CHARTS/f'{year}_prediction_comparison.html',include_plotlyjs=True,config=config)
    display(fig)
