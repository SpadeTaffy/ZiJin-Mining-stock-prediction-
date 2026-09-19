from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/private/tmp/mlp14-mpl'
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import plotly.express as px
from IPython.display import display,Image
OUT=Path(__file__).resolve().parent
p=pd.read_csv(OUT/'predictions.csv',parse_dates=['Date']);s=pd.read_csv(OUT/'summary.csv');m=pd.read_csv(OUT/'monthly_metrics.csv')
labels={'Original':'原金铜3%','Features_only':'只加特征','Triggers_only':'只加触发条件','Features_and_triggers':'特征＋触发条件'}
colors={'Original':'#4e79a7','Features_only':'#b07aa1','Triggers_only':'#e69f00','Features_and_triggers':'#009e73'}
plt.rcParams['font.family']=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf').get_name();plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(3,1,figsize=(14,14),layout='constrained')
axes[0].plot(p.Date,p.Actual,color='#222',lw=1.6,label='真实股价')
for method in ['Original','Features_and_triggers']:axes[0].plot(p.Date,p[method],lw=1.3,color=colors[method],label=labels[method])
axes[0].set(title='2016｜原金铜3% vs 新特征＋多阈值触发（均为半衰期60）',ylabel='股价');axes[0].legend(ncol=3)
wide=m.pivot(index='Month',columns='Method',values='MAPE');pos=np.arange(12)
for i,method in enumerate(labels):axes[1].bar(pos+(i-1.5)*.19,wide[method],width=.19,color=colors[method],label=labels[method])
axes[1].set(xticks=pos,xticklabels=[x[-2:] for x in wide.index],xlabel='2016月份',ylabel='月度MAPE（%）',title='四组逐月误差');axes[1].legend(ncol=4)
ss=s.set_index('Method').loc[list(labels)]
b=axes[2].bar(list(labels.values()),ss.MAPE,color=list(colors.values()),width=.6)
axes[2].bar_label(b,labels=[f'{v:.4f}%' for v in ss.MAPE],padding=5,fontsize=12)
axes[2].set(ylabel='全年MAPE（%）',title='全年误差与更新次数',ylim=(0,ss.MAPE.max()*1.25))
for i,r in enumerate(ss.itertuples()):axes[2].text(i,.15,f'更新{r.Updates}次',ha='center',color='white',fontsize=11)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
long=p.melt(id_vars=['Date','Actual'],value_vars=list(labels),var_name='Method',value_name='Price')
actual=pd.DataFrame({'Date':p.Date,'Method':'Actual','Price':p.Actual});long=pd.concat([long,actual],ignore_index=True)
long['曲线']=long.Method.map({**labels,'Actual':'真实股价'})
f=px.line(long,x='Date',y='Price',color='曲线',color_discrete_map={**{labels[k]:v for k,v in colors.items()},'真实股价':'#222'},title='2016｜原金铜3%与新增特征、触发条件的四组对照',height=650)
for t in f.data:
    if t.name in ['只加特征','只加触发条件']:t.visible='legendonly'
f.update_layout(template='plotly_white',hovermode='x unified',xaxis=dict(title='日期',rangeslider=dict(visible=True)),yaxis_title='股价',legend_title_text='点击切换曲线')
f.write_html(OUT/'predictions_interactive.html',include_plotlyjs=True,config={'displaylogo':False,'scrollZoom':True})
display(Image(filename=str(OUT/'comparison.png')));display(f);display(s.round(6));display(wide.round(4))
