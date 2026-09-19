from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/private/tmp/mlp14-mpl'
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from IPython.display import display,Image
OUT=Path(__file__).resolve().parent
METHODS=['Baseline','RSI','MACD','Both'];LABELS=['基准：特征＋触发','＋RSI14','＋DIFF、DEA','＋RSI、DIFF、DEA'];COLORS=['#7c8798','#4e79a7','#e69f00','#009e73']
a=[];ps=[]
for year in range(2015,2027):
    f=OUT/str(year);assert (f/'done.json').exists()
    s=pd.read_csv(f/'summary.csv');p=pd.read_csv(f/'predictions.csv',parse_dates=['Date']);audit=pd.read_csv(f/'audit.csv',parse_dates=['Train_end','Test_start'])
    assert (audit.Train_end<audit.Test_start).all()
    assert s.Models.nunique()==1 and p.Date.is_unique
    for method in METHODS:
        err=p[method]-p.Actual;r=s[s.Method==method].iloc[0]
        assert np.isclose(r.MAPE,100*(err.abs()/p.Actual.abs()).mean()) and np.isclose(r.MSE,(err**2).mean())
    a.append(s);p['Year']=year;ps.append(p)
a=pd.concat(a,ignore_index=True);p=pd.concat(ps,ignore_index=True);assert len(p)==3124
a.to_csv(OUT/'annual_metrics.csv',index=False);p.to_csv(OUT/'predictions.csv',index=False)
mapetable=a.pivot(index='Year',columns='Method',values='MAPE')[METHODS];msetable=a.pivot(index='Year',columns='Method',values='MSE')[METHODS]
mapetable.to_csv(OUT/'annual_mape.csv');msetable.to_csv(OUT/'annual_mse.csv')
rows=[]
for period,limit in [('2015-2025_complete',2025),('2015-2026_partial',2026)]:
    for method in METHODS:
        g=a[(a.Year<=limit)&(a.Method==method)];years=mapetable.loc[mapetable.index<=limit]
        rows.append(dict(Period=period,Method=method,N=int(g.N.sum()),Annual_equal_MAPE=g.MAPE.mean(),Annual_equal_MSE=g.MSE.mean(),Record_weighted_MAPE=np.average(g.MAPE,weights=g.N),Record_weighted_MSE=np.average(g.MSE,weights=g.N),Years_MAPE_better_than_baseline=int((years[method]<years.Baseline-1e-6).sum()),Years=len(g)))
s=pd.DataFrame(rows);s.to_csv(OUT/'summary.csv',index=False)
plt.rcParams['font.family']=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf').get_name();plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,1,figsize=(17,10),layout='constrained');pos=np.arange(12)
for ax,table,title,ylabel in [(axes[0],mapetable,'2015—2026：滞后RSI与MACD输入对比','年度MAPE（%）'),(axes[1],msetable,'普通MSE（受不同年份股价水平影响）','年度MSE')]:
    for i,method in enumerate(METHODS):
        bars=ax.bar(pos+(i-1.5)*.20,table[method],width=.20,label=LABELS[i],color=COLORS[i]);ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
    ax.set(title=title,ylabel=ylabel,xticks=pos,xticklabels=[str(y) if y!=2026 else '2026*' for y in table.index],ylim=(0,table.to_numpy().max()*1.23));ax.legend(ncol=4);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.supxlabel('*2026截至7月23日。RSI(14)、DIFF/DEA(12,26,9)均由自身股价计算并滞后1条记录；四组更新日、种子和半衰期60相同。',fontsize=11)
fig.savefig(OUT/'annual_bars.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
for ax,(period,title) in zip(axes,[('2015-2025_complete','2015—2025：完整年份'),('2015-2026_partial','2015—2026*：含部分2026')]):
    v=s[s.Period==period].set_index('Method').loc[METHODS,'Annual_equal_MAPE']
    b=ax.bar(['基准','＋RSI','＋DIFF/DEA','三者结合'],v,color=COLORS,width=.65);ax.bar_label(b,labels=[f'{x:.4f}%' for x in v],padding=5,fontsize=11)
    ax.set(title=title,ylabel='年度等权平均MAPE（%）',ylim=(0,s.Annual_equal_MAPE.max()*1.22));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('新增第二阶段输入：平均表现',fontsize=18);fig.savefig(OUT/'average_bars.png',dpi=160,bbox_inches='tight');plt.close(fig)
print(mapetable.round(6).to_string());print(s.round(6).to_string(index=False))
display(Image(filename=str(OUT/'annual_bars.png')));display(Image(filename=str(OUT/'average_bars.png')));display(s.round(6))
