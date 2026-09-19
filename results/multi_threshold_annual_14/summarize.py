from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/private/tmp/mlp14-mpl'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from IPython.display import display,Image
OUT=Path(__file__).resolve().parent
METHODS=['Features_only','Triggers_only','Features_and_triggers']
LABELS={'Features_only':'② 只加特征','Triggers_only':'③ 只加触发条件','Features_and_triggers':'④ 特征＋触发条件'}
COLORS=['#b07aa1','#e69f00','#009e73']
annual=[];allpred=[];allupdates=[]
root=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
prior=OUT.parent/'mlp_four_modes_2015_2026_13'
if not prior.exists():prior=root/'results/mlp_four_modes_2015_2026_13'
for year in range(2015,2027):
    folder=OUT/str(year);cfg=json.loads((folder/'config.json').read_text())
    assert cfg['year']==year and cfg['half_life']==60 and cfg['original_reproduced']
    p=pd.read_csv(folder/'predictions.csv',parse_dates=['Date']);s=pd.read_csv(folder/'summary.csv').set_index('Method')
    u=pd.read_csv(folder/'updates.csv',parse_dates=['Effective_date','Trigger_date']);audit=pd.read_csv(folder/'training_audit.csv',parse_dates=['Train_end','Effective_date'])
    assert (audit.Train_end<audit.Effective_date).all()
    old=pd.read_csv(prior/str(year)/'predictions.csv',parse_dates=['Date'])
    assert p.Date.tolist()==old.Date.tolist() and p.Date.is_unique
    assert np.allclose(p.Actual,old.Actual) and np.allclose(p.Original,old.Crossing_H60,atol=1e-5,rtol=0)
    for method in METHODS:
        err=p[method]-p.Actual;mape=100*(err.abs()/p.Actual.abs()).mean();mse=(err**2).mean()
        assert np.isclose(mape,s.loc[method,'MAPE']) and np.isclose(mse,s.loc[method,'MSE'])
        annual.append(dict(Year=year,Method=method,N=len(p),MAPE=mape,MSE=mse,MAE=err.abs().mean(),Updates=int(s.loc[method,'Updates']),Models=int(s.loc[method,'Models']),Start=str(p.Date.min().date()),End=str(p.Date.max().date())))
    p['Year']=year;u['Year']=year;allpred.append(p[['Date','Year','Actual']+METHODS]);allupdates.append(u[u.Method.isin(METHODS)])
a=pd.DataFrame(annual);pred=pd.concat(allpred,ignore_index=True);updates=pd.concat(allupdates,ignore_index=True)
assert len(pred)==3124
for name,f in [('annual_metrics',a),('predictions',pred),('updates',updates)]:f.to_csv(OUT/f'{name}.csv',index=False)
mapetable=a.pivot(index='Year',columns='Method',values='MAPE')[METHODS];msetable=a.pivot(index='Year',columns='Method',values='MSE')[METHODS]
mapetable.to_csv(OUT/'annual_mape.csv');msetable.to_csv(OUT/'annual_mse.csv')
rows=[]
for period,limit in [('2015-2025_complete',2025),('2015-2026_partial',2026)]:
    for method in METHODS:
        g=a[(a.Year<=limit)&(a.Method==method)]
        rows.append(dict(Period=period,Method=method,Years=len(g),N=int(g.N.sum()),Annual_equal_MAPE=g.MAPE.mean(),Record_weighted_MAPE=np.average(g.MAPE,weights=g.N),Annual_equal_MSE=g.MSE.mean(),Record_weighted_MSE=np.average(g.MSE,weights=g.N),Worst_year=int(g.loc[g.MAPE.idxmax(),'Year']),Worst_year_MAPE=g.MAPE.max(),Total_updates=int(g.Updates.sum())))
summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False)
wins=[]
for left,right in [(METHODS[0],METHODS[1]),(METHODS[1],METHODS[2]),(METHODS[0],METHODS[2])]:
    for period,limit in [('2015-2025_complete',2025),('2015-2026_partial',2026)]:
        selected=mapetable.loc[mapetable.index<=limit]
        wins.append(dict(Period=period,Baseline=left,Alternative=right,Years_better=int((selected[right]<selected[left]-1e-6).sum()),Years=len(selected),Mean_MAPE_difference_pp=(selected[right]-selected[left]).mean()))
pd.DataFrame(wins).to_csv(OUT/'paired_comparisons.csv',index=False)
plt.rcParams['font.family']=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf').get_name();plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,1,figsize=(16,10),layout='constrained');pos=np.arange(12)
for ax,table,title,unit,fmt in [(axes[0],mapetable,'2015—2026：②—④逐年MAPE','MAPE（%，越低越好）','%.2f'),(axes[1],msetable,'逐年普通MSE（受各年股价水平影响）','MSE（越低越好）','%.2f')]:
    for i,method in enumerate(METHODS):
        bars=ax.bar(pos+(i-1)*.25,table[method],width=.25,color=COLORS[i],label=LABELS[method]);ax.bar_label(bars,fmt=fmt,padding=2,fontsize=8)
    ax.set(title=title,ylabel=unit,xticks=pos,xticklabels=[str(y) if y!=2026 else '2026*' for y in table.index],ylim=(0,table.to_numpy().max()*1.2));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(ncol=3,loc='upper left')
fig.supxlabel('*2026仅截至7月23日；三个方案均为半衰期60，第二阶段特征与触发阈值沿用2016实验。',fontsize=11)
fig.savefig(OUT/'annual_bars.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
for ax,(period,title) in zip(axes,[('2015-2025_complete','2015—2025：完整年份'),('2015-2026_partial','2015—2026*：含部分2026')]):
    values=summary[summary.Period==period].set_index('Method').loc[METHODS,'Annual_equal_MAPE']
    bars=ax.bar([LABELS[m] for m in METHODS],values,color=COLORS,width=.6);ax.bar_label(bars,labels=[f'{v:.4f}%' for v in values],padding=5,fontsize=12)
    ax.set(title=title,ylabel='年度等权平均MAPE（%）',ylim=(0,summary.Annual_equal_MAPE.max()*1.23));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('②—④的跨年平均表现',fontsize=18);fig.savefig(OUT/'average_bars.png',dpi=160,bbox_inches='tight');plt.close(fig)
print(mapetable.round(5).to_string());print(summary.round(6).to_string(index=False));print(pd.DataFrame(wins).to_string(index=False))
display(Image(filename=str(OUT/'annual_bars.png')));display(Image(filename=str(OUT/'average_bars.png')));display(summary.round(6))
(OUT/'verification.json').write_text(json.dumps(dict(years=12,N=3124,replicated_original_all_years=True,identical_test_dates=True,chronology_passed=True,reused_2016=True,line_charts_created=False),indent=2))
