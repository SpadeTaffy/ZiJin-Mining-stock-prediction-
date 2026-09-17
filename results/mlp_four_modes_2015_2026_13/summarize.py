from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/private/tmp/mlp13-annual-mpl'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
OUT=Path(__file__).resolve().parent
cfg=json.loads((OUT/'config.json').read_text())
OLD=Path(cfg['old_annual_source'])
METHODS=['Annual_frozen_old','Monthly_equal','Monthly_H60','Crossing_H60']
LABELS={'Annual_frozen_old':'每年一次（旧结果）','Monthly_equal':'每月更新·等权','Monthly_H60':'每月更新·半衰期60','Crossing_H60':'双向触发·半衰期60'}
COLORS={'Annual_frozen_old':'#8a93a2','Monthly_equal':'#4e79a7','Monthly_H60':'#e69f00','Crossing_H60':'#009e73'}
font=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
allpred=[];annual=[];allupdates=[];coverage=[]
oldpred=pd.read_csv(OLD/'predictions.csv',parse_dates=['Date'])
oldpred=oldpred[(oldpred.Model=='Stacking')&(oldpred.Split=='Test')&(oldpred.Budget==100)]
oldmetrics=pd.read_csv(OLD/'metrics.csv')
oldmetrics=oldmetrics[(oldmetrics.Model=='Stacking')&(oldmetrics.Split=='Test')&(oldmetrics.Budget==100)]
assert len(oldmetrics)==120
for year in range(2015,2027):
    folder=OUT/str(year);assert (folder/'done.json').exists()
    p=pd.read_csv(folder/'predictions.csv',parse_dates=['Date']);allpred.append(p)
    u=pd.read_csv(folder/'updates.csv',parse_dates=['Effective_date','Train_end','Applied_start']);allupdates.append(u)
    assert (u.Train_end<u.Applied_start).all()
    done=json.loads((folder/'done.json').read_text());coverage.append(done)
    assert p.Date.is_unique and p.Date.is_monotonic_increasing
    for method in METHODS[1:]:
        err=p[method]-p.Actual;records=u[u.Method==method]
        annual.append(dict(Year=year,Method=method,N=len(p),MAPE=100*(err.abs()/p.Actual.abs()).mean(),MSE=(err**2).mean(),Seed_MAPE_SD=np.nan,Models=len(records),Updates_excluding_initial=len(records)-1,Start=str(p.Date.min().date()),End=str(p.Date.max().date())))
    op=oldpred[oldpred.Year==year]
    assert op.Seed.nunique()==10 and not op.duplicated(['Seed','Date']).any()
    for seed,g in op.groupby('Seed'):
        g=g.sort_values('Date');assert g.Date.tolist()==p.Date.tolist()
        assert np.allclose(g.Actual,p.Actual,rtol=0,atol=1e-9)
        err=g.Predicted-g.Actual;mape=100*(err.abs()/g.Actual.abs()).mean();mse=(err**2).mean()
        metric=oldmetrics[(oldmetrics.Year==year)&(oldmetrics.Seed==seed)].iloc[0]
        assert np.isclose(metric.MAPE,mape,rtol=1e-6) and np.isclose(metric.MSE,mse,rtol=1e-6)
    om=oldmetrics[oldmetrics.Year==year]
    assert (om.N==len(p)).all()
    annual.append(dict(Year=year,Method=METHODS[0],N=len(p),MAPE=om.MAPE.mean(),MSE=om.MSE.mean(),Seed_MAPE_SD=om.MAPE.std(),Models=1,Updates_excluding_initial=0,Start=str(p.Date.min().date()),End=str(p.Date.max().date())))
a=pd.DataFrame(annual);p=pd.concat(allpred,ignore_index=True);u=pd.concat(allupdates,ignore_index=True)
a.to_csv(OUT/'annual_metrics.csv',index=False);p.to_csv(OUT/'predictions.csv',index=False);u.to_csv(OUT/'updates.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
pivot=a.pivot(index='Year',columns='Method',values='MAPE')[METHODS];pivot.to_csv(OUT/'annual_mape.csv')
rows=[]
for period,maxyear in [('2015-2025_complete',2025),('2015-2026_partial',2026)]:
    selected=a[a.Year<=maxyear]
    for method in METHODS:
        g=selected[selected.Method==method]
        rows.append(dict(Period=period,Method=method,Years=len(g),N=int(g.N.sum()),Annual_equal_MAPE=g.MAPE.mean(),Record_weighted_MAPE=np.average(g.MAPE,weights=g.N),Annual_equal_MSE=g.MSE.mean(),Record_weighted_MSE=np.average(g.MSE,weights=g.N),Worst_year=int(g.loc[g.MAPE.idxmax(),'Year']),Worst_year_MAPE=g.MAPE.max(),Total_models=int(g.Models.sum()),Total_updates_excluding_year_initializations=int(g.Updates_excluding_initial.sum())))
summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False)
oldsummary=pd.read_csv(OLD/'summary.csv')
for period,oldperiod in [('2015-2025_complete','2015-2025'),('2015-2026_partial','2015-2026_partial')]:
    current=summary[(summary.Period==period)&(summary.Method==METHODS[0])].iloc[0]
    old=oldsummary[(oldsummary.Period==oldperiod)&(oldsummary.Model=='Stacking')].iloc[0]
    assert np.isclose(current.Annual_equal_MAPE,old.Annual_equal_mean_MAPE)
    assert np.isclose(current.Record_weighted_MAPE,old.Sample_weighted_MAPE)
paired=[]
for baseline,alternative in [('Monthly_equal','Monthly_H60'),('Monthly_H60','Crossing_H60'),('Annual_frozen_old','Monthly_equal'),('Annual_frozen_old','Crossing_H60')]:
    for period,limit in [('2015-2025_complete',2025),('2015-2026_partial',2026)]:
        s=summary[summary.Period==period].set_index('Method');base=s.loc[baseline];alt=s.loc[alternative]
        pp=base.Annual_equal_MAPE-alt.Annual_equal_MAPE;years=pivot.loc[pivot.index<=limit]
        paired.append(dict(Period=period,Baseline=baseline,Alternative=alternative,MAPE_reduction_pp=pp,MAPE_relative_reduction_pct=100*pp/base.Annual_equal_MAPE,Years_improved=int((years[alternative]<years[baseline]).sum()),Years=len(years)))
pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
PLOT_ORDER=['Monthly_equal','Monthly_H60','Crossing_H60','Annual_frozen_old']
# Annual grouped bars: fixed method order, same-date coverage, 2026 explicitly partial.
fig,ax=plt.subplots(figsize=(17,7),layout='constrained');pos=np.arange(12);width=.20
for i,m in enumerate(PLOT_ORDER):
    bars=ax.bar(pos+(i-1.5)*width,pivot[m],width=width,color=COLORS[m],label=LABELS[m])
    ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
ax.set_xticks(pos,[str(y) if y!=2026 else '2026*' for y in pivot.index]);ax.set_ylabel('年度测试 MAPE（%，越低越好）');ax.set_title('2015—2026：四种训练与更新方式的年度误差',fontsize=17,pad=18)
ax.legend(ncol=4,loc='upper left');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_ylim(0,pivot.to_numpy().max()*1.22)
fig.text(.5,-.025,'*2026仅截至7月23日。旧全年模式为10种子误差平均；其余为历史验证选种后的逐年结果。',ha='center',fontsize=11)
fig.savefig(OUT/'annual_bars.png',dpi=170,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(15,6.5),layout='constrained')
short=['每月更新\n等权','每月更新\n半衰期60','双向触发\n半衰期60','每年一次\n旧结果']
for ax,(period,title) in zip(axes,[('2015-2025_complete','2015—2025：11个完整年份'),('2015-2026_partial','2015—2026*：含2026部分年份')]):
    vals=summary[summary.Period==period].set_index('Method').loc[PLOT_ORDER,'Annual_equal_MAPE']
    bars=ax.bar(np.arange(4),vals,color=[COLORS[m] for m in PLOT_ORDER],width=.65)
    ax.bar_label(bars,labels=[f'{v:.3f}%' for v in vals],padding=5,fontsize=12,fontweight='bold')
    ax.set_xticks(np.arange(4),short);ax.set_title(title,fontsize=14,pad=18);ax.set_ylabel('年度等权平均 MAPE（%）');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.set_ylim(0,summary.Annual_equal_MAPE.max()*1.20)
fig.suptitle('跨年平均表现：每个年份权重相同',fontsize=19)
fig.savefig(OUT/'average_bars.png',dpi=170,bbox_inches='tight');plt.close(fig)
print(pivot.round(4).to_string());print(summary.round(5).to_string(index=False));print(pd.DataFrame(paired).round(4).to_string(index=False))
(OUT/'verification.json').write_text(json.dumps(dict(years=12,test_records=len(p),old_seed_year_checks=120,old_annual_summary_reproduced=True,matching_test_dates=True,chronology_verified=True,replicated_2020=True,annual_baseline_retrained=False),indent=2))
