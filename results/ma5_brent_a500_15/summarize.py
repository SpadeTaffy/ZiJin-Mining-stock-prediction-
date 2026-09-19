from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/private/tmp/macro15-mpl'
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
OUT=Path(__file__).resolve().parent
annual=[];pred=[];updates=[]
for year in range(2015,2027):
 folder=OUT/str(year);assert (folder/'done.json').exists()
 p=pd.read_csv(folder/'predictions.csv',parse_dates=['Date']);a=pd.read_csv(folder/'summary.csv');u=pd.read_csv(folder/'updates.csv',parse_dates=['Train_end','Effective_date'])
 assert (u.Train_end<u.Effective_date).all()
 for r in a.itertuples():assert np.isclose(r.MAPE,100*((p[r.Method]-p.Actual).abs()/p.Actual.abs()).mean())
 p['Year']=year;annual.append(a);pred.append(p);updates.append(u)
a=pd.concat(annual,ignore_index=True);p=pd.concat(pred,ignore_index=True);u=pd.concat(updates,ignore_index=True)
assert len(p)==3124 and p.Date.is_unique
for name,frame in [('annual_metrics',a),('predictions',p),('updates',u)]:frame.to_csv(OUT/f'{name}.csv',index=False)
t=a.pivot(index='Year',columns='Method',values='MAPE');t['Difference_pp']=t.Expanded-t.Baseline
t['Baseline_updates']=a[a.Method=='Baseline'].set_index('Year').Updates;t['Expanded_updates']=a[a.Method=='Expanded'].set_index('Year').Updates
t.to_csv(OUT/'annual_comparison.csv')
rows=[]
for label,end in [('2015—2025完整年份',2025),('2015—2026（2026截至7月23日）',2026)]:
 sub=t.loc[:end];sp=p[p.Year<=end]
 rows.append(dict(Period=label,Baseline=sub.Baseline.mean(),Expanded=sub.Expanded.mean(),Difference_pp=sub.Difference_pp.mean(),Relative_improvement_pct=100*(1-sub.Expanded.mean()/sub.Baseline.mean()),Years_better=int((sub.Difference_pp<0).sum()),Baseline_updates=int(sub.Baseline_updates.sum()),Expanded_updates=int(sub.Expanded_updates.sum()),Pooled_baseline=100*((sp.Baseline-sp.Actual).abs()/sp.Actual.abs()).mean(),Pooled_expanded=100*((sp.Expanded-sp.Actual).abs()/sp.Actual.abs()).mean()))
summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False)
plt.rcParams['font.family']=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf').get_name();plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(16,6),layout='constrained');pos=np.arange(12)
for offset,col,label,color in [(-.2,'Baseline','基准：15项第二层输入','#78909c'),(.2,'Expanded','新增布伦特及A500 MA5偏离（17项）','#009e73')]:
 bars=ax.bar(pos+offset,t[col],width=.38,label=label,color=color);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
ax.set(xticks=pos,xticklabels=[str(i) if i!=2026 else '2026*' for i in t.index],ylabel='MAPE（%，越低越好）',title='2015—2026年：新增布伦特与A500 MA5偏离率对比',ylim=(0,t[['Baseline','Expanded']].to_numpy().max()*1.2));ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.supxlabel('*2026截至7月23日；两组半衰期均为60。',fontsize=10);fig.savefig(OUT/'annual_mape.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5),layout='constrained');vals=summary.iloc[-1][['Baseline','Expanded']].astype(float)
bars=ax.bar(['基准模型','新增两项MA5偏离'],vals,color=['#78909c','#009e73'],width=.55);ax.bar_label(bars,fmt='%.4f%%',padding=5);ax.set(ylabel='年度等权平均MAPE（%）',title='2015—2026年平均表现（2026为部分年度）',ylim=(0,vals.max()*1.22));fig.savefig(OUT/'average_mape.png',dpi=160);plt.close(fig)
print(t.round(5).to_string());print(summary.round(6).to_string(index=False))
(OUT/'report.md').write_text('# 新增布伦特与A500 MA5偏离率对比\n\n'+t.round(5).to_markdown()+'\n\n'+summary.round(6).to_markdown(index=False)+'\n\nMAPE差值为新方案减基准，负值表示改善。年度等权平均为各年MAPE的算术平均；2026仅截至7月23日。两组触发条件、更新日期、种子和第一阶段模型完全一致；新方案仅在第二层增加布伦特和A500的MA5偏离率。',encoding='utf-8')
