from pathlib import Path
import nbformat as nb,hashlib
out=Path(__file__).resolve().parent
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/14_继续增加规则.ipynb')
b=src.read_bytes();(out/'14_before.ipynb').write_bytes(b);(out/'original.sha256').write_text(hashlib.sha256(b).hexdigest());n=nb.reads(b.decode(),4)
(out/'start_cell.txt').write_text(str(len(n.cells)))
n.cells.append(nb.v4.new_markdown_cell('''# 特征＋触发＋RSI＋DIFF/DEA：2015—2026逐年预测图

直接读取上一节保存的 `Both` 预测，不重新训练、不改变种子或预测值。黑线为真实股价，绿色为预测股价；先展示12年总览，再给出每年独立大图。各年纵轴独立，便于查看当年的拟合偏差，不能用线条的视觉距离直接比较不同年份误差。每张图标明MAPE、记录数及数据日期。

独立大图底部的绿色短刻度表示**新模型生效日期**，不含年初初始化；按原规则是触发后的下一条有效记录。2026仅截至7月23日。图中的预测仍为使用同日市场输入的历史样本外结果。
'''))
code=r'''from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from IPython.display import display,Image
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
PLOT_OUT=Path(os.environ.get('MLP14_COMBINED_PLOTS_OUT',str(ROOT/'results/rsi_macd_combined_plots_14')))
PLOT_OUT.mkdir(parents=True,exist_ok=True)
PRED_SOURCE=PLOT_OUT.parent/'rsi_macd_annual_14'
if not (PRED_SOURCE/'predictions.csv').exists():PRED_SOURCE=ROOT/'results/rsi_macd_annual_14'
annual_plot_data=pd.read_csv(PRED_SOURCE/'predictions.csv',parse_dates=['Date'])
UPDATE_SOURCE=PLOT_OUT.parent/'multi_threshold_annual_14'
if not UPDATE_SOURCE.exists():UPDATE_SOURCE=ROOT/'results/multi_threshold_annual_14'
annual_plot_updates=pd.read_csv(UPDATE_SOURCE/'updates.csv',parse_dates=['Effective_date'])
annual_plot_metrics=pd.read_csv(PRED_SOURCE/'annual_metrics.csv')
annual_plot_data=annual_plot_data[['Date','Year','Actual','Both']].sort_values('Date')
assert annual_plot_data.Date.is_unique and len(annual_plot_data)==3124
assert set(annual_plot_data.Year)==set(range(2015,2027))
assert np.isfinite(annual_plot_data[['Actual','Both']]).all().all()
annual_plot_updates=annual_plot_updates[annual_plot_updates.Method=='Features_and_triggers'].copy()
font_path=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
if font_path.exists():plt.rcParams['font.family']=FontProperties(fname=str(font_path)).get_name()
plt.rcParams['axes.unicode_minus']=False
ACTUAL_COLOR='#20252b';PRED_COLOR='#009e73'
plot_manifest=[]
def draw_prediction_year(ax,year,large=False):
    g=annual_plot_data[annual_plot_data.Year==year]
    updates=annual_plot_updates[annual_plot_updates.Year==year].sort_values('Effective_date')
    error=g.Both-g.Actual
    mape=float(100*(error.abs()/g.Actual.abs()).mean())
    old=annual_plot_metrics[(annual_plot_metrics.Year==year)&(annual_plot_metrics.Method=='Both')].iloc[0]
    assert len(g)==int(old.N) and np.isclose(mape,old.MAPE)
    ax.plot(g.Date,g.Actual,color=ACTUAL_COLOR,lw=1.6 if large else 1.35,label='真实股价',zorder=3)
    ax.plot(g.Date,g.Both,color=PRED_COLOR,lw=1.5 if large else 1.2,label='预测：特征＋触发＋RSI＋DIFF/DEA',zorder=4)
    title=f'{year}'+('（截至7月23日）' if year==2026 else '')
    ax.set_title(f'{title}  |  MAPE {mape:.2f}%',fontsize=15 if large else 12,pad=10)
    ax.set_ylabel('股价（数据集单位）',fontsize=11 if large else 9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1 if large else 3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    ax.grid(alpha=.16);ax.margins(x=.015,y=.09)
    ax.spines[['top','right']].set_visible(False)
    if large:
        ax.vlines(updates.Effective_date.iloc[1:],0,.045,transform=ax.get_xaxis_transform(),color=PRED_COLOR,alpha=.65,lw=1)
        ax.text(.01,.97,f'{g.Date.min():%Y-%m-%d} 至 {g.Date.max():%Y-%m-%d}  ·  {len(g)}条记录  ·  年内更新{len(updates)-1}次',transform=ax.transAxes,va='top',fontsize=10,color='#515a64')
        ax.legend(loc='upper left',bbox_to_anchor=(0,1.19),ncol=2,frameon=False,fontsize=11)
        ax.set_xlabel('月份；底部绿色短刻度为新模型生效日（不含年初初始化）',fontsize=10,labelpad=12)
    return dict(Year=year,Start=str(g.Date.min().date()),End=str(g.Date.max().date()),N=len(g),MAPE=mape,Updates_excluding_initial=len(updates)-1)
fig,axes=plt.subplots(4,3,figsize=(18,16))
fig.subplots_adjust(top=.93,bottom=.045,left=.065,right=.99,hspace=.28,wspace=.20)
for ax,year in zip(axes.flat,range(2015,2027)):draw_prediction_year(ax,year)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.968),ncol=2,frameon=False,fontsize=13)
fig.suptitle('特征＋触发＋RSI＋DIFF/DEA：逐年真实股价与预测对比',fontsize=21,y=.99)
fig.savefig(PLOT_OUT/'all_years_predictions.png',dpi=180,bbox_inches='tight');plt.close(fig)
for year in range(2015,2027):
    fig,ax=plt.subplots(figsize=(14,5.6),layout='constrained')
    record=draw_prediction_year(ax,year,large=True)
    name=f'{year}_combined_prediction.png'
    fig.savefig(PLOT_OUT/name,dpi=170,bbox_inches='tight');plt.close(fig)
    record['File']=name;plot_manifest.append(record)
pd.DataFrame(plot_manifest).to_csv(PLOT_OUT/'plot_manifest.csv',index=False)
annual_plot_data.to_csv(PLOT_OUT/'plot_data.csv',index=False)
(PLOT_OUT/'config.json').write_text(json.dumps(dict(source=str(PRED_SOURCE),predictions_sha256=hashlib.sha256((PRED_SOURCE/'predictions.csv').read_bytes()).hexdigest(),years=list(range(2015,2027)),method='Both',retrained=False,individual_charts=12,overview_charts=1),ensure_ascii=False,indent=2))
print('12张年度图＋1张总览图已生成；全部MAPE与上一节保存指标核对一致；未重新训练。')
display(Image(filename=str(PLOT_OUT/'all_years_predictions.png')))
'''
n.cells.append(nb.v4.new_code_cell(code))
n.cells.append(nb.v4.new_markdown_cell('## 各年度独立大图'))
n.cells.append(nb.v4.new_code_cell("for year in range(2015,2027):\n    display(Image(filename=str(PLOT_OUT/f'{year}_combined_prediction.png')))"))
nb.validate(n);nb.write(n,out.parent/'14_继续增加规则.ipynb')
s=Path('outputs/mlp_half_life_comparison_13/execute.py').read_text().replace('MLP13_GRID_OUT','MLP14_COMBINED_PLOTS_OUT').replace('mlp_half_life_comparison_13','rsi_macd_combined_plots_14').replace('13_加权MSE的训练','14_继续增加规则').replace("print(scope['monthly'].to_string()); print(scope['summary'].to_string(index=False))","print('Annual prediction charts complete.')")
(out/'execute.py').write_text(s)
