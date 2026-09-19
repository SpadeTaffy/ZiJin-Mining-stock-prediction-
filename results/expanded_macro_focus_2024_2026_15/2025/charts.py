from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
OUT=Path(os.environ.get('MLP15_PX_OUT',str(ROOT/'results/expanded_macro_focus_2024_2026_15'/str(2025))));OUT.mkdir(parents=True,exist_ok=True)
SOURCE=OUT.parent/'mlp_four_modes_2015_2026_13'
if not (SOURCE/'predictions.csv').exists():SOURCE=ROOT/'results/mlp_four_modes_2015_2026_13'
rawdata=pd.read_pickle(ROOT/'src/pickle/Dataset.pkl');raw=rawdata['X'].copy()
assert raw.index.equals(rawdata['Y'].index) and raw.index.is_unique and raw.index.is_monotonic_increasing
names={'A_500':'A500指数','Dollar Index':'美元指数','cn_10y_rate':'中国10年期利率','us_10y_rate':'美国10年期利率','brent':'布伦特原油','Copper Futures Price':'铜','Gold Futures Price':'黄金'}
assert set(raw.columns)==set(names)
levels=raw.rename(columns=names).copy();levels.insert(0,'股价',np.asarray(rawdata['Y']).reshape(-1))
idx=levels.index[levels.index.year==2025];assert len(idx)==268
ma={k:levels.rolling(k,min_periods=k).mean() for k in [5,20]}
daily=100*(levels/levels.shift(1)-1)
dev={k:100*(levels/ma[k]-1) for k in [5,20]}
PRED_SOURCE=OUT.parent/'expanded_macro_rsi_macd_15'
if not (PRED_SOURCE/'predictions.csv').exists():PRED_SOURCE=ROOT/'results/expanded_macro_rsi_macd_15'
UPDATE_SOURCE=OUT.parent/'expanded_macro_rsi_macd_15'
if not (UPDATE_SOURCE/'updates.csv').exists():UPDATE_SOURCE=ROOT/'results/expanded_macro_rsi_macd_15'
pred=pd.read_csv(PRED_SOURCE/'predictions.csv',parse_dates=['Date']).set_index('Date').loc[idx]
assert np.allclose(pred.Actual,levels.loc[idx,'股价'])
updates=pd.read_csv(UPDATE_SOURCE/'updates.csv',parse_dates=['Effective_date','Trigger_date'])
updates=updates[(updates.Year==2025)].copy()
events=updates[updates.Trigger_date.notna()].copy()
assert (events.Trigger_date<events.Effective_date).all()
features=pd.read_csv(SOURCE/'features_and_target.csv',parse_dates=['Date']).set_index('Date').loc[idx]
for eng,cn in [('gold','黄金'),('copper','铜')]:
    assert np.allclose(daily.loc[idx,cn],100*features[f'{eng}_daily'])
    for k in [5,20]:assert np.allclose(dev[k].loc[idx,cn],100*features[f'{eng}_vs_ma{k}'])
base=levels.loc[idx[0]]
indexed=100*levels.loc[idx]/base
# Prediction uses actual stock's year-start denominator: prediction errors are retained.
indexed['最新扩展模型预测']=100*pred.Expanded/base['股价']
assert np.allclose(indexed.drop(columns='最新扩展模型预测').iloc[0],100)
LABELS=['相邻记录变化','相对MA5偏离','相对MA20偏离']
colors={'股价':'#222222','最新扩展模型预测':'#009E73','A500指数':'#6F4E7C','美元指数':'#CC6677','中国10年期利率':'#44AA99','美国10年期利率':'#882255','布伦特原油':'#AA7733','铜':'#0072B2','黄金':'#D59400',LABELS[0]:'#8B9199',LABELS[1]:'#009E73',LABELS[2]:'#CC79A7'}
config={'displaylogo':False,'scrollZoom':True,'responsive':True,'toImageButtonOptions':{'format':'png','scale':2}}
# Saved trigger signals are shared by the latest RSI + MACD model.
rules={'黄金':[3],'铜':[3,8],'A500指数':[5,8],'美国10年期利率':[5,10,20],'中国10年期利率':[3],'布伦特原油':[4,10,15],'美元指数':[1,2]}
variable_names={'gold_vs_ma20':'黄金','copper_vs_ma20':'铜','a500_vs_ma20':'A500指数','us10y_vs_ma20':'美国10年期利率','cn10y_vs_ma20':'中国10年期利率','brent_vs_ma20':'布伦特原油','dollar_vs_ma20':'美元指数'}
signals=pd.read_csv(UPDATE_SOURCE/'2025'/'signals.csv',parse_dates=['Trigger_date','Effective_date'])
assert set(signals.Effective_date)==set(events.Effective_date)
reason_rows=[]
for update in updates.itertuples(index=False):
    trigger=update.Trigger_date
    record={'生效日':update.Effective_date,'触发日':trigger,'类型':'年初初始化' if pd.isna(trigger) else '阈值穿越'}
    parts=[]
    if pd.isna(trigger):
        parts=['年初初始化：按年度实验规则重新建立模型，并非阈值触发'];trigger_text='无（年初初始化）'
    else:
        pos=levels.index.get_loc(trigger);previous_day=levels.index[pos-1]
        assert levels.index[pos+1]==update.Effective_date
        trigger_text=str(trigger.date())
        rows=signals[signals.Effective_date==update.Effective_date]
        assert (rows.Trigger_date==trigger).all()
        for r in rows.itertuples(index=False):
            variable=variable_names[r.Variable];threshold=r.Threshold*100
            before=float(dev[20].loc[previous_day,variable]);after=float(dev[20].loc[trigger,variable])
            assert np.allclose([before,after],[r.Previous_deviation*100,r.Deviation*100])
            assert (abs(before)>=threshold)!=(abs(after)>=threshold)
            direction='进入外侧' if r.Direction=='outward' else '回到区间内'
            parts.append(f'{variable}：{direction}，±{threshold:g}%边界；MA20偏离 {before:+.3f}% → {after:+.3f}%')
        assert parts
    record['原因']='；'.join(parts);record['悬停说明']='<br>'.join(parts);record['触发日期显示']=trigger_text
    reason_rows.append(record)
reasons=pd.DataFrame(reason_rows)
assert len(reasons)==len(updates)
reasons.drop(columns=['悬停说明','触发日期显示']).to_csv(OUT/'update_reasons.csv',index=False,encoding='utf-8-sig')

# Compact view: stock and prediction remain fixed; dropdown changes the explanatory variable.
fig_focus=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.39,.32,.29],vertical_spacing=.085,subplot_titles=['股价与预测（同一基准，保留预测误差）','变量水平、MA5与MA20（年初=100）','变量相邻变化、MA5/MA20偏离（%）'])
for name in ['股价','最新扩展模型预测']:
    trace=px.line(x=idx,y=indexed[name]).data[0]
    trace.update(name=name,showlegend=True,line=dict(color=colors[name],width=2),customdata=(pred.Expanded if name=='最新扩展模型预测' else levels.loc[idx,'股价']).to_numpy(),hovertemplate='%{x|%Y-%m-%d}<br>'+name+'指数 %{y:.3f}<br>原始股价 %{customdata:.4f}<extra></extra>')
    fig_focus.add_trace(trace,row=1,col=1)
fig_focus.add_trace(go.Scatter(x=reasons['生效日'],y=indexed.loc[reasons['生效日'],'最新扩展模型预测'],mode='markers',marker=dict(size=8,color='#C44E52',symbol='diamond',line=dict(color='white',width=1)),name='新模型生效（悬停查看原因）',visible=True,customdata=reasons[['触发日期显示','悬停说明','类型']].to_numpy(),hovertemplate='生效日：%{x|%Y-%m-%d}<br>类型：%{customdata[2]}<br>触发日：%{customdata[0]}<br>%{customdata[1]}<extra></extra>'),row=1,col=1)
fixed=3;groups={};default='黄金'
for name in names.values():
    group=[]
    for label,values,dash in [(name,levels[name],'solid'),(name+' MA5',ma[5][name],'dash'),(name+' MA20',ma[20][name],'dot')]:
        group.append(len(fig_focus.data));fig_focus.add_trace(go.Scatter(x=idx,y=100*values.loc[idx]/base[name],name=label,visible=name==default,customdata=values.loc[idx].to_numpy(),line=dict(color=colors[name],dash=dash,width=1.7),hovertemplate='%{x|%Y-%m-%d}<br>'+label+'指数 %{y:.3f}<br>原始水平 %{customdata:.5f}<extra></extra>'),row=2,col=1)
    for label,values in zip(LABELS,[daily[name],dev[5][name],dev[20][name]]):
        group.append(len(fig_focus.data));fig_focus.add_trace(go.Scatter(x=idx,y=values.loc[idx],name=name+' · '+label,visible=name==default,line=dict(color=colors[label],width=1.5),hovertemplate='%{x|%Y-%m-%d}<br>'+label+' %{y:.3f}%<extra></extra>'),row=3,col=1)
    if name in rules:
        for v in [sign*t for t in rules[name] for sign in [-1,1]]:
            group.append(len(fig_focus.data));fig_focus.add_trace(go.Scatter(x=[idx.min(),idx.max()],y=[v,v],mode='lines',name=f'{v:+}%触发边界',visible=name==default,line=dict(color='#aaa',dash='dash',width=1),showlegend=False,hoverinfo='skip'),row=3,col=1)
    groups[name]=group
buttons=[]
for name,group in groups.items():
    visible=[True,True,True]+[False]*(len(fig_focus.data)-fixed)
    for j in group:visible[j]=True
    vals=np.column_stack([daily.loc[idx,name],dev[5].loc[idx,name],dev[20].loc[idx,name]])
    lim=max(float(np.abs(vals).max()*1.15),max(rules.get(name,[0]))*1.15 or .1)
    buttons.append(dict(label=name,method='update',args=[{'visible':visible},{'title.text':f'2025｜股价、预测与{name}：逐变量细看','yaxis2.autorange':True,'yaxis3.range':[-lim,lim]}]))
fig_focus.update_layout(template='plotly_white',title=dict(text='2025｜股价、预测与黄金：逐变量细看',y=.98,x=.05),height=1100,hovermode='x unified',font=dict(family='Arial, PingFang SC, Microsoft YaHei, sans-serif'),margin=dict(l=80,r=220,t=155,b=80),legend=dict(x=1.03,y=1,title='点击切换曲线'),updatemenus=[dict(type='dropdown',active=list(names.values()).index(default),buttons=buttons,x=0,y=1.075,xanchor='left',yanchor='top')])
fig_focus.update_xaxes(matches='x',showticklabels=True)
fig_focus.update_yaxes(title_text='股价指数',row=1,col=1);fig_focus.update_yaxes(title_text='变量指数',row=2,col=1);fig_focus.update_yaxes(title_text='相对变化（%）',ticksuffix='%',zeroline=True,zerolinecolor='#888',row=3,col=1)
goldlim=max(float(np.abs(np.column_stack([daily.loc[idx,'黄金'],dev[5].loc[idx,'黄金'],dev[20].loc[idx,'黄金']])).max()*1.15),3.5)
fig_focus.update_yaxes(range=[-goldlim,goldlim],row=3,col=1)
fig_focus.update_layout(xaxis3=dict(rangeslider=dict(visible=True,thickness=.06),rangeselector=dict(buttons=[dict(count=1,label='1个月',step='month',stepmode='backward'),dict(count=3,label='3个月',step='month',stepmode='backward'),dict(label='全年',step='all')],y=-.30)))
fig_focus.write_html(OUT/'2025_variable_focus.html',include_plotlyjs=True,config=config);fig_focus.write_json(OUT/'focus_figure.json')
chart=levels.loc[idx].copy();chart['最新扩展模型预测']=pred.Expanded
for name in levels:
    chart[name+'_MA5']=ma[5].loc[idx,name];chart[name+'_MA20']=ma[20].loc[idx,name]
    chart[name+'_相邻变化_pct']=daily.loc[idx,name]
    for k in [5,20]:chart[name+f'_MA{k}偏离_pct']=dev[k].loc[idx,name]
chart['模型生效']=chart.index.isin(reasons['生效日'])
chart['触发日']=chart.index.map(reasons.set_index('生效日')['触发日'])
chart['生效原因']=chart.index.map(reasons.set_index('生效日')['原因'])
chart.to_csv(OUT/'chart_data.csv',index_label='Date',encoding='utf-8-sig')
events.to_csv(OUT/'effective_updates.csv',index=False)

assert len(buttons)==7
assert all(len(b["args"][0]["visible"])==len(fig_focus.data) for b in buttons)
assert np.isfinite(chart.select_dtypes(include='number').to_numpy()).all()
print(f"2025年：{len(idx)}条记录，7项变量可切换，{len(events)}次触发更新。")
display(fig_focus)
