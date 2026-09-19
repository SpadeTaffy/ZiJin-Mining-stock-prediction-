from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')
OUT=Path(os.environ.get('MLP14_PX_OUT',str(ROOT/'results/all_variables_plotly_2016_14')));OUT.mkdir(parents=True,exist_ok=True)
SOURCE=OUT.parent/'mlp_four_modes_2015_2026_13'
if not (SOURCE/'predictions.csv').exists():SOURCE=ROOT/'results/mlp_four_modes_2015_2026_13'
rawdata=pd.read_pickle(ROOT/'src/pickle/Dataset.pkl');raw=rawdata['X'].copy()
assert raw.index.equals(rawdata['Y'].index) and raw.index.is_unique and raw.index.is_monotonic_increasing
names={'A_500':'A500指数','Dollar Index':'美元指数','cn_10y_rate':'中国10年期利率','us_10y_rate':'美国10年期利率','brent':'布伦特原油','Copper Futures Price':'铜','Gold Futures Price':'黄金'}
assert set(raw.columns)==set(names)
levels=raw.rename(columns=names).copy();levels.insert(0,'股价',np.asarray(rawdata['Y']).reshape(-1))
idx=levels.index[levels.index.year==2016];assert len(idx)==268
ma={k:levels.rolling(k,min_periods=k).mean() for k in [5,20]}
daily=100*(levels/levels.shift(1)-1)
dev={k:100*(levels/ma[k]-1) for k in [5,20]}
pred=pd.read_csv(SOURCE/'predictions.csv',parse_dates=['Date']).set_index('Date').loc[idx]
assert np.allclose(pred.Actual,levels.loc[idx,'股价'])
updates=pd.read_csv(SOURCE/'updates.csv',parse_dates=['Effective_date','Trigger_date'])
updates=updates[(updates.Year==2016)&(updates.Method=='Crossing_H60')].copy()
events=updates[updates.Trigger_date.notna()].copy()
assert len(events)==60 and (events.Trigger_date<events.Effective_date).all()
features=pd.read_csv(SOURCE/'features_and_target.csv',parse_dates=['Date']).set_index('Date').loc[idx]
for eng,cn in [('gold','黄金'),('copper','铜')]:
    assert np.allclose(daily.loc[idx,cn],100*features[f'{eng}_daily'])
    for k in [5,20]:assert np.allclose(dev[k].loc[idx,cn],100*features[f'{eng}_vs_ma{k}'])
base=levels.loc[idx[0]]
indexed=100*levels.loc[idx]/base
# Prediction uses actual stock's year-start denominator: prediction errors are retained.
indexed['双向触发60预测']=100*pred.Crossing_H60/base['股价']
assert np.allclose(indexed.drop(columns='双向触发60预测').iloc[0],100)
LABELS=['相邻记录变化','相对MA5偏离','相对MA20偏离']
colors={'股价':'#222222','双向触发60预测':'#009E73','A500指数':'#6F4E7C','美元指数':'#CC6677','中国10年期利率':'#44AA99','美国10年期利率':'#882255','布伦特原油':'#AA7733','铜':'#0072B2','黄金':'#D59400',LABELS[0]:'#8B9199',LABELS[1]:'#009E73',LABELS[2]:'#CC79A7'}
panels=['水平指数（2016首条记录=100）']+[name+'：相对变化（%）' for name in names.values()]
frames=[]
for name in indexed:
    original=pred.Crossing_H60 if name=='双向触发60预测' else levels.loc[idx,name]
    frames.append(pd.DataFrame({'日期':idx,'数值':indexed[name].to_numpy(),'曲线':name,'面板':panels[0],'原始值':original.to_numpy()}))
for name in names.values():
    for label,values in zip(LABELS,[daily[name],dev[5][name],dev[20][name]]):
        frames.append(pd.DataFrame({'日期':idx,'数值':values.loc[idx].to_numpy(),'曲线':label,'面板':name+'：相对变化（%）','原始值':levels.loc[idx,name].to_numpy()}))
long=pd.concat(frames,ignore_index=True);assert np.isfinite(long['数值']).all()
fig_all=px.line(long,x='日期',y='数值',color='曲线',facet_row='面板',category_orders={'面板':panels},color_discrete_map=colors,custom_data=['原始值'],height=2300,facet_row_spacing=.017,render_mode='svg',title='2016｜股价、预测与全部7项输入变量：水平、相邻变化及MA5/MA20偏离')
fig_all.update_yaxes(matches=None,showticklabels=True,title_text=None)
fig_all.update_xaxes(matches='x',showticklabels=True,title_text=None)
seen=set()
for t in fig_all.data:
    t.showlegend=t.name not in seen;seen.add(t.name)
    t.line.width=2.2 if t.name=='股价' else 1.7 if t.name=='双向触发60预测' else 1.3
    suffix='%' if t.name in LABELS else ''
    t.hovertemplate='%{x|%Y-%m-%d}<br>'+t.name+': %{y:.3f}'+suffix+'<br>原始水平: %{customdata[0]:.5f}<extra></extra>'
for i,panel in enumerate(panels):
    row=len(panels)-i
    if i:
        name=list(names.values())[i-1];vals=np.column_stack([daily.loc[idx,name],dev[5].loc[idx,name],dev[20].loc[idx,name]])
        limit=float(np.abs(vals).max()*1.12)
        fig_all.update_yaxes(range=[-limit,limit],ticksuffix='%',zeroline=True,zerolinecolor='#aaa',row=row,col=1)
    else:fig_all.update_yaxes(title_text='水平指数',row=row,col=1)
# Identical range for gold and copper, matching notebook 12.
metal_limit=float(max(np.abs(long[long['面板'].isin(['黄金：相对变化（%）','铜：相对变化（%）'])]['数值']).max()*1.12,3.5))
for metal in ['黄金','铜']:
    row=len(panels)-panels.index(metal+'：相对变化（%）')
    fig_all.update_yaxes(range=[-metal_limit,metal_limit],row=row,col=1)
    for v in [-3,3]:fig_all.add_hline(y=v,row=row,col=1,line_dash='dash',line_color='#b7a78b',line_width=1)
for a,panel in zip(fig_all.layout.annotations,reversed(panels)):a.text=panel
# Trigger markers reflect the original full-history MA20 state, not stock prediction errors.
outside=dev[20][['黄金','铜']].abs().ge(3)
previous=outside.shift(1)
trigger_export=[]
for metal in ['黄金','铜']:
    change=outside[metal].ne(previous[metal])&previous[metal].notna()
    dates=idx[change.loc[idx].to_numpy()]
    row=len(panels)-panels.index(metal+'：相对变化（%）')
    for outward,symbol,label in [(True,'triangle-up','超出±3%'),(False,'triangle-down','回到±3%内')]:
        selected=dates[outside.loc[dates,metal].eq(outward).to_numpy()]
        fig_all.add_trace(go.Scatter(x=selected,y=dev[20].loc[selected,metal],mode='markers',marker=dict(symbol=symbol,size=8,color='#B75D00' if outward else '#4979A6'),name=label,legendgroup=label,showlegend=metal=='黄金',hovertemplate='%{x|%Y-%m-%d}<br>'+metal+' '+label+'<br>MA20偏离 %{y:.3f}%<extra></extra>'),row=row,col=1)
        for day in selected:trigger_export.append(dict(Date=day,Metal=metal,Direction='outward' if outward else 'inward',MA20_deviation_pct=float(dev[20].loc[day,metal])))
# Model-effective dates can be toggled on without cluttering the default price panel.
fig_all.add_trace(go.Scatter(x=events.Effective_date,y=indexed.loc[events.Effective_date,'双向触发60预测'],mode='markers',marker=dict(size=6,color='#C44E52'),name='新模型生效日（点击显示）',visible='legendonly',hovertemplate='%{x|%Y-%m-%d}<br>新模型生效<extra></extra>'),row=len(panels),col=1)
fig_all.update_layout(template='plotly_white',hovermode='x unified',font=dict(family='Arial, PingFang SC, Microsoft YaHei, sans-serif'),margin=dict(l=75,r=270,t=110,b=70),legend=dict(title='点击切换；双击单独显示',x=1.06,y=1,font=dict(size=11)),xaxis=dict(title='日期',rangeslider=dict(visible=True,thickness=.028),rangeselector=dict(buttons=[dict(count=1,label='1个月',step='month',stepmode='backward'),dict(count=3,label='3个月',step='month',stepmode='backward'),dict(label='全年',step='all')],y=-.27)))
config={'displaylogo':False,'scrollZoom':True,'responsive':True,'toImageButtonOptions':{'format':'png','scale':2}}
fig_all.write_html(OUT/'2016_all_variables.html',include_plotlyjs=True,config=config)
fig_all.write_json(OUT/'all_variables_figure.json')
# Compact view: stock and prediction remain fixed; dropdown changes the explanatory variable.
fig_focus=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.39,.32,.29],vertical_spacing=.085,subplot_titles=['股价与预测（同一基准，保留预测误差）','变量水平、MA5与MA20（年初=100）','变量相邻变化、MA5/MA20偏离（%）'])
for name in ['股价','双向触发60预测']:
    fig_focus.add_trace(go.Scatter(x=idx,y=indexed[name],name=name,line=dict(color=colors[name],width=2),hovertemplate='%{x|%Y-%m-%d}<br>'+name+'指数 %{y:.3f}<extra></extra>'),row=1,col=1)
fig_focus.add_trace(go.Scatter(x=events.Effective_date,y=indexed.loc[events.Effective_date,'双向触发60预测'],mode='markers',marker=dict(size=5,color='#C44E52'),name='新模型生效日',visible='legendonly',hovertemplate='%{x|%Y-%m-%d}<br>新模型生效<extra></extra>'),row=1,col=1)
fixed=3;groups={};default='黄金'
for name in names.values():
    group=[]
    for label,values,dash in [(name,levels[name],'solid'),(name+' MA5',ma[5][name],'dash'),(name+' MA20',ma[20][name],'dot')]:
        group.append(len(fig_focus.data));fig_focus.add_trace(go.Scatter(x=idx,y=100*values.loc[idx]/base[name],name=label,visible=name==default,customdata=values.loc[idx].to_numpy(),line=dict(color=colors[name],dash=dash,width=1.7),hovertemplate='%{x|%Y-%m-%d}<br>'+label+'指数 %{y:.3f}<br>原始水平 %{customdata:.5f}<extra></extra>'),row=2,col=1)
    for label,values in zip(LABELS,[daily[name],dev[5][name],dev[20][name]]):
        group.append(len(fig_focus.data));fig_focus.add_trace(go.Scatter(x=idx,y=values.loc[idx],name=name+' · '+label,visible=name==default,line=dict(color=colors[label],width=1.5),hovertemplate='%{x|%Y-%m-%d}<br>'+label+' %{y:.3f}%<extra></extra>'),row=3,col=1)
    if name in ['黄金','铜']:
        for v in [-3,3]:
            group.append(len(fig_focus.data));fig_focus.add_trace(go.Scatter(x=[idx.min(),idx.max()],y=[v,v],mode='lines',name=f'{v:+}%触发边界',visible=name==default,line=dict(color='#aaa',dash='dash',width=1),showlegend=False,hoverinfo='skip'),row=3,col=1)
    groups[name]=group
buttons=[]
for name,group in groups.items():
    visible=[True,True,'legendonly']+[False]*(len(fig_focus.data)-fixed)
    for j in group:visible[j]=True
    vals=np.column_stack([daily.loc[idx,name],dev[5].loc[idx,name],dev[20].loc[idx,name]])
    lim=max(float(np.abs(vals).max()*1.15),3.5 if name in ['黄金','铜'] else .1)
    buttons.append(dict(label=name,method='update',args=[{'visible':visible},{'title.text':f'2016｜股价、预测与{name}：逐变量细看','yaxis2.autorange':True,'yaxis3.range':[-lim,lim]}]))
fig_focus.update_layout(template='plotly_white',title=dict(text='2016｜股价、预测与黄金：逐变量细看',y=.98,x=.05),height=1100,hovermode='x unified',font=dict(family='Arial, PingFang SC, Microsoft YaHei, sans-serif'),margin=dict(l=80,r=220,t=155,b=80),legend=dict(x=1.03,y=1,title='点击切换曲线'),updatemenus=[dict(type='dropdown',active=list(names.values()).index(default),buttons=buttons,x=0,y=1.075,xanchor='left',yanchor='top')])
fig_focus.update_xaxes(matches='x',showticklabels=True)
fig_focus.update_yaxes(title_text='股价指数',row=1,col=1);fig_focus.update_yaxes(title_text='变量指数',row=2,col=1);fig_focus.update_yaxes(title_text='相对变化（%）',ticksuffix='%',zeroline=True,zerolinecolor='#888',row=3,col=1)
goldlim=max(float(np.abs(np.column_stack([daily.loc[idx,'黄金'],dev[5].loc[idx,'黄金'],dev[20].loc[idx,'黄金']])).max()*1.15),3.5)
fig_focus.update_yaxes(range=[-goldlim,goldlim],row=3,col=1)
fig_focus.update_layout(xaxis3=dict(rangeslider=dict(visible=True,thickness=.06),rangeselector=dict(buttons=[dict(count=1,label='1个月',step='month',stepmode='backward'),dict(count=3,label='3个月',step='month',stepmode='backward'),dict(label='全年',step='all')],y=-.30)))
fig_focus.write_html(OUT/'2016_variable_focus.html',include_plotlyjs=True,config=config);fig_focus.write_json(OUT/'focus_figure.json')
long.to_csv(OUT/'chart_long_data.csv',index=False)
chart=levels.loc[idx].copy();chart['双向触发60预测']=pred.Crossing_H60
for name in levels:
    chart[name+'_MA5']=ma[5].loc[idx,name];chart[name+'_MA20']=ma[20].loc[idx,name]
    chart[name+'_相邻变化_pct']=daily.loc[idx,name]
    for k in [5,20]:chart[name+f'_MA{k}偏离_pct']=dev[k].loc[idx,name]
chart.to_csv(OUT/'chart_data.csv',index_label='Date')
events.to_csv(OUT/'effective_updates.csv',index=False)
trigger_table=pd.DataFrame(trigger_export).sort_values(['Date','Metal']);trigger_table.to_csv(OUT/'metal_crossings.csv',index=False)
assert set(trigger_table.Date)==set(events.Trigger_date)
# Export before/after MA20 deviation per effective update to make jumps inspectable.
audit=[]
for r in events.itertuples(index=False):
    day=r.Trigger_date;prevday=levels.index[levels.index.get_loc(day)-1]
    row={'触发日':day,'生效日':r.Effective_date,'种子':r.Seed}
    for metal in ['黄金','铜']:
        row[metal+'_前条MA20偏离_pct']=dev[20].loc[prevday,metal]
        row[metal+'_当条MA20偏离_pct']=dev[20].loc[day,metal]
        row[metal+'_偏离变化_pp']=dev[20].loc[day,metal]-dev[20].loc[prevday,metal]
    audit.append(row)
pd.DataFrame(audit).to_csv(OUT/'trigger_audit.csv',index=False)
(OUT/'config.json').write_text(json.dumps(dict(year=2016,N=len(idx),base_date=str(idx[0].date()),variables=names,derived_model_features=6,added_diagnostics='other variables daily/MA5/MA20 relative changes, not new model inputs',ma_full_history=True,ma_includes_current=True,rate_change_unit='relative percent, NOT percentage point or basis point',prediction_denominator='actual stock first record',model_updates=len(events),retrained=False,new_trigger_rule=False,dataset_sha256=hashlib.sha256((ROOT/'src/pickle/Dataset.pkl').read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
print('268条2016记录；7项原始输入全部覆盖；金铜6项派生特征与模型一致；60次触发与保存更新记录完全一致。')
display(fig_all);display(fig_focus)
