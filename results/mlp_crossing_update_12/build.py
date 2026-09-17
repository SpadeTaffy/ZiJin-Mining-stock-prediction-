from pathlib import Path
import nbformat as nb
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/12_减少预测的时间,扩大验证集的范围.ipynb')
n=nb.read(src,as_version=4);nb.write(n,'outputs/mlp_crossing_update_12/12_before_crossing.ipynb')
Path('outputs/mlp_crossing_update_12/start_cell.txt').write_text(str(len(n.cells)))
start=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and c.source.startswith('# 金铜MA20偏离回归正常后触发更新'))
codes=[c.source for c in n.cells[start:] if c.cell_type=='code']
setup,training,plot=codes[:3]
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 金铜MA20±3%双向穿越触发：超出或回归，次条记录更新

保留上一节仅回归触发的实验，本节改为**金或铜任一种发生状态改变即触发**：`abs(P/MA20−1)<3%`为正常，`>=3%`为超出。从正常到超出、从超出回正常均触发。持续超出不每天重复触发，正负两侧都算；若从+4%直接到−4%，两条已观察记录都处于超出状态，不额外推断盘中穿越，不触发。同日金铜或不同方向信号合并为一次更新，不设冷却期。

其它条件与上一节完全一致：2020年、既定10候选种子、两阶段Stacking各100轮、前三日历月验证选种子，然后合并训练验证从头重训。触发当天继续使用旧模型，下一条有效记录开始时重训并应用；所有训练/验证标签截止触发日，不能使用生效日真实股价。初始化单独计数。

继续与原每月更新＋纳入验证数据重训方案对照；同时列出上一节仅回归触发的结果。仍为原Dataset同日输入的历史样本外回测，非月初/日初已知未来输入的严格提前预测。
''')
setup=setup.replace('MLP12_REENTRY_OUT','MLP12_CROSSING_OUT').replace('results/mlp_reentry_update_12','results/mlp_crossing_update_12')
setup=setup.replace("individual metal abs(MA20 deviation) re-enters below 3% after at least 3%; same-date events merged", "individual metal abs(MA20 deviation) crosses outward or inward at 3%; same-date events merged")
setup=setup.replace("reentry=outside.shift(1,fill_value=False)&~outside", "previous=outside.shift(1)\nvalid_previous=previous.notna().all(axis=1)\nentry=outside & previous.eq(False)\nreentry=~outside & previous.eq(True)\ncrossing=(entry|reentry)&valid_previous.to_numpy()[:,None]")
setup=setup.replace('reentry.any(axis=1)','crossing.any(axis=1)')
setup=setup.replace("bool(reentry.loc[day,'gold_vs_ma20'])", "bool(crossing.loc[day,'gold_vs_ma20'])").replace("bool(reentry.loc[day,'copper_vs_ma20'])", "bool(crossing.loc[day,'copper_vs_ma20'])")
setup=setup.replace('Gold_deviation_percent=100*', "Gold_direction=('outward' if entry.loc[day,'gold_vs_ma20'] else 'inward' if reentry.loc[day,'gold_vs_ma20'] else ''),Copper_direction=('outward' if entry.loc[day,'copper_vs_ma20'] else 'inward' if reentry.loc[day,'copper_vs_ma20'] else ''),Gold_deviation_percent=100*")
code(setup)
training=training.replace("Gold_reentries=int(events.Gold_trigger.sum()),Copper_reentries=int(events.Copper_trigger.sum())", "Gold_crossings=int(events.Gold_trigger.sum()),Copper_crossings=int(events.Copper_trigger.sum()),Gold_outward=int(events.Gold_direction.eq('outward').sum()),Gold_inward=int(events.Gold_direction.eq('inward').sum()),Copper_outward=int(events.Copper_direction.eq('outward').sum()),Copper_inward=int(events.Copper_direction.eq('inward').sum())")
training=training.replace("Method='Return-to-normal updates'", "Method='Outward-or-inward crossing updates'")
training+='''
# 旧回归触发方案只读复用，不重新训练。
REENTRY=OUT.parent/'mlp_reentry_update_12'
if not (REENTRY/'stitched_predictions.csv').exists():REENTRY=ROOT/'results/mlp_reentry_update_12'
prior_event=pd.read_csv(REENTRY/'stitched_predictions.csv',parse_dates=['Date'])
assert pd.DatetimeIndex(prior_event.Date).equals(whole) and np.allclose(prior_event.Actual,stitched.Actual)
stitched['Reentry_only_prediction']=prior_event.Predicted.to_numpy()
stitched['Reentry_only_APE']=100*np.abs(stitched.Reentry_only_prediction-stitched.Actual)/stitched.Actual.abs()
monthly['Reentry_only_MAPE']=stitched.groupby('Month').Reentry_only_APE.mean()
monthly['Both_minus_reentry_pp']=monthly.Event_MAPE-monthly.Reentry_only_MAPE
oldcounts=json.loads((REENTRY/'update_counts.json').read_text())
summary=pd.concat([summary,pd.DataFrame([dict(Method='Inward-only updates',Annual_daily_MAPE=stitched.Reentry_only_APE.mean(),Monthly_equal_MAPE=monthly.Reentry_only_MAPE.mean(),Annual_MSE=np.mean((stitched.Reentry_only_prediction-stitched.Actual)**2),Updates_excluding_initial=oldcounts['Triggered_updates'])])],ignore_index=True)
stitched.to_csv(OUT/'stitched_predictions.csv',index=False);monthly.to_csv(OUT/'monthly_comparison.csv');summary.to_csv(OUT/'summary.csv',index=False)
# 检查两个方向确实涵盖旧回归触发日，且每个事件均有可验证的前后状态改变。
old_events=pd.read_csv(REENTRY/'trigger_events.csv',parse_dates=['Trigger_date'])
assert set(old_events.Trigger_date)<=set(events.Trigger_date)
for r in events.itertuples():
    for metal,flag,direction in [('gold',r.Gold_trigger,r.Gold_direction),('copper',r.Copper_trigger,r.Copper_direction)]:
        if flag:
            col=metal+'_vs_ma20';i=x.index.get_loc(r.Trigger_date)
            before=bool(outside.iloc[i-1][col]);after=bool(outside.iloc[i][col])
            assert before!=after and direction==('outward' if after else 'inward')
print('与仅回归、每月重训的完整对照：');display(summary.round(4))
'''
code(training)
plot=plot.replace("label='Re-entry triggered + refit'", "label='Outward/inward crossing + refit'").replace('Return-to-normal triggered updates','Outward/inward triggered updates').replace('Return below absolute 3% triggers next-record update | triangles: trigger dates','Cross absolute 3% outward or inward | up: outward, down: inward')
plot=plot.replace("marker='v',s=35,zorder=5)","marker='^' if r.Gold_direction=='outward' else 'v',s=35,zorder=5)",1)
# Second scatter uses copper direction.
pos=plot.index('if r.Copper_trigger:')
plot=plot[:pos]+plot[pos:].replace("marker='v',s=35,zorder=5)","marker='^' if r.Copper_direction=='outward' else 'v',s=35,zorder=5)",1)
plot=plot.replace("OUT/'reentry_comparison.png'", "OUT/'crossing_comparison.png'")
code(plot)
nb.validate(n);nb.write(n,'outputs/12_减少预测的时间,扩大验证集的范围.ipynb')
