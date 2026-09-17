from pathlib import Path
import pandas as pd,nbformat as nb
out=Path(__file__).resolve().parent
s=pd.read_csv(out/'summary.csv');a=pd.read_csv(out/'annual_mape.csv').set_index('Year')
labels={'Monthly_equal':'每月更新·等权','Monthly_H60':'每月更新·半衰期60','Crossing_H60':'双向触发·半衰期60','Annual_frozen_old':'每年一次·旧结果'}
order=list(labels)
full=s[s.Period=='2015-2025_complete'].set_index('Method');all_=s[s.Period=='2015-2026_partial'].set_index('Method')
text='## 数字汇总与结论\n\n主表为**各年MAPE等权平均**，越低越好：\n\n| 方案 | 2015—2025完整年份 | 2015—2026（含部分2026） |\n|---|---:|---:|\n'
for m in order:text+=f'| {labels[m]} | {full.loc[m,"Annual_equal_MAPE"]:.4f}% | {all_.loc[m,"Annual_equal_MAPE"]:.4f}% |\n'
text+='\n补充：把所有测试记录合并后的指标如下，旧全年模式仍为各种子指标平均，不是预测集成：\n\n| 方案 | 整体MAPE | 整体普通MSE |\n|---|---:|---:|\n'
for m in order:text+=f'| {labels[m]} | {all_.loc[m,"Record_weighted_MAPE"]:.4f}% | {all_.loc[m,"Record_weighted_MSE"]:.6f} |\n'
text+='\n**本轮排序：双向触发＋半衰期60最佳，其次月度＋半衰期60，再次月度等权，旧全年冻结模式误差最高。** 在2015—2025完整年份和包含2026部分年份两种汇总口径下，排序一致。\n\n'
text+='- 月度等权 → 月度半衰期60：年度平均MAPE从6.6729%降至5.7435%，下降0.9293个百分点（13.93%），9/12年改善；2016、2021、2026略有退步，其中2021几乎持平。\n'
text+='- 月度半衰期60 → 双向触发半衰期60：从5.7435%降至4.8352%，进一步下降0.9083个百分点（15.81%），11/12年改善；仅2018由3.4709%升至3.9202%。\n'
text+='- 旧全年冻结 → 双向触发半衰期60：从12.7872%降至4.8352%，低7.9520个百分点（62.19%）。这是整体方案差异，含训练损失、更新日期、候选种子和选择方式变化，不能解释为单纯触发机制的因果效果。\n'
text+='- 三种更新策略在这12个年份的MAPE均低于旧全年10种子平均。2022年旧全年均值为31.7990%，三个更新方案分别为6.8175%、6.6610%、6.1768%。\n'
text+='- 2015—2026累计：月度方案各部署139个模型（含12次年初初始化，年内更新127次）；触发方案部署689个模型（含12次初始化，年内更新677次）。更高更新频率也是改善的可能来源，尚未与相同更新次数的固定周期对照。\n\n'
text+='逐年MAPE明细（2026截至7月23日）：\n\n| 年份 | 月度等权 | 月度60 | 双向触发60 | 旧全年冻结 |\n|---|---:|---:|---:|---:|\n'
for year,r in a.iterrows():text+=f'| {year}{"*" if year==2026 else ""} | '+ ' | '.join(f'{r[m]:.4f}%' for m in order)+' |\n'
text+='\n检查通过：12年共3124条共同测试记录；旧全年模式120个种子-年份指标及既有汇总一致；2020三种更新策略复现前文；所有模型的训练日期早于应用日期。旧全年模型本轮未重训。不能因2026只有152条记录而将其表现外推为全年。\n'
(out/'report.md').write_text(text)
p=out.parent/'13_加权MSE的训练.ipynb';n=nb.read(p,4);n.cells.append(nb.v4.new_markdown_cell(text));nb.validate(n);nb.write(n,p)
