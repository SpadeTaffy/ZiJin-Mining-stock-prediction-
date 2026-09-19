from pathlib import Path
import nbformat,hashlib
out=Path('outputs/variable_focus_2017_15')
p=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/15_分析每一年数据,细化更新条件.ipynb')
raw=p.read_bytes();(out/'15_before.ipynb').write_bytes(raw);(out/'original.sha256').write_text(hashlib.sha256(raw).hexdigest())
s=Path('outputs/all_variables_plotly_2016_14/charts.py').read_text()
code=s[:s.index("panels=")]+"config={'displaylogo':False,'scrollZoom':True,'responsive':True,'toImageButtonOptions':{'format':'png','scale':2}}\n"+s[s.index('# Compact view:'):s.index("long.to_csv")]
code+=s[s.index('chart=levels.loc'):s.index('trigger_table=')]
code=code.replace('2016','2017').replace('MLP14_PX_OUT','MLP15_PX_OUT').replace('all_variables_plotly_2017_14','variable_focus_2017_15').replace('assert len(idx)==268','assert len(idx)==267').replace('assert len(events)==60 and','assert len(events)==43 and')
code+='\nassert len(buttons)==7\nassert all(len(b["args"][0]["visible"])==len(fig_focus.data) for b in buttons)\nassert np.isfinite(chart.to_numpy()).all()\nprint(f"2017年：{len(idx)}条记录，7项变量可切换，{len(events)}次触发更新。")\ndisplay(fig_focus)\n'
(out/'charts.py').write_text(code)
n=nbformat.read(p,as_version=4);(out/'start_cell.txt').write_text(str(len(n.cells)))
n.cells.extend([nbformat.v4.new_markdown_cell('## 2017年｜逐变量细看\n\n沿用14开头的逐变量交互图，仅保留本图，不生成上方的全部变量总览。下拉切换A500、美元指数、中美10年期利率、原油、铜、黄金。\n\n预测口径沿用该参考图：原金铜3%双向触发、半衰期60；直接读取既有预测，不重新训练。水平按年初首条记录归一化为100；MA5/MA20在完整历史上计算，含当条记录。利率的变化表示相对百分比，不是百分点。'),nbformat.v4.new_code_cell(code)])
nbformat.write(n,Path('outputs')/p.name)
e=Path('outputs/all_variables_plotly_2016_14/execute.py').read_text().replace('MLP14_PX_OUT','MLP15_PX_OUT').replace('all_variables_plotly_2016_14','variable_focus_2017_15').replace('14_继续增加规则.ipynb',p.name).replace('2016 all-variable charts complete.','2017 focus chart complete.')
(out/'execute.py').write_text(e)
