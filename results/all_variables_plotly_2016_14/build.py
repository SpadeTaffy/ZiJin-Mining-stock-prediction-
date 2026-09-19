from pathlib import Path
import nbformat as nb,hashlib
out=Path(__file__).resolve().parent
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/14_继续增加规则.ipynb')
b=src.read_bytes();(out/'14_before.ipynb').write_bytes(b);(out/'original.sha256').write_text(hashlib.sha256(b).hexdigest());n=nb.reads(b.decode(),4)
(out/'start_cell.txt').write_text(str(len(n.cells)))
n.cells.append(nb.v4.new_markdown_cell('''# 2016年：股价与全部输入变量的交互对比（沿用12的Plotly Express图）

本节先观察2016年的变量和现有预测，不重新训练，也不预先添加新的触发限制。

**覆盖范围**：A500指数、美元指数、中国10年期利率、美国10年期利率、布伦特原油、铜、黄金，共7项原始输入；另展示真实股价和“双向触发＋半衰期60”最终预测。金铜相邻记录涨跌幅、MA5/MA20偏离共6项派生曲线逐条与模型输入核对。其他变量的均线和变化率用于诊断，不表示已加入模型。

- 第一张为全部变量总览：顶部统一为2016首条记录=100；下面7个面板分别展示相邻记录变化、相对MA5、相对MA20的偏离。各面板独立尺度，金铜两面板尺度一致。
- 第二张为逐变量细看：顶部保留股价与预测，下拉框在7个变量中切换；中间显示变量水平及MA5/MA20，下方显示相对变化。图例点击开关、双击单独显示；拖动缩放、日期滑块可联动各面板。悬停可查看原始水平。
- 预测曲线使用**真实股价的年初基数**归一化，避免把预测自己的起点也强制设为100而掩盖误差。利率的指数仅表示水平相对变化，并非投资收益。
- 所有MA在完整历史上计算后截取2016，包含当日，按记录计数。“相邻记录变化”为100×[P(t)/P(t−1)−1]；MA偏离为100×[P(t)/MA(t)−1]。利率采用同一相对变化公式，**不是百分点变化或基点变化**。
- 金铜面板的±3%虚线为旧规则边界；向上三角表示从区间内到区间外，向下三角表示回到区间内。顶部“新模型生效日”默认隐藏，点击图例显示；生效日为触发后的下一条有效记录。

**关于2%→5%的情况**：如果两条连续已观察记录分别为2%与5%，旧规则会在5%这条记录上触发，下一条记录生效。它不会重算触发当条已经产生的预测；持续在3%之外继续上升也不会重复触发。观察时应区分“未跨阈值”“已触发但尚未生效”和“区间外继续变动”。本节的触发明细同时保存前后MA20偏离和偏离变化，便于后续提出限制。
'''))
n.cells.append(nb.v4.new_code_cell((out/'charts.py').read_text()))
nb.validate(n);nb.write(n,out.parent/'14_继续增加规则.ipynb')
s=Path('outputs/mlp_crossing_h60_annual_plots_13/execute.py').read_text().replace('MLP13_ANNUAL_PLOTS_OUT','MLP14_PX_OUT').replace('mlp_crossing_h60_annual_plots_13','all_variables_plotly_2016_14').replace('13_加权MSE的训练','14_继续增加规则').replace("print('Annual prediction charts complete.')","print('2016 all-variable charts complete.')")
(out/'execute.py').write_text(s)
