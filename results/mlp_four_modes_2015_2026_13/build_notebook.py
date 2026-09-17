from pathlib import Path
import nbformat as nb,hashlib
out=Path(__file__).resolve().parent
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/13_加权MSE的训练.ipynb')
b=src.read_bytes();(out/'13_before.ipynb').write_bytes(b);(out/'original.sha256').write_text(hashlib.sha256(b).hexdigest())
n=nb.reads(b.decode(),4);(out/'start_cell.txt').write_text(str(len(n.cells)))
def md(s):n.cells.append(nb.v4.new_markdown_cell(s))
def code(s):n.cells.append(nb.v4.new_code_cell(s))
md('''# 2015—2026：四种训练/更新方式的跨年比较

四组均采用两阶段Stacking、每阶段16→32两隐藏层、ReLU、Adam学习率0.01、全批量各100轮，同日特征预测同日股价：

1. **每月更新·等权**：每月验证选种子后，将月前全部有效历史纳入等权重训。
2. **每月更新·半衰期60**：与第一组同训练日期、同种子，只把最终两阶段重训损失改为半衰期60条有效记录的加权MSE。
3. **双向触发·半衰期60**：金或铜相对MA20的绝对偏离在 `<3%` 与 `>=3%` 间切换，下一条有效记录更新；同日信号合并，不设冷却期。仅年初初始化，不额外叠加月度更新。
4. **每年一次·旧结果**：直接读取11的 `mlp_annual_ten_ma_11` 中Stacking的100轮、10种子测试结果，每年仅在年初训练并全年冻结。本轮完全不重训这一组，也不把十个预测平均成集成模型；柱高是十个种子各自误差的均值。

三种更新策略沿用本章固定的10个候选种子。非2020年份在每次更新前，用此前三个日历月验证MAPE选择等权候选，精确并列按候选顺序，再纳入验证数据从头重训；半衰期仅用于最终重训，候选选择仍等权，避免同时改动选择协议。2020复用已保存的选择记录，并验证三组预测复现前文结果。三组每年重新初始化，年初共享1月1日选择起点，确保第一段加权模型一致。

**比较边界**：旧全年模式是另一组10个固定种子的误差均值，更新模式是验证选种后的单一模型序列；旧全年模式还使用等权损失。所以它是历史整体方案参照，不是只改变更新频率的完全配对试验。只比较共同测试日期，验证旧120个种子-年份指标与原结果一致。触发模型不使用生效日或之后的标签。

**平均口径**：主指标是各年MAPE的等权平均；同时列按记录数加权的整体MAPE、MSE。2015—2025为11个完整年份；2015—2026汇总另含2026年1月1日至7月23日的152条记录，不把它称为完整2026年表现。跨年绝对MSE受股价水平影响，作为补充。半衰期60此前在2020筛选过，本轮不是全新盲测。

以下保留完整可运行实现；已完成年份按本节专用目录缓存，重新运行会复用，旧全年结果始终只读。
''')
source=(out/'run.py').read_text().split("if __name__=='__main__':")[0]
source=source.replace('OUT=Path(__file__).resolve().parent\n','')
line="ROOT=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-')"
source=source.replace(line,line+"\nOUT=Path(os.environ.get('MLP13_FOUR_OUT',str(ROOT/'results/mlp_four_modes_2015_2026_13')));OUT.mkdir(parents=True,exist_ok=True)")
code(source)
code("prepare()\ncoverage_results=[year_run(year) for year in range(2015,2027)]\nprint('12 years ready; annual frozen baseline was not trained.')")
summary=(out/'summarize.py').read_text().replace('OUT=Path(__file__).resolve().parent\n','')
code(summary+"\nfrom IPython.display import display, Image\ndisplay(Image(filename=str(OUT/'annual_bars.png')))\ndisplay(Image(filename=str(OUT/'average_bars.png')))\ndisplay(summary.round(5))")
nb.validate(n);nb.write(n,out.parent/'13_加权MSE的训练.ipynb')
s=Path('outputs/mlp_half_life_comparison_13/execute.py').read_text().replace('MLP13_GRID_OUT','MLP13_FOUR_OUT').replace('mlp_half_life_comparison_13','mlp_four_modes_2015_2026_13').replace("print(scope['monthly'].to_string()); print(scope['summary'].to_string(index=False))","print(scope['summary'].to_string(index=False))")
(out/'execute_notebook.py').write_text(s)
