from pathlib import Path
import hashlib,nbformat as nb
out=Path(__file__).resolve().parent;src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/14_继续增加规则.ipynb')
b=src.read_bytes();(out/'14_before.ipynb').write_bytes(b);(out/'original.sha256').write_text(hashlib.sha256(b).hexdigest());n=nb.reads(b.decode(),4);(out/'start_cell.txt').write_text(str(len(n.cells)))
n.cells.append(nb.v4.new_markdown_cell('''# 2016：A500、美债新特征与多阈值双向触发实验

偏离率统一为 `P(t)/MA20(t)-1`，MA在完整历史上计算、包含当日，20指有效记录窗口。美债使用现有`us_10y_rate`（10年期收益率），10%/20%表示收益率相对MA20的偏离，不是10/20个百分点，更不是债券价格变化。

| 变量 | 原触发边界（绝对偏离） | 新触发边界（绝对偏离） |
|---|---|---|
| 黄金 | 3% | 3%（保留） |
| 铜 | 3% | 3%、8% |
| A500 | 无 | 5%、8% |
| 美国10年期收益率 | 无 | 10%、20% |

对每个阈值，`abs(偏离)<阈值`与`>=阈值`的状态双向转换都触发，任一变量/阈值满足即更新。同一条记录跨多个阈值或多个变量触发只合并一次；持续位于同一阈值区间不会逐日重复更新；触发当条仍用旧模型，下一条记录生效。没有增加冷却期或叠加月度更新。

四组控制：①原金铜3%＋原特征；②只加A500、美债MA20偏离两项第二阶段特征；③只增加触发边界，输入不变；④特征和触发都增加。第一阶段仍为7原始变量；第二阶段由原7输入（基础预测＋6金铜特征）变为9输入。全组两个阶段半衰期60、各100轮、16→32隐藏层、ReLU、Adam 0.01、全批量、等权标准化。

为比较特征与更新时间的作用，**同一生效日各组使用相同的基准选定种子**：旧日期复用此前选择；新增日期按原特征、等权候选训练，在此前3个日历月验证MAPE上从原10个候选中选择。然后以半衰期60在生效日前全部历史重训。新特征组不单独选种子，所以不是对新结构充分调参后的最佳性能比较。

2016已用于观察和制定本规则，本轮是开发集实验，不是独立样本外年份验证。原金铜3%预测须复现上一章，所有268条测试日期一致，训练标签严格早于模型生效日。
'''))
s=(out/'experiment.py').read_text().split("if __name__=='__main__':")[0]
s=s.replace("OUT=Path(os.environ.get('MLP14_MULTI_OUT',str(Path(__file__).resolve().parent)))", "OUT=Path(os.environ.get('MLP14_MULTI_OUT','/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/results/multi_threshold_2016_14'));OUT.mkdir(parents=True,exist_ok=True)")
n.cells.append(nb.v4.new_code_cell(s+"\nif not (OUT/'summary.csv').exists():experiment()\n# 如需重新训练，可显式调用 experiment()。\n"))
plot=(out/'plots.py').read_text().replace('OUT=Path(__file__).resolve().parent\n','')
n.cells.append(nb.v4.new_code_cell(plot))
nb.validate(n);nb.write(n,out.parent/'14_继续增加规则.ipynb')
s=Path('outputs/all_variables_plotly_2016_14/execute.py').read_text().replace('MLP14_PX_OUT','MLP14_MULTI_OUT').replace('all_variables_plotly_2016_14','multi_threshold_2016_14');(out/'execute.py').write_text(s)
