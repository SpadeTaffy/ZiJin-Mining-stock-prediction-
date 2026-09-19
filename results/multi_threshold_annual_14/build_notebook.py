from pathlib import Path
import hashlib,nbformat as nb
out=Path(__file__).resolve().parent
src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/14_继续增加规则.ipynb')
b=src.read_bytes();(out/'14_before.ipynb').write_bytes(b);(out/'original.sha256').write_text(hashlib.sha256(b).hexdigest());n=nb.reads(b.decode(),4);(out/'start_cell.txt').write_text(str(len(n.cells)))
n.cells.append(nb.v4.new_markdown_cell('''# ②—④方案：2015—2026逐年与平均表现（仅柱状图）

保持上一节三组定义和超参数，不增加阈值搜索：

| 编号 | 第二阶段新增A500、美债MA20偏离 | 更新规则 |
|---|---|---|
| ② 只加特征 | 是 | 原金铜±3%双向触发 |
| ③ 只加触发条件 | 否 | 黄金3%；铜3%、8%；A500 5%、8%；美债10%、20% |
| ④ 特征＋触发条件 | 是 | 与③相同 |

阈值均针对绝对MA20偏离；任一阈值双向穿越即触发，同日合并，下一条记录生效。各年重新初始化，年初选择日期为1月1日。两阶段均半衰期60条有效记录、各100轮、16→32隐藏层。基准原特征的等权候选在更新前3个日历月验证MAPE上选种；同一更新日期三组共用选定种子，新增特征组不另外调参。先前日期选择直接复用，新日期按相同候选规则计算。

2016复用上一节已完成结果，其余年份按同一协议扩展。内部保留原金铜3%的复现检查，但本节只展示②—④。没有生成新的折线图。

主指标为**年度MAPE等权平均**；并列全年/逐年MSE，以及合并记录后按样本数加权的指标。2015—2025为11个完整年份；2015—2026含2026截至7月23日的152条记录，单独标记。MSE受价格水平影响，不能单凭不同年份的绝对MSE判断哪年更难预测。

这些阈值在观察2016后提出；跨年实验是历史比较，2016仍不是独立验证集。继续沿用固定种子选择以控制对照，不代表新输入结构已充分调参。
'''))
s=(out/'run.py').read_text().split("if __name__=='__main__':")[0]
s=s.replace('RUN_OUT=Path(__file__).resolve().parent',"RUN_OUT=Path(os.environ.get('MLP14_ANNUAL_MULTI_OUT','/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/results/multi_threshold_annual_14'));RUN_OUT.mkdir(parents=True,exist_ok=True)")
n.cells.append(nb.v4.new_code_cell(s+"\n# 完成的年份直接复用；只有缺失的年份才训练。\nfor year in range(2015,2027):year_run(year)\nOUT=RUN_OUT\n"))
s=(out/'summarize.py').read_text().replace('OUT=Path(__file__).resolve().parent','OUT=RUN_OUT')
n.cells.append(nb.v4.new_code_cell(s))
nb.validate(n);nb.write(n,out.parent/'14_继续增加规则.ipynb')
s=Path('outputs/multi_threshold_2016_14/execute.py').read_text().replace('MLP14_MULTI_OUT','MLP14_ANNUAL_MULTI_OUT').replace('multi_threshold_2016_14','multi_threshold_annual_14');(out/'execute.py').write_text(s)
