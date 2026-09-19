from pathlib import Path
import nbformat,hashlib,io,contextlib
from IPython.core.interactiveshell import InteractiveShell
out=Path(__file__).resolve().parent
root=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-');name='15_分析每一年数据,细化更新条件.ipynb';p=root/'notebooks/04_采用神经网络模型'/name
raw=p.read_bytes();(out/'15_before.ipynb').write_bytes(raw);(out/'original.sha256').write_text(hashlib.sha256(raw).hexdigest());n=nbformat.read(p,as_version=4)
md='''## 2015—2026：新增宏观MA20偏离特征与多条件触发

基准：RSI14＋MACD（DIFF、DEA）＋原多条件触发，读取此前 Both 预测。新方案保留全部基准输入，在第二层添加中国10年期国债利率、布伦特原油、美元指数的MA20相对偏离率。

|触发变量|基准阈值|新方案阈值|
|---|---|---|
|黄金|3%|3%|
|铜|3%、8%|3%、8%|
|A500|5%、8%|5%、8%|
|美国10年期利率|10%、20%|5%、10%、20%|
|中国10年期利率|无|3%|
|布伦特原油|无|4%、10%、15%|
|美元指数|无|1%、2%|

触发依据均为绝对MA20偏离在边界内外双向切换，同日信号合并；下一条有效记录生效。MA20在完整历史上计算，含当条记录；利率偏离为相对百分比。RSI、DIFF、DEA滞后1条记录。两组均半衰期60、各阶段100轮、隐藏层16/32。共享更新日沿用基准种子及第一阶段模型；新增更新日沿用原10候选种子、前三个月验证MAPE选种子的规则，仅使用生效前数据。

2026仅截至7月23日。平均表现分别报告年度等权与记录加权；此次同时改变特征及触发频率，不能将改善单独归因于某一项。完整预测、触发原因、模型文件及训练脚本位于 results/expanded_macro_rsi_macd_15。
'''
code="""from pathlib import Path
import pandas as pd
from IPython.display import display, Image
RESULTS=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/results/expanded_macro_rsi_macd_15')
annual=pd.read_csv(RESULTS/'annual_comparison.csv').rename(columns={'Year':'年份','Baseline':'基准MAPE(%)','Expanded':'新方案MAPE(%)','Difference_pp':'差值(百分点)','Baseline_updates':'基准更新次数','Expanded_updates':'新方案更新次数'})
display(annual.round(4))
display(Image(filename=str(RESULTS/'annual_mape.png')))
display(pd.read_csv(RESULTS/'summary.csv').round(6))
display(Image(filename=str(RESULTS/'average_mape.png')))
"""
cell=nbformat.v4.new_code_cell(code);cell.execution_count=1
shell=InteractiveShell.instance()
def publish(data,metadata=None,**kw):cell.outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=metadata or {}))
shell.display_pub.publish=publish
exec(code.replace(str(root/'results/expanded_macro_rsi_macd_15'),str(out)),{})
n.cells.extend([nbformat.v4.new_markdown_cell(md),cell]);nbformat.validate(n);nbformat.write(n,out.parent/name)
