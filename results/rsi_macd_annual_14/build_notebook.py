from pathlib import Path
import hashlib,nbformat as nb
out=Path(__file__).resolve().parent;src=Path('/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/notebooks/04_采用神经网络模型/14_继续增加规则.ipynb')
b=src.read_bytes();(out/'14_before.ipynb').write_bytes(b);(out/'original.sha256').write_text(hashlib.sha256(b).hexdigest());n=nb.reads(b.decode(),4);(out/'start_cell.txt').write_text(str(len(n.cells)))
n.cells.append(nb.v4.new_markdown_cell(r'''# 新实验：基准＋RSI、DIFF/DEA及组合，2015—2026

基准为前节④“新增A500/美债MA20偏离＋多阈值双向触发＋半衰期60”。第一阶段保持冻结并复用已保存模型；四组每次更新的生效日期、训练区间、选中种子完全相同，不重新选种子。新增指标均进入**第二阶段网络的输入**，不是直接接入第二个隐藏层。

| 组别 | 第二阶段输入数 | 在基准9项输入之外增加 |
|---|---:|---|
| 基准 | 9 | 无 |
| ＋RSI | 10 | 自身股价RSI14(t−1) |
| ＋MACD | 11 | 自身股价DIFF(t−1)、DEA(t−1) |
| 三者结合 | 12 | 上述三项 |

“EDA”按常见MACD信号线DEA理解。RSI采用Wilder 14期，前14个价格变化的上涨/下跌均值初始化，再递推为 `(13×上一均值＋当前变化)/14`；RSI=100×平均上涨/(平均上涨＋平均下跌)，两者均为0取50。MACD使用DIFF=EMA12−EMA26，DEA=EMA9(DIFF)，EMA取alpha=2/(周期+1)，从首条价格初始化，递推计算（adjust=False）。不添加MACD柱，因为它由DIFF与DEA决定。

[RSI公式与定义（Fidelity）](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI)；[MACD定义（Fidelity）](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd)。

**避免目标信息泄漏**：所有技术指标在原Dataset完整历史上计算后统一shift(1)，再对齐训练/测试日期；t日指标只使用t前一条原始记录及更早的股价，不使用t日真实股价。这是逐记录已有信息的回测，不是年初一次性知道全年技术指标。已验证：改变当天及未来股价，不改变当天的滞后输入；没有用未来补值填充早期缺失，既有训练起点处所有指标均已有效。

第二阶段保持16→32两隐藏层、ReLU、Adam 0.01、全批量100轮、半衰期60；标准化只拟合生效日前训练集。基准预测直接复用，并逐段恢复旧检查点核对。更新条件不使用RSI/MACD，仍为黄金3%、铜3%/8%、A500 5%/8%、美债10%/20%的绝对MA20偏离双向穿越，同日合并、下条记录生效。

主指标是年度等权平均MAPE，并报告MSE、按记录加权指标。2026仅截至7月23日，完整年份与包含部分2026分别汇总。技术指标加入了自身历史价格信息，因此改善不等价于外部宏观/金铜变量解释能力增强；同一种子在不同输入维度下也不意味着所有初始权重逐元素相同。本轮没有为每组另调参数。
'''))
s=(out/'run.py').read_text().split("if __name__=='__main__':")[0]
s=s.replace("RUN_OUT=Path(os.environ.get('MLP14_RSI_MACD_OUT',str(Path(__file__).resolve().parent)))", "RUN_OUT=Path(os.environ.get('MLP14_RSI_MACD_OUT','/Users/admin/数据分析/ZiJin-Mining/ZiJin-Mining-stock-prediction-/results/rsi_macd_annual_14'));RUN_OUT.mkdir(parents=True,exist_ok=True)")
n.cells.append(nb.v4.new_code_cell(s+"\nprepare()\nfor year in range(2015,2027):run_year(year)\nOUT=RUN_OUT\n"))
s=(out/'summarize.py').read_text().replace('OUT=Path(__file__).resolve().parent','OUT=RUN_OUT');n.cells.append(nb.v4.new_code_cell(s))
nb.validate(n);nb.write(n,out.parent/'14_继续增加规则.ipynb')
s=Path('outputs/multi_threshold_annual_14/execute.py').read_text().replace('MLP14_ANNUAL_MULTI_OUT','MLP14_RSI_MACD_OUT').replace('multi_threshold_annual_14','rsi_macd_annual_14');(out/'execute.py').write_text(s)
