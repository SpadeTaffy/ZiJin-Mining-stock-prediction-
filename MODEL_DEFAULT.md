# 默认模型

用户明确指定：以后默认使用最新的RSI＋MACD（DIFF、DEA）＋多条件触发模型，半衰期60。预测使用 results/rsi_macd_annual_14/predictions.csv 的 Both 列；更新日使用 results/multi_threshold_annual_14/updates.csv 的 Features_and_triggers；原因使用各年 expanded_signals.csv。不得默认退回仅金铜3%的旧模型。
