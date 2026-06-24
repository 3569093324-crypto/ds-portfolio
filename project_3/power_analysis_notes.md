
I类错误 (False Positive) — 拒真
  H0为真时, 错误地拒绝H0。
  概率 = alpha = 0.05 (单次检验)
  多重比较时: P(至少1个假阳性) = 1 - (1-alpha)^k ≈ 64% (k=20次)

II类错误 (False Negative) — 取伪
  H1为真时, 错误地接受H0。
  概率 = beta
  统计功效 = 1 - beta (正确检测到真实效应的概率)

效应量越小 → 所需样本量越大 → 小样本时功效不足

p-hacking 常见方式 (都应该避免!):
  ❌ 边收集数据边检验, p<0.05就停止 (sequential testing)
  ❌ 跑20个指标, 只报告显著的那个 (cherry-picking)
  ❌ 尝试不同的分析方法直到找到显著结果 (fishing expedition)
  ❌ 剔除'异常值'直到p<0.05
  ❌ 改变假设方向(单尾→双尾→单尾)直到显著

正确做法:
  ✅ 实验前确定样本量 (power analysis)
  ✅ 预注册分析计划 (preregistration)
  ✅ 报告所有测试过的指标 (transparency)
  ✅ 使用多重比较校正 (Bonferroni / FDR)
  ✅ 报告效应量和置信区间, 不仅仅报告p值
