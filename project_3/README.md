# 项目 3：实验设计与分析报告

> **数据科学训练 · Phase 3 收尾项目** | 2026-06-24

---

## 📋 项目概述

从零设计并执行一个完整的 A/B 测试——涵盖样本量计算、随机化方案、统计分析、效应量和业务建议。展示统计严谨性和实验设计能力——这是大厂 DS 岗位最看重的技能之一。

## 🗂 项目结构

```
project_3/
├── README.md                    ← 你在这里
├── experiment_report.md         ← A/B测试完整实验报告
├── ab_test_simulation.py        ← A/B测试全流程模拟 (8步骤)
├── descriptive_statistics.py    ← 5种概率分布 + 正态性检验
├── clt_simulation.py            ← 中心极限定理模拟验证
├── confidence_intervals.py      ← 置信区间 (手动/Bootstrap/理论)
├── hypothesis_testing.py        ← t检验 + 卡方检验 + p值反思
├── power_analysis.py            ← 功效分析 + 多重比较 + p-hacking
├── nonparametric_tests.py       ← 非参数检验 + 决策树
├── regression_diagnostics.py    ← 回归诊断 (残差/VIF/Cook's D/DW)
├── anova_analysis.py            ← ANOVA + Tukey HSD 事后比较
├── clt_notes.md                 ← CLT实验总结
├── p_value_reflection.md        ← p=0.049 vs p=0.051 反思
├── power_analysis_notes.md      ← I类/II类错误总结
└── visuals/                     ← 所有分析图表
```

## 🔬 统计方法全景

| 方法 | 文件 | 场景 |
|------|------|------|
| 描述统计 | `descriptive_statistics.py` | 5分布模拟 + Q-Q图 + Shapiro-Wilk |
| 中心极限定理 | `clt_simulation.py` | CLT验证, SE=σ/√n |
| 置信区间 | `confidence_intervals.py` | 手动/理论/Bootstrap三种方法 |
| t检验 + 卡方 | `hypothesis_testing.py` | 4种检验 + H0/H1/p值解读 |
| 统计功效 | `power_analysis.py` | 功效曲线, 多重比较, p-hacking |
| 非参数检验 | `nonparametric_tests.py` | MW-U, Kruskal-Wallis, Wilcoxon |
| 回归诊断 | `regression_diagnostics.py` | 残差/VIF/Cook's D/DW/BP |
| ANOVA | `anova_analysis.py` | 单因素ANOVA + Tukey HSD + eta^2 |
| **A/B全流程** | `ab_test_simulation.py` | **8步骤完整模拟** |

## 📊 A/B 测试关键结论

| 指标 | 对照组 | 实验组 | 提升 | p值 |
|------|--------|--------|------|-----|
| 转化率 | 12.18% | 13.62% | +1.44pp | 0.032 |
| AOV | ¥14.85 | ¥16.61 | +¥1.75 | 0.124 |

**决策**: ✅ 建议全量上线新版页面

## 🛠 技术栈

- **统计**: scipy.stats, statsmodels
- **可视化**: Matplotlib (Q-Q图, 功效曲线, 箱线图, 小提琴图)
- **数据处理**: NumPy, Pandas

## 🚀 快速复现

```bash
# 完整A/B测试模拟
python ab_test_simulation.py

# 功效分析
python power_analysis.py

# 非参数检验决策树
python nonparametric_tests.py
```

---

*项目 3 是 68 天数据科学训练中 Statistics & A/B Testing Phase (Days 25-34) 的最终产出。*
