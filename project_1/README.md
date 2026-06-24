# 项目 1：电商商业数据分析报告

> **数据科学训练 · Phase 1 收尾项目** | 2026-06-24

---

## 📋 项目概述

基于自建的电商数据库（150用户 · 122商品 · 200订单 · 15,000+订单明细），完成6个业务维度的数据分析，输出可视化报告和可复用的SQL查询库。

## 🗂 项目结构

```
project_1/
├── README.md                    ← 你在这里
├── analysis.md                  ← 6个业务问题的完整分析报告
├── optimization_summary.md      ← SQL查询优化总结
├── generate_charts.py           ← 图表生成脚本
├── queries/                     ← SQL查询库（13天练习）
│   ├── day03_basic_queries.sql
│   ├── day04_aggregate_queries.sql
│   ├── day05_join_queries.sql
│   ├── day06_subqueries.sql
│   ├── day07_window_functions_1.sql
│   ├── day08_window_functions_2.sql
│   ├── day09_cte_queries.sql
│   └── day10_case_when.sql
├── visuals/                     ← 分析图表
│   ├── chart1_high_value_users.png
│   ├── chart2_category_pareto.png
│   ├── chart3_monthly_trend.png
│   ├── chart4_repurchase_rate.png
│   ├── chart5_churn_risk.png
│   └── chart6_category_preference.png
├── output/                      ← 导出的数据文件
│   ├── products.csv
│   └── products.parquet
├── day13_pandas_basics.py       ← SQL ↔ Pandas 对比练习
└── sql_optimization.py          ← EXPLAIN + 索引优化实验
```

## 🗃 数据来源

自建 SQLite 数据库 `business.db`（电商场景），包含4张表：

| 表 | 行数 | 说明 |
|----|------|------|
| `customers` | 150 | 用户信息（姓名、城市、注册日期、等级） |
| `products` | 122 | 商品信息（名称、品类、价格、成本） |
| `orders` | 200 | 订单信息（用户、日期、总金额） |
| `order_items` | 15,000+ | 订单明细（商品、数量、单价） |

数据使用 Python Faker 生成，模拟真实电商业务场景。

## 📊 分析方法

| 维度 | 分析内容 | SQL技术 | 图表类型 |
|------|---------|---------|----------|
| 用户分析 | 高价值用户特征 | CTE + 窗口函数 + 聚合 | 柱状图 |
| 商品分析 | 品类收入 + 长尾效应 | GROUP BY + 累计占比 | Pareto图 |
| 趋势分析 | 月度销售趋势 + 环比 | 日期函数 + LAG | 折线+柱状 |
| 复购分析 | Segment复购率对比 | CASE WHEN + JOIN | 分组柱状图 |
| 流失分析 | 用户流失风险分层 | 日期计算 + 条件分类 | 饼图 |
| 交叉分析 | 高/低价值用户品类偏好 | 多CTE + pivot | 对比柱状图 |

## 🔑 关键发现

1. **头部集中**：Electronics品类贡献53.9%的销售额，前2品类合计占比71.3%
2. **复购差异**：Retail和VIP用户复购率约61%，显著高于New用户（42%）
3. **流失风险**：70%+的用户超过90天未下单，需要建立召回机制
4. **品类偏好**：高价值用户更偏好高价Electronics，普通用户在低价品类消费占比更高

## 🛠 技术栈

- **数据库**: SQLite 3.45
- **数据处理**: Python 3 + Pandas
- **可视化**: Matplotlib
- **SQL技术**: JOIN, 子查询, 窗口函数(ROW_NUMBER/RANK/LAG/LEAD), CTE, CASE WHEN, 索引优化

## 📈 查询性能

所有业务查询在15,000行数据下执行时间 < 2ms，通过5个索引（含1个复合索引）实现了全索引覆盖。

详见：[optimization_summary.md](optimization_summary.md)

## 🚀 快速开始

```bash
# 浏览数据库
sqlite3 business.db

# 运行分析查询
sqlite3 business.db < queries/day03_basic_queries.sql

# 生成图表
python generate_charts.py

# 查看报告
open analysis.md
```

---

*项目1是68天数据科学训练中SQL Phase (Days 2-14) 的最终产出。*
