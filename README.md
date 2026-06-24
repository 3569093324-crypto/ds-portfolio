# Data Science Portfolio — 68天系统训练

**北京理工大学珠海学院 · 数据科学与大数据技术**

68天全勤完成6阶段数据科学训练，产出4个完整项目，覆盖 SQL → Python → 统计 → 机器学习 → 工程化 → 面试模拟全链路。

---

## 📊 项目概览

| # | 项目 | 核心技术 | 亮点 |
|---|------|---------|------|
| 1 | **电商商业分析** | SQL (CTE/窗口函数/JOIN/子查询/索引优化), Matplotlib | 8组SQL查询库, 6维度分析报告, 15,000+订单明细 |
| 2 | **Streamlit交互看板** | Streamlit, Pandas, pytest | 35个单元测试, 4维筛选, 5个KPI卡片, 4种图表 |
| 3 | **A/B测试与统计** | scipy.stats, statsmodels, Bootstrap | 转化率+1.44pp (p=0.032), 功效分析, 9组统计脚本 |
| 4 | **用户复购预测ML** | XGBoost, SHAP, Optuna, FastAPI, Docker | AUC 0.96, ROI 2377x, 5模型对比, 12特征工程 |

---

## 🛠 技术栈

```text
SQL: CTE | 窗口函数 | JOIN | 子查询 | 索引优化 | EXPLAIN QUERY PLAN
Python: Pandas | NumPy | Matplotlib | Seaborn | Streamlit | FastAPI
统计: t检验 | ANOVA | Bootstrap | 功效分析 | 非参数检验 | CLT
ML: scikit-learn | XGBoost | SHAP | Optuna | 特征工程 | 模型调优
工程: Git (Conventional Commits) | Docker | pytest | Type Hints | logging
```

---

## 📁 仓库结构

```
├── project_1/              # SQL商业数据分析
│   ├── queries/            # 8组SQL查询 (CTE/窗口/JOIN/子查询)
│   ├── visuals/            # 6张业务可视化图表
│   └── analysis.md         # 6维度分析报告
├── project_2/              # Streamlit交互式数据看板
│   ├── app.py              # 看板主程序
│   ├── tests/              # 35个pytest测试用例
│   └── data_cleaning.py    # 数据清洗管道 (Type Hints + docstring)
├── project_3/              # A/B测试实验设计
│   ├── ab_test_simulation.py  # 完整A/B测试流程
│   ├── experiment_report.md   # 8步骤实验报告
│   └── visuals/            # 11张统计分析图表
├── project_4/              # 用户复购预测ML管道 (旗舰项目)
│   ├── model_comparison.py    # 5模型系统对比 (LR/DT/RF/XGBoost/MLP)
│   ├── feature_engineering.py # 7种方法12特征工程
│   ├── model_interpretability.py # SHAP可解释性分析
│   ├── api_app.py             # FastAPI模型部署
│   ├── Dockerfile             # Docker容器化
│   └── visuals/               # 18张ML分析图表
├── ml_interview_qa.md          # ML面试20问
├── star_interview_stories.md   # STAR面试故事库
├── business_case_frameworks.md # 业务案例分析框架
├── sql_oa_simulation.md        # SQL在线笔试模拟
├── python_oa_simulation.py     # Python在线笔试模拟
└── requirements.txt            # 完整Python依赖
```

---

## 🎯 关键量化成果

- **68天** 系统训练100%全勤完成
- **4个** 完整端到端项目
- **5个** ML模型系统对比 (LR/RF/XGBoost/MLP)
- **AUC 0.96** 最佳模型性能
- **35个** pytest单元测试
- **1.44pp** 转化率提升 (p=0.032)
- **ROI 2,377x** 预估业务回报

---

*训练完成日期: 2026年6月24日*
