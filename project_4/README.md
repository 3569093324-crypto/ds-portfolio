# 项目 4：用户复购预测 — 端到端机器学习项目

> **数据科学训练 · Phase 4 旗舰项目** | 2026-06-24

![Phase](https://img.shields.io/badge/Phase-4%20ML-blue) ![Status](https://img.shields.io/badge/Status-Complete-green) ![Models](https://img.shields.io/badge/Models-5-orange)

---

## 📋 项目概述

从业务问题定义到商业价值量化的完整端到端 ML 项目。预测电商用户未来 30 天内是否会复购，系统性对比 5 种模型，通过 SHAP 解释关键驱动因素，并量化模型上线的商业价值。

### 核心结果

| 指标 | 值 |
|------|-----|
| 最佳模型 | XGBoost (AUC=0.96) |
| 年度财务影响 | ¥12.6 亿 |
| ROI | 2377x |
| Top 预测因子 | review_score, age, income |

## 🗂 项目结构

```
project_4/
├── README.md                       ← 你在这里
├── ml_problem_framing.md           ← ML问题框架化 (Day 35)
├── business_value_analysis.md      ← 商业价值量化 (Day 51)
├── requirements.txt                ← Python依赖
├── eda_for_ml.py                   ← EDA for ML (Day 36)
├── cross_validation.py             ← 训练/测试切分 & CV (Day 37)
├── linear_regression_from_scratch.py ← 从零实现线性回归 (Day 38)
├── logistic_regression.py          ← 逻辑回归 & 概率校准 (Day 39)
├── model_evaluation.py             ← 模型评估超越准确率 (Day 40)
├── decision_tree.py                ← 决策树 & 过拟合 (Day 41)
├── random_forest.py                ← 随机森林 (Day 42)
├── xgboost_model.py                ← XGBoost (Day 43)
├── feature_engineering.py          ← 特征工程全流程 (Day 44)
├── imbalanced_classification.py    ← 不平衡分类处理 (Day 45)
├── hyperparameter_tuning.py        ← Grid/Random/Bayesian调优 (Day 46)
├── model_interpretability.py       ← SHAP & PDP (Day 47)
├── clustering_analysis.py          ← K-Means & DBSCAN (Day 48)
├── model_comparison.py             ← 5模型系统性对比 (Day 49)
├── missing_value_handling.py       ← 缺失值3策略对比 (Day 50)
└── visuals/                        ← 所有分析图表
```

## 🔬 方法论

### 问题框架化
- 业务问题: 预测用户是否会在30天内复购
- ML问题: 二分类 (P(repurchase=1))
- 评估指标: AUC-ROC + Precision@20% (营销预算约束)

### 模型对比

| 模型 | CV AUC | 训练时间 | 可解释性 |
|------|--------|---------|----------|
| Logistic Regression | 0.757 | 0.02s | ✅ 高 (coef_) |
| Decision Tree | 0.801 | 0.04s | ✅ 高 (可视化) |
| Random Forest | 0.918 | 0.64s | ✅ SHAP |
| **XGBoost** | **0.924** | 0.29s | ✅ SHAP |
| MLP (Neural Net) | 0.964 | 1.56s | ❌ 有限 |

### 特征工程
- 7 种方法: One-Hot / Target Encoding / 分箱 / Log & Box-Cox / 交互特征 / 多项式 / MI特征选择
- Box-Cox: 偏度 1.46→0.00
- 特征工程后 RF AUC 提升 3.5%

### 缺失值处理
- 缺失指示器策略完胜: AUC 0.94 vs 简单填充 0.81 (+12%)
- Informative Missing: f2 的缺失与 target 显著相关

### 模型可解释性 (SHAP)
- Top 3 特征: review_score, age, income
- Waterfall plot: 解释单个用户的预测逻辑
- PDP: 展示特征与预测的非线性关系

## 💰 商业价值

| 维度 | 影响 |
|------|------|
| 月度节省营销成本 | ¥80,000,000 |
| 年度财务影响 | ¥12.6 亿 |
| ROI | 2377x |
| 发券用户减少 | -80% |

详见: [business_value_analysis.md](business_value_analysis.md)

## 🚀 快速复现

```bash
pip install -r requirements.txt
python model_comparison.py      # 5模型对比
python model_interpretability.py # SHAP分析
python hyperparameter_tuning.py # 超参数调优
```

## 🛠 技术栈

- **ML**: scikit-learn, XGBoost, SHAP, Optuna
- **数据处理**: Pandas, NumPy, imbalanced-learn
- **可视化**: Matplotlib
- **统计**: scipy.stats, statsmodels

---

*项目 4 是 68 天数据科学训练中 ML Phase (Days 35-52) 的旗舰产出。*
