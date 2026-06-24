#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 43: XGBoost — 工业界的王者
Boosting · early_stopping · 特征重要性 · vs RF/LR
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 数据
X, y = make_classification(n_samples=800, n_features=20, n_informative=8,
                            n_redundant=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
# 分出验证集用于 early stopping
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print("=" * 60)
print("  XGBoost: 工业界分类模型王者")
print("=" * 60)
print(f"  Train: {len(X_train_fit)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ============================================================
# 1 & 3. XGBoost + early_stopping
# ============================================================
xgb = XGBClassifier(
    n_estimators=500,        # 最大树数 (early stopping会提前停止)
    learning_rate=0.1,       # 学习率 (eta)
    max_depth=5,             # 树深度
    subsample=0.8,           # 行采样比例
    colsample_bytree=0.8,    # 列采样比例
    reg_alpha=0.1,           # L1正则化
    reg_lambda=1.0,          # L2正则化
    eval_metric='logloss',
    early_stopping_rounds=20, # 20轮不提升就停止
    random_state=42,
)

xgb.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_train_fit, y_train_fit), (X_val, y_val)],
    verbose=False,
)

best_iteration = xgb.best_iteration
best_score = xgb.best_score

print(f"\n  Best iteration: {best_iteration} (of 500 max)")
print(f"  Best val logloss: {best_score:.4f}")
print(f"  Early stopping saved {(500 - best_iteration) * (1/0.1):.0f} unnecessary trees")

# ============================================================
# 4. 训练/验证损失曲线
# ============================================================
results = xgb.evals_result()
train_loss = results['validation_0']['logloss']
val_loss = results['validation_1']['logloss']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(train_loss, alpha=0.5, color='#2196F3', linewidth=1, label='Train')
ax.plot(val_loss, color='#FF9800', linewidth=2, label='Validation')
ax.axvline(best_iteration, color='red', linestyle='--', alpha=0.7,
           label=f'Best iter={best_iteration}')
ax.set_xlabel('Iteration'); ax.set_ylabel('Log Loss')
ax.set_title('XGBoost Training Curves (Early Stopping)', fontweight='bold')
ax.legend()

# ============================================================
# 3. learning_rate vs n_estimators trade-off
# ============================================================
ax = axes[0, 1]
learning_rates = [0.01, 0.05, 0.1, 0.3]
for lr in learning_rates:
    tmp = XGBClassifier(n_estimators=500, learning_rate=lr, max_depth=4,
                         early_stopping_rounds=20, random_state=42)
    tmp.fit(X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)], verbose=False)
    # best_iteration only available when early_stopping used
    best = getattr(tmp, 'best_iteration', 500)
    ax.plot(tmp.evals_result()['validation_0']['logloss'],
            alpha=0.7, linewidth=1.5, label=f'lr={lr} (best={best})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Val Log Loss')
ax.set_title('Learning Rate Effect\n(Low lr = slow but potentially better)',
             fontweight='bold')
ax.legend(fontsize=8)

# ============================================================
# 6. 特征重要性 (3种度量)
# ============================================================
ax = axes[1, 0]
# gain 是最常用的
importance_types = ['weight', 'gain', 'cover']
importances = {}
for imp_type in importance_types:
    score = xgb.get_booster().get_score(importance_type=imp_type)
    importances[imp_type] = score

# 用 gain 排序
gain_scores = importances['gain']
gain_sorted = sorted(gain_scores.items(), key=lambda x: x[1], reverse=True)[:10]
features_top = [f'f{k[1:]}' for k, v in gain_sorted]
gain_vals = [v for k, v in gain_sorted]
ax.barh(features_top[::-1], gain_vals[::-1], color='#4CAF50', edgecolor='white')
ax.set_xlabel('Gain'); ax.set_title('Feature Importance (Gain)', fontweight='bold')

print(f"\n  Importance types comparison:")
print(f"  {'Feature':8s} | {'Weight':8s} | {'Gain':8s} | {'Cover':8s}")
print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for f in ['f7', 'f6', 'f0', 'f5', 'f9']:
    w = importances['weight'].get(f, 0)
    g = importances['gain'].get(f, 0)
    c = importances['cover'].get(f, 0)
    print(f"  {f:8s} | {w:8.1f} | {g:8.1f} | {c:8.1f}")

print(f"\n  Weight: 该特征被用于分裂的次数")
print(f"  Gain:   该特征带来的平均损失减少 (最重要!)")
print(f"  Cover:  该特征覆盖的样本数")

# ============================================================
# 7. XGBoost vs Random Forest vs Logistic Regression
# ============================================================
ax = axes[1, 1]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Logistic Reg': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                              random_state=42, verbosity=0),
}

names, test_aucs, cv_means, cv_stds = [], [], [], []
for name, model in models.items():
    if name == 'Logistic Reg':
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc')
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    test_auc = roc_auc_score(y_test, y_prob)
    names.append(name)
    test_aucs.append(test_auc)
    cv_means.append(cv.mean())
    cv_stds.append(cv.std())
    print(f"  {name:15s}: Test AUC={test_auc:.4f}, CV AUC={cv.mean():.4f} ± {cv.std():.4f}")

x = np.arange(len(names))
w = 0.3
ax.bar(x - w/2, test_aucs, w, label='Test AUC', color='#FF9800', edgecolor='white')
ax.bar(x + w/2, cv_means, w, label='CV AUC', color='#2196F3', edgecolor='white')
ax.errorbar(x + w/2, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('AUC-ROC')
ax.set_title('Model Comparison: LR vs RF vs XGBoost', fontweight='bold')
ax.legend(fontsize=8)
for i, v in enumerate(test_aucs):
    ax.text(i - w/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'xgboost_analysis.png'), dpi=150)
plt.close()
print("  Saved: xgboost_analysis.png")

# ============================================================
# 面试知识点
# ============================================================
print(f"""
  XGBoost 面试核心知识点:

  1. XGBoost vs GBDT:
     - GBDT只用一阶导数, XGBoost用二阶导数(更快更准)
     - XGBoost支持自定义loss函数(泰勒展开)
     - 内置正则化(L1+L2), GBDT没有
     - 支持列采样(类似Random Forest的特性)

  2. 关键参数:
     - learning_rate (eta): 每棵树的贡献权重, 小=慢但好
     - n_estimators: 树数量, 配合early_stopping
     - max_depth: 树深度, 通常3-6 (比RF浅!)
     - subsample: 行采样, 防止过拟合
     - colsample_bytree: 列采样
     - reg_alpha/reg_lambda: L1/L2正则化

  3. early_stopping: 验证集loss/N轮不降→自动停止
     → 防止过拟合 + 自动确定最优n_estimators

  4. 为什么XGBoost比RF好?
     - Boosting串行纠错 vs Bagging并行平均
     - 梯度提升每次都拟合上一轮的残差
     - RF降低方差, XGBoost降低偏差
     - 实践中XGBoost通常AUC更高 (但也更容易过拟合)
""")

print("✅ Day 43 完成")
