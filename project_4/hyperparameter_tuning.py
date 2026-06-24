#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 46: 超参数调优 — Grid vs Random vs Bayesian
Optuna贝叶斯优化 + 参数重要性可视化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      train_test_split, cross_val_score)
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
import time
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 数据
X, y = make_classification(n_samples=1000, n_features=15, n_informative=6,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print("=" * 60)
print("  超参数调优: Grid vs Random vs Bayesian (Optuna)")
print("=" * 60)
print(f"  Data: {len(X_train)} train, {len(X_test)} test")

# Baseline
xgb_base = XGBClassifier(random_state=42, verbosity=0)
cv_base = cross_val_score(xgb_base, X_train, y_train, cv=5, scoring='roc_auc')
xgb_base.fit(X_train, y_train)
auc_base = roc_auc_score(y_test, xgb_base.predict_proba(X_test)[:, 1])
print(f"\n  Baseline (default params): CV AUC={cv_base.mean():.4f}, Test AUC={auc_base:.4f}")

# ============================================================
# 1. GridSearchCV (小网格)
# ============================================================
print(f"\n  --- 1. GridSearchCV ---")
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
}
grid = GridSearchCV(
    XGBClassifier(random_state=42, verbosity=0),
    param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
)
t0 = time.time()
grid.fit(X_train, y_train)
t_grid = time.time() - t0

print(f"  Grid: {len(param_grid['max_depth'])*len(param_grid['learning_rate'])*len(param_grid['n_estimators'])} combinations × 3 CV = {grid.n_splits_*len(grid.cv_results_['params'])} fits")
print(f"  Time: {t_grid:.1f}s, Best score: {grid.best_score_:.4f}")
print(f"  Best params: {grid.best_params_}")

# ============================================================
# 2. RandomizedSearchCV
# ============================================================
print(f"\n  --- 2. RandomizedSearchCV ---")
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 9],
    'learning_rate': np.linspace(0.01, 0.3, 10).tolist(),
    'n_estimators': [50, 100, 150, 200, 300],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
}
random_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, verbosity=0),
    param_dist, n_iter=12, cv=3, scoring='roc_auc',
    n_jobs=-1, random_state=42, verbose=0
)
t0 = time.time()
random_search.fit(X_train, y_train)
t_random = time.time() - t0

print(f"  Random: 12 iterations × 3 CV = 36 fits")
print(f"  Time: {t_random:.1f}s, Best score: {random_search.best_score_:.4f}")
print(f"  Best params: {random_search.best_params_}")

# ============================================================
# 3. Optuna 贝叶斯优化
# ============================================================
print(f"\n  --- 3. Optuna Bayesian Optimization ---")

def objective(trial):
    """Optuna 目标函数"""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3,
                              scoring='roc_auc', n_jobs=-1)
    return scores.mean()

# 创建 study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)

t0 = time.time()
study.optimize(objective, n_trials=30, show_progress_bar=False)
t_optuna = time.time() - t0

print(f"  Optuna: 30 trials × 3 CV = 90 fits")
print(f"  Time: {t_optuna:.1f}s, Best score: {study.best_value:.4f}")
print(f"  Best params: {study.best_params}")

# ============================================================
# 4. 对比
# ============================================================
print(f"\n  --- 4. 三种方法对比 ---")
optuna_best = XGBClassifier(**{**study.best_params, 'random_state':42, 'verbosity':0})
optuna_best.fit(X_train, y_train)
auc_optuna = roc_auc_score(y_test, optuna_best.predict_proba(X_test)[:, 1])

grid_best = grid.best_estimator_
auc_grid = roc_auc_score(y_test, grid_best.predict_proba(X_test)[:, 1])

rs_best = random_search.best_estimator_
auc_rs = roc_auc_score(y_test, rs_best.predict_proba(X_test)[:, 1])

print(f"  {'Method':15s} | {'Time':8s} | {'Best CV':10s} | {'Test AUC':10s} | {'Params'}")
print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*20}")
print(f"  {'Baseline':15s} | {'-':8s} | {'-':10s} | {auc_base:.4f}      | default")
print(f"  {'GridSearch':15s} | {t_grid:.1f}s     | {grid.best_score_:.4f}     | {auc_grid:.4f}      | {str(grid.best_params_)[:30]}")
print(f"  {'RandomSearch':15s} | {t_random:.1f}s     | {random_search.best_score_:.4f}     | {auc_rs:.4f}      | {str(random_search.best_params_)[:30]}")
print(f"  {'Optuna':15s} | {t_optuna:.1f}s     | {study.best_value:.4f}     | {auc_optuna:.4f}      | {str(study.best_params)[:30]}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 优化历史
ax = axes[0, 0]
trials = [t.number for t in study.trials]
values = [t.value for t in study.trials]
best_sofar = np.maximum.accumulate(values)
ax.plot(trials, values, 'o', alpha=0.4, markersize=4, color='#2196F3', label='Each trial')
ax.plot(trials, best_sofar, 'r-', linewidth=2, label='Best so far')
ax.set_xlabel('Trial'); ax.set_ylabel('CV AUC')
ax.set_title('Optuna Optimization History', fontweight='bold')
ax.legend()

# (2) 参数重要性
ax = axes[0, 1]
importances = optuna.importance.get_param_importances(study)
params_imp = list(importances.keys())[:7]
values_imp = [importances[p] for p in params_imp]
colors_oi = plt.cm.Blues(np.linspace(0.4, 0.9, len(params_imp)))
ax.barh(range(len(params_imp))[::-1], values_imp[::-1],
        color=colors_oi[::-1], edgecolor='white')
ax.set_yticks(range(len(params_imp))[::-1])
ax.set_yticklabels(params_imp[::-1], fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Optuna: Parameter Importance', fontweight='bold')

# (3) 方法对比柱状图
ax = axes[1, 0]
methods = ['Baseline', 'GridSearch', 'RandomSearch', 'Optuna']
aucs = [auc_base, auc_grid, auc_rs, auc_optuna]
times = [0, t_grid, t_random, t_optuna]
colors_method = ['#9E9E9E', '#2196F3', '#FF9800', '#4CAF50']
bars = ax.bar(methods, aucs, color=colors_method, edgecolor='white')
ax.set_ylabel('Test AUC')
ax.set_title('Hyperparameter Tuning: Method Comparison', fontweight='bold')
for bar, auc, t_val in zip(bars, aucs, times):
    label = f'{auc:.4f}' if t_val == 0 else f'{auc:.4f}\n({t_val:.0f}s)'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.03,
            label, ha='center', fontsize=9, fontweight='bold', color='white')

# (4) 调参建议
ax = axes[1, 1]
ax.axis('off')
tuning_guide = """
XGBoost 调参顺序 (由粗到细)
=============================

Step 1: 固定 learning_rate=0.1
        调 n_estimators + early_stopping
        → 确定最优树数量

Step 2: 调 max_depth + min_child_weight
        → 控制模型复杂度

Step 3: 调 subsample + colsample_bytree
        → 增加随机性, 防止过拟合

Step 4: 调 reg_alpha + reg_lambda
        → L1/L2 正则化强度

Step 5: 降低 learning_rate
        重新调 n_estimators
        → 精细打磨

方法选择:
• GridSearch: 小参数空间时用
• RandomSearch: 参数空间大时用
  (同样时间探索更多组合)
• Optuna/Bayesian: 最高效,
  智能选择下一组参数

面试话术:
"我用Optuna做贝叶斯优化,
30次trial就找到了比GridSearch
更好的参数组合。TPE算法根据
历史结果智能选择下一个采样点。"
"""
ax.text(0.05, 0.95, tuning_guide, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'hyperparameter_tuning.png'), dpi=150)
plt.close()
print("  Saved: hyperparameter_tuning.png")

print("\n✅ Day 46 完成")
