#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 50: 缺失值处理 — 从统计问题到建模决策
简单填充 · 缺失指示器 · 迭代插补 — 3策略对比
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 生成完整数据 + 制造缺失
X_full, y = make_classification(n_samples=800, n_features=8, n_informative=5, random_state=42)
feature_names = [f'f{i}' for i in range(8)]

# 制造缺失 (MAR: Missing At Random — 与另一特征相关)
X_missing = X_full.copy()
# f0的缺失依赖 f1的值
mask0 = X_full[:, 1] > np.percentile(X_full[:, 1], 60)
X_missing[mask0, 0] = np.nan
# f2的缺失依赖 y (informative missing!)
mask2 = (y == 1) & (np.random.random(len(y)) < 0.6)
X_missing[mask2, 2] = np.nan
# f5随机缺失
mask5 = np.random.random(len(y)) < 0.3
X_missing[mask5, 5] = np.nan

missing_pcts = np.isnan(X_missing).mean(axis=0)
print("=" * 60)
print("  缺失值处理: 3种策略对比")
print("=" * 60)
for i, name in enumerate(feature_names):
    print(f"  {name}: {missing_pcts[i]*100:.0f}% missing", end='')
    if i == 2:
        print(" (informative — related to target!)")
    elif i == 0:
        print(" (related to f1)")
    else:
        print()

X_tr, X_te, y_tr, y_te = train_test_split(X_missing, y, test_size=0.3, random_state=42)

# ============================================================
# Strategy A: Simple Imputer (均值/中位数)
# ============================================================
imp_simple = SimpleImputer(strategy='median')
X_tr_simple = imp_simple.fit_transform(X_tr)
X_te_simple = imp_simple.transform(X_te)

# ============================================================
# Strategy B: Simple Imputer + Missing Indicator
# ============================================================
# 手动添加缺失指示器
def add_missing_indicators(X_original, X_imputed):
    """为每个有缺失的列添加二值指示器"""
    indicators = np.isnan(X_original).astype(float)
    # 只保留有缺失的列
    missing_cols = np.where(indicators.sum(axis=0) > 0)[0]
    return np.hstack([X_imputed, indicators[:, missing_cols]])

X_tr_simple_ind = add_missing_indicators(X_tr, X_tr_simple)
X_te_simple_ind = add_missing_indicators(X_te, X_te_simple)

# ============================================================
# Strategy C: IterativeImputer (MICE)
# ============================================================
imp_iter = IterativeImputer(max_iter=10, random_state=42)
X_tr_iter = imp_iter.fit_transform(X_tr)
X_te_iter = imp_iter.transform(X_te)

# ============================================================
# 对比
# ============================================================
model = LogisticRegression(max_iter=2000, random_state=42)
results = {}

for name, Xtr, Xte in [
    ('A. Simple Fill (median)', X_tr_simple, X_te_simple),
    ('B. Fill + Missing Indicator', X_tr_simple_ind, X_te_simple_ind),
    ('C. IterativeImputer (MICE)', X_tr_iter, X_te_iter),
]:
    cv = cross_val_score(model, Xtr, y_tr, cv=5, scoring='roc_auc')
    model.fit(Xtr, y_tr)
    test_auc = roc_auc_score(y_te, model.predict_proba(Xte)[:, 1])
    results[name] = {'cv_mean': cv.mean(), 'cv_std': cv.std(), 'test_auc': test_auc}
    print(f"\n  {name}:")
    print(f"    CV AUC = {cv.mean():.4f} ± {cv.std():.4f}, Test AUC = {test_auc:.4f}")

# ============================================================
# 缺失指示器的重要性验证
# ============================================================
best_name = max(results, key=lambda k: results[k]['test_auc'])
print(f"\n  ✅ Best: {best_name} (Test AUC={results[best_name]['test_auc']:.4f})")

# 为什么缺失指示器有帮助？
print(f"""
  为什么缺失指示器有帮助？

  1. Informative Missing: 缺失本身包含预测信息
     例: '收入'缺失 → 可能是自由职业者/不愿意透露
     例: f2的缺失与target相关 → 缺失指示器直接编码了这部分信息

  2. 简单填充的问题:
     - 均值填充 → 丢失了'缺失'本身的信息
     - 所有缺失值变成同一个值 → 方差被低估

  3. IterativeImputer (MICE) 的优势:
     - 用其他特征建模预测缺失值
     - 保留变量间的协方差结构
     - 多次迭代收敛到更准确的值

  4. 面试回答:
     Q: "你怎么处理缺失值?"
     A: "首先分析缺失模式——是MCAR/MAR/MNAR?
         然后对比简单填充+缺失指示器 vs 迭代插补。
         如果缺失是informative的(与目标相关),
         缺失指示器往往能带来显著提升。"
""")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 缺失模式
ax = axes[0]
missing_matrix = np.isnan(X_missing[:100, :])
ax.imshow(missing_matrix, aspect='auto', cmap='Reds', interpolation='none')
ax.set_yticks(range(0, 100, 20))
ax.set_xlabel('Feature')
ax.set_ylabel('Sample (first 100)')
ax.set_title('Missing Value Pattern\n(Red = missing)', fontweight='bold')
for i, name in enumerate(feature_names):
    if missing_pcts[i] > 0:
        ax.text(i, 95, f'{missing_pcts[i]*100:.0f}%', ha='center',
                fontsize=7, fontweight='bold', color='black')

# 策略对比
ax = axes[1]
names_list = list(results.keys())
short_names = ['Simple Fill', 'Fill+Indicator', 'Iterative (MICE)']
aucs = [results[n]['test_auc'] for n in names_list]
colors_bar = ['#9E9E9E', '#FF9800', '#4CAF50']
bars = ax.bar(short_names, aucs, color=colors_bar, edgecolor='white')
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.01,
            f'{auc:.4f}', ha='center', fontweight='bold', fontsize=12, color='white')
ax.set_ylabel('Test AUC')
ax.set_title('Missing Value Strategy Comparison', fontweight='bold')

# 缺失指示器特征重要性
ax = axes[2]
# 训练模型看哪些缺失指示器重要
model_full = LogisticRegression(max_iter=2000, random_state=42)
model_full.fit(X_tr_simple_ind, y_tr)
n_original = X_tr.shape[1]
n_indicator = X_tr_simple_ind.shape[1] - n_original
indicator_importances = np.abs(model_full.coef_[0][-n_indicator:])
indicator_names = [f'f{i}_missing' for i, m in enumerate(missing_pcts) if m > 0][:n_indicator]
if len(indicator_names) > 0:
    colors_ind = ['#f44336' if imp > np.median(indicator_importances) else '#9E9E9E'
                  for imp in indicator_importances]
    ax.barh(range(len(indicator_names))[::-1], indicator_importances[::-1],
            color=colors_ind[::-1], edgecolor='white')
    ax.set_yticks(range(len(indicator_names))[::-1])
    ax.set_yticklabels(indicator_names[::-1], fontsize=9)
    ax.set_xlabel('|Coefficient|')
    ax.set_title('Missing Indicator Importance\n'
                 f'(Informative patterns detected)',
                 fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'missing_value_handling.png'), dpi=150)
plt.close()
print("  Saved: missing_value_handling.png")

print("\n✅ Day 50 完成")
