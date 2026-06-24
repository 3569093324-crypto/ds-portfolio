#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 45: 不平衡分类处理
SMOTE · class_weight · 阈值调整 · PR-AUC评估
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, classification_report)
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 制造不平衡数据 (1:50)
# ============================================================
X, y = make_classification(n_samples=5000, n_features=10, n_informative=4,
                            weights=[0.98, 0.02], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("=" * 60)
print("  不平衡分类处理: 5种策略对比")
print("=" * 60)
print(f"  Train: {len(y_train)} (pos rate={y_train.mean():.1%})")
print(f"  Test:  {len(y_test)} (pos rate={y_test.mean():.1%})")
print(f"  不平衡比: {sum(y_train==0)/sum(y_train==1):.0f}:1")

# ============================================================
# Baseline: 不平衡+默认参数
# ============================================================
lr_base = LogisticRegression(max_iter=2000, random_state=42)
lr_base.fit(X_train, y_train)
y_prob_base = lr_base.predict_proba(X_test)[:, 1]
pr_auc_base = average_precision_score(y_test, y_prob_base)

print(f"\n  Strategy 0 — Baseline (no treatment):")
print(f"    PR-AUC = {pr_auc_base:.4f}, ROC-AUC = {roc_auc_score(y_test, y_prob_base):.4f}")
print(f"    Predictions: pos={(lr_base.predict(X_test)==1).sum()}/{len(y_test)}")

# ============================================================
# 3种策略 + 对比
# ============================================================
strategies = {}

# Strategy 1: class_weight='balanced'
lr_cw = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
lr_cw.fit(X_train, y_train)
strategies['1. Class Weight'] = lr_cw.predict_proba(X_test)[:, 1]

# Strategy 2: SMOTE (过采样)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
lr_smote = LogisticRegression(max_iter=2000, random_state=42)
lr_smote.fit(X_smote, y_smote)
strategies['2. SMOTE (Oversampling)'] = lr_smote.predict_proba(X_test)[:, 1]

# Strategy 3: RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)
lr_rus = LogisticRegression(max_iter=2000, random_state=42)
lr_rus.fit(X_rus, y_rus)
strategies['3. RandomUnderSampler'] = lr_rus.predict_proba(X_test)[:, 1]

# Strategy 4: NearMiss
nm = NearMiss(version=1)
X_nm, y_nm = nm.fit_resample(X_train, y_train)
lr_nm = LogisticRegression(max_iter=2000, random_state=42)
lr_nm.fit(X_nm, y_nm)
strategies['4. NearMiss (Undersampling)'] = lr_nm.predict_proba(X_test)[:, 1]

# Strategy 5: 阈值调整 (基于Baseline模型，找最优阈值)
prec, rec, thresholds = precision_recall_curve(y_test, y_prob_base)
f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-10)
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred_tuned = (y_prob_base >= best_thresh).astype(int)
strategies['5. Threshold Tuning'] = y_prob_base  # 同样概率, 用不同阈值

# ============================================================
# 对比结果
# ============================================================
print(f"\n  {'Strategy':30s} | {'PR-AUC':8s} | {'ROC-AUC':8s} | {'Best Thresh':12s}")
print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")
for name, y_prob in strategies.items():
    pr = average_precision_score(y_test, y_prob)
    roc = roc_auc_score(y_test, y_prob)
    if 'Threshold' in name:
        print(f"  {name:30s} | {pr:.4f}   | {roc:.4f}   | {best_thresh:.4f}")
    else:
        print(f"  {name:30s} | {pr:.4f}   | {roc:.4f}   | 0.5 (default)")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 数据分布
ax = axes[0, 0]
ax.bar(['Negative (0)', 'Positive (1)'],
       [sum(y_train==0), sum(y_train==1)],
       color=['#2196F3', '#f44336'], edgecolor='white')
ax.set_title(f'Imbalanced Training Data\nRatio = {sum(y_train==0)/sum(y_train==1):.0f}:1',
             fontweight='bold')
for i, v in enumerate([sum(y_train==0), sum(y_train==1)]):
    ax.text(i, v + 100, str(v), ha='center', fontweight='bold', fontsize=14)

# (2) PR曲线对比
ax = axes[0, 1]
for name, y_prob in strategies.items():
    p, r, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(r, p, linewidth=1.5, alpha=0.8, label=f'{name}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves\n(PR-AUC matters for imbalanced data!)',
             fontweight='bold')
ax.legend(fontsize=7)

# (3) PR-AUC 柱状图对比
ax = axes[1, 0]
names_list = list(strategies.keys())
pr_aucs = [average_precision_score(y_test, p) for p in strategies.values()]
colors_pr = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names_list)))
bars = ax.barh(range(len(names_list))[::-1], pr_aucs[::-1],
               color=colors_pr, edgecolor='white')
ax.set_yticks(range(len(names_list))[::-1])
ax.set_yticklabels([n[:25] for n in names_list][::-1], fontsize=8)
ax.set_xlabel('PR-AUC')
ax.set_title('PR-AUC Comparison\n(Higher = Better)', fontweight='bold')
ax.axvline(pr_auc_base, color='gray', linestyle='--', alpha=0.5, label='Baseline')
for i, (bar, score) in enumerate(zip(bars, pr_aucs[::-1])):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{score:.4f}', va='center', fontsize=8, fontweight='bold')
ax.legend(fontsize=7)

# (4) SMOTE 原理说明
ax = axes[1, 1]
ax.axis('off')
smote_explanation = """
SMOTE (Synthetic Minority Oversampling Technique)
=================================================

原理: 在特征空间中, 对少数类样本进行插值

算法步骤:
1. 对于每个少数类样本 x_i:
2.   找到它的 k 个最近邻 (同类别)
3.   随机选一个邻居 x_nn
4.   生成新样本: x_new = x_i + λ(x_nn - x_i)
     (λ ~ Uniform(0,1))

关键特点:
✅ 不是简单复制 → 缓解过拟合
✅ 在特征空间插值 → 创建多样性
⚠️ 只在少数类上操作 → 不丢失多数类信息
⚠️ k_neighbors 不能超过少数类样本数

对比总结:
┌────────────────┬──────────┬──────────┐
│ 方法           │ 优点     │ 缺点     │
├────────────────┼──────────┼──────────┤
│ Class Weight   │ 最简单   │ 不改分布 │
│ SMOTE          │ 增多样性 │ 可能噪声 │
│ RandomUnder    │ 速度快   │ 丢信息   │
│ NearMiss       │ 保留边界 │ 计算量大 │
│ Threshold Tune │ 不改模型 │ 需验证集 │
└────────────────┴──────────┴──────────┘

推荐: 先试 class_weight + SMOTE
评估: 用 PR-AUC (不用 Accuracy!)
"""
ax.text(0.05, 0.95, smote_explanation, transform=ax.transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'imbalanced_learning.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: imbalanced_learning.png")

print("\n✅ Day 45 完成")
