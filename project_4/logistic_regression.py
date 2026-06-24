#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 39: 逻辑回归 — 决策边界、概率校准、阈值调优
面试高频: sigmoid, log-loss, coef_ 解读
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 生成数据
# ============================================================
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=4,
    n_redundant=2, n_clusters_per_class=2,
    random_state=42
)
feature_names = [f'feature_{i}' for i in range(10)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("=" * 60)
print("  逻辑回归 — 分类问题基线模型")
print("=" * 60)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Features: {X.shape[1]}, Positive rate: {y.mean():.1%}")

# ============================================================
# 1. 训练逻辑回归
# ============================================================
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

# ============================================================
# 2. predict() vs predict_proba()
# ============================================================
print("\n" + "=" * 60)
print("  2. predict() vs predict_proba()")
print("=" * 60)

y_prob = lr.predict_proba(X_test_s)  # shape (n, 2): [P(class0), P(class1)]
y_pred = lr.predict(X_test_s)        # shape (n,): 0 or 1 (threshold=0.5)

print(f"  predict_proba() 返回形状: {y_prob.shape}")
print(f"  predict_proba()[:5]:")
for i in range(5):
    print(f"    [{i}] P(y=0)={y_prob[i,0]:.4f}, P(y=1)={y_prob[i,1]:.4f} → "
          f"predict={y_pred[i]} (threshold=0.5)")
print(f"\n  predict() = (predict_proba()[:,1] >= 0.5).astype(int)")

# ============================================================
# 3. 决策边界 (2D可视化)
# ============================================================
# 取2个最重要的特征
coef_abs = np.abs(lr.coef_[0])
top2_idx = np.argsort(coef_abs)[-2:]
print(f"\n  Top 2 features for decision boundary: {feature_names[top2_idx[0]]}, "
      f"{feature_names[top2_idx[1]]}")

# 在2D子空间训练
X_2d = X[:, top2_idx]
lr_2d = LogisticRegression(max_iter=1000, random_state=42)
lr_2d.fit(X_2d, y)

# 画决策边界
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax = axes[0, 0]
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))
Z = lr_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
ax.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='blue', edgecolors='white',
           s=30, label='Class 0', alpha=0.7)
ax.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='red', edgecolors='white',
           s=30, label='Class 1', alpha=0.7)
# 决策边界线 (P=0.5)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
ax.set_xlabel(feature_names[top2_idx[0]]); ax.set_ylabel(feature_names[top2_idx[1]])
ax.set_title('Decision Boundary (P=0.5 contour)', fontweight='bold')
ax.legend(fontsize=7)

# (2) Sigmoid 函数
ax = axes[0, 1]
z = np.linspace(-8, 8, 200)
sigmoid = 1 / (1 + np.exp(-z))
ax.plot(z, sigmoid, 'b-', linewidth=3)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(z[z>=0], sigmoid[z>=0], 0.5, alpha=0.2, color='red', label='P>0.5 → Class 1')
ax.fill_between(z[z<0], sigmoid[z<0], 0.5, alpha=0.2, color='blue', label='P<0.5 → Class 0')
ax.set_xlabel('z = w·x + b (log-odds)'); ax.set_ylabel('σ(z) = P(y=1)')
ax.set_title('Sigmoid Function: P(y=1) = 1/(1+e^{-z})', fontweight='bold')
ax.legend(fontsize=7)

# (3) Feature Importance (coef_)
ax = axes[0, 2]
coef_sorted = np.argsort(lr.coef_[0])
colors_coef = ['red' if lr.coef_[0][i] < 0 else 'green' for i in coef_sorted]
ax.barh(range(10), lr.coef_[0][coef_sorted], color=colors_coef, edgecolor='white')
ax.set_yticks(range(10))
ax.set_yticklabels([feature_names[i][:12] for i in coef_sorted], fontsize=8)
ax.set_xlabel('Coefficient (log-odds change per unit)')
ax.set_title('Feature Impact on Log-Odds\n(Green=+effect, Red=-effect)', fontweight='bold')

# (4) 概率校准曲线
ax = axes[1, 0]
prob_true, prob_pred = calibration_curve(y_test, y_prob[:, 1], n_bins=10)
ax.plot(prob_pred, prob_true, 'o-', color='#2196F3', linewidth=2, markersize=8,
        label='Logistic Regression')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve (Reliability Diagram)', fontweight='bold')
ax.legend(fontsize=8)

# 校准评估
# Brier score: 越小越好
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, y_prob[:, 1])
ax.text(0.05, 0.95, f'Brier Score = {brier:.4f}\n(0=perfect, 0.25=useless)',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# (5) 阈值调优: Precision-Recall vs Threshold
ax = axes[1, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob[:, 1])
# 注意: thresholds 长度比 precisions/recalls 少1
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

ax.plot(thresholds, precisions[:-1], 'b-', linewidth=2, label='Precision')
ax.plot(thresholds, recalls[:-1], 'g-', linewidth=2, label='Recall')
ax.plot(thresholds, f1_scores, 'r--', linewidth=2, label='F1 Score')
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default threshold=0.5')

# 找最优阈值 (max F1)
best_thresh = thresholds[np.argmax(f1_scores)]
ax.axvline(best_thresh, color='red', linestyle=':', alpha=0.7,
           label=f'Best F1 threshold={best_thresh:.2f}')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision-Recall-F1 vs Threshold\n(Adjust threshold to balance trade-off)',
             fontweight='bold')
ax.legend(fontsize=7)

# (6) ROC Curve
ax = axes[1, 2]
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
auc = roc_auc_score(y_test, y_prob[:, 1])
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'LR (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve (AUC={auc:.3f})', fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'logistic_regression.png'), dpi=150)
plt.close()
print("  Saved: logistic_regression.png")

# ============================================================
# 7. L1 vs L2 正则化对比
# ============================================================
print("\n" + "=" * 60)
print("  7. L1 vs L2 正则化 — 特征选择效果")
print("=" * 60)

# 生成高维特征 (含大量噪声)
X_high, y_high = make_classification(
    n_samples=300, n_features=50, n_informative=8,
    n_redundant=10, random_state=42
)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_high, y_high, test_size=0.3, random_state=42
)
scaler_h = StandardScaler()
Xh_train_s = scaler_h.fit_transform(Xh_train)
Xh_test_s = scaler_h.transform(Xh_test)

lr_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=2000, C=0.3, random_state=42)
lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=0.3, random_state=42)
lr_l1.fit(Xh_train_s, yh_train)
lr_l2.fit(Xh_train_s, yh_train)

n_l1_nonzero = (lr_l1.coef_[0] != 0).sum()
n_l2_all = X_high.shape[1]
print(f"  L1 (Lasso): {n_l1_nonzero}/{n_l2_all} features selected (others=0)")
print(f"  L2 (Ridge): all {n_l2_all} features used (weights shrunk but not zero)")
print(f"  L1 Test AUC: {roc_auc_score(yh_test, lr_l1.predict_proba(Xh_test_s)[:,1]):.3f}")
print(f"  L2 Test AUC: {roc_auc_score(yh_test, lr_l2.predict_proba(Xh_test_s)[:,1]):.3f}")
print(f"  → L1 自动做特征选择 (稀疏解), L2 保留所有特征")

# 可视化: L1 vs L2 权重分布
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(50)
w = 0.35
ax.bar(x - w/2, lr_l1.coef_[0], w, label='L1 (Lasso)', color='#E91E63', alpha=0.8)
ax.bar(x + w/2, lr_l2.coef_[0], w, label='L2 (Ridge)', color='#2196F3', alpha=0.6)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Feature Index'); ax.set_ylabel('Coefficient')
ax.set_title('L1 vs L2: Feature Selection Effect\n(L1 produces sparse solution)',
             fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'l1_vs_l2.png'), dpi=150)
plt.close()
print("  Saved: l1_vs_l2.png")

# ============================================================
# 面试知识点总结
# ============================================================
print("\n" + "=" * 60)
print("  面试速查: 逻辑回归核心")
print("=" * 60)
print(f"""
  Sigmoid: σ(z) = 1/(1+e^{-z}), z = w·x + b
    → 将任意实数映射到 (0,1), 解释为概率

  Log-Loss (Cross-Entropy):
    J(w) = -(1/n) Σ [y·log(p) + (1-y)·log(1-p)]
    → 完美预测: loss→0; 错误预测: loss→∞

  coef_ 解读:
    某个特征的 coef_ = 该特征每增加1单位, log-odds 的变化量
    exp(coef_) = odds ratio (几率比)
    例: coef=0.5 → exp(0.5)=1.65 → 该特征+1单位, 正类几率增加65%

  predict() vs predict_proba():
    predict(): 阈值0.5 → 0/1 硬分类
    predict_proba(): [P(y=0), P(y=1)] 概率输出
    → 阈值可根据业务需求调整 (如减少假阳性→提高阈值)

  L1 vs L2:
    L1 (Lasso): 产生稀疏解 → 自动特征选择
    L2 (Ridge): 权重收缩但不为零 → 处理共线性
    ElasticNet: L1 + L2 组合

  AUC = {auc:.3f}, Brier Score = {brier:.4f}
  Best threshold by F1: {best_thresh:.2f}
""")

print("✅ Day 39 完成")
