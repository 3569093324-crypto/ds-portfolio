#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 40: 模型评估 — 超越准确率
混淆矩阵 · ROC/PR曲线 · 宏/微平均 · 回归指标
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 数据准备: 平衡 + 不平衡两种场景
# ============================================================
# 平衡数据
X_bal, y_bal = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                    weights=[0.5, 0.5], random_state=42)
# 不平衡数据
X_imb, y_imb = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                    weights=[0.95, 0.05], random_state=42)

for name, (X, y) in [('Balanced', (X_bal, y_bal)), ('Imbalanced', (X_imb, y_imb))]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    print(f"\n{'='*60}")
    print(f"  {name} Data (positive rate={y.mean():.1%})")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_te, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_te, y_pred):.4f}")
    print(f"  F1:        {f1_score(y_te, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_te, y_prob):.4f}")
    if name == 'Imbalanced':
        # 全猜负类的 baseline
        baseline_acc = (y_te == 0).mean()
        print(f"  ⚠️ 全猜负类 Accuracy: {baseline_acc:.4f} (几乎等于模型!)")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, (name, (X, y)) in enumerate([
    ('Balanced', (X_bal, y_bal)),
    ('Imbalanced', (X_imb, y_imb))
]):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    row = idx  # 0=balanced top row, 1=imbalanced bottom row

    # 混淆矩阵
    ax = axes[row, 0]
    cm = confusion_matrix(y_te, y_pred)
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix ({name})', fontweight='bold')

    # ROC
    ax = axes[row, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc_val = roc_auc_score(y_te, y_prob)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color='blue')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'ROC Curve ({name})', fontweight='bold')
    ax.legend()

    # PR
    ax = axes[row, 2]
    prec, rec, _ = precision_recall_curve(y_te, y_prob)
    pr_auc = average_precision_score(y_te, y_prob)
    baseline_pr = y_te.mean()
    ax.plot(rec, prec, 'g-', linewidth=2, label=f'PR-AUC={pr_auc:.3f}')
    ax.axhline(baseline_pr, color='gray', linestyle='--', alpha=0.7,
               label=f'Baseline={baseline_pr:.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve ({name})', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_evaluation.png'), dpi=150)
plt.close()
print("\n  Saved: model_evaluation.png")

# ============================================================
# 3. Precision vs Recall 业务场景
# ============================================================
print("\n" + "=" * 60)
print("  3. 什么时候优先 Precision? 什么时候优先 Recall?")
print("=" * 60)
print("""
  ✅ 优先 Precision (减少误报):
     - 垃圾邮件检测: 不能把重要邮件标为垃圾
     - 信用卡欺诈: 误冻结正常用户代价大
     - 精准营销: 预算有限, 要确保推送的都是高意向

  ✅ 优先 Recall (减少漏报):
     - 癌症筛查: 宁可误诊也不能漏诊
     - 安全监控: 不能漏掉任何潜在威胁
     - 欺诈调查初筛: 宁可多查几个, 不能漏掉

  ✅ 平衡 (F1):
     - 大多数场景的默认选择
""")

# ============================================================
# 5. macro vs micro vs weighted
# ============================================================
print("=" * 60)
print("  5. macro vs micro vs weighted average")
print("=" * 60)

# 多分类模拟
np.random.seed(123)
y_true_multi = np.random.choice([0, 1, 2], 300, p=[0.1, 0.3, 0.6])
y_pred_multi = np.random.choice([0, 1, 2], 300, p=[0.08, 0.32, 0.6])

print(f"  类别分布: 0={sum(y_true_multi==0)}, 1={sum(y_true_multi==1)}, 2={sum(y_true_multi==2)}")
print(f"  Accuracy: {accuracy_score(y_true_multi, y_pred_multi):.3f}")
print(f"  Precision macro:     {precision_score(y_true_multi, y_pred_multi, average='macro'):.3f}")
print(f"  Precision micro:     {precision_score(y_true_multi, y_pred_multi, average='micro'):.3f}")
print(f"  Precision weighted:  {precision_score(y_true_multi, y_pred_multi, average='weighted'):.3f}")

print("""
  macro:    每个类别独立计算 → 算术平均 → 小类和大类同等重要
  micro:    全局计算 → = accuracy (多分类时)
  weighted: 每个类别独立计算 → 按样本数加权 → 大类影响更大
  → 类别不平衡时, macro 更能体现小类的表现
""")

# ============================================================
# 6. 回归指标
# ============================================================
print("=" * 60)
print("  6. 回归问题评估指标")
print("=" * 60)

np.random.seed(42)
y_true_reg = np.random.randn(200) * 50 + 500
y_pred_reg = y_true_reg + np.random.randn(200) * 30

mae = mean_absolute_error(y_true_reg, y_pred_reg)
mse = mean_squared_error(y_true_reg, y_pred_reg)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_true_reg, y_pred_reg)
r2 = r2_score(y_true_reg, y_pred_reg)

print(f"  MAE:  {mae:.2f} — 平均绝对误差, 量纲相同, 对异常值不敏感")
print(f"  MSE:  {mse:.2f} — 均方误差, 放大异常值的影响")
print(f"  RMSE: {rmse:.2f} — 均方根误差, 量纲相同, 对异常值敏感")
print(f"  MAPE: {mape*100:.1f}% — 百分比误差, 适合跨尺度比较 (注意: y=0时不可用)")
print(f"  R²:   {r2:.4f} — 解释方差比例, 1=完美, 0=不如猜均值")

print("""
  选择指南:
  - 异常值不重要 → MAE
  - 异常值很重要 → RMSE (大误差被重点惩罚)
  - 跨产品/跨时间对比 → MAPE
  - 模型整体拟合度 → R²
""")

print("✅ Day 40 完成")
