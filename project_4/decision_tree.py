#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 41: 决策树 — 可视化与过拟合
max_depth · min_samples_split · 剪枝 · 特征重要性
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import make_classification
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 数据
X, y = make_classification(n_samples=500, n_features=8, n_informative=4,
                            random_state=42)
feature_names = [f'f{i}' for i in range(8)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=" * 60)
print("  决策树: 可视化与过拟合")
print("=" * 60)

# ============================================================
# 1. 可视化决策树 (max_depth=3)
# ============================================================
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(dt_viz, feature_names=feature_names, class_names=['Class 0', 'Class 1'],
          filled=True, rounded=True, fontsize=9, ax=ax, impurity=True)
ax.set_title('Decision Tree (max_depth=3)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'decision_tree_viz.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: decision_tree_viz.png")

# ============================================================
# 2 & 3. 深度对比 + 学习曲线
# ============================================================
depths = [1, 2, 3, 5, 7, 10, 15, None]
train_scores, test_scores, train_aucs, test_aucs = [], [], [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test, dt.predict(X_test)))
    train_aucs.append(roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1]))
    test_aucs.append(roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]))

print(f"\n  {'Depth':8s} | {'Train Acc':10s} | {'Test Acc':10s} | {'Test AUC':10s} | {'Gap':8s}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
for d, tr, te, ta in zip(depths, train_scores, test_scores, test_aucs):
    d_str = str(d) if d else 'None'
    gap = tr - te
    flag = '⚠️ OVERFIT' if gap > 0.05 else '✅ OK'
    print(f"  {d_str:8s} | {tr:.4f}      | {te:.4f}      | {ta:.4f}      | {gap:.3f} {flag}")

# 学习曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
depth_labels = [str(d) if d else 'None' for d in depths]
ax.plot(depth_labels, train_scores, 'o-', color='#2196F3', linewidth=2, label='Train')
ax.plot(depth_labels, test_scores, 'o-', color='#FF9800', linewidth=2, label='Test')
ax.fill_between(range(len(depths)), train_scores, test_scores, alpha=0.2, color='red')
ax.set_xlabel('max_depth'); ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve: Depth vs Accuracy', fontweight='bold')
ax.legend()
ax.text(5, train_scores[3] - 0.02, 'Overfitting zone', fontsize=8, color='red', ha='center')

# min_samples_split 影响
ax = axes[1]
min_splits = [2, 5, 10, 20, 50, 100]
split_train, split_test = [], []
for ms in min_splits:
    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=ms, random_state=42)
    dt.fit(X_train, y_train)
    split_train.append(accuracy_score(y_train, dt.predict(X_train)))
    split_test.append(accuracy_score(y_test, dt.predict(X_test)))

ax.plot(min_splits, split_train, 'o-', color='#2196F3', linewidth=2, label='Train')
ax.plot(min_splits, split_test, 'o-', color='#FF9800', linewidth=2, label='Test')
ax.set_xlabel('min_samples_split'); ax.set_ylabel('Accuracy')
ax.set_title('Effect of min_samples_split', fontweight='bold')
ax.legend()

# min_samples_leaf 影响
ax = axes[2]
min_leafs = [1, 5, 10, 20, 50]
leaf_train, leaf_test = [], []
for ml in min_leafs:
    dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=ml, random_state=42)
    dt.fit(X_train, y_train)
    leaf_train.append(accuracy_score(y_train, dt.predict(X_train)))
    leaf_test.append(accuracy_score(y_test, dt.predict(X_test)))

ax.plot(min_leafs, leaf_train, 'o-', color='#2196F3', linewidth=2, label='Train')
ax.plot(min_leafs, leaf_test, 'o-', color='#FF9800', linewidth=2, label='Test')
ax.set_xlabel('min_samples_leaf'); ax.set_ylabel('Accuracy')
ax.set_title('Effect of min_samples_leaf', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'decision_tree_tuning.png'), dpi=150)
plt.close()
print("  Saved: decision_tree_tuning.png")

# ============================================================
# 4. 特征重要性
# ============================================================
dt_best = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_best.fit(X_train, y_train)

print(f"\n  Feature Importance (max_depth=5):")
for i, imp in sorted(enumerate(dt_best.feature_importances_), key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 30)
    print(f"    f{i}: {imp:.4f} {bar}")

# ============================================================
# 5. 基尼系数 vs 信息增益
# ============================================================
print(f"\n  Criterion comparison:")
for crit in ['gini', 'entropy']:
    dt = DecisionTreeClassifier(criterion=crit, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"    {crit:10s}: Test acc = {acc:.4f}")

print(f"""
  Gini Impurity:  G = 1 - Σ p_i^2
    → 计算更快, 倾向于分离出大类

  Entropy (Information Gain):  H = -Σ p_i log(p_i)
    → 倾向于产生更平衡的分裂

  实践中两者差异很小, Gini 是 sklearn 默认值
""")

# ============================================================
# 6. 防止过拟合总结
# ============================================================
print("=" * 60)
print("  防止决策树过拟合的方法")
print("=" * 60)
print("""
  前剪枝 (Pre-pruning):
    1. max_depth: 限制最大深度 (最直接)
    2. min_samples_split: 分裂所需最小样本数
    3. min_samples_leaf: 叶节点最小样本数
    4. min_impurity_decrease: 分裂的最小不纯度下降

  后剪枝 (Post-pruning):
    5. ccp_alpha: 代价复杂度剪枝 (Cost Complexity Pruning)
       → 先种一棵大树, 再根据ccp_alpha逐步剪掉不重要的分支

  集成方法:
    6. Random Forest / XGBoost → 用多棵树平均, 天然抗过拟合

  面试回答模板:
  Q: "决策树过拟合了怎么办?"
  A: "首先限制max_depth, 这是最直接的方法。
      其次增大min_samples_split/min_samples_leaf,
      让树不要对噪声数据过度拟合。
      如果还不行, 考虑ccp_alpha后剪枝,
      或者改用随机森林——多棵树的平均天然降低方差。"
""")

print("✅ Day 41 完成")
