#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 42: 随机森林 — 集成学习入门
Bagging · OOB · 双重随机性 · vs Boosting
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import make_classification
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

X, y = make_classification(n_samples=500, n_features=15, n_informative=6,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=" * 60)
print("  随机森林: Bagging + 特征随机")
print("=" * 60)

# ============================================================
# 1. n_estimators 对比
# ============================================================
n_estimators_list = [5, 10, 30, 50, 100, 200]
train_scores, test_scores, oob_scores = [], [], []

for n in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, rf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, rf.predict(X_test)))
    oob_scores.append(rf.oob_score_)

print(f"\n  {'n_trees':8s} | {'Train':8s} | {'Test':8s} | {'OOB':8s}")
print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for n, tr, te, oob in zip(n_estimators_list, train_scores, test_scores, oob_scores):
    print(f"  {n:8d} | {tr:.4f}   | {te:.4f}   | {oob:.4f}")

print(f"\n  OOB = Out-of-Bag: 用没被bootstrap抽到的样本做验证")
print(f"  OOB ≈ 交叉验证, 不需要额外划分验证集")

# ============================================================
# 2. 两个随机性
# ============================================================
print(f"\n  随机森林的两个随机性:")
print(f"  1. 样本随机: Bootstrap (有放回抽样, 每次约63%样本被选中)")
print(f"  2. 特征随机: 每次分裂只考虑 sqrt(n_features) 个特征")
print(f"     → 在15个特征中, 每次分裂随机选 {int(np.sqrt(15))} 个作为候选")
print(f"  双重随机性 → 树之间相关性降低 → 方差减小")

# ============================================================
# 3 & 5. 特征重要性
# ============================================================
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)
dt_single = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_single.fit(X_train, y_train)

# 对比
imp_comparison = np.column_stack([
    rf_full.feature_importances_,
    dt_single.feature_importances_,
])
top_indices = np.argsort(rf_full.feature_importances_)[::-1][:8]

print(f"\n  Feature Importance (Top 8):")
print(f"  {'Feature':10s} | {'RF Imp':10s} | {'DT Imp':10s}")
print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}")
for idx in top_indices:
    print(f"  f{idx:9d} | {imp_comparison[idx, 0]:.4f}      | {imp_comparison[idx, 1]:.4f}")

# ============================================================
# 6. 调参
# ============================================================
print(f"\n  max_features 影响:")
for mf in ['sqrt', 'log2', 0.3, 0.5, None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42)
    cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"    max_features={str(mf):6s}: CV={cv.mean():.4f} ± {cv.std():.4f}")

# ============================================================
# 7. Bagging vs Boosting
# ============================================================
print(f"""
  Bagging vs Boosting 本质区别:

  Bagging (Random Forest):
    - 并行训练: 每棵树独立
    - 降低方差: 多棵树投票平均 → 减少过拟合
    - 适合: 高方差模型 (如深决策树)
    - 样本: Bootstrap 独立抽样

  Boosting (XGBoost, AdaBoost):
    - 串行训练: 每棵树纠正前一颗的错误
    - 降低偏差: 逐步拟合残差 → 提高准确率
    - 适合: 高偏差模型 (如浅决策树)
    - 样本: 根据上一轮错误率加权

  记忆: Bagging=并行→降方差, Boosting=串行→降偏差
""")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (1) n_estimators 学习曲线
ax = axes[0]
ax.plot(n_estimators_list, train_scores, 'o-', color='#2196F3', linewidth=2, label='Train')
ax.plot(n_estimators_list, test_scores, 'o-', color='#FF9800', linewidth=2, label='Test')
ax.plot(n_estimators_list, oob_scores, 's--', color='#4CAF50', linewidth=2, label='OOB')
ax.set_xlabel('n_estimators'); ax.set_ylabel('Accuracy')
ax.set_title('Random Forest: More Trees = Better\n(but diminishing returns)',
             fontweight='bold')
ax.legend()

# (2) 特征重要性
ax = axes[1]
imp_sorted = np.argsort(rf_full.feature_importances_)
colors_rf = plt.cm.Blues(np.linspace(0.4, 0.9, 15))
ax.barh(range(15), rf_full.feature_importances_[imp_sorted], color=colors_rf,
        edgecolor='white')
ax.set_yticks(range(15))
ax.set_yticklabels([f'f{i}' for i in imp_sorted], fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance', fontweight='bold')

# (3) RF vs Single Tree
ax = axes[2]
models_comp = {'Single Tree\n(max_depth=10)': accuracy_score(y_test, dt_single.predict(X_test)),
               'Random Forest\n(100 trees)': accuracy_score(y_test, rf_full.predict(X_test))}
colors_comp = ['#f44336', '#4CAF50']
ax.bar(list(models_comp.keys()), list(models_comp.values()), color=colors_comp,
       edgecolor='white')
for i, (k, v) in enumerate(models_comp.items()):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold', fontsize=14)
ax.set_ylabel('Test Accuracy')
ax.set_title('Random Forest vs Single Decision Tree', fontweight='bold')
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'random_forest.png'), dpi=150)
plt.close()
print("  Saved: random_forest.png")

print("✅ Day 42 完成")
