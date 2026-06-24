#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 49: 模型系统性对比
5模型 × 统一CV × 多维评估 × 统计检验
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold)
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from scipy import stats
import time
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 数据
X, y = make_classification(n_samples=1000, n_features=15, n_informative=8,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print("=" * 60)
print("  模型系统性对比: 5 Models")
print("=" * 60)

# ============================================================
# 候选模型
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                              random_state=42, verbosity=0),
    'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                       random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 统一CV评估
# ============================================================
results = {}
for name, model in models.items():
    t0 = time.time()
    # Cross-val
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    # Train on full train set
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Test
    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    # Inference time
    t0 = time.time()
    for _ in range(100):
        model.predict(X_test[:10])
    inference_time = (time.time() - t0) / 100 * 1000

    # SHAP compatibility
    shap_ok = name in ['Random Forest', 'XGBoost', 'Decision Tree'] or 'XGB' in name

    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores,
        'test_auc': test_auc,
        'train_time': train_time,
        'inference_time_ms': inference_time,
        'interpretable': 'coef_' in dir(model) or shap_ok,
        'shap_ready': shap_ok,
    }

# ============================================================
# 输出对比表
# ============================================================
print(f"\n  {'Model':22s} | {'CV AUC':12s} | {'Test AUC':10s} | {'Train':8s} | {'Infer':8s} | {'Interp':8s}")
print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for name, r in results.items():
    print(f"  {name:22s} | {r['cv_mean']:.4f} ± {r['cv_std']:.4f} | {r['test_auc']:.4f}     | "
          f"{r['train_time']:.2f}s   | {r['inference_time_ms']:.2f}ms  | "
          f"{'Yes' if r['interpretable'] else 'Limited'}")

# ============================================================
# 5. 统计检验: 最好的两个模型
# ============================================================
sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
best_name, best_r = sorted_models[0]
second_name, second_r = sorted_models[1]

t_stat, p_val = stats.ttest_rel(best_r['cv_scores'], second_r['cv_scores'])
print(f"\n  Paired t-test: {best_name} vs {second_name}")
print(f"    t = {t_stat:.4f}, p = {p_val:.4f}")
if p_val < 0.05:
    print(f"    ✅ {best_name} significantly better than {second_name}")
else:
    print(f"    ❌ Difference not statistically significant")

# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) CV分数箱线图
ax = axes[0, 0]
cv_data = [r['cv_scores'] for r in results.values()]
names = list(results.keys())
bp = ax.boxplot(cv_data, patch_artist=True)
colors_box = plt.cm.Set2(np.linspace(0, 1, len(names)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels([n[:12] for n in names], rotation=20, ha='right', fontsize=8)
ax.set_ylabel('CV AUC')
ax.set_title('5-Fold CV Score Distribution', fontweight='bold')

# (2) 速度 vs 性能散点图
ax = axes[0, 1]
for i, (name, r) in enumerate(results.items()):
    ax.scatter(r['train_time'], r['cv_mean'], s=200, c=[colors_box[i]],
               edgecolors='white', zorder=5, label=name[:15])
    ax.annotate(name[:12], (r['train_time'], r['cv_mean']),
                fontsize=7, ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')
ax.set_xlabel('Training Time (s)'); ax.set_ylabel('CV AUC')
ax.set_title('Speed vs Performance Trade-off', fontweight='bold')
ax.legend(fontsize=6, loc='lower right')

# (3) 雷达图: 多维度对比
ax = axes[1, 0]
# 标准化每个维度到 [0,1]
metrics_radar = ['CV AUC', 'Speed', 'Interpretability', 'Test AUC']
# 评分 (手动归一化)
radar_scores = {}
for name in names:
    r = results[name]
    radar_scores[name] = [
        (r['cv_mean'] - 0.85) / 0.15,  # AUC scaling
        1 - min(r['train_time'], 5) / 5,  # Speed (faster=better)
        1.0 if r['interpretable'] else 0.3,  # Interpretability
        (r['test_auc'] - 0.85) / 0.15,  # Test AUC
    ]

# 简单柱状图替代雷达图
x = np.arange(len(metrics_radar))
w = 0.15
for i, (name, scores) in enumerate(radar_scores.items()):
    ax.bar(x + i * w - w * 2, scores, w, label=name[:12], alpha=0.8, color=colors_box[i])
ax.set_xticks(x)
ax.set_xticklabels(metrics_radar, fontsize=9)
ax.set_ylabel('Normalized Score')
ax.set_title('Multi-Dimensional Comparison', fontweight='bold')
ax.legend(fontsize=6)
ax.set_ylim(0, 1.2)

# (4) 选型建议
ax = axes[1, 1]
ax.axis('off')

best_overall = sorted_models[0][0]
best_fast = sorted(
    [(n, r) for n, r in results.items() if r['interpretable']],
    key=lambda x: x[1]['train_time']
)[0][0]

recommendation = f"""
Model Selection Recommendation
================================

Best overall performance:
→ {best_overall}
  CV AUC: {results[best_overall]['cv_mean']:.4f}
  Train: {results[best_overall]['train_time']:.2f}s

Best with interpretability:
→ {best_fast}
  CV AUC: {results[best_fast]['cv_mean']:.4f}

Decision Framework:

┌──────────────────┬──────────────┐
│ Business Need    │ Recommend    │
├──────────────────┼──────────────┤
│ Max Accuracy     │ {best_overall[:12]:12s} │
│ Interpretability │ {best_fast[:12]:12s} │
│ Fast Training    │ Logistic Reg │
│ Unbalanced Data  │ XGBoost      │
│ Deployment       │ Logistic Reg │
└──────────────────┴──────────────┘

Statistical significance:
{best_name} vs {second_name}:
p = {p_val:.4f} {'(significant)' if p_val<0.05 else '(not significant)'}

Final Recommendation:
Use {best_overall} for maximum performance.
Keep Logistic Regression as interpretable baseline
for stakeholder communication.
"""
ax.text(0.05, 0.95, recommendation, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_comparison.png'), dpi=150)
plt.close()
print("  Saved: model_comparison.png")

print("\n✅ Day 49 完成")
