#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 47: 模型可解释性 — SHAP与部分依赖图
SHAP summary/dependence/waterfall + PDP对比
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import shap
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 生成数据 (带可解释的特征名)
X, y = make_classification(n_samples=800, n_features=10, n_informative=6,
                            random_state=42)
feature_names = [f'age', f'income', f'order_count', f'avg_order_value',
                 f'days_since_last', f'review_score', f'return_rate',
                 f'VIP_level', f'session_count', f'discount_usage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 训练 XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                     random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
auc_test = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"  XGBoost Test AUC: {auc_test:.4f}")

# ============================================================
# SHAP 分析
# ============================================================
# 创建 explainer
explainer = shap.TreeExplainer(xgb)
# 取样本计算 SHAP (TreeExplainer 用 background 数据集)
shap_values = explainer.shap_values(X_test[:200])  # 取200个样本加速

print(f"  SHAP values shape: {shap_values.shape}")
print(f"  Average |SHAP| for each feature:")
for i, name in enumerate(feature_names):
    print(f"    {name:20s}: {np.abs(shap_values[:, i]).mean():.4f}")

# ============================================================
# 图表
# ============================================================
fig = plt.figure(figsize=(18, 14))

# (1) SHAP Summary Plot
ax = plt.subplot(3, 3, 1)
shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names,
                  show=False, max_display=10)
ax.set_title('SHAP Summary Plot\n(Global importance + direction)', fontweight='bold', fontsize=10)

# (2) SHAP Bar Plot (mean |SHAP|)
ax = plt.subplot(3, 3, 2)
shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names,
                  plot_type='bar', show=False, max_display=10)
ax.set_title('SHAP Feature Importance\n(mean |SHAP|)', fontweight='bold', fontsize=10)

# (3) SHAP Dependence Plot (最重要特征)
ax = plt.subplot(3, 3, 3)
# 找最重要的特征
top_feat_idx = np.argmax(np.abs(shap_values).mean(axis=0))
shap.dependence_plot(top_feat_idx, shap_values, X_test[:200],
                     feature_names=feature_names, show=False, ax=ax)
ax.set_title(f'SHAP Dependence: {feature_names[top_feat_idx]}', fontweight='bold', fontsize=10)

# (4) SHAP Waterfall (单个样本解释)
ax = plt.subplot(3, 3, 4)
# 选一个正类预测样本
pos_idx = np.where(y_test[:200] == 1)[0][0]
shap.waterfall_plot(
    shap.Explanation(values=shap_values[pos_idx],
                     base_values=explainer.expected_value,
                     data=X_test[pos_idx],
                     feature_names=feature_names),
    show=False, max_display=10
)
ax = plt.gca()
ax.set_title(f'Waterfall: Sample {pos_idx} (predicted POSITIVE)', fontweight='bold', fontsize=10)

# (5) SHAP Waterfall (负类样本)
ax = plt.subplot(3, 3, 5)
neg_idx = np.where(y_test[:200] == 0)[0][0]
shap.waterfall_plot(
    shap.Explanation(values=shap_values[neg_idx],
                     base_values=explainer.expected_value,
                     data=X_test[neg_idx],
                     feature_names=feature_names),
    show=False, max_display=10
)
ax = plt.gca()
ax.set_title(f'Waterfall: Sample {neg_idx} (predicted NEGATIVE)', fontweight='bold', fontsize=10)

# (6) SHAP vs XGBoost Importance 对比
ax = plt.subplot(3, 3, 6)
shap_imp = np.abs(shap_values).mean(axis=0)
xgb_imp = xgb.feature_importances_
# 归一化对比
shap_imp_norm = shap_imp / shap_imp.sum()
xgb_imp_norm = xgb_imp / xgb_imp.sum()
x = np.arange(len(feature_names))
w = 0.3
ax.bar(x - w/2, shap_imp_norm, w, label='SHAP Importance', color='#E91E63', alpha=0.8)
ax.bar(x + w/2, xgb_imp_norm, w, label='XGBoost Gain', color='#2196F3', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([n[:8] for n in feature_names], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Normalized Importance')
ax.set_title('SHAP vs XGBoost Feature Importance', fontweight='bold', fontsize=10)
ax.legend(fontsize=7)

# (7) Partial Dependence Plot (PDP) — 前2个特征
ax = plt.subplot(3, 3, 7)
top2_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-2:]
PartialDependenceDisplay.from_estimator(
    xgb, X_train, features=top2_idx,
    feature_names=feature_names, ax=ax, line_kw={'linewidth': 2}
)
ax.set_title('Partial Dependence Plot (Top 2)', fontweight='bold', fontsize=10)

# (8) SHAP Dependence Plot (第二个最重要特征)
ax = plt.subplot(3, 3, 8)
second_feat_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-2]
shap.dependence_plot(second_feat_idx, shap_values, X_test[:200],
                     feature_names=feature_names, show=False, ax=ax)
ax.set_title(f'SHAP Dependence: {feature_names[second_feat_idx]}', fontweight='bold', fontsize=10)

# (9) 业务建议
ax = plt.subplot(3, 3, 9)
ax.axis('off')
top3_shap = np.argsort(np.abs(shap_values).mean(axis=0))[-3:][::-1]
business_insights = f"""
SHAP-Based Business Recommendations
====================================

Top 3 Most Important Features:
1. {feature_names[top3_shap[0]]}
2. {feature_names[top3_shap[1]]}
3. {feature_names[top3_shap[2]]}

Actionable Insights:

1. {feature_names[top3_shap[0]]} is the strongest predictor
   → Target users with LOW values for
     reactivation campaigns
   → Offer incentives to improve this metric

2. {feature_names[top3_shap[1]]} shows non-linear effect
   → Segment users by this feature
   → Different strategies for different tiers

3. Dependence plots reveal non-linear effects
   → Segment users based on feature thresholds
   → Combined targeting more effective
     than single-feature approach

SHAP enables:
• Global: Which features matter most?
• Local: Why did THIS user get THIS prediction?
• Interaction: How do features work together?

Model AUC: {auc_test:.3f}
"""
ax.text(0.05, 0.95, business_insights, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: shap_analysis.png")

print("\n✅ Day 47 完成")
