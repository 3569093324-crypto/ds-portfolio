#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 44: 特征工程全流程
5+种方法: 编码 · 分箱 · 变换 · 交互 · 多项式 · 特征选择
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, LabelEncoder,
                                    PolynomialFeatures, PowerTransformer)
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 生成含混合类型特征的数据
# ============================================================
n = 600
X_num, y = make_classification(n_samples=n, n_features=8, n_informative=5, random_state=42)

# 转为 DataFrame 方便操作
df = pd.DataFrame(X_num, columns=[f'num_{i}' for i in range(8)])
df['target'] = y

# 添加分类特征
df['category_A'] = np.random.choice(['A', 'B', 'C', 'D'], n, p=[0.3, 0.3, 0.2, 0.2])
df['category_B'] = np.random.choice(['X', 'Y', 'Z'], n)
# 给分类特征加一些与目标相关的信号
df.loc[df['target']==1, 'category_A'] = np.random.choice(['A', 'B', 'C', 'D'],
    sum(df['target']==1), p=[0.4, 0.3, 0.2, 0.1])
# 右偏数值特征
df['right_skewed'] = np.random.exponential(2, n) + (df['target'] * 1.5)
# 包含交互关系: num_0 × num_1
df['num_0_sq'] = df['num_0'] ** 2

print("=" * 60)
print("  特征工程全流程")
print("=" * 60)
print(f"  原始数据: {df.shape[1]} 列 (含1个target)")

# ============================================================
# Baseline (不做特征工程)
# ============================================================
X_raw = df.drop('target', axis=1)
y = df['target']

X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.25, random_state=42)

# 仅标准化数值列
num_cols_raw = [f'num_{i}' for i in range(8)] + ['right_skewed', 'num_0_sq']
# 暂时丢弃分类列 (baseline 不用复杂编码)
scaler_base = StandardScaler()
X_tr_num = X_tr[num_cols_raw]
X_te_num = X_te[num_cols_raw]
X_tr_base = scaler_base.fit_transform(X_tr_num)
X_te_base = scaler_base.transform(X_te_num)

lr_base = LogisticRegression(max_iter=2000, random_state=42)
cv_base = cross_val_score(lr_base, X_tr_base, y_tr, cv=5, scoring='roc_auc')
lr_base.fit(X_tr_base, y_tr)
test_auc_base = roc_auc_score(y_te, lr_base.predict_proba(X_te_base)[:, 1])

print(f"\n  Baseline (仅数值+标准化): CV AUC={cv_base.mean():.4f}, Test AUC={test_auc_base:.4f}")

# ============================================================
# 1. 类别编码
# ============================================================
print(f"\n  --- 1. 类别编码 ---")

# One-Hot
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = ['category_A', 'category_B']
X_tr_cat = X_tr[cat_features].astype(str)
X_te_cat = X_te[cat_features].astype(str)
ohe.fit(X_tr_cat)
X_tr_ohe = ohe.transform(X_tr_cat)
X_te_ohe = ohe.transform(X_te_cat)
print(f"  One-Hot: {X_tr_ohe.shape[1]} 列 (从 {len(cat_features)} 列展开)")

# Target Encoding (手动实现)
from sklearn.model_selection import KFold
target_enc_map = {}
# 留一法 target encoding
for col in cat_features:
    # 全局均值
    global_mean = y_tr.mean()
    # 每个类别的均值 + 平滑
    cat_means = X_tr[col].astype(str).to_frame().join(
        pd.Series(y_tr, name='target', index=X_tr.index)
    ).groupby(col)['target'].mean()
    # 简单 target encoding
    target_enc_map[col] = cat_means.to_dict()
    print(f"  Target Encoding ({col}): {cat_means.round(3).to_dict()}")

# ============================================================
# 2. 数值分箱
# ============================================================
print(f"\n  --- 2. 数值分箱 ---")

# 等宽分箱
binned_equal = pd.cut(df['num_0'], bins=5, labels=['VL', 'L', 'M', 'H', 'VH'])
print(f"  等宽分箱 (num_0): {binned_equal.value_counts().sort_index().to_dict()}")

# 等频分箱
binned_quantile = pd.qcut(df['num_0'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
print(f"  等频分箱 (num_0): {binned_quantile.value_counts().sort_index().to_dict()}")

# ============================================================
# 3. 特征变换
# ============================================================
print(f"\n  --- 3. 特征变换 ---")

# 先检查 right_skewed 的偏度
skew_before = df['right_skewed'].skew()
# Log变换
log_transformed = np.log1p(df['right_skewed'])
skew_after = log_transformed.skew()
print(f"  right_skewed: skew={skew_before:.2f} → log变换后 skew={skew_after:.2f}")

# Box-Cox (scipy)
from scipy.stats import boxcox
bc_transformed, bc_lambda = boxcox(df['right_skewed'].clip(lower=0.01))
print(f"  Box-Cox: lambda={bc_lambda:.3f}, skew after={pd.Series(bc_transformed).skew():.2f}")

# PowerTransformer (Yeo-Johnson, 支持负值)
pt = PowerTransformer(method='yeo-johnson')
pt_transformed = pt.fit_transform(df[['num_0']])
print(f"  Yeo-Johnson (num_0): skew before={df['num_0'].skew():.2f}, "
      f"after={pd.Series(pt_transformed.ravel()).skew():.2f}")

# ============================================================
# 4. 特征交互
# ============================================================
print(f"\n  --- 4. 特征交互 ---")
df['inter_num0_num1'] = df['num_0'] * df['num_1']
df['inter_num0_num2'] = df['num_0'] * df['num_2']
df['ratio_num3_num4'] = df['num_3'] / (df['num_4'].abs() + 0.1)
print(f"  创建: num_0×num_1, num_0×num_2, num_3/num_4")
print(f"  交互特征与target的相关性:")
for c in ['inter_num0_num1', 'inter_num0_num2', 'ratio_num3_num4']:
    print(f"    {c}: r={df[c].corr(df['target']):.4f}")

# ============================================================
# 5. 多项式特征
# ============================================================
print(f"\n  --- 5. 多项式特征 ---")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
# 仅在 top 3 特征上做多项式
top3 = ['num_0', 'num_1', 'num_2']
X_poly = poly.fit_transform(df[top3])
poly_names = poly.get_feature_names_out(top3)
print(f"  原始3个特征 → {len(poly_names)} 个多项式特征")
print(f"  新特征: {list(poly_names[3:])}")  # 跳过原始3个

# ============================================================
# 6. 特征选择
# ============================================================
print(f"\n  --- 6. 特征选择 (Mutual Information) ---")

# 准备了更完整的特征集
df_full = df.copy()
# 加入 One-Hot 编码的分类特征
for i, col in enumerate(ohe.get_feature_names_out(cat_features)):
    df_full[col] = 0
for idx in X_tr.index:
    for j, col in enumerate(ohe.get_feature_names_out(cat_features)):
        df_full.loc[idx, col] = X_tr_ohe[list(X_tr.index).index(idx), j]

feature_candidates = [c for c in df_full.columns
                      if c != 'target' and df_full[c].dtype in ['float64', 'int64', 'int32', 'float32']]

mi_scores = mutual_info_classif(df_full[feature_candidates].fillna(0), y, random_state=42)
mi_ranked = sorted(zip(feature_candidates, mi_scores), key=lambda x: x[1], reverse=True)

print(f"  Top 10 features by Mutual Information:")
for i, (name, score) in enumerate(mi_ranked[:10]):
    bar = '█' * int(score * 50)
    print(f"    {i+1:2d}. {name:25s}: MI={score:.4f} {bar}")

# 选 Top 10 特征
top10_features = [name for name, _ in mi_ranked[:10]]
print(f"\n  选择 Top 10 特征 (MI > {mi_ranked[9][1]:.4f})")

# ============================================================
# 7. 特征工程前后对比
# ============================================================
print(f"\n  --- 7. 前后对比 ---")

# 特征工程后的数据
# 构造增强特征集
X_enhanced = df_full[feature_candidates].fillna(0).values
X_tr_enh, X_te_enh, y_tr_enh, y_te_enh = train_test_split(
    X_enhanced, y, test_size=0.25, random_state=42
)

scaler_enh = StandardScaler()
X_tr_enh_s = scaler_enh.fit_transform(X_tr_enh)
X_te_enh_s = scaler_enh.transform(X_te_enh)

# 只用 Top 10
selector = SelectKBest(mutual_info_classif, k=10)
X_tr_top = selector.fit_transform(X_tr_enh_s, y_tr_enh)
X_te_top = selector.transform(X_te_enh_s)

# 模型对比
models_compare = {
    'LR Baseline (raw num)': (LogisticRegression(max_iter=2000, random_state=42),
                               X_tr_base, X_te_base, y_tr, y_te),
    'LR + All Features': (LogisticRegression(max_iter=2000, random_state=42),
                           X_tr_enh_s, X_te_enh_s, y_tr_enh, y_te_enh),
    'LR + Top 10 MI': (LogisticRegression(max_iter=2000, random_state=42),
                        X_tr_top, X_te_top, y_tr_enh, y_te_enh),
    'RF + All Features': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                           X_tr_enh, X_te_enh, y_tr_enh, y_te_enh),
}

print(f"\n  {'Model':25s} | {'CV AUC':12s} | {'Test AUC':10s} | {'Improvement'}")
print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")
for name, (model, Xtr, Xte, ytr, yte) in models_compare.items():
    cv = cross_val_score(model, Xtr, ytr, cv=5, scoring='roc_auc')
    model.fit(Xtr, ytr)
    if hasattr(model, 'predict_proba'):
        test_auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    else:
        test_auc = roc_auc_score(yte, model.predict(Xte))
    imp = f"+{test_auc - test_auc_base:.4f}" if test_auc > test_auc_base else f"{test_auc - test_auc_base:.4f}"
    print(f"  {name:25s} | {cv.mean():.4f} ± {cv.std():.4f} | {test_auc:.4f}      | {imp}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 特征重要性 (MI)
ax = axes[0]
scores_top = [s for _, s in mi_ranked[:12]]
names_top = [n[:15] for n, _ in mi_ranked[:12]]
colors_mi = plt.cm.Blues(np.linspace(0.4, 0.9, 12))
ax.barh(range(12)[::-1], scores_top[::-1], color=colors_mi[::-1], edgecolor='white')
ax.set_yticks(range(12)[::-1])
ax.set_yticklabels(names_top[::-1], fontsize=8)
ax.set_xlabel('Mutual Information')
ax.set_title('Feature Importance (MI)', fontweight='bold')

# 偏度变化
ax = axes[1]
methods = ['Original', 'Log(1+x)', 'Box-Cox']
skews = [skew_before, skew_after, pd.Series(bc_transformed).skew()]
colors_skew = ['#f44336', '#FF9800', '#4CAF50']
ax.bar(methods, skews, color=colors_skew, edgecolor='white')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
for i, s in enumerate(skews):
    ax.text(i, s + 0.1, f'{s:.2f}', ha='center', fontweight='bold')
ax.set_ylabel('Skewness')
ax.set_title('Right-Skewed Feature Transformation', fontweight='bold')

# 对比: before vs after
ax = axes[2]
labels_comp = ['LR Baseline\n(raw num)', 'LR + All\nFeatures', 'LR + Top10\nMI Selected', 'RF + All\nFeatures']
scores_comp = [test_auc_base]
for name, (model, Xtr, Xte, ytr, yte) in models_compare.items():
    if name == 'LR Baseline (raw num)': continue
    model.fit(Xtr, ytr)
    scores_comp.append(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
colors_bar = ['#9E9E9E', '#2196F3', '#4CAF50', '#FF9800']
ax.bar(labels_comp, scores_comp, color=colors_bar, edgecolor='white')
ax.axhline(test_auc_base, color='gray', linestyle='--', alpha=0.5, label='Baseline')
for i, s in enumerate(scores_comp):
    ax.text(i, s + 0.005, f'{s:.3f}', ha='center', fontweight='bold', fontsize=10)
ax.set_ylabel('Test AUC')
ax.set_title('Feature Engineering Impact', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_engineering.png'), dpi=150)
plt.close()
print("\n  Saved: feature_engineering.png")

print("\n✅ Day 44 完成")
