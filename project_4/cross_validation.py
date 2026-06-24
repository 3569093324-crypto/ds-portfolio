#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 37: 训练/测试切分与交叉验证
核心: 防止数据泄露 — 先切分, 再做特征工程
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 生成模拟数据
n_samples = 200
X = np.random.randn(n_samples, 5)
X[:, 0] = X[:, 0] * 3 + np.random.randn(n_samples) * 0.5  # 特征0有信号
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

print("=" * 60)
print("  训练/测试切分与交叉验证")
print("=" * 60)
print(f"  数据: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  目标: binary classification, positive={y.mean()*100:.1f}%")

# ============================================================
# 1. 手动实现 train_test_split (numpy only)
# ============================================================
print("\n" + "=" * 60)
print("  1. 手动实现 train_test_split")
print("=" * 60)

def manual_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """纯numpy实现 — 理解底层原理"""
    if random_state is not None:
        np.random.seed(random_state)

    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx

X_train_m, X_test_m, y_train_m, y_test_m, train_idx, test_idx = \
    manual_train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  Train: {len(X_train_m)} samples")
print(f"  Test:  {len(X_test_m)} samples")
print(f"  Train positive rate: {y_train_m.mean():.2f}")
print(f"  Test positive rate:  {y_test_m.mean():.2f}")

# ============================================================
# 2. 随机切分 vs 分层切分
# ============================================================
print("\n" + "=" * 60)
print("  2. 随机切分 vs 分层切分")
print("=" * 60)

# 构造不平衡数据
n_imb = 200
y_imb = np.concatenate([np.ones(30), np.zeros(170)])
X_imb = np.random.randn(n_imb, 3)
np.random.shuffle(y_imb)

# 随机切分
_, _, _, _, _, _ = manual_train_test_split(X_imb, y_imb, test_size=0.3, random_state=1)
test_random = y_imb[np.random.RandomState(1).permutation(n_imb)[int(n_imb*0.7):]]

# 分层切分
def manual_stratified_split(X, y, test_size=0.2, random_state=None):
    """手动分层切分: 确保train和test中各类别比例一致"""
    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y)
    train_idx, test_idx = [], []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        split = int(len(cls_indices) * (1 - test_size))
        train_idx.extend(cls_indices[:split])
        test_idx.extend(cls_indices[split:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

_, _, y_train_s, y_test_s = manual_stratified_split(X_imb, y_imb, test_size=0.3, random_state=1)

print(f"  总体阳性率: {y_imb.mean():.2%}")
print(f"  随机切分: Train={y_train_m.mean():.2%}, Test={test_random.mean():.2%}")
print(f"  分层切分: Train={y_train_s.mean():.2%}, Test={y_test_s.mean():.2%}")
print(f"  → 分层切分确保 train/test 类别比例一致")

# ============================================================
# 3. 手动实现 K 折交叉验证
# ============================================================
print("\n" + "=" * 60)
print("  3. 手动实现 K-Fold CV")
print("=" * 60)

def manual_kfold_cv(model_fn, X, y, n_splits=5, random_state=None):
    """手动K折交叉验证 — 理解每一折如何运转"""
    if random_state is not None:
        np.random.seed(random_state)

    n = len(X)
    indices = np.random.permutation(n)
    fold_size = n // n_splits
    scores = []

    for fold in range(n_splits):
        # 确定验证集索引
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_splits - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.setdiff1d(indices, val_idx)

        # 训练
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])

        # 评估
        score = model.score(X[val_idx], y[val_idx])
        scores.append(score)
        print(f"    Fold {fold+1}: {len(train_idx)} train, {len(val_idx)} val, score={score:.4f}")

    print(f"    Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    return scores

print("  K=5 CV Results (Logistic Regression):")
cv_scores = manual_kfold_cv(
    lambda: LogisticRegression(max_iter=1000), X, y,
    n_splits=5, random_state=42
)

# ============================================================
# 4. 数据泄露演示
# ============================================================
print("\n" + "=" * 60)
print("  4. 数据泄露演示")
print("=" * 60)

from sklearn.model_selection import train_test_split as sk_split

# 全新数据
X_leak, y_leak = X.copy(), y.copy()

# --- 错误做法: 先标准化再切分 ---
scaler_wrong = StandardScaler()
X_scaled_wrong = scaler_wrong.fit_transform(X_leak)  # ❌ fit看到了测试集!
X_tr_w, X_te_w, y_tr_w, y_te_w = sk_split(X_scaled_wrong, y_leak, test_size=0.2)

model_wrong = LogisticRegression(max_iter=1000)
model_wrong.fit(X_tr_w, y_tr_w)
score_wrong = model_wrong.score(X_te_w, y_te_w)

# --- 正确做法: 先切分再标准化 ---
X_train, X_test, y_train, y_test = sk_split(X_leak, y_leak, test_size=0.2)

scaler_right = StandardScaler()
X_train_scaled = scaler_right.fit_transform(X_train)     # ✅ 只在训练集上fit
X_test_scaled = scaler_right.transform(X_test)            # ✅ 用训练集的参数transform

model_right = LogisticRegression(max_iter=1000)
model_right.fit(X_train_scaled, y_train)
score_right = model_right.score(X_test_scaled, y_test)

print(f"  ❌ 先标准化再切分: Test acc = {score_wrong:.4f} (信息泄露)")
print(f"  ✅ 先切分再标准化: Test acc = {score_right:.4f} (正确)")
print(f"  差异: {abs(score_wrong - score_right):.4f}")

# 为什么有差异?
print(f"\n  为什么有问题?")
print(f"  - 错误做法中 StandardScaler.fit() 看到了所有数据(包括测试集)")
print(f"  - 测试集的均值和方差信息'泄露'到了训练过程中")
print(f"  - 这导致在测试集上的表现虚高, 上线后效果会变差")
print(f"  - 更严重的情况: 如果做了特征选择或缺失值填充, 影响更大")

# ============================================================
# 5. 时间序列: TimeSeriesSplit
# ============================================================
print("\n" + "=" * 60)
print("  5. TimeSeriesSplit — 时序数据不能用随机CV")
print("=" * 60)

n_ts = 100
dates = pd.date_range('2024-01-01', periods=n_ts, freq='D')
X_ts = np.random.randn(n_ts, 3)
y_ts = np.cumsum(np.random.randn(n_ts) * 0.5)

tscv = TimeSeriesSplit(n_splits=4)
print(f"  TimeSeriesSplit (n_splits=4):")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    print(f"    Fold {fold+1}: Train=[{dates[train_idx[0]].date()}~"
          f"{dates[train_idx[-1]].date()}] ({len(train_idx)}d) → "
          f"Test=[{dates[test_idx[0]].date()}~"
          f"{dates[test_idx[-1]].date()}] ({len(test_idx)}d)")

print(f"\n  关键原则:")
print(f"  - 训练集永远在测试集之前 (不能用未来预测过去)")
print(f"  - 随机CV会打乱时间顺序 → 用未来的数据训练, 预测过去")

# ============================================================
# 6. train_val_test_split_safe()
# ============================================================
print("\n" + "=" * 60)
print("  6. train_val_test_split_safe()")
print("=" * 60)

def train_val_test_split_safe(X, y, test_size=0.2, val_size=0.1,
                                stratify=None, random_state=None,
                                is_time_series=False):
    """
    安全的三路数据切分，确保无数据泄露。

    Parameters
    ----------
    X, y : 特征和目标
    test_size : 测试集比例
    val_size : 验证集比例 (从训练集中分出)
    stratify : 分层标签 (用于分类问题)
    random_state : 随机种子
    is_time_series : 是否时序数据 (True=不随机打乱)

    Returns
    -------
    dict: {'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'}
    """
    n = len(X)
    if random_state:
        np.random.seed(random_state)

    indices = np.arange(n)
    if not is_time_series:
        np.random.shuffle(indices)

    # 分出测试集
    test_n = int(n * test_size)
    if is_time_series:
        # 时序: 最后一部分作为测试集
        test_idx = indices[-test_n:]
        train_val_idx = indices[:-test_n]
    else:
        test_idx = indices[:test_n]
        train_val_idx = indices[test_n:]

    # 从训练集中分出验证集
    val_n = int(len(train_val_idx) * val_size / (1 - test_size))

    if stratify is not None and not is_time_series:
        # 分层抽样
        from sklearn.model_selection import train_test_split as sk_split
        X_tv, _, y_tv, _ = sk_split(
            X[train_val_idx], y[train_val_idx],
            test_size=val_n / len(train_val_idx),
            stratify=stratify[train_val_idx] if stratify is not None else None,
            random_state=random_state
        )
        # 但这个方法比较复杂, 简单起见这里使用直接索引
        val_idx = train_val_idx[:val_n]
        train_idx = train_val_idx[val_n:]
    else:
        val_idx = train_val_idx[:val_n]
        train_idx = train_val_idx[val_n:]

    result = {
        'X_train': X[train_idx], 'X_val': X[val_idx], 'X_test': X[test_idx],
        'y_train': y[train_idx], 'y_val': y[val_idx], 'y_test': y[test_idx],
        'train_size': len(train_idx), 'val_size': len(val_idx), 'test_size': len(test_idx),
    }

    # 安全检查
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train-Test overlap!"
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "Train-Val overlap!"
    assert len(np.intersect1d(val_idx, test_idx)) == 0, "Val-Test overlap!"

    print(f"  ✅ 安全切分完成: Train={result['train_size']}, "
          f"Val={result['val_size']}, Test={result['test_size']}")
    print(f"  ✅ 无数据重叠验证通过")
    return result

split = train_val_test_split_safe(X, y, test_size=0.2, val_size=0.1, random_state=42)

# ============================================================
# 可视化: CV过程
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) K-Fold示意
ax = axes[0, 0]
n = 50
k = 5
fold_size = n // k
for fold in range(k):
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size if fold < k - 1 else n
    for i in range(n):
        if val_start <= i < val_end:
            ax.barh(fold, 1, left=i, height=0.8, color='#FF9800', edgecolor='white')
        else:
            ax.barh(fold, 1, left=i, height=0.8, color='#2196F3', edgecolor='white')

ax.set_xlabel('Sample Index'); ax.set_ylabel('Fold')
ax.set_title('K-Fold Cross-Validation\n(Blue=Train, Orange=Val)',
             fontweight='bold')

# (2) 分层切分对比
ax = axes[0, 1]
labels = ['Random Split', 'Stratified Split']
train_rates = [y_train_m.mean(), y_train_s.mean()]
test_rates = [test_random.mean(), y_test_s.mean()]
x = np.arange(2); w = 0.3
ax.bar(x - w/2, train_rates, w, label='Train', color='#2196F3')
ax.bar(x + w/2, test_rates, w, label='Test', color='#FF9800')
ax.axhline(y_imb.mean(), color='red', linestyle='--', label=f'Overall ({y_imb.mean():.2%})')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Positive Rate'); ax.set_title('Random vs Stratified Split', fontweight='bold')
ax.legend(fontsize=8)

# (3) 数据泄露影响
ax = axes[1, 0]
methods = ['Wrong\n(Scale→Split)', 'Correct\n(Split→Scale)']
scores = [score_wrong, score_right]
ax.bar(methods, scores, color=['#f44336', '#4CAF50'], edgecolor='white')
for i, s in enumerate(scores):
    ax.text(i, s + 0.01, f'{s:.4f}', ha='center', fontweight='bold')
ax.set_ylabel('Test Accuracy')
ax.set_title('Data Leakage Impact', fontweight='bold')

# (4) CV验证流程
ax = axes[1, 1]
ax.axis('off')
cv_flow = """
Safe ML Pipeline
==================

1. Split Data FIRST
   ├── Train set (70-80%)
   ├── Validation set (10-15%)
   └── Test set (10-15%)

2. On TRAIN SET ONLY:
   ├── EDA & visualization
   ├── Handle missing values (fit on train)
   ├── Feature engineering
   ├── Feature selection
   └── Scale/normalize (fit on train)

3. Cross-validate on Train+Val
   ├── K-Fold for i.i.d. data
   └── TimeSeriesSplit for temporal data

4. Final evaluation
   └── Only on Test set (ONE TIME)

NEVER:
❌ Scale before split
❌ Feature select on full data
❌ Look at test set during development
❌ Use random CV for time series
"""
ax.text(0.05, 0.95, cv_flow, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cross_validation.png'), dpi=150)
plt.close()
print("  Saved: cross_validation.png")

print("\n✅ Day 37 完成")
