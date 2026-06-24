#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 38: 从零实现线性回归 — 正规方程 + 梯度下降 + Ridge
对比 sklearn，验证实现正确性
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# 生成数据
n_train, n_features = 200, 3
true_weights = np.array([2.5, -1.8, 1.2])
X_raw = np.random.randn(n_train, n_features)
y_raw = X_raw @ true_weights + np.random.randn(n_train) * 1.5 + 5.0  # 截距=5

# 加偏置列
X = np.c_[np.ones(n_train), X_raw]  # shape (n, d+1)

print("=" * 60)
print("  从零实现线性回归")
print("=" * 60)
print(f"  数据: {n_train} samples × {n_features} features")
print(f"  真实权重: bias=5.0, w={true_weights}")

# ============================================================
# 1. 正规方程 (Normal Equation)
# ============================================================
class LinearRegressionNormalEquation:
    """线性回归 — 解析解: w = (X^T X)^{-1} X^T y"""

    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.weights

    @property
    def intercept_(self):
        return self.weights[0]

    @property
    def coef_(self):
        return self.weights[1:]


# ============================================================
# 2. 梯度下降三种变体
# ============================================================
class LinearRegressionGD:
    """线性回归 — 梯度下降 (支持 Batch / SGD / Mini-batch)"""

    def __init__(self, lr=0.01, n_iters=500, batch_size=None, l2_lambda=0):
        """
        Args:
            lr: 学习率
            n_iters: 迭代次数
            batch_size: None=Batch GD, 1=SGD, >1=Mini-batch GD
            l2_lambda: L2正则化系数 (0=无正则化)
        """
        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        self.weights = np.random.randn(d) * 0.01
        self.loss_history = []

        for _ in range(self.n_iters):
            if self.batch_size is None:
                # Batch GD: 全量数据
                X_batch, y_batch = X, y
            else:
                # Mini-batch / SGD
                idx = np.random.choice(n, self.batch_size, replace=False)
                X_batch, y_batch = X[idx], y[idx]

            # 梯度计算
            y_pred = X_batch @ self.weights
            error = y_pred - y_batch
            grad = (X_batch.T @ error) / len(y_batch)

            # L2 正则化梯度 (不对 bias 做正则化)
            if self.l2_lambda > 0:
                grad[1:] += self.l2_lambda * self.weights[1:]

            # 更新
            self.weights -= self.lr * grad

            # 记录全量 loss
            full_pred = X @ self.weights
            mse = np.mean((full_pred - y) ** 2)
            if self.l2_lambda > 0:
                mse += self.l2_lambda * np.sum(self.weights[1:] ** 2)
            self.loss_history.append(mse)

        return self

    def predict(self, X):
        return X @ self.weights

    @property
    def intercept_(self):
        return self.weights[0]

    @property
    def coef_(self):
        return self.weights[1:]


# ============================================================
# 训练并对比
# ============================================================
print("\n" + "=" * 60)
print("  模型对比")
print("=" * 60)

# sklearn
sk_model = LinearRegression().fit(X_raw, y_raw)

# 正规方程
ne_model = LinearRegressionNormalEquation().fit(X, y_raw)

# Batch GD
bgd_model = LinearRegressionGD(lr=0.01, n_iters=500).fit(X, y_raw)

# Mini-batch GD
mbgd_model = LinearRegressionGD(lr=0.01, n_iters=500, batch_size=32).fit(X, y_raw)

# SGD
sgd_model = LinearRegressionGD(lr=0.005, n_iters=500, batch_size=1).fit(X, y_raw)

models = {
    'sklearn': sk_model,
    'Normal Eq': ne_model,
    'Batch GD': bgd_model,
    'Mini-Batch GD(32)': mbgd_model,
    'SGD': sgd_model,
}

for name, m in models.items():
    if name == 'sklearn':
        y_pred = m.predict(X_raw)
        coef = m.coef_
        intercept = m.intercept_
    else:
        y_pred = m.predict(X)
        coef = m.coef_
        intercept = m.intercept_
    r2 = r2_score(y_raw, y_pred)
    print(f"  {name:20s}: R²={r2:.4f}, w={np.round(coef, 3)}, bias={intercept:.3f}")

# ============================================================
# 3. 损失下降曲线对比
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 损失曲线对比
ax = axes[0, 0]
for name, model in [('Batch GD', bgd_model), ('Mini-Batch(32)', mbgd_model), ('SGD', sgd_model)]:
    ax.plot(model.loss_history, alpha=0.8, linewidth=1.5, label=name)
ax.set_xlabel('Iteration'); ax.set_ylabel('MSE Loss')
ax.set_title('Loss Convergence: Batch vs Mini-Batch vs SGD', fontweight='bold')
ax.legend()
ax.set_yscale('log')

# (2) 预测 vs 真实
ax = axes[0, 1]
y_pred = bgd_model.predict(X)
ax.scatter(y_raw, y_pred, alpha=0.5, c='#2196F3', edgecolors='white')
ax.plot([y_raw.min(), y_raw.max()], [y_raw.min(), y_raw.max()], 'r--', linewidth=2)
ax.set_xlabel('True y'); ax.set_ylabel('Predicted y')
ax.set_title(f'Predictions vs True (R²={r2_score(y_raw, y_pred):.4f})', fontweight='bold')

# (3) 权重对比
ax = axes[1, 0]
n_weights_total = n_features + 1  # bias + weights
x_pos = np.arange(n_weights_total)
w = 0.2
for i, (name, m) in enumerate([('sklearn', sk_model),
                                 ('Normal Eq', ne_model),
                                 ('Batch GD', bgd_model)]):
    if name == 'sklearn':
        weights_plot = np.concatenate([[m.intercept_], m.coef_])
    else:
        weights_plot = m.weights
    ax.bar(x_pos + i * w - w, weights_plot, w, label=name, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(['bias'] + [f'w{i}' for i in range(n_features)])
ax.set_title('Weight Comparison: sklearn vs Our Implementation', fontweight='bold')
ax.legend(fontsize=8)

# (4) 学习率影响
ax = axes[1, 1]
for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
    m = LinearRegressionGD(lr=lr, n_iters=300).fit(X, y_raw)
    ax.plot(m.loss_history, alpha=0.7, linewidth=1, label=f'lr={lr}')
ax.set_xlabel('Iteration'); ax.set_ylabel('MSE Loss')
ax.set_title('Effect of Learning Rate', fontweight='bold')
ax.legend(fontsize=7)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'linear_regression.png'), dpi=150)
plt.close()
print("  Saved: linear_regression.png")

# ============================================================
# 6. Ridge 回归 (L2)
# ============================================================
print("\n" + "=" * 60)
print("  6. Ridge 回归 (L2 正则化)")
print("=" * 60)

# 生成共线性数据 (Ridge 的优势场景)
n_samples_col, n_col = 50, 30
X_base = np.random.randn(n_samples_col, 5) * 2
X_collinear = np.zeros((n_samples_col, n_col))
X_collinear[:, :5] = X_base
# 其余列是前5列的线性组合+噪声 (=高度共线)
for j in range(5, n_col):
    X_collinear[:, j] = X_base[:, j % 5] * (0.5 + np.random.random()) + np.random.randn(n_samples_col) * 0.3
y_collinear = X_base[:, :3].sum(axis=1) + np.random.randn(n_samples_col) * 0.8

from sklearn.model_selection import train_test_split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_collinear, y_collinear, test_size=0.3, random_state=42
)

# sklearn Ridge
sk_ridge = Ridge(alpha=1.0).fit(Xc_train, yc_train)
sk_lr = LinearRegression().fit(Xc_train, yc_train)

# 我们的 Ridge (用 Batch GD + L2)
our_ridge = LinearRegressionGD(lr=0.01, n_iters=1000, l2_lambda=1.0).fit(
    np.c_[np.ones(len(Xc_train)), Xc_train], yc_train
)
our_lr = LinearRegressionGD(lr=0.01, n_iters=1000).fit(
    np.c_[np.ones(len(Xc_train)), Xc_train], yc_train
)

print(f"  {n_col}个特征 (存在共线性):")
print(f"    sklearn LR:     Train R²={sk_lr.score(Xc_train, yc_train):.4f}, "
      f"Test R²={sk_lr.score(Xc_test, yc_test):.4f}")
print(f"    sklearn Ridge:  Train R²={sk_ridge.score(Xc_train, yc_train):.4f}, "
      f"Test R²={sk_ridge.score(Xc_test, yc_test):.4f}")
print(f"  → Ridge 的 Test R² 更高: L2 正则化缓解了过拟合")

# 不同 lambda 的影响
lambdas = [0, 0.01, 0.1, 1, 10, 100]
ridge_coefs = []
for lam in lambdas:
    m = LinearRegressionGD(lr=0.01, n_iters=1000, l2_lambda=lam).fit(
        np.c_[np.ones(len(Xc_train)), Xc_train], yc_train
    )
    ridge_coefs.append(m.coef_)

# 可视化 Ridge 路径
fig, ax = plt.subplots(figsize=(8, 4))
for j in range(min(n_col, 20)):
    ax.plot(lambdas, [c[j] for c in ridge_coefs], 'o-', markersize=3, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Lambda (L2 penalty)')
ax.set_ylabel('Coefficient Value')
ax.set_title('Ridge Regularization Path\n(Larger lambda → coefficients shrink to 0)',
             fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ridge_path.png'), dpi=150)
plt.close()
print("  Saved: ridge_path.png")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("  面试速查: 线性回归核心公式")
print("=" * 60)
print("""
  损失函数 (MSE): J(w) = (1/n) * sum((y - Xw)^2)

  梯度: ∇J = -(2/n) * X^T (y - Xw)

  正规方程: w = (X^T X)^{-1} X^T y
    ✅ 一步到位, 无需选学习率
    ❌ O(d³) 计算量, d>10K 时不可行

  Batch GD:  每次用全部样本 → 稳定但慢
  SGD:       每次用1个样本 → 快但震荡
  Mini-batch: 每次用32-256个 → 最佳平衡

  Ridge (L2): J(w) = MSE + lambda * ||w||²
    → 惩罚大权重, 防止过拟合, 处理共线性
""")

print("✅ Day 38 完成")
