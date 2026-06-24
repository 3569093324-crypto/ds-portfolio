#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 25: 描述统计与概率分布
5种核心分布 + Q-Q图 + 正态性检验 + 偏度峰度
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 1 & 2. 模拟并可视化 5 种分布
# ============================================================
print("=" * 60)
print("  1 & 2. 5种概率分布 — 直方图 + 理论PDF")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

distributions = [
    {
        'name': 'Normal(μ=0, σ=1)',
        'data': np.random.normal(0, 1, 10000),
        'x_range': np.linspace(-4, 4, 200),
        'pdf': lambda x: stats.norm.pdf(x, 0, 1),
        'color': '#2196F3',
        'use_case': '身高、测量误差、自然现象',
    },
    {
        'name': 'Binomial(n=20, p=0.3)',
        'data': np.random.binomial(20, 0.3, 10000),
        'x_range': np.arange(0, 21),
        'pdf': lambda x: stats.binom.pmf(x, 20, 0.3),
        'color': '#4CAF50',
        'use_case': '转化率、点击率、A/B测试',
    },
    {
        'name': 'Poisson(λ=4)',
        'data': np.random.poisson(4, 10000),
        'x_range': np.arange(0, 16),
        'pdf': lambda x: stats.poisson.pmf(x, 4),
        'color': '#FF9800',
        'use_case': '网站访问量、客服来电数',
    },
    {
        'name': 'Exponential(β=2)',
        'data': np.random.exponential(2, 10000),
        'x_range': np.linspace(0, 15, 200),
        'pdf': lambda x: stats.expon.pdf(x, scale=2),
        'color': '#E91E63',
        'use_case': '等待时间、故障间隔',
    },
    {
        'name': 'Uniform(0, 10)',
        'data': np.random.uniform(0, 10, 10000),
        'x_range': np.linspace(-1, 11, 200),
        'pdf': lambda x: stats.uniform.pdf(x, 0, 10),
        'color': '#9C27B0',
        'use_case': '随机抽样、蒙特卡洛模拟',
    },
]

for i, dist in enumerate(distributions):
    ax = axes[i]
    data = dist['data']

    # 直方图
    ax.hist(data, bins=40, density=True, alpha=0.6, color=dist['color'],
            edgecolor='white', linewidth=0.3)

    # 理论PDF/PMF
    x = dist['x_range']
    if dist['name'].startswith('Binomial'):
        # PMF 用stem图更合适
        y = dist['pdf'](x)
        ax.stem(x, y, linefmt=f'{dist["color"]}', markerfmt='o', basefmt=' ',
                )
    else:
        y = dist['pdf'](x)
        ax.plot(x, y, 'k-', linewidth=2, label='Theoretical PDF')

    # 标注均值线
    mean_val = data.mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean={mean_val:.2f}')
    ax.axvline(mean_val + data.std(), color='orange', linestyle=':', linewidth=1)
    ax.axvline(mean_val - data.std(), color='orange', linestyle=':', linewidth=1,
               label=f'±1σ')

    ax.set_title(dist['name'], fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, loc='upper right')

# 第6格: 总览
ax = axes[5]
ax.axis('off')
summary_text = "\n".join([
    f"• {d['name'].split('(')[0]}: {d['use_case']}"
    for d in distributions
])
ax.text(0.1, 0.5, "应用场景:\n\n" + summary_text,
        fontsize=8, verticalalignment='center',
        fontfamily='monospace')
ax.set_title('Distribution Use Cases', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'distributions_overview.png'), dpi=150)
plt.close()
print("  Saved: distributions_overview.png")


# ============================================================
# 3. 真实数据正态性检验 (Q-Q图 + Shapiro-Wilk)
# ============================================================
print("\n" + "=" * 60)
print("  3. 真实数据正态性检验")
print("=" * 60)

# 从 business.db 读取数据
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "business.db")
conn = sqlite3.connect(DB_PATH)

# 取订单金额数据
orders = pd.read_sql("SELECT total_amount FROM orders", conn)
amounts = orders['total_amount'].values
conn.close()

print(f"  数据: orders.total_amount (n={len(amounts)})")
print(f"  均值={amounts.mean():.2f}, 标准差={amounts.std():.2f}")

# Q-Q图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 左: 原始数据的Q-Q图
stats.probplot(amounts, dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot: Order Amounts vs Normal', fontweight='bold')
axes[0].get_lines()[0].set_markerfacecolor('#2196F3')
axes[0].get_lines()[0].set_markeredgecolor('#2196F3')
axes[0].get_lines()[1].set_color('red')

# 右: 直方图 + 正态拟合
ax = axes[1]
ax.hist(amounts, bins=30, density=True, alpha=0.6, color='#4CAF50', edgecolor='white')
x = np.linspace(amounts.min(), amounts.max(), 200)
mu, sigma = amounts.mean(), amounts.std()
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'N({mu:.0f}, {sigma:.0f})')
ax.set_title('Histogram + Normal Fit', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'normality_test.png'), dpi=150)
plt.close()
print("  Saved: normality_test.png")

# Shapiro-Wilk 检验
stat, p_value = stats.shapiro(amounts[:500])  # Shapiro 限制最大5000样本
print(f"\n  Shapiro-Wilk test:")
print(f"    W-statistic = {stat:.4f}")
print(f"    p-value     = {p_value:.4f}")
if p_value < 0.05:
    print(f"    ❌ 拒绝H0: 数据不服从正态分布 (p={p_value:.4f} < 0.05)")
else:
    print(f"    ✅ 不能拒绝H0: 数据可能服从正态分布")

# 额外检验: D'Agostino's K² test
k2_stat, k2_p = stats.normaltest(amounts)
print(f"\n  D'Agostino K² test:")
print(f"    K² = {k2_stat:.4f}, p = {k2_p:.4f}")
print(f"    {'❌ 非正态' if k2_p < 0.05 else '✅ 可能正态'}")


# ============================================================
# 4. 偏度 (Skewness) 和 峰度 (Kurtosis)
# ============================================================
print("\n" + "=" * 60)
print("  4. 偏度 & 峰度")
print("=" * 60)

# 生成不同偏度和峰度的数据来说明
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

examples = [
    ('Right Skewed (Skew > 0)', np.random.exponential(2, 5000), '#FF9800',
     '长尾在右侧\n均值 > 中位数'),
    ('Symmetric (Skew ≈ 0)', np.random.normal(0, 1, 5000), '#4CAF50',
     '左右对称\n均值 ≈ 中位数'),
    ('Left Skewed (Skew < 0)', -np.random.exponential(2, 5000) + 10, '#2196F3',
     '长尾在左侧\n均值 < 中位数'),
]

for ax, (title, data, color, note) in zip(axes, examples):
    ax.hist(data, bins=40, density=True, alpha=0.7, color=color, edgecolor='white')
    s = stats.skew(data)
    k = stats.kurtosis(data)  # excess kurtosis (Fisher)
    ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(data):.2f}')
    ax.axvline(np.median(data), color='blue', linestyle='-', linewidth=2, label=f'Median={np.median(data):.2f}')
    ax.set_title(f'{title}\nSkew={s:.2f}, Kurt={k:.2f}', fontweight='bold')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'skewness_kurtosis.png'), dpi=150)
plt.close()
print("  Saved: skewness_kurtosis.png")

# 对真实数据计算
print(f"\n  真实数据 (order amounts):")
print(f"    Skewness = {stats.skew(amounts):.4f}")
print(f"    Kurtosis = {stats.kurtosis(amounts):.4f} (excess)")
print(f"    解读: ", end='')
s = stats.skew(amounts)
k = stats.kurtosis(amounts)
if s > 0.5:
    skew_desc = "右偏分布(长尾在右)，少数大额订单拉高了均值"
elif s < -0.5:
    skew_desc = "左偏分布(长尾在左)"
else:
    skew_desc = "近似对称分布"
if k > 1:
    kurt_desc = "厚尾分布(比正态分布有更多极端值)"
elif k < -1:
    kurt_desc = "薄尾分布(极端值较少)"
else:
    kurt_desc = "峰度接近正态分布"
print(f"{skew_desc}, {kurt_desc}")


# ============================================================
# 5. check_distribution() 函数
# ============================================================
print("\n" + "=" * 60)
print("  5. check_distribution() — 自动判断分布类型")
print("=" * 60)

def check_distribution(data, verbose=True):
    """
    自动判断数据大致符合哪种概率分布。

    通过比较数据的偏度、峰度、离散特征和拟合优度，
    给出最可能的分布类型及置信度评估。

    Args:
        data: array-like, 输入数据
        verbose: 是否打印详细结果

    Returns:
        dict: {
            'best_fit': 最匹配的分布名,
            'normal_pvalue': Shapiro-Wilk p值,
            'skewness': 偏度,
            'kurtosis': 峰度 (excess),
            'is_discrete': 是否为离散分布,
            'recommendation': 统计方法建议,
        }
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]  # 去除NaN和Inf
    n = len(data)

    if n < 8:
        return {'best_fit': 'insufficient_data', 'recommendation': '样本量太小,无法判断'}

    # 基本统计量
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)  # excess kurtosis
    mean_val = np.mean(data)
    var_val = np.var(data)
    unique_ratio = len(np.unique(data)) / n

    # 离散性判断
    is_discrete = (unique_ratio < 0.1 and n > 30)

    # 正态性检验
    if n > 5000:
        # Shapiro 只支持最多5000
        sample = np.random.choice(data, 5000, replace=False)
        _, normal_p = stats.shapiro(sample)
    else:
        _, normal_p = stats.shapiro(data)

    # 判断逻辑
    candidates = []

    # 正态分布: 偏度≈0, 峰度≈0
    if abs(skew) < 0.5 and abs(kurt) < 1.0 and normal_p > 0.01:
        candidates.append(('Normal', 0.9))

    # 指数分布: 右偏, 方差≈均值²
    if skew > 0.5 and abs(var_val / (mean_val ** 2) - 1) < 0.3 and np.min(data) >= 0:
        candidates.append(('Exponential', 0.7))

    # 泊松分布: 离散, 方差≈均值
    if is_discrete and abs(var_val / mean_val - 1) < 0.3 and np.min(data) >= 0:
        candidates.append(('Poisson', 0.65))

    # 均匀分布: 偏度≈0, 峰度≈-1.2
    if abs(skew) < 0.3 and kurt < -0.8 and kurt > -1.5:
        candidates.append(('Uniform', 0.6))

    # 对数正态: 强右偏, 正值
    if skew > 1.0 and np.min(data) > 0:
        candidates.append(('Log-normal', 0.55))

    # 二项分布: 离散, 方差<均值
    if is_discrete and var_val < mean_val:
        candidates.append(('Binomial', 0.5))

    # 选最佳候选
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_fit = candidates[0][0]
    else:
        best_fit = 'unknown'

    # 推荐后续方法
    if normal_p > 0.05 or best_fit == 'Normal':
        recommendation = '可使用参数检验(t-test, ANOVA, Pearson相关)'
    else:
        recommendation = '建议使用非参数检验(Mann-Whitney, Spearman相关)或对数据做log变换'

    result = {
        'best_fit': best_fit,
        'normal_pvalue': normal_p,
        'skewness': round(skew, 4),
        'kurtosis': round(kurt, 4),
        'is_discrete': is_discrete,
        'recommendation': recommendation,
        'n': n,
    }

    if verbose:
        print(f"\n  📊 分布检测结果 (n={n})")
        print(f"  偏度={skew:.4f}, 峰度(excess)={kurt:.4f}")
        print(f"  Shapiro-Wilk p={normal_p:.4f}")
        print(f"  离散分布: {'是' if is_discrete else '否'}")
        print(f"  最佳匹配: {best_fit}")
        print(f"  建议: {recommendation}")
        if candidates:
            print(f"  候选分布: {[(c[0], f'{c[1]:.0%}') for c in candidates[:3]]}")

    return result


# 测试函数
print("\n  测试 check_distribution():")
for label, test_data in [
    ('Normal', np.random.normal(100, 15, 2000)),
    ('Exponential', np.random.exponential(5, 2000)),
    ('Poisson', np.random.poisson(8, 2000)),
    ('Uniform', np.random.uniform(0, 100, 2000)),
    ('Real order amounts', amounts),
]:
    print(f"\n  --- {label} ---")
    r = check_distribution(test_data)


# ============================================================
# 6. 非正态分布的影响
# ============================================================
print("\n" + "=" * 60)
print("  6. 非正态分布对后续分析的影响")
print("=" * 60)
print("""
  如果数据不是正态分布:

  影响1: t检验不可靠
    → t检验假设数据来自正态分布。非正态时改用 Mann-Whitney U (非参数)。

  影响2: 均值不是好的中心度量
    → 右偏分布中均值被极端值拉高，中位数更能代表"典型值"。

  影响3: 线性回归的残差假设
    → 回归假设残差正态。非正态时考虑:
      1) 对目标变量做 log/sqrt/Box-Cox 变换
      2) 使用稳健回归 (Huber, Quantile)
      3) 使用非参数方法

  影响4: 置信区间可能不准确
    → Bootstrap 方法不依赖分布假设，适用于任意分布。

  影响5: 中心极限定理 (CLT) 的救场
    → 当 n > 30 时，样本均值的分布趋近正态。
      这意味着即使原始数据不服从正态分布，
      大样本下 t 检验仍然近似有效。
      "统计学的免费午餐" — CLT 拯救你的假设检验。
""")


print("=" * 60)
print("  Day 25 完成！")
print("=" * 60)
