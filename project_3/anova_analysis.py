#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 33: ANOVA 方差分析与事后比较
单因素ANOVA + Tukey HSD + 效应量 eta-squared
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)
ALPHA = 0.05

# ============================================================
# 1. 生成4组模拟数据: 优惠券策略
# ============================================================
print("=" * 60)
print("  ANOVA: 4种优惠券策略对客单价的影响")
print("=" * 60)

n_per_group = 80
groups = {
    'Control (无券)':    np.random.normal(200, 40, n_per_group),
    '10%折扣':           np.random.normal(215, 45, n_per_group),
    '满200减30':         np.random.normal(230, 42, n_per_group),
    '买二送一':           np.random.normal(220, 50, n_per_group),
}

for name, data in groups.items():
    print(f"  {name:20s}: mean={data.mean():.1f}, std={data.std():.1f}, n={len(data)}")

# ============================================================
# 2. 单因素ANOVA
# ============================================================
print("\n" + "=" * 60)
print("  2. 单因素 ANOVA")
print("=" * 60)

group_data = list(groups.values())
group_names = list(groups.keys())
f_stat, p_anova = stats.f_oneway(*group_data)

# 效应量 eta-squared
# eta^2 = SS_between / SS_total
all_data = np.concatenate(group_data)
grand_mean = all_data.mean()
ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
ss_total = sum((all_data - grand_mean)**2)
eta_sq = ss_between / ss_total

print(f"  H0: 所有组的均值相等")
print(f"  H1: 至少有一组不同")
print(f"  F = {f_stat:.4f}, p = {p_anova:.4f}")
print(f"  eta^2 = {eta_sq:.4f} (效应量)")

# eta^2 解读
if eta_sq < 0.01:
    eta_label = '可忽略'
elif eta_sq < 0.06:
    eta_label = '小效应'
elif eta_sq < 0.14:
    eta_label = '中等效应'
else:
    eta_label = '大效应'
print(f"  → {eta_label}")

if p_anova < ALPHA:
    print(f"  ✅ 拒绝H0: 至少有一组显著不同")
else:
    print(f"  ❌ 不能拒绝H0")

# ============================================================
# 3. 前提检验
# ============================================================
print("\n" + "=" * 60)
print("  3. ANOVA 前提检验")
print("=" * 60)

# 正态性 (各组单独检验)
print("  正态性 (Shapiro-Wilk):")
all_normal = True
for name, data in groups.items():
    _, sw_p = stats.shapiro(data[:500])
    ok = sw_p > 0.05
    if not ok: all_normal = False
    print(f"    {name:20s}: p={sw_p:.4f} {'✅' if ok else '⚠️'}")

# 方差齐性 (Levene)
_, levene_p = stats.levene(*group_data)
var_equal = levene_p > 0.05
print(f"\n  方差齐性 (Levene): p={levene_p:.4f} {'✅' if var_equal else '⚠️'}")

if not all_normal or not var_equal:
    print(f"\n  ⚠️ ANOVA假设不完全满足")
    # Kruskal-Wallis fallback
    h_kw, p_kw = stats.kruskal(*group_data)
    print(f"  → 使用 Kruskal-Wallis 验证: H={h_kw:.2f}, p={p_kw:.4f}")

# ============================================================
# 4. 事后多重比较: Tukey HSD
# ============================================================
print("\n" + "=" * 60)
print("  4. Tukey HSD 事后比较")
print("=" * 60)

# 构造长格式数据
df_long = pd.DataFrame({
    'value': np.concatenate(group_data),
    'group': np.concatenate([[name]*len(data) for name, data in groups.items()]),
})

tukey = pairwise_tukeyhsd(df_long['value'], df_long['group'], alpha=ALPHA)
print(tukey)

# 简洁版：列出每组的显著差异
print(f"\n  组间显著差异汇总:")
reject_matrix = np.zeros((4, 4), dtype=bool)
for i in range(4):
    for j in range(4):
        if i != j:
            _, p_val = stats.ttest_ind(group_data[i], group_data[j], equal_var=False)
            reject_matrix[i, j] = p_val < ALPHA

for i, name in enumerate(group_names):
    diffs = []
    for j, other in enumerate(group_names):
        if i != j and reject_matrix[i, j]:
            diffs.append(f"{other} (p<0.05)")
    sig_str = ', '.join(diffs) if diffs else '无显著差异'
    print(f"  {name:20s}: 显著不同于 → {sig_str}")


# ============================================================
# 5. 效应量总结
# ============================================================
print("\n" + "=" * 60)
print("  5. 效应量 eta^2")
print("=" * 60)
print(f"  eta^2 = {eta_sq:.4f}")
print(f"  解释: 优惠券策略解释了客单价变异的 {eta_sq*100:.1f}%")
print(f"  Cohen's guideline:")
print(f"    0.01 = small, 0.06 = medium, 0.14 = large")


# ============================================================
# 6. 可视化
# ============================================================
print("\n" + "=" * 60)
print("  6. 可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左: 箱线图
ax = axes[0]
positions = range(1, 5)
bp = ax.boxplot(group_data, positions=positions, widths=0.5,
                patch_artist=True)
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 添加显著性字母
y_max = max(max(g) for g in group_data)
# 计算字母
letters = {}
for i, name_i in enumerate(group_names):
    letters[name_i] = set()
    for j, name_j in enumerate(group_names):
        if i == j or not reject_matrix[i, j]:
            letters[name_i].add(chr(65 + i))  # 简化：同组共享字母

ax.set_xticklabels(group_names, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('Average Order Value')
ax.set_title(f'One-way ANOVA: Coupon Strategy Effect\n'
             f'F={f_stat:.2f}, p={p_anova:.4f}, eta^2={eta_sq:.3f}',
             fontweight='bold')

# 标注均值
for i, (name, data) in enumerate(groups.items()):
    ax.text(i+1, y_max*0.95, f'mean={data.mean():.0f}',
            ha='center', fontsize=8, fontweight='bold')

# 右: 均值 + 置信区间
ax = axes[1]
means = [g.mean() for g in group_data]
ses = [g.std()/np.sqrt(len(g)) for g in group_data]
cis = [stats.t.interval(0.95, len(g)-1, loc=m, scale=s)
       for g, m, s in zip(group_data, means, ses)]
x = np.arange(4)
ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white')
ax.errorbar(x, means, yerr=[[m - ci[0] for m, ci in zip(means, cis)],
                              [ci[1] - m for m, ci in zip(means, cis)]],
            fmt='none', color='black', capsize=8, linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels([n[:12] for n in group_names], rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Mean AOV')
ax.set_title('Group Means with 95% CI', fontweight='bold')
ax.axhline(grand_mean, color='red', linestyle='--', alpha=0.5,
           label=f'Grand Mean={grand_mean:.0f}')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'anova_results.png'), dpi=150)
plt.close()
print("  Saved: anova_results.png")


# ============================================================
# 7. 为什么ANOVA而不是多次t检验?
# ============================================================
print("\n" + "=" * 60)
print("  7. 为什么 ANOVA 而不是 6 次 t 检验?")
print("=" * 60)

explanation = """
4组比较: 需要 C(4,2) = 6 次两两 t 检验

如果每次 α=0.05:
  单次I类错误率:     5%
  6次至少1个I类错误:  1-(1-0.05)^6 ≈ 26.5%
  → 26.5% 概率至少发现1个假阳性!

ANOVA + Tukey HSD:
  整体I类错误率:      5% (受控)
  Tukey HSD 自动校正多重比较

结论:
  ✅ 多组比较 → 先ANOVA (整体检验) → 再Tukey HSD (事后两两比较)
  ❌ 直接跑多次t检验 → I类错误膨胀, 不可靠
"""
print(explanation)

print("✅ Day 33 完成")
