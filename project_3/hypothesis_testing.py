#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 28: 假设检验实战 — t检验 + 卡方检验
A/B测试的统计学基础
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)
ALPHA = 0.05

# ============================================================
# 场景模拟
# ============================================================
control = np.clip(np.random.normal(200, 50, 200), 0, None)
treatment = np.clip(np.random.normal(215, 55, 200), 0, None)

print("=" * 60)
print("  假设检验实战: A/B测试模拟")
print("=" * 60)
print(f"  对照组 A (原页面): n={len(control)}, mean={control.mean():.1f}")
print(f"  实验组 B (新版面): n={len(treatment)}, mean={treatment.mean():.1f}")
print(f"  均值差异: {treatment.mean() - control.mean():.1f}")

# ============================================================
# 1. 单样本 t 检验
# ============================================================
print("\n" + "=" * 60)
print("  1. 单样本 t 检验")
print("=" * 60)

target_value = 200
t_stat_1s, p_value_1s = stats.ttest_1samp(control, target_value)
ci_1s = stats.t.interval(0.95, df=len(control)-1, loc=control.mean(), scale=stats.sem(control))

print(f"  H0: mu = {target_value}")
print(f"  H1: mu != {target_value}")
print(f"  t = {t_stat_1s:.4f}, p = {p_value_1s:.4f}")
print(f"  95% CI: [{ci_1s[0]:.1f}, {ci_1s[1]:.1f}]")
print(f"  {'✅ 拒绝H0' if p_value_1s < ALPHA else '❌ 不能拒绝H0'}")

# ============================================================
# 2. 独立双样本 t 检验 (Welch)
# ============================================================
print("\n" + "=" * 60)
print("  2. 独立双样本 t 检验 (Welch)")
print("=" * 60)

t_stat_2s, p_value_2s = stats.ttest_ind(treatment, control, equal_var=False)
pooled_std = np.sqrt((treatment.var(ddof=1) + control.var(ddof=1)) / 2)
cohens_d = (treatment.mean() - control.mean()) / pooled_std

# 效应量标签
d_abs = abs(cohens_d)
if d_abs < 0.2:
    d_label = '可忽略的效应'
elif d_abs < 0.5:
    d_label = '小效应'
elif d_abs < 0.8:
    d_label = '中等效应'
else:
    d_label = '大效应'

print(f"  H0: mu_B - mu_A = 0")
print(f"  H1: mu_B - mu_A != 0")
print(f"  t = {t_stat_2s:.4f}, p = {p_value_2s:.4f}")
print(f"  Cohen's d = {cohens_d:.3f} ({d_label})")

if p_value_2s < ALPHA:
    uplift = (treatment.mean() - control.mean()) / control.mean() * 100
    print(f"  ✅ 拒绝H0: 新版面显著提升 {uplift:.1f}%")
else:
    print(f"  ❌ 不能拒绝H0")


# ============================================================
# 3. 配对 t 检验
# ============================================================
print("\n" + "=" * 60)
print("  3. 配对 t 检验 — 同一用户前后对比")
print("=" * 60)

n_paired = 50
before = np.random.normal(180, 40, n_paired)
after = np.clip(before + np.random.normal(25, 30, n_paired), 0, None)
diff = after - before

t_stat_paired, p_value_paired = stats.ttest_rel(after, before)

print(f"  H0: mu_diff = 0 (无变化)")
print(f"  H1: mu_diff != 0 (有变化)")
print(f"  使用前均值: {before.mean():.1f}")
print(f"  使用后均值: {after.mean():.1f}")
print(f"  平均变化: {diff.mean():.1f}")
print(f"  t = {t_stat_paired:.4f}, p = {p_value_paired:.4f}")
print(f"  {'✅ 拒绝H0' if p_value_paired < ALPHA else '❌ 不能拒绝H0'}")


# ============================================================
# 4. 卡方检验
# ============================================================
print("\n" + "=" * 60)
print("  4. 卡方检验 — 用户等级 × 购买行为")
print("=" * 60)

observed = np.array([
    [35, 20, 28, 8],    # 买过高价
    [8,  12, 11, 28],   # 没买过
])
chi2, p_chi2, dof, expected = stats.chi2_contingency(observed)

print(f"  H0: 等级与购买行为独立")
print(f"  H1: 等级与购买行为不独立")
print(f"  chi2 = {chi2:.4f}, df = {dof}, p = {p_chi2:.6f}")
print(f"  {'✅ 拒绝H0: 显著相关' if p_chi2 < ALPHA else '❌ 不能拒绝H0'}")

# 标准化残差
residuals = (observed - expected) / np.sqrt(expected)
print(f"  标准化残差 (|z|>2 = 显著偏离):")
for label, row in zip(['买过高价', '未买过'], residuals):
    print(f"    {label}: {np.round(row, 2)}")


# ============================================================
# 5. 汇总
# ============================================================
print("\n" + "=" * 60)
print("  5. 检验结果汇总")
print("=" * 60)
results = [
    ('单样本t', f'mu={target_value}', p_value_1s, p_value_1s < ALPHA),
    ('独立双样本t', 'mu_B-mu_A=0', p_value_2s, p_value_2s < ALPHA),
    ('配对t', 'mu_diff=0', p_value_paired, p_value_paired < ALPHA),
    ('卡方', '等级与购买独立', p_chi2, p_chi2 < ALPHA),
]
for name, h0, p, sig in results:
    print(f"  {name:15s} | H0: {h0:20s} | p={p:.6f} | {'SIGNIFICANT' if sig else 'NOT SIG'}")


# ============================================================
# 6. 可视化
# ============================================================
print("\n" + "=" * 60)
print("  6. 可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 两组对比
ax = axes[0, 0]
bp = ax.boxplot([control, treatment], positions=[1, 2], widths=0.4, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#2196F3', '#FF9800']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(['Control (A)', 'Treatment (B)'])
ax.set_ylabel('Spending Amount')
ax.set_title(f'Two-Sample t-test\np={p_value_2s:.4f} '
             f'({"SIGNIFICANT" if p_value_2s<ALPHA else "NOT SIG"})',
             fontweight='bold')

# 右上: 配对前后
ax = axes[0, 1]
for i in range(min(20, n_paired)):
    ax.plot([0, 1], [before[i], after[i]], 'o-', color='gray', alpha=0.4, markersize=3)
bp2 = ax.boxplot([before, after], positions=[0, 1], widths=0.3, patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('#4CAF50')
    patch.set_alpha(0.7)
ax.set_xticklabels(['Before', 'After'])
ax.set_ylabel('Spending')
ax.set_title(f'Paired t-test\np={p_value_paired:.4f}', fontweight='bold')

# 左下: 卡方
ax = axes[1, 0]
cats = ['VIP', 'Retail', 'Wholesale', 'New']
x = np.arange(len(cats))
w = 0.35
ax.bar(x - w/2, observed[0], w, label='Bought High-Value', color='#E91E63', alpha=0.8)
ax.bar(x + w/2, observed[1], w, label='Did Not Buy', color='#9E9E9E', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel('Count')
ax.set_title(f'Chi-squared: chi2={chi2:.1f}, p={p_chi2:.4f}', fontweight='bold')
ax.legend(fontsize=8)

# 右下: p值分布 (H0下)
ax = axes[1, 1]
p_under_h0 = np.random.uniform(0, 1, 1000)
ax.hist(p_under_h0, bins=30, density=True, alpha=0.6, color='#2196F3', edgecolor='white')
ax.axvline(ALPHA, color='red', linestyle='--', linewidth=2, label=f'alpha={ALPHA}')
ax.axvspan(0, ALPHA, alpha=0.1, color='red')
ax.set_xlabel('p-value')
ax.set_title('p-value Under H0\n(Uniform if H0 true)', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'hypothesis_testing.png'), dpi=150)
plt.close()
print("  Saved: hypothesis_testing.png")


# ============================================================
# 7. p=0.049 vs p=0.051 反思
# ============================================================
print("\n" + "=" * 60)
print("  7. p=0.049 vs p=0.051 — 反思")
print("=" * 60)

reflection = """
p=0.049 和 p=0.051 在统计学上没有本质区别。

1. 二分思维的问题
   alpha=0.05 不是自然规律 — 这是 Fisher 当年随意选的惯例。
   p=0.049 "显著" 而 p=0.051 "不显著" 是人为二分法的荒谬之处。

2. ASA (2016) 声明
   "p值不衡量效应大小或结果的重要性"
   "不要仅凭p值做二元决策"
   "统计显著性 != 科学显著性 != 业务重要性"

3. 正确决策框架
   p<0.05 + 效应量大 → 上线
   p<0.05 + 效应量小 → 不值得上线
   p>0.05 + 效应量大 → 扩大样本再测
   p>0.05 + 效应量小 → 放弃

4. 面试回答
   Q: "p=0.051 你怎么决策?"
   A: "我不会仅凭p值决策。我会看效应量大小和置信区间。
       如果效应量够大但power不足, 建议扩大样本重测。
       如果效应量本身很小, 即使p<0.05也缺乏业务价值。"
"""
print(reflection)

print("\n✅ Day 28 完成")
