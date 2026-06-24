#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 31: 非参数检验 — 当数据不满足正态假设
Mann-Whitney U / Kruskal-Wallis / Wilcoxon signed-rank
"""

import numpy as np
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
# 1 & 2. 高度偏态数据: t检验 vs Mann-Whitney U
# ============================================================
print("=" * 60)
print("  1 & 2. 偏态数据: t-test vs Mann-Whitney U")
print("=" * 60)

# 生成对数正态分布（高度右偏）
n = 100
# 两组: A来自对数正态(0, 0.8), B来自对数正态(0.3, 0.8)
group_a = np.random.lognormal(mean=0, sigma=0.8, size=n)
group_b = np.random.lognormal(mean=0.3, sigma=0.8, size=n)

print(f"  Group A: median={np.median(group_a):.2f}, skew={stats.skew(group_a):.2f}")
print(f"  Group B: median={np.median(group_b):.2f}, skew={stats.skew(group_b):.2f}")
print(f"  Shapiro-Wilk p (A) = {stats.shapiro(group_a)[1]:.4f}")
print(f"  Shapiro-Wilk p (B) = {stats.shapiro(group_b)[1]:.4f}")
print(f"  → 两组都显著偏离正态 (p<0.05)")

# t-test (parametric, assumes normality)
t_stat, p_t = stats.ttest_ind(group_a, group_b, equal_var=False)

# Mann-Whitney U (non-parametric, no normality assumption)
u_stat, p_mw = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

print(f"\n  {'Method':20s} | {'Statistic':12s} | {'p-value':10s} | {'Significant?':12s}")
print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")
print(f"  {'t-test (Welch)':20s} | t={t_stat:<10.3f} | {p_t:<10.4f} | {'YES' if p_t<ALPHA else 'NO':12s}")
print(f"  {'Mann-Whitney U':20s} | U={u_stat:<10.1f} | {p_mw:<10.4f} | {'YES' if p_mw<ALPHA else 'NO':12s}")

# 差距说明
print(f"\n  两种检验可能给出不同结果的原因:")
print(f"  - t-test 比较的是 均值, 对异常值和偏态敏感")
print(f"  - MW-U 比较的是 中位数(秩), 对偏态分布更稳健")


# ============================================================
# 3. 对比：哪个更适合偏态数据？
# ============================================================
print("\n" + "=" * 60)
print("  3. 对比：为什么MW-U更适合偏态数据？")
print("=" * 60)

# 模拟: 在偏态数据中注入极端值, 观察两个检验的稳健性
np.random.seed(123)
n_sim = 500
t_power_t, t_power_mw = [], []

for _ in range(n_sim):
    # 两组: A无效应, B有轻微中位数偏移
    a = np.random.lognormal(0, 1.0, 50)
    b = np.random.lognormal(0.2, 1.0, 50)

    # 随机注入1个极端值
    if np.random.random() < 0.1:
        idx = np.random.randint(0, 50)
        a[idx] = np.random.lognormal(5, 1.0)

    _, p1 = stats.ttest_ind(a, b, equal_var=False)
    _, p2 = stats.mannwhitneyu(a, b, alternative='two-sided')
    t_power_t.append(p1 < ALPHA)
    t_power_mw.append(p2 < ALPHA)

print(f"  500次模拟 (偏态数据, 10%概率注入极端值):")
print(f"    t-test 检出率: {np.mean(t_power_t)*100:.1f}%")
print(f"    MW-U   检出率: {np.mean(t_power_mw)*100:.1f}%")
print(f"    → MW-U在偏态+异常值场景下更可靠")


# ============================================================
# 4. Kruskal-Wallis (多组比较, 非参数ANOVA)
# ============================================================
print("\n" + "=" * 60)
print("  4. Kruskal-Wallis — 多组非参数比较")
print("=" * 60)

# 场景: 4个用户等级的消费金额比较
segments = ['VIP', 'Retail', 'Wholesale', 'New']
data_by_seg = {
    'VIP': np.random.lognormal(5.5, 0.6, 80),
    'Retail': np.random.lognormal(5.2, 0.7, 80),
    'Wholesale': np.random.lognormal(5.0, 0.8, 80),
    'New': np.random.lognormal(4.8, 0.9, 80),
}

# Kruskal-Wallis H-test
h_stat, p_kw = stats.kruskal(*data_by_seg.values())

# One-way ANOVA (做对比, 虽然数据非正态)
f_stat, p_anova = stats.f_oneway(*data_by_seg.values())

print(f"  比较4组用户等级的消费金额:")
for seg, d in data_by_seg.items():
    print(f"    {seg:12s}: median=¥{np.median(d):.1f}, mean=¥{d.mean():.1f}")
print(f"\n  Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.4f} "
      f"{'✅' if p_kw < ALPHA else '❌'}")
print(f"  ANOVA (对比):    F={f_stat:.2f},  p={p_anova:.4f} "
      f"{'✅' if p_anova < ALPHA else '❌'}")

# 事后检验: Dunn's test (Mann-Whitney + Bonferroni)
if p_kw < ALPHA:
    print(f"\n  事后两两比较 (Mann-Whitney + Bonferroni):")
    seg_names = list(data_by_seg.keys())
    k = len(seg_names)
    n_comparisons = k * (k-1) // 2
    bonf_alpha = ALPHA / n_comparisons
    for i in range(k):
        for j in range(i+1, k):
            _, p = stats.mannwhitneyu(
                data_by_seg[seg_names[i]], data_by_seg[seg_names[j]],
                alternative='two-sided'
            )
            sig = '✓' if p < bonf_alpha else ''
            print(f"    {seg_names[i]:12s} vs {seg_names[j]:12s}: p={p:.4f} {sig}")


# ============================================================
# 5. Wilcoxon 符号秩检验 (配对非参数)
# ============================================================
print("\n" + "=" * 60)
print("  5. Wilcoxon signed-rank — 配对非参数")
print("=" * 60)

# 场景: 30个用户体验新功能前后的满意度评分 (1-5分，明显非正态)
n_paired = 30
before_satisfaction = np.random.choice([1, 2, 3, 4, 5], n_paired, p=[0.05, 0.15, 0.4, 0.3, 0.1])
after_satisfaction = np.clip(before_satisfaction + np.random.choice([-1, 0, 1, 2], n_paired,
                             p=[0.05, 0.25, 0.4, 0.3]), 1, 5)

w_stat, p_wilcoxon = stats.wilcoxon(after_satisfaction, before_satisfaction)
t_paired, p_paired_t = stats.ttest_rel(after_satisfaction, before_satisfaction)

print(f"  满意度评分 (1-5离散值, 非正态)")
print(f"    使用前: median={np.median(before_satisfaction)}, mean={before_satisfaction.mean():.2f}")
print(f"    使用后: median={np.median(after_satisfaction)}, mean={after_satisfaction.mean():.2f}")
print(f"    Wilcoxon: W={w_stat:.1f}, p={p_wilcoxon:.4f} {'✅' if p_wilcoxon<ALPHA else '❌'}")
print(f"    Paired t: t={t_paired:.3f}, p={p_paired_t:.4f} {'✅' if p_paired_t<ALPHA else '❌'}")


# ============================================================
# 6. choose_test() 函数
# ============================================================
print("\n" + "=" * 60)
print("  6. choose_test() — 自动选择检验方法")
print("=" * 60)

def choose_test(data, groups=None, paired=False):
    """
    自动判断该用参数检验还是非参数检验。

    Parameters
    ----------
    data : array-like or dict
        如果是array: 单组或两组数据
        如果是dict: {group_name: array} 多组数据
    groups : array-like, optional
        分组标签 (如果data是单array)
    paired : bool
        是否为配对设计

    Returns
    -------
    dict: {
        'recommended_test': 推荐的检验名,
        'reason': 推荐理由,
        'result': (statistic, p_value),
        'normality_check': 正态性检验结果,
    }
    """
    # 解析输入
    if isinstance(data, dict):
        group_names = list(data.keys())
        group_arrays = list(data.values())
        n_groups = len(group_arrays)
    elif groups is not None:
        unique_groups = np.unique(groups)
        group_arrays = [data[groups == g] for g in unique_groups]
        group_names = list(unique_groups)
        n_groups = len(group_names)
        data = np.asarray(data)
    else:
        data = np.asarray(data)
        group_names = None
        n_groups = 1

    # 正态性检验
    normality_results = {}
    all_normal = True
    for name, arr in (zip(group_names, group_arrays) if group_names
                      else [('data', data)]):
        if len(arr) >= 3:
            _, sw_p = stats.shapiro(arr[:500])  # 截断到500
        else:
            sw_p = 1.0
        normality_results[name] = sw_p
        if sw_p < 0.05:
            all_normal = False

    # 决策逻辑
    if paired:
        if n_groups != 2:
            return {'error': 'Paired test requires exactly 2 groups'}
        arr1, arr2 = group_arrays[0], group_arrays[1]
        if all_normal and len(arr1) >= 30:
            result = stats.ttest_rel(arr1, arr2)
            recommended = 'Paired t-test'
            reason = '数据近似正态 + n≥30'
        else:
            result = stats.wilcoxon(arr1, arr2)
            recommended = 'Wilcoxon signed-rank'
            reason = '非正态或小样本, 使用非参数配对检验'
        return {
            'recommended_test': recommended,
            'reason': reason,
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'normality_check': normality_results,
        }

    if n_groups == 1:
        # 单样本
        arr = data
        if all_normal and len(arr) >= 30:
            result = stats.ttest_1samp(arr, np.mean(arr))
            recommended = 'One-sample t-test'
            reason = '数据近似正态 + n≥30'
        else:
            result = stats.wilcoxon(arr - np.median(arr))
            recommended = 'Wilcoxon (one-sample)'
            reason = '非正态或小样本, 使用非参数'
        return {
            'recommended_test': recommended,
            'reason': reason,
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'normality_check': normality_results,
        }

    if n_groups == 2:
        arr1, arr2 = group_arrays[0], group_arrays[1]
        if all_normal and len(arr1) >= 30 and len(arr2) >= 30:
            _, p_var = stats.levene(arr1, arr2)
            if p_var < 0.05:
                result = stats.ttest_ind(arr1, arr2, equal_var=False)
                recommended = "Welch's t-test"
                reason = '正态但方差不齐 (Welch校正)'
            else:
                result = stats.ttest_ind(arr1, arr2, equal_var=True)
                recommended = "Student's t-test"
                reason = '正态 + 方差齐性'
        else:
            result = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
            recommended = 'Mann-Whitney U'
            reason = '非正态或小样本, 使用非参数'
        return {
            'recommended_test': recommended,
            'reason': reason,
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'normality_check': normality_results,
        }

    if n_groups >= 3:
        if all_normal:
            result = stats.f_oneway(*group_arrays)
            recommended = 'One-way ANOVA'
            reason = '多组 + 数据正态'
        else:
            result = stats.kruskal(*group_arrays)
            recommended = 'Kruskal-Wallis'
            reason = '多组 + 非正态, 使用非参数'
        return {
            'recommended_test': recommended,
            'reason': reason,
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'normality_check': normality_results,
        }

    return {'error': 'Unable to determine test'}


# 测试
print("  测试 choose_test():")
print()
# Test 1: 两组偏态数据
combined = np.concatenate([group_a, group_b])
labels = np.array(['A']*len(group_a) + ['B']*len(group_b))
r = choose_test(combined, groups=labels)
print(f"  偏态两组: → {r['recommended_test']} (p={r['p_value']:.4f}) — {r['reason']}")

# Test 2: 多组
r2 = choose_test(data_by_seg)
print(f"  多组偏态: → {r2['recommended_test']} (p={r2['p_value']:.4f}) — {r2['reason']}")

# Test 3: 正态数据
normal_data = {'X': np.random.normal(0, 1, 100), 'Y': np.random.normal(0.5, 1, 100)}
r3 = choose_test(normal_data)
print(f"  两组正态: → {r3['recommended_test']} (p={r3['p_value']:.4f}) — {r3['reason']}")


# ============================================================
# 7. 决策树可视化
# ============================================================
print("\n" + "=" * 60)
print("  7. 参数 vs 非参数检验 — 决策树")
print("=" * 60)

tree = """
  统计检验选择决策树
  ===================

  [研究设计类型]
      │
      ├── 配对设计 (同一对象前后测)
      │   ├── 正态 + n≥30 → Paired t-test
      │   └── 非正态      → Wilcoxon signed-rank
      │
      ├── 独立两组
      │   ├── 正态 + n≥30
      │   │   ├── 方差齐 → Student's t-test
      │   │   └── 方差不齐 → Welch's t-test
      │   └── 非正态或小样本 → Mann-Whitney U
      │
      ├── 独立三组及以上
      │   ├── 正态 → One-way ANOVA
      │   └── 非正态 → Kruskal-Wallis
      │
      └── 分类变量关联
          ├── 2×2表 → Fisher's exact / Chi-squared
          └── 大表 → Chi-squared (期望频数>5)

  非参数检验的 Trade-off:
    ✅ 优点:
      - 不依赖分布假设 (稳健)
      - 对异常值不敏感
      - 适用于序数数据 (如评分1-5)
    ❌ 缺点:
      - 功效略低 (需要更大样本检测同一效应)
      - 只报告显著性, 不直接给出效应量
      - 比较的是中位数(秩), 而非均值
"""
print(tree)

print("✅ Day 31 完成")
