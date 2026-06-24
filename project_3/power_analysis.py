#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 29: I类/II类错误与统计功效
模拟功效曲线、多重比较、p-hacking
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 1 & 2. 模拟功效曲线
# ============================================================
print("=" * 60)
print("  1 & 2. 功效曲线模拟")
print("=" * 60)

# 参数
ALPHA = 0.05
EFFECT_SIZE = 0.3   # Cohen's d: 小效应
N_SIMULATIONS = 2000  # 每种样本量的模拟次数
sample_sizes = np.arange(20, 301, 20)

# 模拟：固定效应量，变化样本量
simulated_power = []

for n in sample_sizes:
    significant = 0
    for _ in range(N_SIMULATIONS):
        # 生成有真实效应量的两组数据
        group1 = np.random.normal(0, 1, n)
        group2 = np.random.normal(EFFECT_SIZE, 1, n)  # 均值偏移 EFFECT_SIZE
        _, p = stats.ttest_ind(group1, group2, equal_var=False)
        if p < ALPHA:
            significant += 1
    power = significant / N_SIMULATIONS
    simulated_power.append(power)

# 理论功效 (用statsmodels)
power_analysis = TTestIndPower()
theoretical_power = [
    power_analysis.power(effect_size=EFFECT_SIZE, nobs1=n, alpha=ALPHA,
                          ratio=1.0, alternative='two-sided')
    for n in sample_sizes
]

# 计算达到 80% 功效所需样本量
required_n = power_analysis.solve_power(
    effect_size=EFFECT_SIZE, power=0.8, alpha=ALPHA,
    ratio=1.0, alternative='two-sided'
)
print(f"  效应量 d={EFFECT_SIZE}")
print(f"  达到 80% 功效所需每组样本量: {required_n:.0f}")
print(f"  样本量={sample_sizes[-1]}时功效: {simulated_power[-1]:.3f}")

# 画功效曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(sample_sizes, simulated_power, 'o-', color='#2196F3', linewidth=2,
        markersize=6, label=f'Simulated ({N_SIMULATIONS} reps)')
ax.plot(sample_sizes, theoretical_power, 'r--', linewidth=2,
        label=f'Theory (statsmodels)')
ax.axhline(0.8, color='gray', linestyle=':', alpha=0.7, label='80% power')
ax.axvline(required_n, color='green', linestyle='--', alpha=0.7,
           label=f'Required n={required_n:.0f}')
ax.set_xlabel('Sample Size per Group (n)')
ax.set_ylabel('Statistical Power (1 - beta)')
ax.set_title(f'Power Curve: d={EFFECT_SIZE}, alpha={ALPHA}', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ============================================================
# 3. 功效分析: 计算所需样本量 (不同效应量)
# ============================================================
ax = axes[1]
effect_sizes = np.linspace(0.1, 1.0, 10)
required_ns = [
    power_analysis.solve_power(effect_size=es, power=0.8, alpha=ALPHA,
                                ratio=1.0, alternative='two-sided')
    for es in effect_sizes
]

ax.plot(effect_sizes, required_ns, 'o-', color='#FF9800', linewidth=2, markersize=8)
ax.set_xlabel("Cohen's d (Effect Size)")
ax.set_ylabel('Required Sample Size per Group (for 80% power)')
ax.set_title('Required Sample Size vs Effect Size', fontweight='bold')
ax.grid(True, alpha=0.3)

# 标注关键点
annotate_indices = {'small': 1, 'medium': 4, 'large': 7}
for label, idx in annotate_indices.items():
    es = effect_sizes[idx]
    n = required_ns[idx]
    ax.annotate(f'd={es:.1f} ({label})\nn={n:.0f}',
                xy=(es, n), fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'power_analysis.png'), dpi=150)
plt.close()
print("  Saved: power_analysis.png")


# ============================================================
# 4. 多重比较: 对随机噪声跑20次检验
# ============================================================
print("\n" + "=" * 60)
print("  4. 多重比较: 20次检验 × 纯随机数据")
print("=" * 60)

n_per_group = 50
n_tests = 20

# 重复实验多次
n_experiments = 100
false_positive_rates = []

for exp in range(n_experiments):
    p_values = []
    for _ in range(n_tests):
        # 两组都从同一个分布抽样 (H0为真)
        g1 = np.random.normal(0, 1, n_per_group)
        g2 = np.random.normal(0, 1, n_per_group)
        _, p = stats.ttest_ind(g1, g2, equal_var=False)
        p_values.append(p)
    false_positive_rates.append(sum(p < ALPHA for p in p_values))

avg_fp = np.mean(false_positive_rates)
print(f"  每次实验: {n_tests} 次独立 t 检验")
print(f"  重复实验: {n_experiments} 次")
print(f"  平均'显著'发现数: {avg_fp:.1f}/{n_tests} (预期: {n_tests*ALPHA:.1f})")
print(f"  至少1个'显著'的概率: {1 - (1-ALPHA)**n_tests:.3f} (= {100*(1-(1-ALPHA)**n_tests):.1f}%)")
print(f"  ⚠️ 跑20次检验有{100*(1-(1-ALPHA)**n_tests):.0f}%概率至少发现1个假阳性!")


# ============================================================
# 5. Bonferroni 校正 & FDR
# ============================================================
print("\n" + "=" * 60)
print("  5. Bonferroni 校正 & FDR")
print("=" * 60)

# 生成一些 p 值 (模拟)
np.random.seed(123)
all_pvalues = []
# 前5个有真实效应 (H1为真), 后15个无效应 (H0为真)
for i in range(20):
    g1 = np.random.normal(0, 1, 100)
    g2 = np.random.normal(0.5 if i < 5 else 0, 1, 100)  # 前5组有0.5效应
    _, p = stats.ttest_ind(g1, g2, equal_var=False)
    all_pvalues.append(p)

all_pvalues = np.array(all_pvalues)

# Bonferroni
bonf_reject, bonf_p_corrected, _, _ = multipletests(all_pvalues, alpha=ALPHA, method='bonferroni')

# Benjamini-Hochberg (FDR)
fdr_reject, fdr_p_corrected, _, _ = multipletests(all_pvalues, alpha=ALPHA, method='fdr_bh')

print(f"  {'Test':6s} | {'Raw p':10s} | {'Bonferroni':12s} | {'FDR(BH)':12s} | {'Truth':6s}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}")
for i in range(len(all_pvalues)):
    truth = 'H1' if i < 5 else 'H0'
    bonf_sig = '✓' if bonf_reject[i] else ''
    fdr_sig = '✓' if fdr_reject[i] else ''
    print(f"  {i+1:4d}  | p={all_pvalues[i]:.4f}  | "
          f"p'={bonf_p_corrected[i]:.4f} {bonf_sig:2s} | "
          f"q={fdr_p_corrected[i]:.4f} {fdr_sig:2s} | {truth}")

print(f"\n  Bonferroni: 发现 {sum(bonf_reject)} 个显著 (更保守, 控制FWER)")
print(f"  FDR (BH):   发现 {sum(fdr_reject)} 个显著 (更敏感, 控制FDR)")
print(f"  真阳性: 前5个有真实效应 (d=0.5)")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 左: 多重比较演示
ax = axes[0]
fp_counts = np.random.binomial(n_tests, ALPHA, 10000)
ax.hist(fp_counts, bins=range(0, n_tests+1), density=True, alpha=0.7,
        color='#2196F3', edgecolor='white')
ax.axvline(n_tests * ALPHA, color='red', linestyle='--', linewidth=2,
           label=f'Expected: {n_tests*ALPHA:.1f}')
ax.set_xlabel('Number of "Significant" Findings')
ax.set_ylabel('Probability')
ax.set_title('Under H0: 20 tests on random noise\nBinomial(20, 0.05)',
             fontweight='bold', fontsize=9)
ax.legend(fontsize=8)

# 中: 校正方法对比
ax = axes[1]
x = range(1, 21)
raw_sig = all_pvalues < ALPHA
ax.scatter(np.arange(20)[raw_sig], all_pvalues[raw_sig],
           c='red', s=80, zorder=5, label=f'Raw significant ({sum(raw_sig)})')
ax.scatter(np.arange(20)[~raw_sig], all_pvalues[~raw_sig],
           c='gray', s=40, zorder=4, label=f'Not significant ({sum(~raw_sig)})')
ax.axhline(ALPHA, color='gray', linestyle='--', alpha=0.5, label=f'alpha={ALPHA}')
ax.axhline(ALPHA / 20, color='green', linestyle=':', alpha=0.7,
           label=f'Bonferroni threshold={ALPHA}/20={ALPHA/20:.3f}')
ax.set_xlabel('Test Index')
ax.set_ylabel('p-value')
ax.set_title('Multiple Testing Correction', fontweight='bold', fontsize=9)
ax.legend(fontsize=7)

# 右: p-hacking 图示
ax = axes[2]
# 模拟 p-hacking: 逐渐增加样本量直到显著
np.random.seed(99)
phacked_results = []
for trial in range(200):
    n = 10
    while n <= 500:
        g1 = np.random.normal(0, 1, n)
        g2 = np.random.normal(0, 1, n)  # 无真实效应!
        _, p = stats.ttest_ind(g1, g2, equal_var=False)
        if p < 0.05:
            phacked_results.append((trial, n, p))
            break
        n += 10
    else:
        phacked_results.append((trial, 500, 1.0))

phacked_ns = [r[1] for r in phacked_results]
phacked_ps = [r[2] for r in phacked_results]
ax.scatter(phacked_ns, phacked_ps, alpha=0.4, s=20, c='#E91E63')
ax.axhline(0.05, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Sample Size When "Significant"')
ax.set_ylabel('p-value')
ax.set_title('p-Hacking Demo\n(Sequential testing until p<0.05)',
             fontweight='bold', fontsize=9)

pct_significant = sum(1 for p in phacked_ps if p < 0.05) / len(phacked_ps) * 100
ax.text(0.95, 0.90, f'{pct_significant:.0f}% eventually "significant"\n'
        f'(H0 was TRUE!)',
        transform=ax.transAxes, ha='right', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'multiple_testing.png'), dpi=150)
plt.close()
print("  Saved: multiple_testing.png")


# ============================================================
# 6. 总结: p-hacking 为什么危险
# ============================================================
print("\n" + "=" * 60)
print("  6. 为什么 p-hacking 是个问题")
print("=" * 60)

conclusion = """
I类错误 (False Positive) — 拒真
  H0为真时, 错误地拒绝H0。
  概率 = alpha = 0.05 (单次检验)
  多重比较时: P(至少1个假阳性) = 1 - (1-alpha)^k ≈ 64% (k=20次)

II类错误 (False Negative) — 取伪
  H1为真时, 错误地接受H0。
  概率 = beta
  统计功效 = 1 - beta (正确检测到真实效应的概率)

效应量越小 → 所需样本量越大 → 小样本时功效不足

p-hacking 常见方式 (都应该避免!):
  ❌ 边收集数据边检验, p<0.05就停止 (sequential testing)
  ❌ 跑20个指标, 只报告显著的那个 (cherry-picking)
  ❌ 尝试不同的分析方法直到找到显著结果 (fishing expedition)
  ❌ 剔除'异常值'直到p<0.05
  ❌ 改变假设方向(单尾→双尾→单尾)直到显著

正确做法:
  ✅ 实验前确定样本量 (power analysis)
  ✅ 预注册分析计划 (preregistration)
  ✅ 报告所有测试过的指标 (transparency)
  ✅ 使用多重比较校正 (Bonferroni / FDR)
  ✅ 报告效应量和置信区间, 不仅仅报告p值
"""
print(conclusion)

with open(os.path.join(os.path.dirname(__file__), 'power_analysis_notes.md'),
          'w', encoding='utf-8') as f:
    f.write(conclusion)

print("\n✅ Day 29 完成")
