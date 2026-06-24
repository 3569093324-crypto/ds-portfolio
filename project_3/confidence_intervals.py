#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 27: 置信区间 — 计算与正确解释
手动计算 vs scipy vs Bootstrap，三种方法对比
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

# 生成模拟数据（从真实数据库抽取的感觉）
# 模拟 100 个用户的订单金额
true_mean = 250
true_std = 80
n = 100
data = np.random.normal(true_mean, true_std, n)
# 加点噪声让数据更真实
data = np.clip(data, 0, None)  # 金额不能为负

sample_mean = data.mean()
sample_std = data.std(ddof=1)  # 样本标准差 (ddof=1 for unbiased)
n = len(data)

print("=" * 60)
print("  置信区间计算实验")
print("=" * 60)
print(f"  样本量 n={n}")
print(f"  样本均值={sample_mean:.2f}")
print(f"  样本标准差={sample_std:.2f}")
print(f"  真实均值(已知)={true_mean}")

# ============================================================
# 1. 手动计算 95% CI（仅用numpy）
# ============================================================
print("\n" + "=" * 60)
print("  1. 手动计算 (numpy only)")
print("=" * 60)

confidence = 0.95
alpha = 1 - confidence

# 标准误
se = sample_std / np.sqrt(n)
print(f"  标准误 SE = s/√n = {sample_std:.2f}/√{n} = {se:.4f}")

# t 临界值 (手动用 scipy 验证，然后纯 numpy 实现近似)
t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
print(f"  t临界值 (df={n-1}, α/2={alpha/2}) = {t_crit:.4f}")

# 置信区间 = mean ± t_crit × SE
ci_lower_manual = sample_mean - t_crit * se
ci_upper_manual = sample_mean + t_crit * se
print(f"  95% CI = [{ci_lower_manual:.2f}, {ci_upper_manual:.2f}]")
print(f"  区间宽度 = {ci_upper_manual - ci_lower_manual:.2f}")

# 验证：真实均值是否在区间内？
captures_true = ci_lower_manual <= true_mean <= ci_upper_manual
print(f"  真实均值 {true_mean} 在区间内？ {'YES ✅' if captures_true else 'NO ❌'}")


# ============================================================
# 2. scipy.stats 验证
# ============================================================
print("\n" + "=" * 60)
print("  2. scipy.stats 验证")
print("=" * 60)

ci_scipy = stats.t.interval(confidence, df=n-1, loc=sample_mean, scale=se)
print(f"  scipy.t.interval() = [{ci_scipy[0]:.2f}, {ci_scipy[1]:.2f}]")
print(f"  手动 vs scipy 差异: {abs(ci_lower_manual - ci_scipy[0]):.10f}")

# 同时用 norm.interval (z检验，大样本近似)
ci_z = stats.norm.interval(confidence, loc=sample_mean, scale=se)
print(f"  z-interval (正态近似) = [{ci_z[0]:.2f}, {ci_z[1]:.2f}]")
print(f"  t vs z 差异: {abs(ci_scipy[0] - ci_z[0]):.4f} (n≥30时差异很小)")


# ============================================================
# 3. Bootstrap 法计算置信区间
# ============================================================
print("\n" + "=" * 60)
print("  3. Bootstrap 法 — 重采样10000次")
print("=" * 60)

N_BOOTSTRAP = 10000
bootstrap_means = np.zeros(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    # 从原始数据中有放回地抽取 n 个样本
    resample = np.random.choice(data, size=n, replace=True)
    bootstrap_means[i] = resample.mean()

# 百分位数法：取 2.5% 和 97.5% 分位数
ci_boot_lower = np.percentile(bootstrap_means, 2.5)
ci_boot_upper = np.percentile(bootstrap_means, 97.5)

print(f"  Bootstrap 95% CI = [{ci_boot_lower:.2f}, {ci_boot_upper:.2f}]")
print(f"  Bootstrap 均值 = {bootstrap_means.mean():.2f}")
print(f"  Bootstrap SE    = {bootstrap_means.std():.4f}")
print(f"  理论 SE          = {se:.4f}")

# 更精确的 Bootstrap: BCa 方法
# (用 scipy 的 bootstrap 函数)
from scipy.stats import bootstrap as scipy_bootstrap
res = scipy_bootstrap(
    (data,), np.mean,
    confidence_level=0.95,
    n_resamples=10000,
    method='BCa',
    random_state=42,
)
print(f"\n  Bootstrap BCa 95% CI = [{res.confidence_interval.low:.2f}, "
      f"{res.confidence_interval.high:.2f}]")


# ============================================================
# 4. 理论 vs Bootstrap 对比
# ============================================================
print("\n" + "=" * 60)
print("  4. 理论公式 vs Bootstrap — 对比分析")
print("=" * 60)

print(f"""
  方法               | 95% CI                     | 宽度
  ------------------|----------------------------|-------
  理论公式 (t分布)   | [{ci_lower_manual:.2f}, {ci_upper_manual:.2f}] | {ci_upper_manual-ci_lower_manual:.2f}
  Bootstrap 百分位数 | [{ci_boot_lower:.2f}, {ci_boot_upper:.2f}] | {ci_boot_upper-ci_boot_lower:.2f}
  Bootstrap BCa      | [{res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f}] | {res.confidence_interval.high-res.confidence_interval.low:.2f}

  什么时候用 Bootstrap？
  ✅ 数据不服从正态分布时 (CLT需要n≥30, Bootstrap没有这个限制)
  ✅ 统计量不是均值时 (如中位数、相关系数、分位数 — 这些没有简单公式)
  ✅ 样本量小且分布未知时
  ✅ 想要更准确的区间估计时 (BCa方法修正偏度)

  什么时候用理论公式？
  ✅ 数据近似正态且统计量是均值 → t检验公式足够
  ✅ 计算资源有限 → Bootstrap需要大量重采样
""")

# ============================================================
# 5. 可视化
# ============================================================
print("=" * 60)
print("  5. 可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: 数据分布 + CI
ax = axes[0]
ax.hist(data, bins=20, density=True, alpha=0.6, color='#2196F3', edgecolor='white')
ax.axvline(sample_mean, color='red', linewidth=2, label=f'Mean={sample_mean:.1f}')
ax.axvline(ci_lower_manual, color='green', linestyle='--', linewidth=2,
           label=f'95% CI Lower={ci_lower_manual:.1f}')
ax.axvline(ci_upper_manual, color='green', linestyle='--', linewidth=2,
           label=f'95% CI Upper={ci_upper_manual:.1f}')
ax.axvspan(ci_lower_manual, ci_upper_manual, alpha=0.1, color='green')
ax.axvline(true_mean, color='gray', linestyle=':', linewidth=2,
           label=f'True Mean={true_mean}')
ax.set_title('Data Distribution + 95% CI', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlabel('Order Amount')

# 右: Bootstrap 均值分布
ax = axes[1]
ax.hist(bootstrap_means, bins=50, density=True, alpha=0.6, color='#FF9800', edgecolor='white')
ax.axvline(bootstrap_means.mean(), color='red', linewidth=2,
           label=f'Bootstrap Mean={bootstrap_means.mean():.1f}')
ax.axvline(ci_boot_lower, color='purple', linestyle='--', linewidth=2,
           label=f'2.5%={ci_boot_lower:.1f}')
ax.axvline(ci_boot_upper, color='purple', linestyle='--', linewidth=2,
           label=f'97.5%={ci_boot_upper:.1f}')
# 覆盖理论正态曲线
x = np.linspace(bootstrap_means.min(), bootstrap_means.max(), 200)
ax.plot(x, stats.norm.pdf(x, sample_mean, se), 'k-', linewidth=1.5,
        label=f'Theory N({sample_mean:.1f}, {se:.2f})')
ax.set_title(f'Bootstrap Distribution (n={N_BOOTSTRAP})', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlabel('Sample Mean')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confidence_intervals.png'), dpi=150)
plt.close()
print("  Saved: confidence_intervals.png")


# ============================================================
# 6. 正确解释 95% 置信区间
# ============================================================
print("\n" + "=" * 60)
print("  6. 正确解释 95% CI — 纠正常见误解")
print("=" * 60)

explanation = """
95% 置信区间: 正确 vs 错误解释
=================================

❌ 错误1: "真值有95%的概率落在 [a, b] 区间内"
   为什么错: 在频率学派框架下, 真值是常量(不是随机变量),
   没有"概率"。区间才是随机的(每次抽样得到的区间不同)。

❌ 错误2: "95%的样本落在这个区间内"
   为什么错: 混淆了置信区间和预测区间。
   置信区间描述的是"均值的精度", 不是"个体的分布范围"。

✅ 正确解释:
   "如果我们从同一总体中重复抽样无数次, 每次用同样的方法
   计算95%置信区间, 那么这些区间中有95%会包含真实的总体参数。"

   通俗版: "我们有95%的把握认为真实均值在 [a, b] 之间"
   (虽然技术上不严谨, 但在业务沟通中广泛接受)。

📊 模拟验证: 重复1000次抽样, 统计包含真值的比例
"""
print(explanation)

# 模拟验证
np.random.seed(123)
n_simulations = 1000
captured = 0

for _ in range(n_simulations):
    sim_data = np.random.normal(true_mean, true_std, 50)
    sim_mean = sim_data.mean()
    sim_se = sim_data.std(ddof=1) / np.sqrt(50)
    sim_ci = stats.t.interval(0.95, df=49, loc=sim_mean, scale=sim_se)
    if sim_ci[0] <= true_mean <= sim_ci[1]:
        captured += 1

print(f"  模拟 {n_simulations} 次独立抽样:")
print(f"  包含真值的区间: {captured}/{n_simulations} = {captured/n_simulations*100:.1f}%")
print(f"  预期: 95% (因为用了95%置信水平)")
print(f"  → 验证了正确解释: 约95%的CI包含真值 ✓")

print("\n✅ Day 27 完成")
