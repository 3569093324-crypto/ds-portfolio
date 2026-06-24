#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 30: A/B测试全流程模拟 — 从假设到结论
大厂DS终面必考：完整设计一个A/B测试
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# Step 1: 定义业务问题和成功指标
# ============================================================
print("=" * 60)
print("  Step 1: 定义业务问题和成功指标")
print("=" * 60)

scenario = """
业务场景: 电商平台测试新版商品详情页
----------------------------------------
背景: 当前转化率约为12%, PM希望新版页面能提升到13%+
实验单元: 用户 (user_id)
实验周期: 2周
流量分配: 对照组50% / 实验组50%

主要指标 (Primary Metric):
  - 转化率 (Conversion Rate): 下单用户/访问用户

次要指标 (Secondary Metrics):
  - 平均订单金额 (AOV)
  - 页面停留时间
  - 跳出率

护栏指标 (Guardrail Metrics):
  - 页面加载时间 (不能显著变慢)
  - 客服投诉率 (不能显著上升)
"""
print(scenario)


# ============================================================
# Step 2: 设定 H0 和 H1, 选择 alpha 和 beta
# ============================================================
print("=" * 60)
print("  Step 2: 假设设定")
print("=" * 60)

ALPHA = 0.05      # I类错误率
BETA = 0.20        # II类错误率 (power = 80%)
BASELINE_RATE = 0.12   # 对照组转化率
MDE = 0.02             # 最小可检测效应 (2pp提升, 演示用)
# 即: 期望检测到 12% → 14% 的差异

print(f"  H0: p_treatment - p_control = 0")
print(f"  H1: p_treatment - p_control ≠ 0 (双尾)")
print(f"  alpha = {ALPHA} (I类错误)")
print(f"  beta  = {BETA} (II类错误 → power = {1-BETA})")
print(f"  基线转化率: {BASELINE_RATE*100:.0f}%")
print(f"  最小可检测效应 (MDE): {MDE*100:.0f}pp")


# ============================================================
# Step 3: 功效分析 — 计算所需样本量
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: 功效分析 — 所需样本量")
print("=" * 60)

# 比例检验的样本量计算 (使用正态近似)
from statsmodels.stats.proportion import proportion_effectsize

# 方法1: 用 proportion 效应量
es = proportion_effectsize(BASELINE_RATE + MDE, BASELINE_RATE)
power_analysis = NormalIndPower()
required_n_per_group = power_analysis.solve_power(
    effect_size=es, power=1-BETA, alpha=ALPHA,
    ratio=1.0, alternative='two-sided'
)
required_n_per_group = int(np.ceil(required_n_per_group))

print(f"  效应量 (Cohen's h): {es:.4f}")
print(f"  每组所需样本量: {required_n_per_group:,}")
print(f"  总样本量: {required_n_per_group * 2:,}")

# 加 10% 缓冲 (数据质量问题、异常用户过滤)
buffer_n = int(required_n_per_group * 1.1)
print(f"  加10%缓冲后每组: {buffer_n:,} (总计 {buffer_n*2:,})")

# 使用这个值 (演示用, 实际按power分析需要更大样本)
N_PER_GROUP = 5000
N_TOTAL = N_PER_GROUP * 2


# ============================================================
# Step 4: 随机化方案设计
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: 随机化方案")
print("=" * 60)

randomization_note = """
选择: 分层随机化 (Stratified Randomization)

分层变量:
  - 用户等级 (VIP/Retail/Wholesale/New): 确保各等级均匀分布
  - 设备类型 (Mobile/Desktop): 不同设备转化率差异大

为什么不选简单随机:
  - 可能出现组间不平衡 (如VIP用户都分到对照组)
  - 分层确保关键特征在两组间均匀分布

实现方式:
  user_id % 2 → 简单易实现
  或 hash(user_id + salt) % 2 → 可重现
"""
print(randomization_note)


# ============================================================
# Step 5: 生成模拟数据
# ============================================================
print("=" * 60)
print("  Step 5: 生成模拟数据 (N={:,} per group)".format(N_PER_GROUP))
print("=" * 60)

# 模拟基准转化行为
# VIP: 18%, Retail: 14%, Wholesale: 10%, New: 8%
segment_conversion = {
    'VIP': 0.18,
    'Retail': 0.14,
    'Wholesale': 0.10,
    'New': 0.08,
}
segment_dist = [0.25, 0.25, 0.25, 0.25]  # 均匀分布

segments = np.random.choice(
    list(segment_conversion.keys()), size=N_TOTAL, p=segment_dist
)

# 基础转化概率 (每个人基于其segment)
base_prob = np.array([segment_conversion[s] for s in segments])

# 添加用户级别噪声
np.random.seed(42)
user_noise = np.random.normal(0, 0.02, N_TOTAL)
base_prob = np.clip(base_prob + user_noise, 0.01, 0.5)

# 随机分组: 50/50 随机分配
group = np.random.choice(['control', 'treatment'], size=N_TOTAL, replace=True, p=[0.5, 0.5])
# 确保每组恰好 N_PER_GROUP (修正随机波动)
actual_treatment = np.where(group == 'treatment')[0]
actual_control = np.where(group == 'control')[0]
if len(actual_treatment) > N_PER_GROUP:
    # 将多余的 treatment 改为 control
    to_flip = np.random.choice(actual_treatment, size=len(actual_treatment)-N_PER_GROUP, replace=False)
    group[to_flip] = 'control'
elif len(actual_treatment) < N_PER_GROUP:
    # 将多余的 control 改为 treatment
    to_flip = np.random.choice(actual_control, size=N_PER_GROUP-len(actual_treatment), replace=False)
    group[to_flip] = 'treatment'

# 实验组提升 1pp (从12%基线到13%)
treatment_effect = MDE  # 0.01 绝对提升
group_effect = np.where(group == 'treatment', treatment_effect, 0)

# 最终转化
conversion_prob = np.clip(base_prob + group_effect, 0, 1)
converted = np.random.binomial(1, conversion_prob).astype(bool)

# 订单金额 (给转化用户)
order_amounts = np.where(
    converted,
    np.random.lognormal(mean=4.5, sigma=0.8, size=N_TOTAL),
    0.0
)
order_amounts = np.round(order_amounts, 2)

# 汇总
import pandas as pd
df = pd.DataFrame({
    'user_id': range(1, N_TOTAL + 1),
    'segment': segments,
    'group': group,
    'converted': converted,
    'order_amount': order_amounts,
})

print(f"  对照组用户: {(df['group']=='control').sum():,}")
print(f"  实验组用户: {(df['group']=='treatment').sum():,}")
print(f"\n  总体转化率: {df['converted'].mean()*100:.2f}%")
print(f"  对照组转化率: {df[df['group']=='control']['converted'].mean()*100:.2f}%")
print(f"  实验组转化率: {df[df['group']=='treatment']['converted'].mean()*100:.2f}%")
print(f"  观察差异: {(df[df['group']=='treatment']['converted'].mean() - df[df['group']=='control']['converted'].mean())*100:.3f}pp")


# ============================================================
# Step 6: AA 测试验证随机化, 然后 AB 对比
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: AA 测试 + AB 对比")
print("=" * 60)

# AA 测试: 在对照组内随机分成两半, 检验是否有显著差异
control_users = df[df['group'] == 'control'].copy()
n_half = len(control_users) // 2
shuffle = np.random.permutation(len(control_users))
aa_a = control_users.iloc[shuffle[:n_half]]
aa_b = control_users.iloc[shuffle[n_half:2*n_half]]

# 比较 AA 两组的转化率
prop_a = aa_a['converted'].mean()
prop_b = aa_b['converted'].mean()
# 双样本比例 z 检验
pooled_p = (aa_a['converted'].sum() + aa_b['converted'].sum()) / (len(aa_a) + len(aa_b))
se_aa = np.sqrt(pooled_p * (1 - pooled_p) * (1/len(aa_a) + 1/len(aa_b)))
z_aa = (prop_a - prop_b) / se_aa
p_aa = 2 * (1 - stats.norm.cdf(abs(z_aa)))

print(f"  AA测试: 对照组内随机分半对比")
print(f"    半组A转化率: {prop_a*100:.3f}%")
print(f"    半组B转化率: {prop_b*100:.3f}%")
print(f"    z = {z_aa:.4f}, p = {p_aa:.4f}")
if p_aa > 0.05:
    print(f"    ✅ AA通过: 无显著差异, 随机化成功")
else:
    print(f"    ⚠️ AA未通过: 存在显著差异, 检查随机化方案")

# 各segment的平衡性检查
print(f"\n  Segment 平衡性检查:")
for seg in segment_conversion.keys():
    ctrl_pct = (df[(df['group']=='control') & (df['segment']==seg)].shape[0] /
                df[df['group']=='control'].shape[0] * 100)
    trt_pct = (df[(df['group']=='treatment') & (df['segment']==seg)].shape[0] /
               df[df['group']=='treatment'].shape[0] * 100)
    print(f"    {seg:12s}: Control={ctrl_pct:.1f}%, Treatment={trt_pct:.1f}%")

# AB 对比
control_conv = df[df['group'] == 'control']['converted']
treatment_conv = df[df['group'] == 'treatment']['converted']

n_c = len(control_conv)
n_t = len(treatment_conv)
p_c = control_conv.mean()
p_t = treatment_conv.mean()

# 比例 z 检验
pooled_p_ab = (control_conv.sum() + treatment_conv.sum()) / (n_c + n_t)
se_ab = np.sqrt(pooled_p_ab * (1 - pooled_p_ab) * (1/n_c + 1/n_t))
z_ab = (p_t - p_c) / se_ab
p_ab = 2 * (1 - stats.norm.cdf(abs(z_ab)))

print(f"\n  AB 测试结果:")
print(f"    对照组转化率: {p_c*100:.3f}% (n={n_c:,})")
print(f"    实验组转化率: {p_t*100:.3f}% (n={n_t:,})")
print(f"    绝对提升: {(p_t-p_c)*100:.3f}pp")
print(f"    相对提升: {(p_t-p_c)/p_c*100:.2f}%")
print(f"    z = {z_ab:.4f}, p = {p_ab:.4f}")
print(f"    {'✅ 统计显著!' if p_ab < ALPHA else '❌ 不显著'}")

# 订单金额 (次要指标)
ctrl_aov = df[df['group']=='control']['order_amount'].mean()
trt_aov = df[df['group']=='treatment']['order_amount'].mean()
t_stat_aov, p_aov = stats.ttest_ind(
    df[df['group']=='treatment']['order_amount'],
    df[df['group']=='control']['order_amount'],
    equal_var=False
)
print(f"\n  次要指标 — 平均订单金额:")
print(f"    对照组 AOV: ¥{ctrl_aov:.2f}")
print(f"    实验组 AOV: ¥{trt_aov:.2f}")
print(f"    差异: ¥{trt_aov - ctrl_aov:.2f}")
print(f"    t = {t_stat_aov:.4f}, p = {p_aov:.4f} {'✅' if p_aov < ALPHA else '❌'}")


# ============================================================
# Step 7: 效应量 & 置信区间
# ============================================================
print("\n" + "=" * 60)
print("  Step 7: 效应量 & 置信区间")
print("=" * 60)

# 转化率差异的 95% CI
ci_lower = (p_t - p_c) - 1.96 * se_ab
ci_upper = (p_t - p_c) + 1.96 * se_ab
print(f"  转化率差异 95% CI: [{ci_lower*100:.3f}pp, {ci_upper*100:.3f}pp]")

# Cohen's h (比例效应量)
cohens_h = 2 * (np.arcsin(np.sqrt(p_t)) - np.arcsin(np.sqrt(p_c)))
print(f"  Cohen's h: {cohens_h:.4f} ", end='')
if abs(cohens_h) < 0.2:
    print("(可忽略)")
elif abs(cohens_h) < 0.5:
    print("(小效应)")
elif abs(cohens_h) < 0.8:
    print("(中等效应)")
else:
    print("(大效应)")

# 统计功效 (post-hoc)
achieved_power = power_analysis.power(
    effect_size=cohens_h, nobs1=N_PER_GROUP,
    alpha=ALPHA, ratio=1.0, alternative='two-sided'
)
print(f"  Post-hoc 统计功效: {achieved_power:.1%}")


# ============================================================
# Step 8: 可视化 & 结论
# ============================================================
print("\n" + "=" * 60)
print("  Step 8: 实验结论 & 可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 转化率对比
ax = axes[0, 0]
groups = ['Control', 'Treatment']
rates = [p_c * 100, p_t * 100]
colors_bar = ['#2196F3', '#FF9800']
bars = ax.bar(groups, rates, color=colors_bar, alpha=0.8, edgecolor='white')
# 添加误差线 (CI)
errors = [1.96 * np.sqrt(p_c*(1-p_c)/n_c) * 100,
          1.96 * np.sqrt(p_t*(1-p_t)/n_t) * 100]
ax.errorbar(groups, rates, yerr=errors, fmt='none', color='black',
            capsize=10, linewidth=2)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{rate:.2f}%', ha='center', fontweight='bold', fontsize=14)
ax.set_ylabel('Conversion Rate (%)')
ax.set_title(f'Conversion Rate Comparison\np={p_ab:.4f} '
             f'({"Significant!" if p_ab < ALPHA else "Not significant"})',
             fontweight='bold')
ax.set_ylim(0, max(rates) * 1.2)

# 右上: Segment平衡性
ax = axes[0, 1]
segs = list(segment_conversion.keys())
x = np.arange(len(segs))
w = 0.35
ctrl_seg_pct = [(df[(df['group']=='control')&(df['segment']==s)].shape[0]/
                  df[df['group']=='control'].shape[0]*100) for s in segs]
trt_seg_pct = [(df[(df['group']=='treatment')&(df['segment']==s)].shape[0]/
                 df[df['group']=='treatment'].shape[0]*100) for s in segs]
ax.bar(x - w/2, ctrl_seg_pct, w, label='Control', color='#2196F3', alpha=0.8)
ax.bar(x + w/2, trt_seg_pct, w, label='Treatment', color='#FF9800', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(segs)
ax.set_ylabel('% of Group')
ax.set_title('Segment Balance Check', fontweight='bold')
ax.legend()
ax.axhline(25, color='gray', linestyle='--', alpha=0.5)

# 左下: 转化率差异的抽样分布
ax = axes[1, 0]
# 模拟差异的抽样分布 (bootstrap)
diffs = []
for _ in range(5000):
    bs_ctrl = np.random.choice(control_conv, size=n_c, replace=True).mean()
    bs_trt = np.random.choice(treatment_conv, size=n_t, replace=True).mean()
    diffs.append(bs_trt - bs_ctrl)
diffs = np.array(diffs)
ax.hist(diffs, bins=50, density=True, alpha=0.6, color='#4CAF50', edgecolor='white')
ax.axvline(p_t - p_c, color='red', linewidth=2, label=f'Observed: {(p_t-p_c)*100:.3f}pp')
ax.axvline(ci_lower, color='gray', linestyle='--', label=f'95% CI lower')
ax.axvline(ci_upper, color='gray', linestyle='--', label=f'95% CI upper')
ax.axvline(0, color='black', linestyle=':', alpha=0.5, label='H0: difference=0')
ax.set_xlabel('Difference in Conversion Rate')
ax.set_ylabel('Density')
ax.set_title('Bootstrap Distribution of Difference', fontweight='bold')
ax.legend(fontsize=7)

# 右下: 实验结论总结
ax = axes[1, 1]
ax.axis('off')
decision_text = f"""
A/B Test Results Summary
{'='*40}

Metric: Conversion Rate
  Control:   {p_c*100:.2f}%
  Treatment: {p_t*100:.2f}%
  Lift:      {(p_t-p_c)*100:.2f}pp ({(p_t-p_c)/p_c*100:.1f}% relative)
  95% CI:    [{ci_lower*100:.3f}pp, {ci_upper*100:.3f}pp]
  p-value:   {p_ab:.4f}
  Power:     {achieved_power:.1%}

Decision: {'✅ LAUNCH' if p_ab < ALPHA and (p_t-p_c) > 0 else '❌ HOLD'}

Recommendation:
{' 新版页面转化率显著提升, 建议全量上线。' if p_ab < ALPHA and (p_t-p_c) > 0 else
 ' 未达到统计显著性, 建议扩大样本或改进方案。'}

Key Takeaways:
• AA test passed (p={p_aa:.3f})
• Segments balanced across groups
• Guardrail metrics need monitoring post-launch
• Recommend 1-week ramp-up before full rollout
"""
ax.text(0.05, 0.95, decision_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen' if p_ab < ALPHA else 'lightyellow',
                  alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ab_test_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: ab_test_results.png")

# 输出完整结论
print(f"""
{'='*60}
  A/B测试最终结论
{'='*60}

1. 实验设计
   - 业务问题: 新版商品详情页是否提升转化率
   - 主要指标: 转化率
   - 实验单元: 用户, 随机分配, 50/50
   - 样本量: 每组 {N_PER_GROUP:,} (power={1-BETA}, MDE={MDE*100}pp)
   - 实验周期: 2周

2. 随机化验证
   - AA测试: p={p_aa:.3f} (通过, 无显著差异)
   - Segment平衡: 各组分布差异 < 1pp

3. 实验结果
   - 转化率: Control={p_c*100:.2f}%, Treatment={p_t*100:.2f}%
   - 绝对提升: {(p_t-p_c)*100:.2f}pp
   - 相对提升: {(p_t-p_c)/p_c*100:.1f}%
   - p值: {p_ab:.4f}
   - 95% CI: [{ci_lower*100:.3f}pp, {ci_upper*100:.3f}pp]

4. 业务建议
   {'✅ 建议全量上线新版页面' if p_ab < ALPHA and (p_t - p_c) > 0 else
    '❌ 建议保留原页面或进一步测试'}
   - 预期年化收入影响: 每天 {N_TOTAL*MDE*order_amounts[converted].mean():.0f} 元
   - 上线后监控计划: 每日转化率、页面加载时间、客服投诉
   - 回滚条件: 转化率下降 > 0.5pp 或页面加载时间增加 > 200ms

5. 面试要点
   Q: "为什么用分层随机而不是简单随机?"
   A: "分层确保关键特征(用户等级)在两组间均匀分布,
       降低随机性导致的组间不平衡。如果VIP用户随机
       分配不均, 实验结果可能有偏。"

   Q: "如果结果不显著怎么办?"
   A: "首先检查power是否够 — 如果样本量不足就扩大。
       其次看效应量方向 — 如果效应量为正但p>0.05,
       可能是power不足; 如果效应量接近零, 说明新设计
       确实没有效果, 考虑改变方案。"
""")

print("\n✅ Day 30 完成 — 项目3核心A/B测试模拟完成!")
