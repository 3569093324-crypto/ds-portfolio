#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 32: 回归诊断 — 你的模型靠谱吗？
残差分析 · VIF · 异方差 · Cook's Distance · Durbin-Watson
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# 生成模拟数据: 房价预测
# ============================================================
n = 200

# 特征
sqft = np.random.normal(2000, 500, n)          # 面积
bedrooms = np.random.randint(2, 6, n)           # 卧室数
age = np.random.gamma(shape=4, scale=5, size=n) # 房龄
dist_to_center = np.random.exponential(3, n)    # 距市中心距离
noise_level = np.random.normal(0, 4, n)         # 噪声水平（用于异方差）

# 价格 (带一些非线性 + 异方差)
price = (
    150 +                          # 截距
    0.25 * sqft +                  # 面积贡献
    15 * bedrooms +                # 卧室贡献
    -2.5 * age +                   # 房龄贡献 (负)
    -20 * dist_to_center +         # 距离贡献 (负)
    0.1 * (sqft - 2000) * bedrooms / 4 +  # 交互项 (面积×卧室)
    noise_level
)

# 引入一些影响点
price[0] = price[0] + 200       # 高影响点
sqft[1] = 5000                  # 超大房子 (高杠杆点)
price[1] = 800
price[2] = 0                    # 异常低值

df = pd.DataFrame({
    'price': price,
    'sqft': sqft,
    'bedrooms': bedrooms,
    'age': age,
    'dist_to_center': dist_to_center,
})

print("=" * 60)
print("  回归诊断: 房价预测模型")
print("=" * 60)
print(f"  样本量: {n}")
print(f"  特征: sqft, bedrooms, age, dist_to_center")
print(f"  目标: price")

# ============================================================
# 1. 拟合多元线性回归 (statsmodels)
# ============================================================
print("\n" + "=" * 60)
print("  1. 拟合 OLS 回归")
print("=" * 60)

X = sm.add_constant(df[['sqft', 'bedrooms', 'age', 'dist_to_center']])
y = df['price']

model = sm.OLS(y, X).fit()
print(model.summary().tables[1])  # 系数表

print(f"\n  R² = {model.rsquared:.4f}")
print(f"  Adj R² = {model.rsquared_adj:.4f}")
print(f"  F-statistic = {model.fvalue:.2f}, p = {model.f_pvalue:.6f}")


# ============================================================
# 2. 残差正态性
# ============================================================
print("\n" + "=" * 60)
print("  2. 残差正态性检验")
print("=" * 60)

residuals = model.resid
# studentized residuals
influence = model.get_influence()
studentized_resid = influence.resid_studentized_internal

# Shapiro-Wilk
sw_stat, sw_p = stats.shapiro(residuals[:500])
print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}")
print(f"  {'✅ 残差近似正态' if sw_p > 0.05 else '⚠️ 残差偏离正态 (p<0.05)'}")

# Jarque-Bera
jb_stat, jb_p = stats.jarque_bera(residuals)
print(f"  Jarque-Bera:  JB={jb_stat:.4f}, p={jb_p:.4f}")


# ============================================================
# 3. 异方差性: 残差 vs 拟合值
# ============================================================
print("\n" + "=" * 60)
print("  3. 异方差性检验")
print("=" * 60)

fitted = model.fittedvalues

# Breusch-Pagan 检验
bp_lm, bp_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(residuals, X)
print(f"  Breusch-Pagan: LM={bp_lm:.4f}, p={bp_pvalue:.4f}")
if bp_pvalue < 0.05:
    print(f"  ⚠️ 存在异方差 (p={bp_pvalue:.4f} < 0.05)")
    print(f"     建议: 使用稳健标准误 (HC3) 或 WLS")
else:
    print(f"  ✅ 无异方差 (同方差假设成立)")


# ============================================================
# 4. 多重共线性: VIF
# ============================================================
print("\n" + "=" * 60)
print("  4. 多重共线性: VIF")
print("=" * 60)

feature_names = ['const', 'sqft', 'bedrooms', 'age', 'dist_to_center']
vif_data = []
for i, name in enumerate(feature_names):
    vif = variance_inflation_factor(X.values, i)
    vif_data.append((name, vif))
    level = '✅' if vif < 5 else ('⚠️' if vif < 10 else '❌ 严重共线性')
    print(f"  VIF({name:20s}) = {vif:.2f} {level}")

print(f"\n  VIF 解读:")
print(f"    VIF < 5: 无共线性问题")
print(f"    5 ≤ VIF < 10: 中度共线性")
print(f"    VIF ≥ 10: 严重共线性, 需处理")


# ============================================================
# 5. 影响点: Cook's Distance
# ============================================================
print("\n" + "=" * 60)
print("  5. 影响点检测: Cook's Distance")
print("=" * 60)

cooks_d = influence.cooks_distance[0]
# 常用阈值: 4/n
threshold_cook = 4 / n
n_influential = (cooks_d > threshold_cook).sum()

print(f"  Cook's D 阈值 (4/n): {threshold_cook:.4f}")
print(f"  高影响点: {n_influential} 个")
if n_influential > 0:
    high_cook_idx = np.where(cooks_d > threshold_cook)[0]
    for idx in high_cook_idx[:5]:
        print(f"    obs {idx}: Cook's D={cooks_d[idx]:.4f}, "
              f"price={df.iloc[idx]['price']:.1f}, sqft={df.iloc[idx]['sqft']:.0f}")

# 杠杆值
leverage = influence.hat_matrix_diag
high_leverage_threshold = 2 * X.shape[1] / n
n_high_leverage = (leverage > high_leverage_threshold).sum()
print(f"\n  高杠杆点 (>{2*X.shape[1]/n:.4f}): {n_high_leverage} 个")


# ============================================================
# 6. Durbin-Watson
# ============================================================
print("\n" + "=" * 60)
print("  6. Durbin-Watson — 残差自相关")
print("=" * 60)

dw = sm.stats.durbin_watson(residuals)
print(f"  Durbin-Watson = {dw:.4f}")
print(f"  参考范围: DW≈2 = 无自相关, DW<1 = 正自相关, DW>3 = 负自相关")
if 1.5 < dw < 2.5:
    print(f"  ✅ DW接近2, 无显著自相关")
elif dw < 1.5:
    print(f"  ⚠️ DW<1.5, 可能存在正自相关")
else:
    print(f"  ⚠️ DW>2.5, 可能存在负自相关")


# ============================================================
# 7. 诊断总结 + 可视化
# ============================================================
print("\n" + "=" * 60)
print("  7. 诊断总结 & 可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (1) 残差 vs 拟合值 (检查异方差 + 线性)
ax = axes[0, 0]
ax.scatter(fitted, residuals, alpha=0.5, c='#2196F3', edgecolors='white')
ax.axhline(0, color='red', linestyle='--', alpha=0.7)
# Lowess 平滑线
from statsmodels.nonparametric.smoothers_lowess import lowess
lowess_fit = lowess(residuals, fitted, frac=0.3)
ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], 'r-', linewidth=2, label='LOWESS')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted\n(Check: linearity, homoscedasticity)',
             fontweight='bold')
ax.legend(fontsize=7)

# (2) Q-Q 图 (检查正态性)
ax = axes[0, 1]
stats.probplot(studentized_resid, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Studentized Residuals\n(Check: normality)', fontweight='bold')
ax.get_lines()[0].set_markerfacecolor('#2196F3')
ax.get_lines()[1].set_color('red')

# (3) Scale-Location (sqrt(|标准化残差|) vs 拟合值)
ax = axes[0, 2]
sqrt_abs_resid = np.sqrt(np.abs(studentized_resid))
ax.scatter(fitted, sqrt_abs_resid, alpha=0.5, c='#4CAF50', edgecolors='white')
lowess_sqrt = lowess(sqrt_abs_resid, fitted, frac=0.3)
ax.plot(lowess_sqrt[:, 0], lowess_sqrt[:, 1], 'r-', linewidth=2)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('sqrt(|Standardized Residuals|)')
ax.set_title('Scale-Location\n(Check: homoscedasticity)', fontweight='bold')

# (4) Cook's Distance
ax = axes[1, 0]
stem_colors = ['#f44336' if d > threshold_cook else '#2196F3' for d in cooks_d]
ax.stem(range(n), cooks_d, linefmt='grey', markerfmt=' ', basefmt=' ')
ax.scatter(range(n), cooks_d, c=stem_colors, s=20, alpha=0.7, edgecolors='white')
ax.axhline(threshold_cook, color='red', linestyle='--', alpha=0.7,
           label=f'Threshold (4/n) = {threshold_cook:.3f}')
ax.set_xlabel('Observation Index')
ax.set_ylabel("Cook's Distance")
ax.set_title("Cook's Distance\n(Check: influential points)", fontweight='bold')
ax.legend(fontsize=7)

# (5) 残差直方图
ax = axes[1, 1]
ax.hist(residuals, bins=25, density=True, alpha=0.6, color='#FF9800', edgecolor='white')
x_range = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(x_range, stats.norm.pdf(x_range, 0, residuals.std()),
        'k-', linewidth=2, label=f'N(0,{residuals.std():.1f})')
ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.set_title('Residual Distribution\n(Check: normality)', fontweight='bold')
ax.legend(fontsize=7)

# (6) 诊断总结卡片
ax = axes[1, 2]
ax.axis('off')
diagnostic_summary = f"""
Regression Diagnostic Summary
{'='*35}

1. Normality of Residuals
   Shapiro: p={sw_p:.4f} {'✅' if sw_p>0.05 else '⚠️'}
   Jarque-Bera: p={jb_p:.4f}

2. Homoscedasticity
   Breusch-Pagan: p={bp_pvalue:.4f}
   {'✅ No heteroscedasticity' if bp_pvalue>0.05 else '⚠️ Heteroscedasticity detected'}

3. Multicollinearity
   Max VIF = {max(v[1] for v in vif_data):.2f}
   {'✅ No collinearity' if max(v[1] for v in vif_data)<5 else '⚠️ Collinearity present'}

4. Influential Points
   Cook's D > 4/n: {n_influential}
   High leverage: {n_high_leverage}

5. Autocorrelation
   DW = {dw:.3f} {'✅' if 1.5<dw<2.5 else '⚠️'}

Overall Assessment:
{'✅ Model assumptions reasonably met' if (sw_p>0.01 and bp_pvalue>0.01 and max(v[1] for v in vif_data)<10) else '⚠️ Some assumptions violated — see recommendations'}

Recommendations:
• {'Use robust SE (HC3)' if bp_pvalue<0.05 else 'Standard SE adequate'}
• {'Investigate high influence points' if n_influential>0 else 'No influential cases'}
• {'Consider removing/combining collinear vars' if max(v[1] for v in vif_data)>=5 else 'Variables OK'}
"""
ax.text(0.05, 0.95, diagnostic_summary, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'regression_diagnostics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: regression_diagnostics.png")

print(diagnostic_summary)

print("\n✅ Day 32 完成")
