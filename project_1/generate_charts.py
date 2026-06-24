#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目1 — 生成6张分析图表
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import numpy as np

# 尝试使用中文字体
for font_name in ['Microsoft YaHei', 'SimHei', 'Microsoft JhengHei', 'DejaVu Sans']:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "business.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "visuals")
os.makedirs(OUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

# ============================================================
# Chart 1: 高价值用户 Segment 分布
# ============================================================
print("Chart 1: 高价值用户 Segment 分布")
cur = conn.cursor()
cur.execute('''WITH ranked AS (
    SELECT c.segment, COALESCE(SUM(o.total_amount),0) AS spent,
           NTILE(4) OVER (ORDER BY SUM(o.total_amount) DESC) AS q
    FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id
    GROUP BY c.customer_id, c.segment)
SELECT segment, COUNT(*) AS cnt, ROUND(AVG(spent),2) AS avg_spent
FROM ranked WHERE q=1 GROUP BY segment ORDER BY avg_spent DESC''')
rows = cur.fetchall()
segments = [r[0] for r in rows]
counts = [r[1] for r in rows]
avgs = [r[2] for r in rows]

fig, ax1 = plt.subplots(figsize=(8, 5))
bars = ax1.bar(segments, counts, color=['#2196F3','#4CAF50','#FF9800','#9C27B0'])
ax1.set_ylabel('User Count', fontsize=12)
ax1.set_title('Q1: High-Value User Segment Distribution\n(Top 25% Spenders)', fontsize=14, fontweight='bold')
for bar, cnt in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(cnt),
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart1_high_value_users.png'), dpi=150)
plt.close()
print(f"  Saved: chart1_high_value_users.png")

# ============================================================
# Chart 2: 品类销售额 + 累计占比 (Pareto)
# ============================================================
print("Chart 2: 品类销售额 Pareto")
cur.execute('''SELECT p.category, ROUND(SUM(oi.quantity*oi.unit_price),2) AS rev
FROM order_items oi JOIN products p ON oi.product_id=p.product_id
GROUP BY p.category ORDER BY rev DESC''')
rows = cur.fetchall()
cats = [r[0] for r in rows]
revs = [r[1] for r in rows]
total = sum(revs)
cum_pct = np.cumsum(revs) / total * 100
rev_pct = [r/total*100 for r in revs]

fig, ax1 = plt.subplots(figsize=(8, 5))
bars = ax1.bar(cats, revs, color='#2196F3', alpha=0.8)
ax1.set_ylabel('Revenue ($)', fontsize=12)
ax1.set_title('Q2: Category Revenue (Pareto Analysis)', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=30)

ax2 = ax1.twinx()
ax2.plot(cats, cum_pct, 'ro-', linewidth=2, markersize=6)
ax2.set_ylabel('Cumulative %', fontsize=12, color='red')
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% line')
ax2.legend(loc='lower right')

for bar, pct in zip(bars, rev_pct):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{pct:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart2_category_pareto.png'), dpi=150)
plt.close()
print(f"  Saved: chart2_category_pareto.png")

# ============================================================
# Chart 3: 月度销售趋势 + 环比
# ============================================================
print("Chart 3: 月度销售趋势")
cur.execute('''SELECT strftime('%Y-%m', order_date) AS mo,
    COUNT(*) AS oc, ROUND(SUM(total_amount),2) AS rev
FROM orders GROUP BY mo ORDER BY mo''')
rows = cur.fetchall()
months = [r[0] for r in rows]
orders_count = [r[1] for r in rows]
revenue = [r[2] for r in rows]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.fill_between(range(len(months)), revenue, alpha=0.3, color='#2196F3')
ax1.plot(range(len(months)), revenue, 'o-', color='#2196F3', linewidth=2, markersize=4)
ax1.set_ylabel('Monthly Revenue ($)', fontsize=12)
ax1.set_title('Q3: Monthly Revenue Trend', fontsize=14, fontweight='bold')
ax1.set_xticks(range(0, len(months), 3))
ax1.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.bar(range(len(months)), orders_count, alpha=0.3, color='#FF9800')
ax2.set_ylabel('Order Count', fontsize=12, color='#FF9800')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart3_monthly_trend.png'), dpi=150)
plt.close()
print(f"  Saved: chart3_monthly_trend.png")

# ============================================================
# Chart 4: 各 Segment 复购率
# ============================================================
print("Chart 4: 复购率分析")
cur.execute('''WITH uo AS (SELECT customer_id, COUNT(*) AS oc FROM orders GROUP BY customer_id)
SELECT c.segment, COUNT(DISTINCT c.customer_id) AS total,
    SUM(CASE WHEN uo.oc>=2 THEN 1 ELSE 0 END) AS rep
FROM customers c JOIN uo ON c.customer_id=uo.customer_id
GROUP BY c.segment ORDER BY rep DESC''')
rows = cur.fetchall()
segs = [r[0] for r in rows]
totals = [r[1] for r in rows]
reps = [r[2] for r in rows]
onetime = [t - r for t, r in zip(totals, reps)]
rates = [round(r/t*100, 1) for r, t in zip(reps, totals)]

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(segs))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], totals, width, label='One-time Buyers', color='#FF9800', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], reps, width, label='Repeat Buyers', color='#4CAF50', alpha=0.9)
ax.set_ylabel('User Count', fontsize=12)
ax.set_title('Q4: Repurchase Rate by Segment', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(segs)
ax.legend()

for i, (r, t) in enumerate(zip(reps, totals)):
    ax.text(i, t + 1, f'{rates[i]}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart4_repurchase_rate.png'), dpi=150)
plt.close()
print(f"  Saved: chart4_repurchase_rate.png")

# ============================================================
# Chart 5: 用户流失风险分布
# ============================================================
print("Chart 5: 流失风险")
cur.execute('''WITH lo AS (
    SELECT customer_id, MAX(order_date) AS last_dt, SUM(total_amount) AS ts
    FROM orders GROUP BY customer_id),
cr AS (SELECT c.customer_id,
    CAST(JULIANDAY('2026-06-24')-JULIANDAY(last_dt) AS INT) AS days,
    CASE WHEN JULIANDAY('2026-06-24')-JULIANDAY(last_dt)>90 THEN 'High Risk >90d'
    WHEN JULIANDAY('2026-06-24')-JULIANDAY(last_dt)>60 THEN 'Med Risk 60-90d'
    WHEN JULIANDAY('2026-06-24')-JULIANDAY(last_dt)>30 THEN 'Low Risk 30-60d'
    ELSE 'Active <30d' END AS risk
FROM customers c JOIN lo ON c.customer_id=lo.customer_id)
SELECT risk, COUNT(*) AS cnt FROM cr GROUP BY risk
ORDER BY MIN(days) DESC''')
rows = cur.fetchall()
risks = [r[0] for r in rows]
counts_risk = [r[1] for r in rows]
colors = ['#f44336','#FF9800','#FFEB3B','#4CAF50']

fig, ax = plt.subplots(figsize=(8, 5))
wedges, texts, autotexts = ax.pie(counts_risk, labels=risks, autopct='%1.1f%%',
    colors=colors, startangle=90, explode=(0.05,0,0,0))
ax.set_title('Q5: User Churn Risk Distribution', fontsize=14, fontweight='bold')
for at in autotexts:
    at.set_fontweight('bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart5_churn_risk.png'), dpi=150)
plt.close()
print(f"  Saved: chart5_churn_risk.png")

# ============================================================
# Chart 6: 高价值 vs 普通用户品类偏好对比
# ============================================================
print("Chart 6: 品类偏好对比")
cur.execute('''WITH uts AS (
    SELECT customer_id, SUM(total_amount) AS ts, NTILE(2) OVER (ORDER BY SUM(total_amount) DESC) AS tier FROM orders GROUP BY customer_id),
ul AS (SELECT customer_id, CASE WHEN tier=1 THEN 'High Value' ELSE 'Normal Value' END AS lv FROM uts),
cp AS (SELECT ul.lv, p.category, SUM(oi.quantity*oi.unit_price) AS rev
    FROM ul JOIN orders o ON ul.customer_id=o.customer_id
    JOIN order_items oi ON o.order_id=oi.order_id
    JOIN products p ON oi.product_id=p.product_id
    GROUP BY ul.lv, p.category),
lt AS (SELECT lv, SUM(rev) AS total FROM cp GROUP BY lv)
SELECT cp.lv, cp.category, ROUND(cp.rev*100.0/lt.total, 1) AS pct
FROM cp JOIN lt ON cp.lv=lt.lv ORDER BY cp.lv, cp.rev DESC''')
df = pd.DataFrame(cur.fetchall(), columns=['segment', 'category', 'pct'])

pivot = df.pivot(index='category', columns='segment', values='pct').fillna(0)

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(pivot.index))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], pivot['High Value'], width,
               label='High Value Users', color='#E91E63', alpha=0.9)
bars2 = ax.bar([i + width/2 for i in x], pivot['Normal Value'], width,
               label='Normal Users', color='#9E9E9E', alpha=0.7)
ax.set_ylabel('% of Total Spending', fontsize=12)
ax.set_title('Q6: Category Preference: High Value vs Normal Users', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=30, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'chart6_category_preference.png'), dpi=150)
plt.close()
print(f"  Saved: chart6_category_preference.png")

conn.close()
print(f"\n6 charts generated in: {OUT_DIR}")
