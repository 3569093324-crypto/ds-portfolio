#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 17: 日期时间数据处理
涵盖：多种日期格式解析、提取、resample、rolling、时区、date_range
"""

import pandas as pd
import numpy as np
import sqlite3
import os

# ============================================================
# 1. 解析多种日期格式
# ============================================================
print("=" * 60)
print("  1. 解析多种日期格式")
print("=" * 60)

raw_dates = pd.Series([
    '2024-01-15',              # ISO format
    '01/20/2024',              # US MM/DD/YYYY
    '2024/03/10',              # YYYY/MM/DD
    '15-06-2024',              # DD-MM-YYYY
    1704067200,                # Unix timestamp (2024-01-01 00:00:00 UTC)
    '2024-W02-3',              # ISO week date (2024 Week 2 Wednesday)
    'invalid_date_here',       # 无效日期
    '2024-12-25 14:30:00',     # 带时间
])

# 使用 pd.to_datetime 统一解析
parsed = pd.to_datetime(raw_dates, errors='coerce', unit='s')
# 注意：Unix timestamp 的 unit='s' 会影响整列，所以需要分开处理

# 更稳健的方式：逐个尝试
def parse_mixed_dates(series):
    """稳健解析混合日期格式"""
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    # 先尝试默认解析（处理字符串格式）
    result = pd.to_datetime(series, errors='coerce')
    # 对仍然NaT且为数字的，尝试Unix timestamp
    still_nat = result.isna()
    numeric_mask = pd.to_numeric(series[still_nat], errors='coerce').notna()
    if numeric_mask.any():
        idx_to_fix = still_nat.index[still_nat][numeric_mask[still_nat].values]
        if len(idx_to_fix) > 0:
            result[idx_to_fix] = pd.to_datetime(
                pd.to_numeric(series[idx_to_fix]), unit='s'
            )
    return result

parsed_dates = parse_mixed_dates(raw_dates)
comparison = pd.DataFrame({'原始': raw_dates, '解析后': parsed_dates})
print(comparison.to_string())
print(f"\n  成功解析: {parsed_dates.notna().sum()}/{len(raw_dates)}")


# ============================================================
# 2. 提取日期组件
# ============================================================
print("\n" + "=" * 60)
print("  2. 提取日期组件")
print("=" * 60)

dates = pd.date_range('2024-01-01', periods=10, freq='D')
df_dates = pd.DataFrame({'date': dates})
df_dates['year']        = df_dates['date'].dt.year
df_dates['month']       = df_dates['date'].dt.month
df_dates['day']         = df_dates['date'].dt.day
df_dates['day_of_week'] = df_dates['date'].dt.dayofweek     # 0=Monday
df_dates['day_name']    = df_dates['date'].dt.day_name()
df_dates['quarter']     = df_dates['date'].dt.quarter
df_dates['is_weekend']  = df_dates['date'].dt.dayofweek >= 5
df_dates['week_of_year']= df_dates['date'].dt.isocalendar().week
df_dates['year_month']  = df_dates['date'].dt.strftime('%Y-%m')

print(df_dates.to_string())
print(f"\n  组件提取: year, month, day, dayofweek, day_name, quarter, is_weekend, week_of_year")


# ============================================================
# 3. 日期运算
# ============================================================
print("\n" + "=" * 60)
print("  3. 日期运算")
print("=" * 60)

# 两个日期之间的天数
d1 = pd.Timestamp('2024-01-01')
d2 = pd.Timestamp('2026-06-24')
diff_days = (d2 - d1).days
print(f"  2024-01-01 → 2026-06-24: {diff_days} 天")

# 日期加减
print(f"  今天 + 30天: {pd.Timestamp.now().normalize() + pd.Timedelta(days=30)}")
print(f"  今天 - 7天:   {pd.Timestamp.now().normalize() - pd.Timedelta(days=7)}")
print(f"  月初: {pd.Timestamp.now().normalize() - pd.offsets.MonthBegin()}")
print(f"  月末: {pd.Timestamp.now().normalize() + pd.offsets.MonthEnd()}")

# 工作日计算
from pandas.tseries.offsets import BDay
print(f"  10个工作日后的日期: {pd.Timestamp('2024-01-01') + BDay(10)}")

# 计算年龄
birth_date = pd.Timestamp('1998-06-15')
today = pd.Timestamp('2026-06-24')
age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
print(f"  1998-06-15 出生的人在 2026-06-24 的年龄: {age} 岁")


# ============================================================
# 4. 重采样 (resample) — 日→周/月
# ============================================================
print("\n" + "=" * 60)
print("  4. 重采样 (resample)")
print("=" * 60)

# 模拟每日销售数据
daily = pd.DataFrame({
    'date': pd.date_range('2024-01-01', '2024-06-30', freq='D'),
    'sales': np.random.randint(100, 1000, 182)
}).set_index('date')

print(f"  原始日数据: {len(daily)} 天")
print(f"  前5天:\n{daily.head()}")

# 周聚合
weekly = daily.resample('W-MON').agg({
    'sales': ['sum', 'mean', 'count']
})
weekly.columns = ['weekly_sales', 'avg_daily', 'days']
print(f"\n  周聚合 (W-MON): {len(weekly)} 周")
print(weekly.head().to_string())

# 月聚合
monthly = daily.resample('ME').agg({
    'sales': ['sum', 'mean', 'min', 'max']
})
monthly.columns = ['total', 'avg', 'min', 'max']
print(f"\n  月聚合 (ME): {len(monthly)} 月")
print(monthly.to_string())


# ============================================================
# 5. Rolling — 7日/30日移动平均
# ============================================================
print("\n" + "=" * 60)
print("  5. Rolling — 移动平均")
print("=" * 60)

daily['ma_7'] = daily['sales'].rolling(window=7, center=True).mean()
daily['ma_30'] = daily['sales'].rolling(window=30, center=True).mean()

# 扩展窗口（从开始到现在）
daily['expanding_avg'] = daily['sales'].expanding().mean()

print(f"  原始 + 7日MA + 30日MA + 累计均值:")
print(daily.head(15)[['sales', 'ma_7', 'ma_30', 'expanding_avg']].round(1).to_string())


# ============================================================
# 6. 时区处理 — UTC → 北京
# ============================================================
print("\n" + "=" * 60)
print("  6. 时区处理")
print("=" * 60)

# 创建 UTC 时间戳
utc_times = pd.date_range('2024-06-01', periods=5, freq='6h', tz='UTC')
print(f"  UTC 时间:\n{utc_times}")

# 转换为北京时间 (UTC+8)
beijing_times = utc_times.tz_convert('Asia/Shanghai')
print(f"\n  北京时间 (UTC+8):\n{beijing_times}")

# 处理不带时区的时间戳（假设是UTC）
naive_times = pd.to_datetime(['2024-06-01 12:00:00', '2024-06-01 18:00:00',
                               '2024-06-02 00:00:00', '2024-06-02 06:00:00'])
# 1. 先本地化到UTC
localized = naive_times.tz_localize('UTC')
# 2. 再转换到北京时间
in_beijing = localized.tz_convert('Asia/Shanghai')
print(f"\n  原始 (视为UTC):   {naive_times.tolist()}")
print(f"  北京时间:          {in_beijing.tolist()}")


# ============================================================
# 7. date_range — 填充缺失日期
# ============================================================
print("\n" + "=" * 60)
print("  7. date_range — 构造日期范围")
print("=" * 60)

# 模拟有缺失日期的数据
sparse = pd.DataFrame({
    'date': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05',
                             '2024-01-08', '2024-01-12']),
    'value': [10, 20, 15, 30, 25]
})

print(f"  原始稀疏数据:\n{sparse}")

# 构造完整日期范围
full_range = pd.date_range(
    start=sparse['date'].min(),
    end=sparse['date'].max(),
    freq='D'
)
full_df = pd.DataFrame({'date': full_range})
filled = full_df.merge(sparse, on='date', how='left')
print(f"\n  填充后（缺失日期 value=NaN）:\n{filled.to_string()}")

# 前向填充
filled['value_ffill'] = filled['value'].ffill()
print(f"\n  前向填充后:\n{filled.to_string()}")


# ============================================================
# 8. 实际应用：business.db 时间维度分析
# ============================================================
print("\n" + "=" * 60)
print("  8. 实际应用 — business.db 时间维度分析")
print("=" * 60)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "business.db")
conn = sqlite3.connect(DB_PATH)

orders = pd.read_sql("SELECT * FROM orders", conn)
orders['order_date'] = pd.to_datetime(orders['order_date'])

print(f"  订单日期范围: {orders['order_date'].min()} → {orders['order_date'].max()}")

# 8a. 按周统计订单量
orders_weekly = (orders
    .set_index('order_date')
    .resample('W-MON')
    .agg(
        order_count=('order_id', 'count'),
        total_revenue=('total_amount', 'sum')
    )
)
print(f"\n  周度订单趋势 (最近5周):")
print(orders_weekly.tail(5).round(1).to_string())

# 8b. 按星期几分组
orders['day_of_week'] = orders['order_date'].dt.day_name()
orders['is_weekend'] = orders['order_date'].dt.dayofweek >= 5
dow_stats = orders.groupby('day_of_week').agg(
    orders=('order_id', 'count'),
    avg_revenue=('total_amount', 'mean')
).round(1)
# 按周一→周日排序
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_stats = dow_stats.reindex(dow_order)
print(f"\n  按星期几分组:")
print(dow_stats.to_string())

# 8c. 每月新用户注册趋势（从customers表）
customers = pd.read_sql("SELECT * FROM customers", conn)
customers['join_date'] = pd.to_datetime(customers['join_date'])
customers['join_month'] = customers['join_date'].dt.strftime('%Y-%m')
monthly_joins = customers.groupby('join_month').size()
print(f"\n  月度新用户注册 (最近5个月):")
print(monthly_joins.tail(5).to_string())

# 8d. 季度汇总
orders['quarter'] = orders['order_date'].dt.quarter
orders['year'] = orders['order_date'].dt.year
quarterly = orders.groupby(['year', 'quarter']).agg(
    orders=('order_id', 'count'),
    revenue=('total_amount', 'sum')
).round(1)
print(f"\n  季度汇总:")
print(quarterly.to_string())

conn.close()

print("\n" + "=" * 60)
print("  Day 17 总结：Pandas DateTime 核心能力")
print("=" * 60)
print("""
  1. pd.to_datetime() — 解析几乎所有日期格式
  2. .dt accessor — 提取 year/month/day/dayofweek/quarter...
  3. .resample() — 日→周/月/季/年聚合
  4. .rolling() — 移动窗口（7日MA、30日MA）
  5. .tz_localize() + .tz_convert() — 时区处理
  6. pd.date_range() — 构造日期序列、填充缺失
  7. pd.Timedelta / pd.offsets — 日期运算
""")
