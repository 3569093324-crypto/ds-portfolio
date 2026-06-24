#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 62: 模拟 Python OA — 限时 60 分钟 5 题
=========================================
指令: 在不看答案的情况下完成以下5题
时间: 60分钟
"""

import numpy as np
import pandas as pd
import re
from collections import Counter

# ============================================================
# 题 1 (pandas数据清洗) — 15 分
# ============================================================
"""
题目: 给定以下脏数据，完成清洗：
  - 缺失值: age 用中位数填充, city 用 'Unknown' 填充
  - 重复行: 删除
  - 列名: 全部小写
  - score: 限制在 [0, 100] (超出范围的截断)

输入:
  data = {
    'Name': ['Alice', 'Bob', None, 'Alice', 'Eve'],
    'Age': [25, None, 30, 25, 200],
    'City': ['NYC', 'LA', None, 'NYC', 'SF'],
    'Score': [85, 92, -5, 85, 150]
  }
  df = pd.DataFrame(data)

输出: 清洗后的 DataFrame
"""

# --- 你的代码 ---
def clean_exam_data(df):
    pass  # TODO


# ============================================================
# 题 2 (pandas groupby+merge) — 20 分
# ============================================================
"""
题目:
  users.csv 有 user_id, name, city
  orders.csv 有 order_id, user_id, amount
  计算每个城市的:
    - 用户数
    - 总订单数
    - 总消费金额
    - 平均客单价
  按总消费金额降序排列，Top 5
"""

# --- 你的代码 ---
def city_analytics(users_df, orders_df):
    pass  # TODO


# ============================================================
# 题 3 (正则提取) — 20 分
# ============================================================
"""
题目: 从以下日志文本中提取所有:
  - IP地址
  - 时间戳
  - HTTP状态码
  返回一个 list of dict
"""

log_text = """
192.168.1.1 - [2024-01-15 10:30:00] "GET /api HTTP/1.1" 200
10.0.0.5 - [2024-01-15 10:31:00] "POST /login HTTP/1.1" 403
172.16.0.100 - [2024-01-15 10:32:00] "GET /home HTTP/1.1" 200
"""

# --- 你的代码 ---
def parse_logs(text):
    pass  # TODO


# ============================================================
# 题 4 (numpy向量化) — 20 分
# ============================================================
"""
题目: 不用 for 循环，计算以下:
  给定一个数组 scores shape=(1000, 5)，
  1. 对每列做 Z-score 标准化: (x - mean) / std
  2. 找出每行最大值所在的列索引
  3. 找出所有大于 2 倍标准差的"异常值"位置
"""

# --- 你的代码 ---
def numpy_vectorized_ops(scores):
    pass  # TODO


# ============================================================
# 题 5 (算法) — 25 分
# ============================================================
"""
题目:
  给定一个用户行为序列 logs = ['login','view','view','purchase','login','view','purchase']

  1. 用 Counter 统计每种行为的频次
  2. 找出最常见的3种行为
  3. 计算转化率: purchase / (login 之后的事件数)
  4. 找出连续相同行为的最长序列长度
     例: ['A','A','B','A','A','A'] → 最长连续是 3 (3个A)
"""

# --- 你的代码 ---
def behavior_analytics(logs):
    pass  # TODO


# ============================================================
# 答案 (做完后再看!)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Python OA 模拟 — 答案验证")
    print("=" * 60)

    # ANS 1
    data = {
        'Name': ['Alice', 'Bob', None, 'Alice', 'Eve'],
        'Age': [25, None, 30, 25, 200],
        'City': ['NYC', 'LA', None, 'NYC', 'SF'],
        'Score': [85, 92, -5, 85, 150]
    }
    df = pd.DataFrame(data)
    df_clean = df.copy()
    df_clean.columns = [c.lower() for c in df_clean.columns]
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['city'] = df_clean['city'].fillna('Unknown')
    df_clean['name'] = df_clean['name'].fillna('Unknown')
    df_clean['score'] = df_clean['score'].clip(0, 100)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"\nQ1: Cleaned df shape={df_clean.shape}, age nulls={df_clean['age'].isnull().sum()}")
    print(df_clean.to_string())

    # ANS 2
    users = pd.DataFrame({'user_id': [1,2,3,4,5], 'name': list('ABCDE'), 'city': ['NYC','LA','NYC','SF','LA']})
    orders = pd.DataFrame({'order_id': range(1,9), 'user_id': [1,1,2,3,3,3,4,5],
                           'amount': [100,200,150,300,250,100,50,400]})
    merged = users.merge(orders, on='user_id', how='left')
    city_stats = merged.groupby('city').agg(
        users=('user_id', 'nunique'), orders=('order_id', 'count'),
        total=('amount', 'sum')
    ).reset_index()
    city_stats['avg_order'] = (city_stats['total'] / city_stats['orders']).round(1)
    city_stats = city_stats.sort_values('total', ascending=False)
    print(f"\nQ2: City analytics:\n{city_stats.to_string()}")

    # ANS 3
    pattern = r'(\d+\.\d+\.\d+\.\d+).*?\[(.*?)\].*?"\w+ .*?"\s+(\d{3})'
    results = []
    for match in re.finditer(pattern, log_text):
        results.append({'ip': match.group(1), 'timestamp': match.group(2), 'status': match.group(3)})
    print(f"\nQ3: Parsed logs: {results}")

    # ANS 4
    np.random.seed(42)
    scores = np.random.randn(1000, 5) * 20 + 70
    z_scores = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    max_col_idx = np.argmax(scores, axis=1)
    outliers = np.where(np.abs(z_scores) > 2)
    print(f"\nQ4: z_scores shape={z_scores.shape}, max_col examples={max_col_idx[:10]}, outliers={len(outliers[0])}")

    # ANS 5
    logs = ['login','view','view','purchase','login','view','purchase']
    counter = Counter(logs)
    top3 = counter.most_common(3)
    purchases = counter.get('purchase', 0)
    logins = counter.get('login', 0)
    conv_rate = purchases / logins if logins > 0 else 0
    max_consecutive = 1
    current = 1
    for i in range(1, len(logs)):
        if logs[i] == logs[i-1]:
            current += 1
        else:
            max_consecutive = max(max_consecutive, current)
            current = 1
    max_consecutive = max(max_consecutive, current)
    print(f"\nQ5: Counter={counter}, Top3={top3}, Conversion={conv_rate:.2f}, MaxConsecutive={max_consecutive}")

    print("\n✅ Answer verification complete")
