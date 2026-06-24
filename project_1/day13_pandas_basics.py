#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 13: Pandas 基础 — 从 SQL 到 DataFrame
对比 SQL 和 Pandas 语法，复现 5 个核心 SQL 查询
"""

import sqlite3
import pandas as pd
import os

# ============================================================
# 1. 从 SQLite 读取数据到 DataFrame
# ============================================================
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "business.db")

conn = sqlite3.connect(DB_PATH)

# 一次性读取所有表
customers = pd.read_sql("SELECT * FROM customers", conn)
products  = pd.read_sql("SELECT * FROM products",  conn)
orders    = pd.read_sql("SELECT * FROM orders",    conn)
order_items = pd.read_sql("SELECT * FROM order_items", conn)

print("=" * 60)
print("  1. 数据加载完成")
print("=" * 60)
print(f"  customers:    {customers.shape}")
print(f"  products:     {products.shape}")
print(f"  orders:       {orders.shape}")
print(f"  order_items:  {order_items.shape}")

# ============================================================
# 2. DataFrame 基础操作
# ============================================================
print("\n" + "=" * 60)
print("  2. DataFrame 基础操作")
print("=" * 60)

print("\n  --- head() - 前5行预览 ---")
print(customers.head(3).to_string())

print("\n  --- info() - 结构概览 ---")
print(customers.info())

print("\n  --- describe() - 数值列统计 ---")
print(orders[['total_amount']].describe().to_string())

print("\n  --- value_counts() - 分类计数 ---")
print(customers['segment'].value_counts().to_string())

print("\n  --- shape / columns / dtypes ---")
print(f"  orders shape: {orders.shape}")
print(f"  orders columns: {list(orders.columns)}")
print(f"  orders dtypes:\n{orders.dtypes}")


# ============================================================
# 3. 用 Pandas 复现 5 个 SQL 查询（附 SQL vs Pandas 对比）
# ============================================================
print("\n" + "=" * 60)
print("  3. SQL vs Pandas：5 个核心查询对比")
print("=" * 60)

# --- Query 1: WHERE + ORDER BY ---
# SQL:  SELECT * FROM products WHERE price > 100 ORDER BY price DESC
print("\n  --- Query 1: 过滤 + 排序 ---")
print("  SQL:  SELECT * FROM products WHERE price > 100 ORDER BY price DESC")
print("  Pandas: products[products['price'] > 100].sort_values('price', ascending=False)")

pandas_q1 = products[products['price'] > 100].sort_values('price', ascending=False)
print(f"\n  结果: {len(pandas_q1)} 行")
print(pandas_q1[['name', 'category', 'price']].head(5).to_string())

# --- Query 2: GROUP BY + 聚合 ---
# SQL:  SELECT category, COUNT(*), AVG(price), SUM(price)
#       FROM products GROUP BY category ORDER BY AVG(price) DESC
print("\n  --- Query 2: GROUP BY + 聚合 ---")
print("  SQL:  SELECT category, COUNT(*), AVG(price), MAX(price)")
print("        FROM products GROUP BY category")
print("  Pandas: products.groupby('category')['price'].agg(['count','mean','max'])")

pandas_q2 = (products
    .groupby('category')
    .agg(
        product_count=('product_id', 'count'),
        avg_price=('price', 'mean'),
        max_price=('price', 'max'),
        total_cost=('cost', 'sum')
    )
    .round(2)
    .sort_values('avg_price', ascending=False)
)
print(f"\n{pandas_q2.to_string()}")

# --- Query 3: JOIN + 聚合 (替代 GROUP BY + JOIN) ---
# SQL:  SELECT c.name, COUNT(o.order_id), SUM(o.total_amount)
#       FROM customers c JOIN orders o ON c.customer_id = o.customer_id
#       GROUP BY c.customer_id, c.name
print("\n  --- Query 3: JOIN (merge) + 聚合 ---")
print("  SQL:  SELECT c.name, COUNT(o.order_id), SUM(o.total_amount)")
print("        FROM customers c JOIN orders o USING(customer_id)")
print("        GROUP BY c.customer_id, c.name")
print("  Pandas: merge then groupby")

# Pandas: merge → groupby
customer_orders = customers.merge(orders, on='customer_id', how='inner')
pandas_q3 = (customer_orders
    .groupby(['customer_id', 'name'])
    .agg(
        order_count=('order_id', 'count'),
        total_spent=('total_amount', 'sum')
    )
    .round(2)
    .sort_values('total_spent', ascending=False)
)
print(f"\n  结果: {len(pandas_q3)} 行")
print(pandas_q3.head(8).to_string())

# --- Query 4: 窗口函数 — 每个品类内价格排名 ---
# SQL:  SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS rn
#       FROM products
print("\n  --- Query 4: 窗口函数 (ROW_NUMBER) ---")
print("  SQL:  SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC)")
print("        FROM products")
print("  Pandas: groupby + rank(method='first')")

products_with_rank = products.copy()
products_with_rank['price_rank'] = (products_with_rank
    .groupby('category')['price']
    .rank(method='first', ascending=False)
    .astype(int)
)
top3_per_cat = products_with_rank[products_with_rank['price_rank'] <= 3]
print(f"\n  各品类 Top 3:")
print(top3_per_cat[['category', 'name', 'price', 'price_rank']]
      .sort_values(['category', 'price_rank']).to_string())

# --- Query 5: 子查询 — 消费超过平均值的用户 ---
# SQL:  SELECT * FROM (...) WHERE total_spent > (SELECT AVG(total_spent) FROM ...)
print("\n  --- Query 5: 子查询 (高于平均值) ---")
print("  SQL:  WITH t AS (SELECT customer_id, SUM(total_amount) AS spent")
print("        FROM orders GROUP BY customer_id)")
print("        SELECT * FROM t WHERE spent > (SELECT AVG(spent) FROM t)")
print("  Pandas: 先算均值，再过滤")

user_spent = orders.groupby('customer_id')['total_amount'].sum().reset_index()
avg_spent = user_spent['total_amount'].mean()
above_avg = user_spent[user_spent['total_amount'] > avg_spent]
above_avg = above_avg.merge(customers[['customer_id', 'name']], on='customer_id')
print(f"\n  均值: ¥{avg_spent:,.2f}")
print(f"  超过均值的用户: {len(above_avg)} 人")
print(above_avg.sort_values('total_amount', ascending=False).head(8).to_string())


# ============================================================
# 4. 数据透视表 — Pandas pivot_table vs SQL GROUP BY
# ============================================================
print("\n" + "=" * 60)
print("  4. 数据透视表 pivot_table")
print("=" * 60)
print("  场景：每个品类 x 每年 = 销售额矩阵（对比 SQL 的 CASE WHEN + GROUP BY）")

# 构建完整数据集
full_data = (order_items
    .merge(orders, on='order_id')
    .merge(products[['product_id', 'category']], on='product_id')
)
full_data['year'] = pd.to_datetime(full_data['order_date']).dt.year
full_data['revenue'] = full_data['quantity'] * full_data['unit_price']

# Pandas pivot_table
pivot = pd.pivot_table(
    full_data,
    values='revenue',
    index='category',
    columns='year',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='TOTAL'
).round(2)
print(f"\n{pivot.to_string()}")

print("\n  SQL 等价写法:")
print("  SELECT category,")
print("    SUM(CASE WHEN year=2024 THEN revenue END) AS 2024,")
print("    SUM(CASE WHEN year=2025 THEN revenue END) AS 2025,")
print("    ...")
print("  FROM ... GROUP BY category")


# ============================================================
# 5. 保存结果
# ============================================================
print("\n" + "=" * 60)
print("  5. 保存数据")
print("=" * 60)

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# CSV
csv_path = os.path.join(output_dir, "products.csv")
products.to_csv(csv_path, index=False)
print(f"  CSV:  {csv_path} ({os.path.getsize(csv_path):,} bytes)")

# Parquet (更高效的列式存储)
parquet_path = os.path.join(output_dir, "products.parquet")
products.to_parquet(parquet_path, index=False)
print(f"  Parquet: {parquet_path} ({os.path.getsize(parquet_path):,} bytes)")

# 对比大小
print(f"  Parquet 比 CSV 小 {os.path.getsize(csv_path) / os.path.getsize(parquet_path):.1f}x")


# ============================================================
# 6. 链式操作演示
# ============================================================
print("\n" + "=" * 60)
print("  6. Pandas 链式操作 (Method Chaining)")
print("=" * 60)
print("  一条链完成：过滤 → 分组 → 聚合 → 排序 → 取Top")

result = (full_data
    .query("category == 'Electronics'")                    # WHERE
    .groupby(['customer_id', 'order_id'])                   # GROUP BY
    .agg(total=('revenue', 'sum'))                          # SELECT SUM()
    .reset_index()
    .sort_values('total', ascending=False)                  # ORDER BY
    .head(10)                                               # LIMIT 10
    .merge(customers[['customer_id', 'name']], on='customer_id')  # JOIN
)
print(f"\n  Electronics 订单 Top 10:")
print(result[['name', 'order_id', 'total']].to_string())


# ============================================================
# 7. SQL ↔ Pandas 速查表
# ============================================================
print("\n" + "=" * 60)
print("  7. SQL ↔ Pandas 速查表")
print("=" * 60)

cheatsheet = """
  SQL                          | Pandas
  -----------------------------|----------------------------------
  SELECT col1, col2            | df[['col1', 'col2']]
  WHERE condition              | df[df['col'] > 100] 或 df.query()
  ORDER BY col DESC            | df.sort_values('col', ascending=False)
  LIMIT 10                     | df.head(10)
  GROUP BY col                 | df.groupby('col')
  COUNT(*), AVG(col)           | .agg(count=('col','count'), avg=('col','mean'))
  HAVING count > 3             | .query('count > 3') 在 agg 之后
  JOIN ... ON ...              | df.merge(other, on='key', how='inner')
  LEFT JOIN                    | df.merge(other, on='key', how='left')
  ROW_NUMBER() OVER (...)      | df.groupby().rank(method='first')
  CASE WHEN ... THEN ... END   | np.where() 或 pd.cut()
  DISTINCT                     | df['col'].unique() 或 df.drop_duplicates()
  UNION ALL                    | pd.concat([df1, df2])
  strftime('%Y-%m', date)      | pd.to_datetime(df['date']).dt.strftime('%Y-%m')
"""
print(cheatsheet)

conn.close()
print("\n✅ Day 13 Pandas 练习完成")
