#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 12: SQL 查询优化 — EXPLAIN 与索引
扩展 order_items 到 10000+ 行，对比有/无索引的查询性能
"""

import sqlite3
import os
import time
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "business.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("=" * 60)
print("  Day 12: SQL 查询优化实验")
print("=" * 60)

# ============================================================
# 0. 获取现有 ID 列表
# ============================================================
cur.execute("SELECT product_id, price FROM products")
products = cur.fetchall()

cur.execute("SELECT order_id, customer_id, order_date FROM orders")
orders = cur.fetchall()

print(f"\n当前数据规模:")
print(f"  products:     {len(products)} 行")
print(f"  orders:       {len(orders)} 行")
cur.execute("SELECT COUNT(*) FROM order_items")
current_oi = cur.fetchone()[0]
print(f"  order_items:  {current_oi} 行")

# ============================================================
# 1. 扩展 order_items 到 15000+ 行
# ============================================================
print(f"\n正在扩展 order_items 到 15000+ 行...")

new_items = []
target = 15000
needed = target - current_oi

for _ in range(needed):
    oid = random.choice(orders)[0]
    pid, price = random.choice(products)
    qty = random.randint(1, 5)
    new_items.append((oid, pid, qty, price))

cur.executemany(
    "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
    new_items,
)

# 回填订单总额
cur.execute("""
    UPDATE orders SET total_amount = (
        SELECT COALESCE(SUM(quantity * unit_price), 0.01)
        FROM order_items WHERE order_items.order_id = orders.order_id
    )
""")

conn.commit()

cur.execute("SELECT COUNT(*) FROM order_items")
final_oi = cur.fetchone()[0]
print(f"  order_items 扩展至: {final_oi} 行 ✓")

# ============================================================
# 2. 准备性能测试辅助函数
# ============================================================
def benchmark(label, sql, params=()):
    """执行查询并计时，返回 (结果行数, 耗时ms)"""
    start = time.perf_counter()
    cur.execute(sql, params)
    rows = cur.fetchall()
    elapsed = (time.perf_counter() - start) * 1000
    count = len(rows)
    print(f"  {label:40s}  {count:>6} rows  {elapsed:>8.2f} ms")
    return count, elapsed

def explain(sql):
    """打印 EXPLAIN QUERY PLAN"""
    cur.execute(f"EXPLAIN QUERY PLAN {sql}")
    plan = cur.fetchall()
    for row in plan:
        print(f"    {row[0]} | {row[1]} | {row[2]} | {row[3]}")
    return plan

print("\n" + "=" * 60)
print("  2. 性能对比实验")
print("=" * 60)

# ============================================================
# 3. 实验1: 单列索引 — orders.customer_id
# ============================================================
print("\n--- 实验1: 主键查询（基准）---")
benchmark("PK lookup (customer_id=50)",
    "SELECT * FROM customers WHERE customer_id = ?", (50,))

print("\n--- 实验2: 索引 vs 无索引（模拟）---")
# 先确认有索引
print("  现有索引:")
cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
for idx in cur.fetchall():
    cur.execute(f"PRAGMA index_info({idx[0]})")
    cols = [c[2] for c in cur.fetchall()]
    print(f"    - {idx[0]} ({', '.join(cols)})")

print("\n  有索引查询 (orders.customer_id, idx_orders_customer_id):")
benchmark("JOIN with index",
    """SELECT c.name, COUNT(o.order_id)
       FROM customers c JOIN orders o ON c.customer_id = o.customer_id
       GROUP BY c.customer_id""")

# 演示：如果索引不存在会怎样（用 EXPLAIN 展示）
print("\n  EXPLAIN QUERY PLAN:")
explain("""SELECT c.name, COUNT(o.order_id)
    FROM customers c JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.customer_id = 50
    GROUP BY c.customer_id""")

# ============================================================
# 4. 实验3: 复合索引 + 最左前缀原则
# ============================================================
print("\n--- 实验3: 复合索引 + 最左前缀 ---")

# 创建复合索引
try:
    cur.execute("CREATE INDEX idx_oi_order_product ON order_items(order_id, product_id)")
    print("  创建复合索引: idx_oi_order_product (order_id, product_id)")
except sqlite3.OperationalError:
    print("  复合索引已存在")

conn.commit()

# 测试1: 使用最左列 (order_id) → 应该用索引
print("\n  测试1: WHERE order_id = ? (使用最左列 → 可用索引)")
explain("SELECT * FROM order_items WHERE order_id = 100")

# 测试2: 使用两列 → 应该用索引
print("\n  测试2: WHERE order_id = ? AND product_id = ? (使用全部列 → 可用索引)")
explain("SELECT * FROM order_items WHERE order_id = 100 AND product_id = 50")

# 测试3: 只使用第二列 (product_id) → 无法用复合索引的最左前缀
print("\n  测试3: WHERE product_id = ? (跳过最左列 → 索引失效)")
explain("SELECT * FROM order_items WHERE product_id = 50")

# 说明：product_id 上有单独索引，所以实际还是会用 idx_order_items_product_id
print("  (注: product_id 有单独索引 idx_order_items_product_id，所以仍会走索引)")

# 掉单独索引来演示最左前缀原则
cur.execute("DROP INDEX IF EXISTS idx_order_items_product_id")
print("\n  测试4: 删除 product_id 单独索引后，WHERE product_id = ?")
explain("SELECT * FROM order_items WHERE product_id = 50")
print("  → SCAN TABLE order_items (无法用复合索引，因为跳过了最左列 order_id)")

# 恢复
cur.execute("CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id)")
conn.commit()

# ============================================================
# 5. 实验4: 子查询 vs JOIN 性能对比
# ============================================================
print("\n--- 实验4: 子查询 vs JOIN ---")

print("  JOIN 方式:")
benchmark("INNER JOIN",
    """SELECT DISTINCT c.name FROM customers c
       JOIN orders o ON c.customer_id = o.customer_id
       JOIN order_items oi ON o.order_id = oi.order_id
       JOIN products p ON oi.product_id = p.product_id
       WHERE p.category = 'Electronics'""")

print("  子查询 (IN) 方式:")
benchmark("WHERE ... IN (subquery)",
    """SELECT name FROM customers WHERE customer_id IN (
       SELECT DISTINCT o.customer_id FROM orders o
       JOIN order_items oi ON o.order_id = oi.order_id
       JOIN products p ON oi.product_id = p.product_id
       WHERE p.category = 'Electronics')""")

print("  子查询 (EXISTS) 方式:")
benchmark("WHERE EXISTS (subquery)",
    """SELECT name FROM customers c WHERE EXISTS (
       SELECT 1 FROM orders o
       JOIN order_items oi ON o.order_id = oi.order_id
       JOIN products p ON oi.product_id = p.product_id
       WHERE o.customer_id = c.customer_id AND p.category = 'Electronics')""")

# ============================================================
# 6. 实验5: 分析 Day 11 业务查询
# ============================================================
print("\n--- 实验5: Day 11 业务查询 EXPLAIN 分析 ---")

# 复购率查询
print("  复购率查询 EXPLAIN:")
explain("""WITH uo AS (
    SELECT customer_id, COUNT(*) AS oc, SUM(total_amount) AS ts FROM orders GROUP BY customer_id)
SELECT c.segment, COUNT(DISTINCT c.customer_id) AS total,
    SUM(CASE WHEN uo.oc>=2 THEN 1 ELSE 0 END) AS rep
FROM customers c JOIN uo ON c.customer_id=uo.customer_id GROUP BY c.segment""")

# 流失查询
print("\n  流失查询 EXPLAIN:")
explain("""SELECT c.customer_id, c.name, MAX(o.order_date) AS last_order
FROM customers c JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id""")

print("\n" + "=" * 60)
print("  优化实验完成")
print("=" * 60)

conn.close()
