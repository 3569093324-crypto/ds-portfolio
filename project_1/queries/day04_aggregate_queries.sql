-- ============================================================
-- Day 4: 聚合查询 — GROUP BY + HAVING
-- 数据库: business.db (电商场景)
-- ============================================================

-- 查询 1: 统计每个品类的商品数量和平均价格
SELECT category,
       COUNT(*)        AS product_count,
       ROUND(AVG(price), 2) AS avg_price,
       ROUND(MIN(price), 2) AS min_price,
       ROUND(MAX(price), 2) AS max_price
FROM products
GROUP BY category
ORDER BY avg_price DESC;

-- 查询 2: 计算每个用户的订单数和总消费金额
SELECT c.customer_id,
       c.name,
       COUNT(o.order_id)              AS order_count,
       ROUND(SUM(o.total_amount), 2)  AS total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC;

-- 查询 3: 按月份统计订单量（用 strftime 提取年-月）
SELECT strftime('%Y-%m', order_date) AS order_month,
       COUNT(*)                      AS order_count,
       ROUND(SUM(total_amount), 2)   AS monthly_revenue,
       ROUND(AVG(total_amount), 2)   AS avg_order_value
FROM orders
GROUP BY order_month
ORDER BY order_month;

-- 查询 4: 找出订单数超过3单的用户（HAVING）
SELECT c.customer_id,
       c.name,
       c.segment,
       COUNT(o.order_id)              AS order_count,
       ROUND(SUM(o.total_amount), 2)  AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.segment
HAVING order_count > 3
ORDER BY order_count DESC, total_spent DESC;

-- 查询 5: 统计每个城市的用户数量和平均消费
SELECT c.city,
       COUNT(DISTINCT c.customer_id) AS user_count,
       COUNT(o.order_id)             AS order_count,
       ROUND(AVG(o.total_amount), 2) AS avg_order_value,
       ROUND(SUM(o.total_amount), 2) AS total_revenue
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY total_revenue DESC;

-- 查询 6: 计算每个品类的销售额排名
SELECT p.category,
       COUNT(DISTINCT p.product_id)    AS product_count,
       SUM(oi.quantity)                AS units_sold,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.category
ORDER BY total_revenue DESC;

-- 查询 7: 找出平均订单金额超过200元的用户
SELECT c.customer_id,
       c.name,
       c.segment,
       COUNT(o.order_id)              AS order_count,
       ROUND(AVG(o.total_amount), 2)  AS avg_order_amount,
       ROUND(SUM(o.total_amount), 2)  AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.segment
HAVING avg_order_amount > 200
ORDER BY avg_order_amount DESC;

-- 查询 8: 按用户等级(segment)统计人数和总消费
SELECT c.segment,
       COUNT(DISTINCT c.customer_id)  AS user_count,
       COUNT(o.order_id)              AS total_orders,
       ROUND(SUM(o.total_amount), 2)  AS total_revenue,
       ROUND(AVG(o.total_amount), 2)  AS avg_order_value,
       ROUND(SUM(o.total_amount) * 1.0 / COUNT(DISTINCT c.customer_id), 2) AS revenue_per_user
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.segment
ORDER BY total_revenue DESC;

-- 查询 9: 统计每天的订单趋势（按日聚合）
SELECT order_date,
       COUNT(*)                    AS daily_orders,
       ROUND(SUM(total_amount), 2) AS daily_revenue
FROM orders
GROUP BY order_date
ORDER BY order_date;

-- 查询 10: 找出购买过2个以上不同品类商品的用户
SELECT c.customer_id,
       c.name,
       COUNT(DISTINCT p.category)    AS categories_bought,
       COUNT(DISTINCT oi.product_id) AS distinct_products,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
FROM customers c
JOIN orders o         ON c.customer_id = o.customer_id
JOIN order_items oi   ON o.order_id = oi.order_id
JOIN products p       ON oi.product_id = p.product_id
GROUP BY c.customer_id, c.name
HAVING categories_bought > 2
ORDER BY categories_bought DESC, total_spent DESC;

-- ============================================================
-- 补充：COUNT(*) vs COUNT(column) vs COUNT(DISTINCT) 对比
-- 帮助理解面试常考的三种 COUNT 差异
-- ============================================================

-- COUNT(*): 统计所有行（含 NULL）
-- COUNT(column): 统计该列非 NULL 的行数
-- COUNT(DISTINCT column): 统计该列不重复的非 NULL 值数

-- 示例：对比三种 COUNT
SELECT COUNT(*)                  AS total_customers,
       COUNT(city)               AS customers_with_city,
       COUNT(DISTINCT city)      AS distinct_cities,
       COUNT(DISTINCT segment)   AS distinct_segments
FROM customers;
