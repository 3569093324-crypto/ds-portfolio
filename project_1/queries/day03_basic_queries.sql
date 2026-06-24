-- ============================================================
-- Day 3: 基础查询 — SELECT, WHERE, ORDER BY
-- 数据库: business.db (电商场景)
-- ============================================================

-- 查询 1: 找出所有2024年之后注册的用户，按注册日期降序排列
SELECT customer_id, name, city, join_date, segment
FROM customers
WHERE join_date >= '2024-01-01'
ORDER BY join_date DESC;

-- 查询 2: 查询价格高于100的产品，按价格降序显示
SELECT product_id, name, category, price
FROM products
WHERE price > 100
ORDER BY price DESC;

-- 查询 3: 找出订单总金额大于500的订单
SELECT order_id, customer_id, order_date, total_amount
FROM orders
WHERE total_amount > 500
ORDER BY total_amount DESC;

-- 查询 4: 查询城市名包含 'New' 的用户列表（LIKE 模糊匹配）
SELECT customer_id, name, city, join_date
FROM customers
WHERE city LIKE '%New%'
ORDER BY city, name;

-- 查询 5: 找出产品名中包含特定关键词的产品（LIKE）
-- 示例：搜索包含 'Pro' 或 'Premium' 的产品
SELECT product_id, name, category, price
FROM products
WHERE name LIKE '%Pro%'
   OR name LIKE '%Premium%'
ORDER BY price DESC;

-- 查询 6: 使用 BETWEEN 查询价格在 50 到 200 之间的产品
SELECT product_id, name, category, price
FROM products
WHERE price BETWEEN 50 AND 200
ORDER BY price DESC;

-- 查询 7: 使用 IN 查询特定品类的产品
SELECT product_id, name, category, price
FROM products
WHERE category IN ('Electronics', 'Books', 'Sports')
ORDER BY category, price DESC;

-- 查询 8: 找出从未下过订单的用户（LEFT JOIN + IS NULL）
-- 这是面试中 IS NULL 的经典实战用法
SELECT c.customer_id, c.name, c.city, c.segment
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL
ORDER BY c.join_date DESC;

-- 查询 9: 使用 DISTINCT 查询所有不重复的产品品类
SELECT DISTINCT category
FROM products
ORDER BY category;

-- 查询 10: 组合 AND/OR 进行复合条件查询
-- 查找：Electronics 品类中价格 > 500 的产品，
--       或 Home 品类中价格 > 100 的产品
SELECT product_id, name, category, price
FROM products
WHERE (category = 'Electronics' AND price > 500)
   OR (category = 'Home' AND price > 100)
ORDER BY category, price DESC;
