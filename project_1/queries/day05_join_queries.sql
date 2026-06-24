-- ============================================================
-- Day 5: 表连接 — INNER JOIN vs LEFT JOIN
-- 数据库: business.db (电商场景)
-- ============================================================

-- ============================================================
-- 查询 1: INNER JOIN orders + customers
-- 显示每笔订单及对应的用户名（只保留有匹配的行）
-- ============================================================
SELECT o.order_id,
       o.order_date,
       o.total_amount,
       c.name  AS customer_name,
       c.segment
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
ORDER BY o.order_date DESC
LIMIT 10;


-- ============================================================
-- 查询 2: LEFT JOIN customers + orders
-- 找出从未下过单的用户（右表无匹配 → NULL）
-- ============================================================
SELECT c.customer_id,
       c.name,
       c.city,
       c.segment,
       c.join_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL
ORDER BY c.join_date DESC;


-- ============================================================
-- 查询 3: INNER JOIN orders + order_items + products
-- 显示每笔订单的详细商品信息（3表JOIN）
-- ============================================================
SELECT o.order_id,
       o.order_date,
       c.name     AS customer_name,
       p.name     AS product_name,
       p.category,
       oi.quantity,
       oi.unit_price,
       ROUND(oi.quantity * oi.unit_price, 2) AS line_total
FROM orders o
INNER JOIN order_items oi ON o.order_id   = oi.order_id
INNER JOIN products p     ON oi.product_id = p.product_id
INNER JOIN customers c    ON o.customer_id = c.customer_id
ORDER BY o.order_id, p.category
LIMIT 15;


-- ============================================================
-- 查询 4: LEFT JOIN products + order_items
-- 找出从未被购买过的商品（冷门商品预警）
-- ============================================================
SELECT p.product_id,
       p.name,
       p.category,
       p.price
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.item_id IS NULL
ORDER BY p.category, p.price DESC;


-- ============================================================
-- 查询 5: 多表JOIN — 完整购买链路
-- 用户 → 订单 → 订单明细 → 商品，形成端到端的购买视图
-- ============================================================
SELECT c.customer_id,
       c.name         AS customer_name,
       c.segment,
       o.order_id,
       o.order_date,
       p.name         AS product_name,
       p.category,
       oi.quantity,
       oi.unit_price,
       ROUND(oi.quantity * oi.unit_price, 2) AS item_total,
       o.total_amount AS order_total
FROM customers c
INNER JOIN orders o      ON c.customer_id = o.customer_id
INNER JOIN order_items oi ON o.order_id   = oi.order_id
INNER JOIN products p    ON oi.product_id = p.product_id
ORDER BY o.order_date DESC, c.customer_id
LIMIT 20;


-- ============================================================
-- 查询 6: 别名简化 + JOIN后WHERE过滤
-- 找出VIP用户在2025年购买的所有Electronics品类商品
-- ============================================================
SELECT c.name   AS vip_name,
       o.order_date,
       p.name   AS product_name,
       p.price,
       oi.quantity,
       ROUND(oi.quantity * oi.unit_price, 2) AS spent
FROM customers c
JOIN orders o      ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id   = oi.order_id
JOIN products p    ON oi.product_id = p.product_id
WHERE c.segment = 'VIP'
  AND o.order_date >= '2025-01-01'
  AND p.category = 'Electronics'
ORDER BY o.order_date DESC, spent DESC
LIMIT 10;


-- ============================================================
-- 查询 7: INNER JOIN vs LEFT JOIN 对比
-- 同一查询场景，两种JOIN的结果差异（带注释说明）
-- ============================================================

-- 【场景】查询每位用户的订单数
-- 关键问题：如果某用户从未下单，INNER JOIN会直接丢弃该用户，
-- LEFT JOIN会保留该用户（订单数 = 0 或 NULL）

-- 7a: INNER JOIN — 只返回下过单的用户（丢掉了38个未下单用户）
SELECT c.customer_id, c.name,
       COUNT(o.order_id) AS order_count
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY order_count DESC;
-- 结果：仅 112 行（150个用户 - 38个未下单的）

-- 7b: LEFT JOIN — 保留所有用户，未下单的显示 order_count = 0
SELECT c.customer_id, c.name,
       COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY order_count DESC;
-- 结果：完整 150 行（含38个 order_count=0 的用户）

-- 面试关键结论：
-- INNER JOIN = 交集（只保留两表都有匹配的行）
-- LEFT JOIN  = 左表全保留（右表无匹配时填 NULL）
-- 用COUNT(o.order_id)而非COUNT(*)来正确统计订单数（COUNT忽略NULL）


-- ============================================================
-- 查询 8: SELF JOIN — 找出同一segment的用户对
-- 自连接场景：同等级用户交叉分析 → 可用于精准营销/社群运营
-- 注意：Faker生成的城市名几乎不重复，改用segment做SELF JOIN
-- 以保证查询有实际结果
-- ============================================================
SELECT a.name    AS user_a,
       b.name    AS user_b,
       a.segment AS shared_segment,
       a.city    AS user_a_city,
       b.city    AS user_b_city
FROM customers a
JOIN customers b ON a.segment = b.segment
WHERE a.customer_id < b.customer_id  -- 避免重复对和自己配自己
ORDER BY shared_segment, user_a
LIMIT 15;


-- ============================================================
-- 补充：JOIN 面试知识卡
-- ============================================================
-- INNER JOIN : 两表都有匹配才返回（最常用）
-- LEFT JOIN  : 左表全保留，右表无匹配填NULL（找"没有XXX的"）
-- RIGHT JOIN : 右表全保留（SQLite不支持，可用LEFT JOIN交换表序实现）
-- FULL JOIN  : 两表全保留（SQLite不支持，用LEFT JOIN + UNION + RIGHT JOIN模拟）
-- CROSS JOIN : 笛卡尔积（每行×每行），慎用——结果行数 = 左表行数 × 右表行数
--
-- JOIN 重复行陷阱：
-- 当右表有多条匹配时，左表的该行会被复制。例如一个用户有3个订单，
-- JOIN后该用户行出现3次。此时用SUM/COUNT聚合要特别注意去重：
--   COUNT(DISTINCT user_id) vs COUNT(*)
--   SUM(order_amount) 可能重复计算
