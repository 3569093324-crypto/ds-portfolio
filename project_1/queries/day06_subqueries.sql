-- ============================================================
-- Day 6: 子查询 — 在 WHERE / FROM / SELECT 中使用子查询
-- 数据库: business.db (电商场景)
-- ============================================================

-- ============================================================
-- 查询 1: 标量子查询 — 找出消费超过所有用户平均消费的用户
-- 标量子查询 = 返回单个值的子查询
-- ============================================================
SELECT c.customer_id,
       c.name,
       c.segment,
       ROUND(SUM(o.total_amount), 2) AS total_spent,
       (SELECT ROUND(AVG(total_amount), 2) FROM orders) AS overall_avg
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.segment
HAVING total_spent > (SELECT AVG(total_amount) FROM orders)
ORDER BY total_spent DESC
LIMIT 10;


-- ============================================================
-- 查询 2: WHERE 子查询（IN）— 找出购买了 Electronics 品类的用户
-- IN + 子查询：先查出目标品类商品ID，再反向找用户
-- ============================================================
SELECT DISTINCT c.customer_id,
       c.name,
       c.segment,
       c.city
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IN (
    SELECT DISTINCT oi.order_id
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    WHERE p.category = 'Electronics'
)
ORDER BY c.name
LIMIT 10;


-- ============================================================
-- 查询 3: 关联子查询 — 找出消费超过该城市平均消费的用户
-- 关联子查询 = 内层查询引用外层查询的列（逐行执行）
-- ============================================================
SELECT c1.customer_id,
       c1.name,
       c1.city,
       ROUND(SUM(o1.total_amount), 2) AS user_spent,
       (SELECT ROUND(AVG(o2.total_amount), 2)
        FROM orders o2
        JOIN customers c2 ON o2.customer_id = c2.customer_id
        WHERE c2.city = c1.city) AS city_avg
FROM customers c1
JOIN orders o1 ON c1.customer_id = o1.customer_id
GROUP BY c1.customer_id, c1.name, c1.city
HAVING user_spent > city_avg
ORDER BY user_spent DESC
LIMIT 10;


-- ============================================================
-- 查询 4: FROM 子查询 — 将聚合结果作为临时表再查询
-- 把月销售额统计作为子表，再在上面做环比分析
-- ============================================================
SELECT monthly.order_month,
       monthly.monthly_revenue,
       ROUND(monthly.avg_order_value, 2) AS avg_order_value,
       monthly.order_count,
       -- 计算环比增长（本月 vs 上月）
       ROUND(
           (monthly.monthly_revenue - COALESCE(prev.revenue_prev, monthly.monthly_revenue))
           * 100.0 / COALESCE(NULLIF(prev.revenue_prev, 0), monthly.monthly_revenue), 2
       ) AS revenue_growth_pct
FROM (
    SELECT strftime('%Y-%m', order_date) AS order_month,
           ROUND(SUM(total_amount), 2)   AS monthly_revenue,
           ROUND(AVG(total_amount), 2)   AS avg_order_value,
           COUNT(*)                      AS order_count
    FROM orders
    GROUP BY order_month
) AS monthly
LEFT JOIN (
    SELECT strftime('%Y-%m', order_date) AS order_month,
           ROUND(SUM(total_amount), 2)   AS revenue_prev
    FROM orders
    GROUP BY order_month
) AS prev ON prev.order_month = (
    SELECT strftime('%Y-%m', date(monthly.order_month || '-01', '-1 month'))
)
ORDER BY monthly.order_month;


-- ============================================================
-- 查询 5: SELECT 子查询 — 在 SELECT 中用子查询添加统计列
-- 为每位用户添加"全平台平均订单金额"和"用户排名"列
-- ============================================================
SELECT c.customer_id,
       c.name,
       COUNT(o.order_id) AS order_count,
       ROUND(SUM(o.total_amount), 2) AS total_spent,
       -- SELECT 中的标量子查询：全局平均值
       (SELECT ROUND(AVG(total_amount), 2) FROM orders) AS platform_avg,
       -- SELECT 中的标量子查询：用户总数
       (SELECT COUNT(DISTINCT customer_id) FROM orders) AS active_users,
       -- 用户自身 vs 平台均值的差异
       ROUND(SUM(o.total_amount) -
             (SELECT AVG(total_amount) FROM orders), 2) AS diff_from_avg
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 10;


-- ============================================================
-- 查询 6: EXISTS vs IN 对比
-- 两种方式找出有购买记录的用户
-- ============================================================

-- 6a: 使用 IN — 子查询先完整执行，外层用IN匹配
SELECT c.customer_id, c.name, c.segment
FROM customers c
WHERE c.customer_id IN (
    SELECT DISTINCT customer_id FROM orders
)
ORDER BY c.name
LIMIT 8;
-- IN 特点：子查询独立执行一次，适合子查询结果集小的场景
-- 如果子查询返回NULL，IN不会出错但不会匹配任何行

-- 6b: 使用 EXISTS — 外层每行执行一次子查询，找到第一个匹配就返回TRUE
SELECT c.customer_id, c.name, c.segment
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id
)
ORDER BY c.name
LIMIT 8;
-- EXISTS 特点：关联子查询，逐行检查，适合外层表小的场景
-- EXISTS 检查的是"是否存在"，不关心具体值，性能通常优于IN（大表场景）


-- ============================================================
-- 查询 7: NOT EXISTS — 找出从未下过单的用户
-- 对比：LEFT JOIN + IS NULL 方案
-- ============================================================

-- 7a: NOT EXISTS 方案
SELECT c.customer_id, c.name, c.city, c.segment
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id
)
ORDER BY c.join_date DESC
LIMIT 8;

-- 7b: LEFT JOIN + IS NULL 方案（等价，结果相同）
SELECT c.customer_id, c.name, c.city, c.segment
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL
ORDER BY c.join_date DESC
LIMIT 8;
-- 面试要点：两种写法等价，NOT EXISTS 语义更直观（"不存在订单"），
-- LEFT JOIN 在某些查询优化器中可能更快


-- ============================================================
-- 查询 8: 多层嵌套子查询（3层+）
-- 业务场景：找出购买过"销售额Top3品类中价格最高商品"的用户
-- 第1层：找出销售额Top3品类
-- 第2层：在每个Top3品类中找价格最高的商品
-- 第3层：找出购买过这些商品的用户
-- ============================================================
SELECT DISTINCT c.customer_id,
       c.name,
       c.segment,
       top_products.category,
       top_products.product_name,
       top_products.price AS top_product_price
FROM customers c
JOIN orders o      ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id   = oi.order_id
JOIN (
    -- 第2层子查询：Top3品类中价格最高的商品
    SELECT p.product_id,
           p.name     AS product_name,
           p.category,
           p.price
    FROM products p
    WHERE p.category IN (
        -- 第1层子查询：销售额Top3品类
        SELECT category
        FROM (
            SELECT p2.category,
                   SUM(oi2.quantity * oi2.unit_price) AS total_revenue
            FROM order_items oi2
            JOIN products p2 ON oi2.product_id = p2.product_id
            GROUP BY p2.category
            ORDER BY total_revenue DESC
            LIMIT 3
        ) AS top_categories
    )
    AND p.price = (
        -- 该品类中的最高价格
        SELECT MAX(p3.price)
        FROM products p3
        WHERE p3.category = p.category
    )
) AS top_products ON oi.product_id = top_products.product_id
ORDER BY top_products.category, c.name;


-- ============================================================
-- 补充：子查询 vs JOIN 选择指南
-- ============================================================
-- 用子查询的场景：
--   - 问题自然表达为"先算A，再基于A算B"（如"高于平均值的..."）
--   - 需要标量值（1行1列）作为比较基准
--   - 数据量小、可读性优先
--
-- 用 JOIN 的场景：
--   - 需要显示多张表的列
--   - 大表关联（查询优化器对JOIN优化更好）
--   - 需要更新/删除操作
--
-- 关联子查询 vs 非关联子查询：
--   - 非关联：子查询独立执行一次，结果用于外层
--   - 关联：外层每行触发一次子查询（引用外层列），适合逐行判断
--   - 性能：非关联通常更快（执行1次 vs N次），但EXISTS有时比IN快
