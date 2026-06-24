-- ============================================================
-- Day 7: 窗口函数（一）— ROW_NUMBER, RANK, DENSE_RANK
-- 数据库: business.db (电商场景)
-- 📌 中国DS面试SQL OA最高频考点，占SQL题目50%+
-- ============================================================


-- ============================================================
-- 查询 1: ROW_NUMBER() — 每个品类内按价格排序编号
-- ROW_NUMBER() 为每行分配唯一序号，即使值相同也不并列
-- PARTITION BY = 分组窗口，ORDER BY = 窗口内排序
-- ============================================================
SELECT p.category,
       p.name,
       p.price,
       ROW_NUMBER() OVER (
           PARTITION BY p.category
           ORDER BY p.price DESC
       ) AS price_rank
FROM products p
ORDER BY p.category, price_rank;


-- ============================================================
-- 查询 2: RANK() vs DENSE_RANK() — 理解并列行为差异
-- 场景：按消费总额给所有用户排名
-- RANK()      = 有间隔排名（1, 2, 2, 4...）— 跳跃
-- DENSE_RANK()= 无间隔排名（1, 2, 2, 3...）— 连续
-- ROW_NUMBER()= 无并列排名（1, 2, 3, 4...）— 唯一
-- ============================================================
SELECT c.name,
       ROUND(SUM(o.total_amount), 2) AS total_spent,
       RANK()       OVER (ORDER BY SUM(o.total_amount) DESC) AS rank_gap,
       DENSE_RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS rank_dense,
       ROW_NUMBER() OVER (ORDER BY SUM(o.total_amount) DESC) AS rank_unique
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 15;

-- 📝 关键观察：
-- 如果有2个用户消费都是 ¥5000，排名表现为：
--   RANK:       3, 3, 5   （并列后跳过4）
--   DENSE_RANK: 3, 3, 4   （并列后连续）
--   ROW_NUMBER: 3, 4, 5   （随机决定谁3谁4，无并列）
-- 面试常问：有并列时你选哪个？答：看业务需求。


-- ============================================================
-- 查询 3: PARTITION BY + ORDER BY — 每个城市内用户消费排名
-- 窗口函数 + 分组 = 组内排名（面试必考）
-- ============================================================
SELECT c.city,
       c.name,
       ROUND(SUM(o.total_amount), 2) AS user_spent,
       RANK() OVER (
           PARTITION BY c.city
           ORDER BY SUM(o.total_amount) DESC
       ) AS city_rank
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.city
ORDER BY c.city, city_rank
LIMIT 20;


-- ============================================================
-- 查询 4: 用 ROW_NUMBER() 去重 — 保留每组中最新的一条记录
-- 经典用法：每个用户只保留最近一笔订单
-- ============================================================
SELECT *
FROM (
    SELECT o.order_id,
           o.customer_id,
           o.order_date,
           o.total_amount,
           ROW_NUMBER() OVER (
               PARTITION BY o.customer_id
               ORDER BY o.order_date DESC, o.order_id DESC
           ) AS rn
    FROM orders o
) AS ranked_orders
WHERE rn = 1
ORDER BY customer_id
LIMIT 15;

-- 📝 面试话术：
-- "我用 ROW_NUMBER() + PARTITION BY 分组，ORDER BY 时间降序，
--  取 rn=1 即为每组最新记录。这是 SQL 去重保留最新数据的标准做法。"


-- ============================================================
-- 查询 5: 找出每个品类中销售额前3的商品（🔥 经典面试题！）
-- 字节/美团/滴滴 OA 原题
-- ============================================================
SELECT category, product_name, total_revenue, cat_rank
FROM (
    SELECT p.category,
           p.name AS product_name,
           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue,
           ROW_NUMBER() OVER (
               PARTITION BY p.category
               ORDER BY SUM(oi.quantity * oi.unit_price) DESC
           ) AS cat_rank
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY p.category, p.product_id, p.name
) AS ranked_products
WHERE cat_rank <= 3
ORDER BY category, cat_rank;


-- ============================================================
-- 查询 6: 给每个用户的订单按时间顺序编号
-- 场景：用户行为路径分析——第1单、第2单、第3单...
-- ============================================================
SELECT c.name,
       o.order_id,
       o.order_date,
       o.total_amount,
       ROW_NUMBER() OVER (
           PARTITION BY o.customer_id
           ORDER BY o.order_date, o.order_id
       ) AS order_seq
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
ORDER BY c.name, order_seq
LIMIT 20;


-- ============================================================
-- 查询 7: NTILE(4) — 将用户按消费额分为4个等级
-- NTILE(N) 把结果集均匀分成N个桶
-- 场景：RFM模型中的消费分层（高/中高/中/低）
-- ============================================================
SELECT name,
       total_spent,
       tile AS spending_tier,
       CASE tile
           WHEN 1 THEN '高消费 (Top 25%)'
           WHEN 2 THEN '中高消费 (25-50%)'
           WHEN 3 THEN '中消费 (50-75%)'
           WHEN 4 THEN '低消费 (Bottom 25%)'
       END AS tier_label
FROM (
    SELECT c.name,
           ROUND(SUM(o.total_amount), 2) AS total_spent,
           NTILE(4) OVER (ORDER BY SUM(o.total_amount) DESC) AS tile
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.name
) AS tiered_users
ORDER BY total_spent DESC
LIMIT 15;


-- ============================================================
-- 面试速查卡：窗口函数对比
-- ============================================================
-- 函数           | 处理并列    | 适合场景
-- --------------|-----------|------------------
-- ROW_NUMBER()  | 不并列(随机) | 去重、编号、分页
-- RANK()        | 并列+跳跃   | 比赛排名（1,2,2,4）
-- DENSE_RANK()  | 并列+连续   | 等级排名（1,2,2,3）
-- NTILE(N)      | 均分N桶    | 分层分析、百分位
-- LAG/LEAD      | 不适用      | 环比、同比（见Day 8）
--
-- 窗口函数语法骨架：
-- function() OVER (PARTITION BY col1 ORDER BY col2 ROWS/RANGE ...)
--   PARTITION BY → 分组（可选）
--   ORDER BY     → 窗口内排序
--   ROWS/RANGE   → 窗口范围（可选，见Day 8）
