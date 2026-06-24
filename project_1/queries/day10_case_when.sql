-- ============================================================
-- Day 10: CASE WHEN 与条件逻辑
-- 数据库: business.db (电商场景)
-- ============================================================


-- ============================================================
-- 查询 1: 用户消费分层 — 3档分类
-- 场景：RFM模型中的 Monetary（消费金额）维度
-- ============================================================
SELECT c.name,
       c.segment,
       ROUND(COALESCE(SUM(o.total_amount), 0), 2) AS total_spent,
       COUNT(o.order_id) AS order_count,
       CASE
           WHEN SUM(o.total_amount) > 10000 THEN '高价值'
           WHEN SUM(o.total_amount) BETWEEN 5000 AND 10000 THEN '中价值'
           WHEN SUM(o.total_amount) > 0 THEN '普通'
           ELSE '无消费'
       END AS value_tier
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.segment
ORDER BY total_spent DESC
LIMIT 15;


-- ============================================================
-- 查询 2: 在 GROUP BY 中使用 CASE WHEN 做交叉统计
-- 场景：按"年+价值层级"交叉统计订单
-- ============================================================
SELECT
    strftime('%Y', o.order_date) AS order_year,
    CASE
        WHEN o.total_amount > 5000 THEN '大额订单'
        WHEN o.total_amount BETWEEN 1000 AND 5000 THEN '中额订单'
        ELSE '小额订单'
    END AS order_tier,
    COUNT(*) AS order_count,
    ROUND(SUM(o.total_amount), 2) AS total_revenue,
    ROUND(AVG(o.total_amount), 2) AS avg_amount
FROM orders o
GROUP BY order_year, order_tier
ORDER BY order_year, order_count DESC;


-- ============================================================
-- 查询 3: 多条件 CASE — 按品类和价格综合打标签
-- 场景：商品策略——识别高利润品类中的高价商品（重点推广）
-- ============================================================
SELECT p.name AS product_name,
       p.category,
       p.price,
       p.cost,
       ROUND(p.price - p.cost, 2) AS profit_margin,
       ROUND((p.price - p.cost) * 100.0 / p.price, 1) AS margin_pct,
       CASE
           WHEN p.price > 1000 AND (p.price - p.cost) > 500
               THEN '明星商品：高价高利润'
           WHEN p.price > 1000 AND (p.price - p.cost) <= 500
               THEN '流量商品：高价低利润'
           WHEN p.price <= 1000 AND (p.price - p.cost) > 200
               THEN '现金牛：中价高利润'
           WHEN p.price <= 1000 AND (p.price - p.cost) <= 200
               THEN '长尾商品：中价低利润'
           ELSE '未分类'
       END AS product_tag
FROM products p
ORDER BY p.price DESC
LIMIT 15;


-- ============================================================
-- 查询 4: CASE WHEN + 聚合 — 统计各标签的用户数/订单数/总消费
-- 全景式统计每个价值层级的核心指标
-- ============================================================
WITH
labeled_users AS (
    SELECT c.customer_id,
           COALESCE(SUM(o.total_amount), 0) AS total_spent,
           COUNT(o.order_id) AS order_count,
           CASE
               WHEN SUM(o.total_amount) > 15000 THEN '钻石用户'
               WHEN SUM(o.total_amount) BETWEEN 5000 AND 15000 THEN '金卡用户'
               WHEN SUM(o.total_amount) BETWEEN 500 AND 5000 THEN '银卡用户'
               WHEN SUM(o.total_amount) > 0 THEN '铜卡用户'
               ELSE '未激活'
           END AS user_level
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id
)
SELECT user_level,
       COUNT(*) AS user_count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS user_pct,
       SUM(order_count) AS total_orders,
       ROUND(AVG(order_count), 1) AS avg_orders_per_user,
       ROUND(SUM(total_spent), 2) AS total_revenue,
       ROUND(AVG(total_spent), 2) AS avg_spent_per_user
FROM labeled_users
GROUP BY user_level
ORDER BY MIN(CASE user_level
    WHEN '钻石用户' THEN 1 WHEN '金卡用户' THEN 2
    WHEN '银卡用户' THEN 3 WHEN '铜卡用户' THEN 4 ELSE 5 END);


-- ============================================================
-- 查询 5: 用 CASE WHEN 替代多个 UNION
-- 场景：一份查询同时输出"整体统计"和"各segment统计"
-- 传统做法：2个SELECT用UNION ALL
-- CASE WHEN做法：1次扫描，用GROUPING SETS替代逻辑
-- ============================================================

-- 【传统 UNION 方式 — 扫描2次表】
/*
SELECT 'ALL' AS segment, COUNT(*) AS user_count, ROUND(AVG(total_spent),2) AS avg_spent
FROM (SELECT c.customer_id, SUM(o.total_amount) AS total_spent ...)
UNION ALL
SELECT c.segment, COUNT(*), ROUND(AVG(SUM(o.total_amount)),2)
FROM ... GROUP BY c.segment;
*/

-- 【CASE WHEN 方式 — 1次扫描】
-- 注意：SQLite不支持GROUPING SETS，这里演示条件聚合的等价写法
SELECT
    COALESCE(c.segment, 'ALL') AS segment,
    COUNT(DISTINCT c.customer_id) AS user_count,
    COUNT(o.order_id) AS order_count,
    ROUND(COALESCE(SUM(o.total_amount), 0), 2) AS total_revenue
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.segment
-- 再 UNION 一个总计行（模拟 GROUPING SETS）
UNION ALL
SELECT
    'TOTAL' AS segment,
    COUNT(DISTINCT c.customer_id),
    COUNT(o.order_id),
    ROUND(COALESCE(SUM(o.total_amount), 0), 2)
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
ORDER BY segment;

-- 说明：纯 CASE WHEN 替代多个 UNION 的典型场景是
-- 将不同条件下的统计值放在同一行的不同列，例如：
-- SELECT
--   SUM(CASE WHEN segment='VIP' THEN 1 ELSE 0 END) AS vip_count,
--   SUM(CASE WHEN segment='Retail' THEN 1 ELSE 0 END) AS retail_count
-- FROM customers;
-- 这样一次查询就能得到所有分类的计数，无需多个UNION


-- ============================================================
-- 查询 6: 条件聚合 — CASE WHEN inside SUM/COUNT
-- 面试高频考点：在聚合函数内嵌CASE WHEN
-- 场景：一次性统计各品类中"高价商品"和"低价商品"的数量和销售额
-- ============================================================
SELECT p.category,
       -- 商品数统计
       COUNT(*) AS total_products,
       SUM(CASE WHEN p.price > 500 THEN 1 ELSE 0 END) AS high_price_count,
       SUM(CASE WHEN p.price <= 500 THEN 1 ELSE 0 END) AS low_price_count,
       -- 销售额统计（仅已售出商品）
       SUM(CASE WHEN p.price > 500
           THEN COALESCE(s.sold_revenue, 0) ELSE 0 END) AS high_price_revenue,
       SUM(CASE WHEN p.price <= 500
           THEN COALESCE(s.sold_revenue, 0) ELSE 0 END) AS low_price_revenue
FROM products p
LEFT JOIN (
    SELECT product_id,
           SUM(quantity * unit_price) AS sold_revenue
    FROM order_items
    GROUP BY product_id
) s ON p.product_id = s.product_id
GROUP BY p.category
ORDER BY total_products DESC;


-- ============================================================
-- 查询 7: 数据透视（Pivot）— 用 CASE WHEN + GROUP BY 行转列
-- 场景：横向展示每个品类在各年份的销售额（列 = 年份，行 = 品类）
-- ============================================================
SELECT p.category,
       ROUND(SUM(CASE WHEN strftime('%Y', o.order_date) = '2024'
           THEN oi.quantity * oi.unit_price ELSE 0 END), 2) AS revenue_2024,
       ROUND(SUM(CASE WHEN strftime('%Y', o.order_date) = '2025'
           THEN oi.quantity * oi.unit_price ELSE 0 END), 2) AS revenue_2025,
       ROUND(SUM(CASE WHEN strftime('%Y', o.order_date) = '2026'
           THEN oi.quantity * oi.unit_price ELSE 0 END), 2) AS revenue_2026,
       -- 总计列
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue_total
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o      ON oi.order_id  = o.order_id
GROUP BY p.category
ORDER BY revenue_total DESC;


-- ============================================================
-- 面试速查卡：CASE WHEN 常见模式
-- ============================================================
-- 1. 简单分类（替代多个WHERE条件）:
--    CASE WHEN condition1 THEN 'A' WHEN condition2 THEN 'B' ELSE 'C' END
--
-- 2. 条件聚合（CASE inside SUM/COUNT/AVG）:
--    SUM(CASE WHEN condition THEN value ELSE 0 END)
--    COUNT(CASE WHEN condition THEN 1 ELSE NULL END)
--    -- 注意：COUNT(NULL)=0, SUM(0)=0
--
-- 3. 行转列（Pivot）:
--    SELECT group_col,
--           SUM(CASE WHEN pivot_col='A' THEN value ELSE 0 END) AS col_A,
--           SUM(CASE WHEN pivot_col='B' THEN value ELSE 0 END) AS col_B
--    FROM t GROUP BY group_col;
--
-- 4. 在ORDER BY中使用CASE实现自定义排序:
--    ORDER BY CASE status WHEN 'urgent' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END
--
-- 5. CASE WHEN vs IFNULL/COALESCE:
--    CASE WHEN x IS NULL THEN 'unknown' ELSE x END  ≈  COALESCE(x, 'unknown')
--    COALESCE更简洁，但CASE可以做更复杂的NULL处理
