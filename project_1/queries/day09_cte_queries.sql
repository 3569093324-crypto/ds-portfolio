-- ============================================================
-- Day 9: CTE（公用表表达式）— 写出生产级 SQL
-- 数据库: business.db (电商场景)
-- ============================================================


-- ============================================================
-- 查询 1: 用 CTE 改写多层嵌套子查询
-- 场景：复用 Day6 Q8 — Top3品类中价格最高商品的购买者
-- 对比：子查询版本 vs CTE 版本
-- ============================================================

-- 子查询 v1.0（Day 6 原版）：3层嵌套，难读
-- SELECT ... FROM (SELECT ... WHERE ... IN (SELECT ... FROM (SELECT...)))

-- CTE v2.0：层次分明，每个CTE有明确命名
WITH
-- Step 1: 按品类汇总销售额
category_revenue AS (
    SELECT p.category,
           SUM(oi.quantity * oi.unit_price) AS total_revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY p.category
),
-- Step 2: 取销售额 Top 3 品类
top_categories AS (
    SELECT category
    FROM category_revenue
    ORDER BY total_revenue DESC
    LIMIT 3
),
-- Step 3: 每个 Top 品类中价格最高的商品
top_products AS (
    SELECT p.product_id,
           p.name AS product_name,
           p.category,
           p.price
    FROM products p
    WHERE p.category IN (SELECT category FROM top_categories)
      AND p.price = (
          SELECT MAX(p2.price)
          FROM products p2
          WHERE p2.category = p.category
      )
),
-- Step 4: 找出购买过这些商品的用户
buyers AS (
    SELECT DISTINCT c.customer_id, c.name, c.segment,
           tp.category, tp.product_name, tp.price
    FROM customers c
    JOIN orders o      ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id  = oi.order_id
    JOIN top_products tp ON oi.product_id = tp.product_id
)
SELECT * FROM buyers
ORDER BY category, name;


-- ============================================================
-- 查询 2: CTE 分步计算 — 高价值用户定义 → 筛选 → 特征分析
-- 业务场景：定义"高价值用户"（消费>全局均值+1标准差），
-- 然后分析他们的特征（segment分布、平均订单数、偏好品类）
-- ============================================================
WITH
-- Step 1: 全局消费统计
global_stats AS (
    SELECT AVG(total_spent) AS avg_spent,
           AVG(total_spent) + (
               SELECT AVG((total_spent - (SELECT AVG(total_spent) FROM (
                   SELECT SUM(total_amount) AS total_spent
                   FROM orders GROUP BY customer_id
               ))) * (total_spent - (SELECT AVG(total_spent) FROM (
                   SELECT SUM(total_amount) AS total_spent
                   FROM orders GROUP BY customer_id
               ))))
               FROM (SELECT SUM(total_amount) AS total_spent FROM orders GROUP BY customer_id)
           ) AS threshold
    FROM (
        SELECT SUM(total_amount) AS total_spent
        FROM orders GROUP BY customer_id
    )
),
-- Step 2: 每个用户的消费
user_spending AS (
    SELECT c.customer_id, c.name, c.segment, c.city,
           COALESCE(SUM(o.total_amount), 0) AS total_spent,
           COUNT(o.order_id) AS order_count
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.name, c.segment, c.city
),
-- Step 3: 筛选高价值用户（改用：总消费 > 10000 的 Top 用户）
high_value_users AS (
    SELECT * FROM user_spending
    WHERE total_spent > 10000
),
-- Step 4: 高价值用户的品类偏好
hvu_category_pref AS (
    SELECT hv.customer_id, hv.name, hv.segment, p.category,
           SUM(oi.quantity * oi.unit_price) AS category_spent
    FROM high_value_users hv
    JOIN orders o      ON hv.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id   = oi.order_id
    JOIN products p    ON oi.product_id = p.product_id
    GROUP BY hv.customer_id, hv.name, hv.segment, p.category
)
-- 汇总：高价值用户 segment 分布 + 偏好品类
SELECT segment,
       COUNT(DISTINCT customer_id) AS hvu_count,
       ROUND(AVG(category_spent), 2) AS avg_category_spent,
       category AS top_category
FROM hvu_category_pref
GROUP BY segment
ORDER BY hvu_count DESC;


-- ============================================================
-- 查询 3: 多个 CTE 串联 — 用户分层 → 品类分析 → 交叉统计
-- ============================================================
WITH
-- Step 1: 用户消费分层
user_tier AS (
    SELECT customer_id, name, segment, total_spent,
           NTILE(3) OVER (ORDER BY total_spent DESC) AS tier
    FROM (
        SELECT c.customer_id, c.name, c.segment,
               COALESCE(SUM(o.total_amount), 0) AS total_spent
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name, c.segment
    )
),
-- Step 2: 品类销售统计
category_stats AS (
    SELECT p.category,
           COUNT(DISTINCT oi.order_id) AS order_count,
           SUM(oi.quantity) AS units_sold,
           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY p.category
),
-- Step 3: 每层用户在各品类的消费
tier_category_spending AS (
    SELECT ut.tier, p.category,
           COUNT(DISTINCT ut.customer_id) AS user_count,
           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS tier_revenue
    FROM user_tier ut
    JOIN orders o      ON ut.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id    = oi.order_id
    JOIN products p    ON oi.product_id  = p.product_id
    WHERE ut.tier = 1  -- 只看高消费层（Top 33%）
    GROUP BY ut.tier, p.category
)
-- 交叉统计：高消费层 × 各品类
SELECT tcs.tier,
       CASE tcs.tier WHEN 1 THEN '高消费层 (Top 33%)'
                     WHEN 2 THEN '中消费层'
                     WHEN 3 THEN '低消费层' END AS tier_label,
       tcs.category,
       tcs.user_count,
       tcs.tier_revenue,
       ROUND(tcs.tier_revenue * 100.0 / cs.revenue, 1) AS pct_of_category
FROM tier_category_spending tcs
JOIN category_stats cs ON tcs.category = cs.category
ORDER BY tcs.tier_revenue DESC
LIMIT 10;


-- ============================================================
-- 查询 4: 递归 CTE — 生成过去30天的日期序列
-- 用途：填补日报中缺失的日期（让每天都有行，即使当天无订单）
-- ============================================================
WITH RECURSIVE
date_series(d) AS (
    -- 锚点：起始日期（30天前）
    SELECT date('now', '-30 days')
    UNION ALL
    -- 递归：每天+1，直到今天
    SELECT date(d, '+1 day')
    FROM date_series
    WHERE d < date('now')
)
-- 日期序列 LEFT JOIN 每日销售 → 无订单的日期显示 0
SELECT ds.d AS date,
       COALESCE(daily.order_count, 0) AS orders,
       COALESCE(daily.daily_revenue, 0) AS revenue
FROM date_series ds
LEFT JOIN (
    SELECT order_date,
           COUNT(*) AS order_count,
           ROUND(SUM(total_amount), 2) AS daily_revenue
    FROM orders
    GROUP BY order_date
) daily ON ds.d = daily.order_date
ORDER BY ds.d;


-- ============================================================
-- 查询 5: 递归 CTE — 模拟品类层级树
-- 虽然我们的电商数据没有真正的层级结构，
-- 但可以演示递归CTE的语法：生成一个模拟的"类目树"
-- ============================================================
WITH RECURSIVE
category_tree(id, name, parent_id, depth, path) AS (
    -- 根节点（depth=0）
    SELECT 1, 'All Products', NULL, 0, 'All Products'
    UNION ALL
    -- 第1层：大类
    SELECT 2, 'Electronics', 1, 1, 'All Products > Electronics'
    UNION ALL
    SELECT 3, 'Home & Living', 1, 1, 'All Products > Home & Living'
    UNION ALL
    SELECT 4, 'Sports & Outdoors', 1, 1, 'All Products > Sports & Outdoors'
    UNION ALL
    -- 第2层：子类
    SELECT 5, 'Phones', 2, 2, 'All Products > Electronics > Phones'
    UNION ALL
    SELECT 6, 'Computers', 2, 2, 'All Products > Electronics > Computers'
    UNION ALL
    SELECT 7, 'Audio', 2, 2, 'All Products > Electronics > Audio'
    UNION ALL
    SELECT 8, 'Furniture', 3, 2, 'All Products > Home & Living > Furniture'
    UNION ALL
    SELECT 9, 'Kitchen', 3, 2, 'All Products > Home & Living > Kitchen'
    UNION ALL
    SELECT 10, 'Outdoor Gear', 4, 2, 'All Products > Sports & Outdoors > Outdoor Gear'
)
SELECT id, name, parent_id, depth,
       -- 缩进显示层级关系
       CASE depth
           WHEN 0 THEN name
           WHEN 1 THEN '  ├─ ' || name
           WHEN 2 THEN '  │   └─ ' || name
       END AS hierarchy_display,
       path
FROM category_tree
ORDER BY id;


-- ============================================================
-- 查询 6: CTE + 窗口函数 — 先分组计算排名，再筛选 Top N
-- 经典面试组合拳：CTE 算排名 → 外层取 Top N
-- ============================================================
WITH
-- Step 1: 计算每个品类中各商品的销售额排名
product_rankings AS (
    SELECT p.category,
           p.name AS product_name,
           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue,
           ROW_NUMBER() OVER (
               PARTITION BY p.category
               ORDER BY SUM(oi.quantity * oi.unit_price) DESC
           ) AS rank_in_category,
           RANK() OVER (
               PARTITION BY p.category
               ORDER BY SUM(oi.quantity * oi.unit_price) DESC
           ) AS rank_dense
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.category, p.product_id, p.name
),
-- Step 2: 同时计算品类总销售额（用于计算占比）
category_totals AS (
    SELECT p.category,
           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS cat_total_revenue
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.category
)
-- 取每个品类 Top 3，并显示占该品类销售额的百分比
SELECT pr.category,
       pr.product_name,
       pr.revenue,
       pr.rank_in_category,
       ROUND(pr.revenue * 100.0 / ct.cat_total_revenue, 1) AS pct_of_category
FROM product_rankings pr
JOIN category_totals ct ON pr.category = ct.category
WHERE pr.rank_in_category <= 3
ORDER BY pr.category, pr.rank_in_category;


-- ============================================================
-- 查询 7: 同一查询 — 子查询 v1.0 vs CTE v2.0 对比
-- 业务问题：找出"消费Top 10用户中，每个品类消费最多的那个用户"
-- ============================================================

-- ─── 子查询 v1.0 ──────────────────────────────────────────
-- 问题：4层嵌套，逻辑难以追踪，调试时需要从内向外读
/*
SELECT ...
FROM (
    SELECT ..., ROW_NUMBER() OVER (...) AS rn
    FROM (
        SELECT ...
        FROM orders o JOIN customers c ON ...
        WHERE customer_id IN (
            SELECT customer_id FROM (
                SELECT customer_id, SUM(total_amount) AS spent
                FROM orders GROUP BY customer_id
                ORDER BY spent DESC LIMIT 10
            )
        )
        GROUP BY ...
    )
) WHERE rn = 1;
*/

-- ─── CTE v2.0 ─────────────────────────────────────────────
-- 优点：自上而下阅读，每个CTE独立可测，修改变得容易
WITH
-- 1. Top 10 消费用户
top10_users AS (
    SELECT customer_id,
           SUM(total_amount) AS total_spent
    FROM orders
    GROUP BY customer_id
    ORDER BY total_spent DESC
    LIMIT 10
),
-- 2. 这些用户在每个品类的消费
user_category_spending AS (
    SELECT c.customer_id,
           c.name,
           p.category,
           SUM(oi.quantity * oi.unit_price) AS category_spent
    FROM customers c
    JOIN orders o      ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id   = oi.order_id
    JOIN products p    ON oi.product_id = p.product_id
    WHERE c.customer_id IN (SELECT customer_id FROM top10_users)
    GROUP BY c.customer_id, c.name, p.category
),
-- 3. 每个品类中消费最多的用户排名
category_top_user AS (
    SELECT name,
           category,
           ROUND(category_spent, 2) AS spent,
           ROW_NUMBER() OVER (
               PARTITION BY category
               ORDER BY category_spent DESC
           ) AS rn
    FROM user_category_spending
)
-- 4. 最终输出
SELECT category,
       name AS top_spender,
       spent AS amount
FROM category_top_user
WHERE rn = 1
ORDER BY spent DESC;

-- 📝 总结：CTE vs 子查询
-- CTE 优势：
--   ✅ 可读性：自上而下，每个步骤有名字，像文档
--   ✅ 可维护：修改某一步不影响其他步骤
--   ✅ 可复用：同一CTE可在后续多次引用
--   ✅ 可调试：每个CTE可独立运行验证
-- 子查询适用场景：
--   ✅ 简单标量子查询（如 SELECT (SELECT AVG(...)) ）
--   ✅ 单层 IN / EXISTS
--   ❌ 3层+嵌套 → 应该改用CTE
