-- ============================================================
-- Day 8: 窗口函数（二）— LAG, LEAD, 累计求和, 移动平均
-- 数据库: business.db (电商场景)
-- ============================================================


-- ============================================================
-- 查询 1: LAG() — 计算每个用户相邻两笔订单的间隔天数
-- LAG(column, offset, default) = 向上取前 offset 行的值
-- ============================================================
SELECT c.name,
       o.order_date,
       o.total_amount,
       LAG(o.order_date, 1) OVER (
           PARTITION BY o.customer_id
           ORDER BY o.order_date
       ) AS prev_order_date,
       -- 计算与上一笔订单的间隔天数
       JULIANDAY(o.order_date) -
       JULIANDAY(LAG(o.order_date, 1) OVER (
           PARTITION BY o.customer_id
           ORDER BY o.order_date
       )) AS days_since_last
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
ORDER BY c.name, o.order_date
LIMIT 20;


-- ============================================================
-- 查询 2: LEAD() — 获取每个商品在当前品类中的下一个价格
-- LEAD(column, offset, default) = 向下取后 offset 行的值
-- 场景：对比同品类相邻价格区间商品
-- ============================================================
SELECT p.category,
       p.name,
       p.price,
       LEAD(p.name, 1, '--品类最便宜--') OVER (
           PARTITION BY p.category
           ORDER BY p.price DESC
       ) AS next_cheaper_product,
       LEAD(p.price, 1, NULL) OVER (
           PARTITION BY p.category
           ORDER BY p.price DESC
       ) AS next_lower_price,
       -- 价格差
       p.price - LEAD(p.price, 1, NULL) OVER (
           PARTITION BY p.category
           ORDER BY p.price DESC
       ) AS price_gap
FROM products p
ORDER BY p.category, p.price DESC
LIMIT 15;


-- ============================================================
-- 查询 3: 用 LAG 计算月度销售额的环比增长率
-- 环比 = (本月 - 上月) / 上月 × 100%
-- ============================================================
SELECT order_month,
       monthly_revenue,
       prev_month_revenue,
       ROUND(
           (monthly_revenue - prev_month_revenue) * 100.0
           / NULLIF(prev_month_revenue, 0), 2
       ) AS mom_growth_pct
FROM (
    SELECT strftime('%Y-%m', order_date) AS order_month,
           ROUND(SUM(total_amount), 2)   AS monthly_revenue,
           LAG(ROUND(SUM(total_amount), 2), 1) OVER (
               ORDER BY strftime('%Y-%m', order_date)
           ) AS prev_month_revenue
    FROM orders
    GROUP BY order_month
) AS monthly
ORDER BY order_month;


-- ============================================================
-- 查询 4: SUM() OVER (ORDER BY) — 累计销售额随时间变化
-- 无 PARTITION BY → 全局累加
-- 有 ORDER BY    → 按指定顺序累加
-- ============================================================
SELECT order_date,
       daily_revenue,
       ROUND(SUM(daily_revenue) OVER (
           ORDER BY order_date
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ), 2) AS cumulative_revenue
FROM (
    SELECT order_date,
           ROUND(SUM(total_amount), 2) AS daily_revenue
    FROM orders
    GROUP BY order_date
) AS daily
ORDER BY order_date;


-- ============================================================
-- 查询 5: SUM() OVER (PARTITION BY ... ORDER BY) — 每个用户的累计消费
-- 带 PARTITION BY → 每组独立累加
-- ============================================================
SELECT c.name,
       o.order_id,
       o.order_date,
       o.total_amount,
       ROUND(SUM(o.total_amount) OVER (
           PARTITION BY o.customer_id
           ORDER BY o.order_date, o.order_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ), 2) AS cumulative_spent
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
ORDER BY c.name, o.order_date
LIMIT 20;


-- ============================================================
-- 查询 6: 移动平均 — 用 ROWS BETWEEN 计算7日移动平均
-- 场景：平滑每日波动，观察销售趋势
-- 前3天 + 当天 + 后3天 = 7天窗口（居中移动平均）
-- ============================================================
SELECT order_date,
       daily_revenue,
       ROUND(AVG(daily_revenue) OVER (
           ORDER BY order_date
           ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
       ), 2) AS ma_7day_centered
FROM (
    SELECT order_date,
           ROUND(SUM(total_amount), 2) AS daily_revenue
    FROM orders
    GROUP BY order_date
) AS daily
ORDER BY order_date;


-- ============================================================
-- 查询 7: 用 LAG 找出连续3天以上有订单的用户
-- 思路：
--   1. 每天有订单的日期去重
--   2. 用 LAG 取前1天和前2天
--   3. 判断是否连续3天（当天-1天=1 AND 当天-2天=2 或 前1天-前2天=1）
-- ============================================================
SELECT name, streak_start, streak_end, streak_days
FROM (
    SELECT c.name,
           o.order_date,
           -- 检查是否连续3天
           JULIANDAY(o.order_date) - JULIANDAY(
               LAG(o.order_date, 1) OVER (PARTITION BY o.customer_id ORDER BY o.order_date)
           ) AS day_diff_1,
           JULIANDAY(o.order_date) - JULIANDAY(
               LAG(o.order_date, 2) OVER (PARTITION BY o.customer_id ORDER BY o.order_date)
           ) AS day_diff_2,
           LAG(o.order_date, 2) OVER (
               PARTITION BY o.customer_id ORDER BY o.order_date
           ) AS streak_start,
           o.order_date AS streak_end,
           CAST(JULIANDAY(o.order_date) - JULIANDAY(
               LAG(o.order_date, 2) OVER (PARTITION BY o.customer_id ORDER BY o.order_date)
           ) + 1 AS INTEGER) AS streak_days
    FROM (SELECT DISTINCT customer_id, order_date FROM orders) o
    JOIN customers c ON o.customer_id = c.customer_id
) AS streaks
WHERE day_diff_1 = 1 AND day_diff_2 = 2  -- 连续3天的标志
ORDER BY name, streak_start
LIMIT 15;


-- ============================================================
-- 查询 8: FIRST_VALUE / LAST_VALUE — 每个用户第一笔和最新一笔订单
-- FIRST_VALUE(column) = 窗口内第一行的值
-- LAST_VALUE(column)  = 窗口内最后一行的值（注意窗口帧范围！）
-- ============================================================
SELECT DISTINCT
       c.name,
       FIRST_VALUE(o.total_amount) OVER w AS first_order_amount,
       FIRST_VALUE(o.order_date)    OVER w AS first_order_date,
       LAST_VALUE(o.total_amount)  OVER w AS last_order_amount,
       LAST_VALUE(o.order_date)    OVER w AS last_order_date,
       -- 首末消费变化
       LAST_VALUE(o.total_amount)  OVER w -
       FIRST_VALUE(o.total_amount) OVER w AS spending_change
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WINDOW w AS (
    PARTITION BY o.customer_id
    ORDER BY o.order_date, o.order_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)
ORDER BY c.name
LIMIT 15;


-- ============================================================
-- 窗口帧速查卡：ROWS vs RANGE
-- ============================================================
-- ROWS BETWEEN ... AND ...     → 物理行数范围
-- RANGE BETWEEN ... AND ...    → 逻辑值范围（按ORDER BY列的值）
--
-- 常用帧定义：
-- UNBOUNDED PRECEDING AND CURRENT ROW         → 从第一行到当前行（累计）
-- N PRECEDING AND CURRENT ROW                  → 前N行到当前行
-- N PRECEDING AND M FOLLOWING                  → 前N行到后M行（移动窗口）
-- UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  → 整个分区（用于FIRST/LAST_VALUE）
--
-- 默认帧（只写ORDER BY不写ROWS时）：
-- RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- 注意：RANGE 按值范围计算，比 ROWS 慢；通常用 ROWS 更直观

-- ============================================================
-- 面试速查：窗口函数应用场景
-- ============================================================
-- 需求                        | 窗口函数
-- ---------------------------|------------------------
-- 组内排名                    | ROW_NUMBER / RANK / DENSE_RANK
-- 取每组Top N                 | ROW_NUMBER + WHERE rn <= N
-- 环比/同比                   | LAG(column, N)
-- 累计求和                    | SUM(column) OVER (ORDER BY date)
-- 移动平均                    | AVG(column) OVER (ROWS N PRECEDING)
-- 首次/末次值                 | FIRST_VALUE / LAST_VALUE
-- 第N个值                     | NTH_VALUE(column, N)
-- 百分比排名                  | PERCENT_RANK / CUME_DIST
