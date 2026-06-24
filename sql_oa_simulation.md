# 模拟 SQL OA — 限时 45 分钟

> Day 61 | Phase 6: 面试模拟冲刺
> 数据库: business.db | 自行计时

---

## 📋 考试说明

- **时长**: 45 分钟
- **题目数**: 5 题
- **数据库**: business.db (4张表: customers, products, orders, order_items)
- **评分**: 每题 20 分，满分 100 分
- **规则**: 不看答案，不看笔记，模拟真实 OA

---

## 题 1 (基础) — 过滤 + 排序 — 10 分

**题目**: 查询价格高于 500 的商品，按价格从高到低排列，只显示前 10 个。

**预期输出**: product_name, category, price 三列，共最多 10 行。

<details>
<summary>点击查看答案</summary>

```sql
SELECT name, category, price
FROM products
WHERE price > 500
ORDER BY price DESC
LIMIT 10;
```
</details>

---

## 题 2 (聚合) — GROUP BY + 日期提取 — 15 分

**题目**: 统计 2025 年每个月的订单数和总销售额，按月份排序。月份格式为 '2025-01'。

**预期输出**: month, order_count, total_revenue 三列。

<details>
<summary>点击查看答案</summary>

```sql
SELECT strftime('%Y-%m', order_date) AS month,
       COUNT(*) AS order_count,
       ROUND(SUM(total_amount), 2) AS total_revenue
FROM orders
WHERE strftime('%Y', order_date) = '2025'
GROUP BY month
ORDER BY month;
```
</details>

---

## 题 3 (JOIN + 子查询) — 20 分

**题目**: 找出购买过 Electronics 品类的所有用户，显示用户名和等级。结果按用户名排序。

**预期输出**: customer_name, segment 两列。

<details>
<summary>点击查看答案</summary>

```sql
SELECT DISTINCT c.name AS customer_name, c.segment
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IN (
    SELECT DISTINCT oi.order_id
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    WHERE p.category = 'Electronics'
)
ORDER BY c.name;
```
</details>

---

## 题 4 (窗口函数) — 25 分

**题目**: 给每个用户按订单时间顺序编号，找出每个用户的第 1 笔订单和最后 1 笔订单。只显示用户名、首单日期、首单金额、末单日期、末单金额。

**预期输出**: customer_name, first_order_date, first_order_amount, last_order_date, last_order_amount。

<details>
<summary>点击查看答案</summary>

```sql
WITH ranked_orders AS (
    SELECT c.name AS customer_name,
           o.order_date,
           o.total_amount,
           ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS rn_asc,
           ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date DESC) AS rn_desc
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
)
SELECT customer_name,
       MAX(CASE WHEN rn_asc = 1 THEN order_date END) AS first_order_date,
       MAX(CASE WHEN rn_asc = 1 THEN total_amount END) AS first_order_amount,
       MAX(CASE WHEN rn_desc = 1 THEN order_date END) AS last_order_date,
       MAX(CASE WHEN rn_desc = 1 THEN total_amount END) AS last_order_amount
FROM ranked_orders
GROUP BY customer_name
ORDER BY customer_name;
```
</details>

---

## 题 5 (综合 — CTE + 窗口函数 + 多表 JOIN) — 30 分

**题目**: 找出每个品类中累计销售金额排名前 3 的商品。显示品类名、商品名、销售额、品类内排名。

**预期输出**: category, product_name, revenue, rank_in_category 四列。

<details>
<summary>点击查看答案</summary>

```sql
WITH product_revenue AS (
    SELECT p.category,
           p.name AS product_name,
           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue,
           ROW_NUMBER() OVER (
               PARTITION BY p.category
               ORDER BY SUM(oi.quantity * oi.unit_price) DESC
           ) AS rank_in_category
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.category, p.product_id, p.name
)
SELECT category, product_name, revenue, rank_in_category
FROM product_revenue
WHERE rank_in_category <= 3
ORDER BY category, rank_in_category;
```
</details>

---

## 📊 自评表

| 题号 | 题型 | 满分 | 自评得分 | 失误分析 |
|------|------|------|---------|---------|
| 1 | 过滤+排序 | 10 | /10 | |
| 2 | GROUP BY | 15 | /15 | |
| 3 | JOIN+子查询 | 20 | /20 | |
| 4 | 窗口函数 | 25 | /25 | |
| 5 | CTE+综合 | 30 | /30 | |
| **总分** | | **100** | **/100** | |

---

## 🔍 薄弱环节诊断

- **得分 < 60**: 需要回顾 Phase 1 基础（Day 3-6）
- **题1-2 错**: 基础过滤/聚合不熟练 → Day 3-4
- **题3 错**: JOIN/子查询薄弱 → Day 5-6
- **题4 错**: 窗口函数需要加强 → Day 7-8
- **题5 错**: CTE 综合能力不足 → Day 9

---

## ⏱️ 时间分配建议

| 时间段 | 任务 |
|--------|------|
| 0-5 min | 读题1-2, 快速解决 |
| 5-15 min | 做题3 (JOIN 需要仔细) |
| 15-30 min | 做题4 (窗口函数是重点) |
| 30-45 min | 做题5 (综合题, 留足时间) |
| 最后 2 min | 检查所有答案 |

---

*建议: 先在不看答案的情况下计时45分钟完成。之后再对照答案自评。*
