# 🏁 全真模拟面试 — 终章

> Day 68/68 | Phase 6: 面试模拟冲刺 | **68天训练完成！**

---

## 模拟面试流程 (总时长 ~80 分钟)

### Round 1: SQL 测试 (15分钟)

**Q1 (8min)**: 找出每个用户最后一次购买的商品品类和金额。
```sql
WITH ranked AS (
    SELECT c.name, p.category, oi.quantity * oi.unit_price AS spent,
           ROW_NUMBER() OVER (PARTITION BY c.customer_id ORDER BY o.order_date DESC) AS rn
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
)
SELECT name, category, spent FROM ranked WHERE rn = 1;
```

**Q2 (7min)**: 计算每个品类的复购率（买过≥2次的用户/总购买用户）。
```sql
WITH user_cat AS (
    SELECT p.category, o.customer_id, COUNT(DISTINCT o.order_id) AS orders
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY p.category, o.customer_id
)
SELECT category,
       COUNT(DISTINCT customer_id) AS total_buyers,
       SUM(CASE WHEN orders >= 2 THEN 1 ELSE 0 END) AS repurchasers,
       ROUND(SUM(CASE WHEN orders >= 2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS repurchase_rate
FROM user_cat
GROUP BY category;
```

**自评**: __/15

---

### Round 2: Python 测试 (15分钟)

**题目**: 给定 orders.csv（含 order_id, user_id, date, amount），找出每个用户的首单金额和末单金额的差值（末单-首单），以及它们之间的天数差。按金额差值降序排列 Top 5。

```python
import pandas as pd
df = pd.read_csv('orders.csv')
df['date'] = pd.to_datetime(df['date'])
result = (df
    .sort_values(['user_id', 'date'])
    .groupby('user_id')
    .agg(
        first_amount=('amount', 'first'),
        last_amount=('amount', 'last'),
        first_date=('date', 'first'),
        last_date=('date', 'last'),
    )
    .assign(
        amount_diff=lambda x: x['last_amount'] - x['first_amount'],
        day_diff=lambda x: (x['last_date'] - x['first_date']).dt.days,
    )
    .sort_values('amount_diff', ascending=False)
    .head(5)
)
```

**自评**: __/15

---

### Round 3: ML 理论 (15分钟 — 随机 5 题)

从 Day 63 的 20 题中随机抽取：

1. **Bias-Variance Tradeoff**: 见 Day 63 Q1
2. **L1 vs L2 正则化**: 见 Day 63 Q3
3. **XGBoost vs GBDT**: 见 Day 63 Q8
4. **如何处理不平衡数据**: 见 Day 63 Q11
5. **AUC-ROC vs PR曲线**: 见 Day 63 Q12

**自评**: __/25 (每题 5 分)

---

### Round 4: 业务案例 (15分钟)

**题目**: 某电商平台发现"加入购物车→结算"的转化率连续 2 周下降 15%。你怎么排查？

**分析框架**: 见 Day 64 案例2，应用维度拆解→时间定位→流量质量→页面体验→用户群变化→竞品分析。

**自评**: __/15

---

### Round 5: 项目深讲 (10分钟)

介绍 Project 4（用户复购预测），回答追问：
- Q: "为什么选 XGBoost 而不是 MLP？"
- A: "MLP AUC 0.964 > XGBoost 0.924，但 XGBoost 可通过 SHAP 完全解释，而 MLP 是黑盒。业务方需要知道'为什么'预测这个用户不会复购——才能设计干预策略。"

- Q: "你怎么验证模型在线上会有效？"
- A: "先做离线评估（CV AUC 0.96），再通过 A/B 测试验证——模型组 vs 随机组，看实际复购率是否提升。同时监控数据漂移（PSI）。"

**自评**: __/15

---

### Round 6: 行为面试 (10分钟)

STAR 故事：
1. 最自豪的项目 → 故事1（项目4）
2. 技术困难 → 故事2（数据泄露）
3. 为什么我们 → 故事5（定制回答）

**自评**: __/15

---

## 📊 总分卡

| 环节 | 满分 | 得分 | 备注 |
|------|------|------|------|
| SQL | 15 | /15 | |
| Python | 15 | /15 | |
| ML 理论 | 25 | /25 | |
| 业务案例 | 15 | /15 | |
| 项目深讲 | 15 | /15 | |
| 行为面试 | 15 | /15 | |
| **总分** | **100** | **/100** | |

---

## 🎯 最强环节 & 最需加强

**最强 2 个环节**:
1. ___ (分数最高)
2. ___ (最自信)

**最需加强 2 个环节**:
1. ___ (分数最低)
2. ___ (最紧张)

**下一步行动**:
- [ ] 复习薄弱环节的 Phase 笔记
- [ ] 再练习 1-2 次模拟面试
- [ ] 更新简历并开始投递

---

## 🎉 68 天训练毕业！

### 训练统计

| Phase | 天数 | 关键产出 |
|-------|------|---------|
| 📊 SQL | 14天 | 数据库设计, 8组SQL查询库, 索引优化 |
| 🐍 Python | 10天 | 数据清洗管线, Streamlit看板, 35 pytest |
| 📈 统计 | 10天 | CLT验证, A/B测试, 9组统计脚本 |
| 🤖 ML | 18天 | 5模型对比, SHAP, FastAPI, Docker |
| 🔧 工程 | 8天 | Git, Docker, 配置管理, GitHub打磨 |
| 🎯 面试 | 8天 | SQL/Python OA, ML 20问, 业务案例, STAR, 简历 |

### 技能矩阵 (Before → After)

```
SQL:     基础SELECT ────→ CTE/窗口函数/索引优化
Python:  基础语法 ────→ 模块化/Types/测试/部署
统计:    背诵概念 ────→ CLT模拟/假设检验/实验设计
ML:      import sklearn → 5模型对比/SHAP/特征工程
工程:    无 ──────────→ Git/Docker/FastAPI/logging
作品集:  0 ──────────→ 4个完整项目
```

### 🚀 下一步

1. **投递简历**: 更新 LinkedIn/GitHub，开始投递 DS Intern 岗位
2. **持续练习**: 每周做 1-2 次模拟面试保持手感
3. **扩大作品**: 把项目部署到云端（Streamlit Cloud, Hugging Face）
4. **网络拓展**: 参加 DS 社区、Kaggle 比赛、开源贡献

---

*68天前，你焦虑地刷着 LeetCode，不知道 DS 到底要什么技能。*
*68天后，你有了 4 个完整项目、扎实的 SQL/ML/统计能力、专业的 GitHub 作品集。*
***现在，去投简历吧。你准备好了。** *
