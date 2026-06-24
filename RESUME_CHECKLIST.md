# 简历终极检查清单

> Day 67 | Phase 6: 面试模拟冲刺

---

## 一页简历模板 (DS Intern)

```markdown
[Your Name]
[Email] | [Phone] | [GitHub] | [LinkedIn]

EDUCATION
[University], [Degree] in [Major] | [Start Date] - [Expected Graduation]
GPA: [if >3.5/4.0 or >85/100]

TECHNICAL SKILLS
Languages: Python, SQL
ML/DL: scikit-learn, XGBoost, SHAP, Optuna
Data: Pandas, NumPy, Streamlit, Matplotlib
Stats: Hypothesis Testing, A/B Testing, Power Analysis, Experiment Design
Engineering: Git, Docker, FastAPI, pytest, Logging
Databases: SQLite

PROJECTS

User Repurchase Prediction | Python, XGBoost, SHAP, FastAPI | [GitHub Link]
- Built end-to-end ML pipeline predicting e-commerce user repurchase (30-day window)
- Compared 5 models (LR, DT, RF, XGBoost, MLP) using unified 5-fold CV; XGBoost achieved AUC 0.96
- Applied SHAP for model interpretability, identifying review_score as top predictor (|SHAP|=1.73)
- Deployed model as REST API using FastAPI + Docker; estimated annual business impact of ¥1.26B
- Engineered 12 features from raw SQL database; missing indicator strategy boosted AUC by 12%

A/B Testing Experiment Design | Python, scipy, statsmodels | [GitHub Link]
- Designed end-to-end A/B test for new product page; power analysis determined 17K sample per group
- Implemented stratified randomization, AA validation, and two-sample z-test for conversion rate comparison
- Treatment group showed +1.44pp conversion lift (12.18% → 13.62%, p=0.032)
- Quantified business impact: ¥80M/month saved in unnecessary marketing spend

Interactive E-commerce Dashboard | Streamlit, Pandas, SQL | [GitHub Link]
- Built interactive analytics dashboard with 5 KPIs, 4 chart types, and searchable data table
- Integrated SQLite backend with multi-dimensional filters (date, category, price, segment)
- Wrote 35 pytest unit tests for data cleaning pipeline, achieving >90% code coverage

SQL Business Analysis Suite | SQL, Matplotlib | [GitHub Link]
- Designed and populated e-commerce database (4 tables, 1,330+ rows spanning 6 product categories)
- Developed 8 SQL query libraries (JOIN, CTE, Window Functions, Subqueries) for business analysis
- Delivered 6-dimension analysis report with visualizations (Pareto, trend, cohort analysis)
- Optimized queries with 5 indexes + EXPLAIN QUERY PLAN; JOIN queries executed in <2ms

ADDITIONAL
- 68-day structured Data Science training program covering SQL → ML → Deployment
- Languages: English (Fluent), Chinese (Native)
```

---

## ❌ 删除清单

| 删除内容 | 原因 |
|---------|------|
| "热爱数据科学，学习能力强" | 空洞，无法验证。用68天训练结果证明 |
| "熟练使用Python进行数据分析" | 太泛。用具体项目证明 |
| "曾在学生会担任XX职务" | 与DS无关 |
| "熟悉Office办公软件" | DS不需要强调这个 |
| GPA < 3.5 | 不是加分项就省略 |
| "课程项目"这样的标题 | 直接叫"PROJECTS" |
| 每个项目2行以内 | 没有细节=没有价值 |

---

## ✅ 强化清单

| 原写法 | 改进后 |
|--------|--------|
| "用了XGBoost做预测" | "XGBoost AUC 0.96，对比5模型，用SHAP解释关键特征" |
| "做了数据分析" | "分析150用户×200订单，识别出Electronics贡献53.9%收入" |
| "写了测试" | "35个pytest测试，覆盖缺失值/异常值/边界情况" |
| "搭建了看板" | "Streamlit看板支持4维筛选，5个KPI卡片，日均使用10+次" |

---

## 3 个真实 JD 关键词对查

### JD 1: 字节跳动 DS Intern
关键词: SQL, A/B Testing, Python, 指标体系, 数据可视化
✅ 覆盖: SQL(项目1), A/B Testing(项目3), Python(全部), 指标体系(Day65文档), 可视化(项目1/2)

### JD 2: 美团 DS Intern
关键词: 机器学习, 特征工程, 用户画像, 业务分析, Hive/Spark
✅ 覆盖: ML(项目4), 特征工程(项目4), 用户画像(项目1/4), 业务分析(项目1)

### JD 3: 滴滴 DS Intern
关键词: 实验设计, 假设检验, Python, 因果推断, 数据看板
✅ 覆盖: 实验设计(项目3), 假设检验(Day28), Python(全部), 看板(项目2)

---

## 简历自查清单

```
[ ] 严格一页 (超过一页 = 直接被扔)
[ ] 每个项目 3-4 行，每行有数字
[ ] GitHub 链接跟在项目后面
[ ] 技术栈关键词出现在项目描述中 (ATS扫描)
[ ] 无空洞形容词 (热爱/熟练/擅长)
[ ] 无拼写错误
[ ] 日期格式统一
[ ] 用PDF发送 (不是Word——排版不会乱)
[ ] 文件命名: [Name]_DS_Intern_Resume.pdf (不是Resume_final_v3.pdf)
```

---

*你的简历只有 15-30 秒的注意力。让每一行都在说：'这个人能给我的团队带来价值'。*
