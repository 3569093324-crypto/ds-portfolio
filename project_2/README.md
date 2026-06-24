# 项目 2：交互式数据分析看板

> **数据科学训练 · Phase 2 收尾项目** | 2026-06-24

---

## 📋 项目概述

基于自建电商数据库构建的 Streamlit 交互式数据看板，支持多维筛选、KPI 实时展示、4 种可视化图表和可搜索数据表格。

## 🚀 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行看板
streamlit run app.py

# 3. 浏览器打开 http://localhost:8501
```

## 🗂 项目结构

```
project_2/
├── README.md                    ← 你在这里
├── app.py                       ← Streamlit 看板主程序
├── requirements.txt             ← Python 依赖
├── data_cleaning.py             ← 通用数据清洗模块（含类型标注）
├── data_quality_report.py       ← 自动数据质量报告生成器
├── data_pipeline.py             ← 多数据源整合管道
├── datetime_processing.py       ← 日期时间处理练习
├── numpy_vectorization.py       ← NumPy 向量化运算
├── regex_extraction.py          ← 正则表达式提取
├── tests/
│   ├── __init__.py
│   └── test_data_cleaning.py    ← pytest 单元测试 (35 tests)
├── pipeline_data/               ← 数据管道示例文件
│   ├── users.csv
│   ├── orders.json
│   └── memberships.xlsx
└── mock_nginx.log               ← 模拟服务器日志
```

## 🎛 看板功能

### 筛选器（侧边栏）
- 📅 **日期范围选择器** — 按订单日期筛选
- 📦 **品类多选** — 选择要分析的品类
- 💰 **价格范围滑块** — 按商品价格过滤
- 👤 **用户等级多选** — 按 VIP/Retail/Wholesale/New 筛选

### KPI 卡片
| 指标 | 说明 |
|------|------|
| 总销售额 | 筛选条件下的总收入 |
| 订单数 | 唯一订单计数 |
| 活跃用户 | 唯一用户计数 |
| 客单价 | 平均每笔订单金额 |
| 毛利率 | (收入-成本)/收入 |

### 可视化图表
1. **月度销售趋势** — 折线+面积图，展示收入和订单趋势
2. **品类销售额对比** — 水平柱状图，显示各品类收入
3. **用户等级分布** — 饼图，展示各等级占比
4. **价格 vs 销量散点图** — 气泡大小 = 销售额

### 数据表格
- 支持关键词搜索（商品名/用户名）
- 按日期排序
- 可调整列宽、隐藏索引

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| 数据层 | SQLite + SQL（JOIN/窗口函数/CTE/聚合） |
| 处理层 | Pandas + NumPy（清洗/聚合/向量化） |
| 可视化 | Matplotlib + Streamlit |
| 测试 | pytest（35 测试用例） |
| 代码质量 | Type Hints + Google-style docstrings + SRP |

## 🔑 配套模块

### data_cleaning.py
通用数据清洗管线，支持：
- 列名标准化（snake_case）
- 缺失值填充（数值→中位数，分类→'Unknown'）
- 重复行删除
- IQR 异常值标记
- 日期解析
- 完整 Type Hints + docstring + 输入验证

### data_quality_report.py
自动生成 Markdown 格式的数据质量报告，包含：
- 列概览（类型/缺失率/唯一值）
- 数值列统计（均值/标准差/分位数/偏度/峰度）
- 分类列频率分布
- 异常值检测结果

### data_pipeline.py
多数据源整合管道（CSV + JSON + Excel → 合并 → 干净 DataFrame）

## 📊 数据规模

- 用户：150 人
- 商品：122 种（6 品类）
- 订单：200 笔
- 订单明细：15,000+ 行

## 🖼 看板截图

运行 `streamlit run app.py` 后：
- 侧边栏包含 4 组筛选控件
- 主区域顶部 5 个 KPI 指标卡片
- 4 张图表分两行排列
- 底部可搜索数据表格

---

*项目 2 是 68 天数据科学训练中 Python Phase (Days 15-24) 的最终产出。*
