"""
项目2: 电商运营数据看板 — Streamlit 交互式 Dashboard

运行: streamlit run app.py
"""

import sqlite3
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="电商运营看板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 数据加载
# ============================================================
@st.cache_data(ttl=600)
def load_data():
    """加载并缓存数据（TTL=10分钟）"""
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "business.db")
    conn = sqlite3.connect(DB_PATH)

    # 完整 JOIN 数据集
    df = pd.read_sql("""
        SELECT
            c.customer_id, c.name AS customer_name, c.segment, c.city,
            c.join_date,
            o.order_id, o.order_date, o.total_amount,
            p.product_id, p.name AS product_name, p.category,
            p.price, p.cost,
            oi.quantity, oi.unit_price
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
    """, conn)
    conn.close()

    # 日期解析
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['revenue'] = df['quantity'] * df['unit_price']
    df['profit'] = df['revenue'] - df['quantity'] * df['cost']
    df['order_month'] = df['order_date'].dt.strftime('%Y-%m')
    df['day_of_week'] = df['order_date'].dt.day_name()

    return df


df = load_data()

# ============================================================
# 侧边栏 — 筛选控件
# ============================================================
st.sidebar.title("🎛️ 筛选器")

# 日期范围
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()
date_range = st.sidebar.date_input(
    "📅 日期范围",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# 品类多选
all_categories = sorted(df['category'].unique())
selected_categories = st.sidebar.multiselect(
    "📦 商品品类",
    options=all_categories,
    default=all_categories[:3],
)

# 价格范围滑块
price_min = float(df['price'].min())
price_max = float(df['price'].max())
price_range = st.sidebar.slider(
    "💰 商品价格范围 (¥)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    step=10.0,
)

# 用户等级
all_segments = sorted(df['segment'].unique())
selected_segments = st.sidebar.multiselect(
    "👤 用户等级",
    options=all_segments,
    default=all_segments,
)

st.sidebar.divider()
st.sidebar.caption(f"数据更新: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption(f"总数据量: {len(df):,} 行")

# ============================================================
# 数据过滤
# ============================================================
mask = (
    (df['order_date'].dt.date >= date_range[0])
    & (df['order_date'].dt.date <= date_range[1])
    & (df['category'].isin(selected_categories))
    & (df['price'].between(price_range[0], price_range[1]))
    & (df['segment'].isin(selected_segments))
)
filtered = df[mask]

# ============================================================
# 主区域
# ============================================================
st.title("📊 电商运营数据看板")
st.caption("基于自建电商数据库 | SQL + Pandas + Streamlit")

# --- KPI 卡片 ---
st.divider()
st.subheader("📈 核心指标")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    total_revenue = filtered['revenue'].sum()
    st.metric("总销售额", f"¥{total_revenue:,.0f}")

with kpi2:
    total_orders = filtered['order_id'].nunique()
    st.metric("订单数", f"{total_orders:,}")

with kpi3:
    total_users = filtered['customer_id'].nunique()
    st.metric("活跃用户", f"{total_users:,}")

with kpi4:
    avg_order = filtered.groupby('order_id')['revenue'].sum().mean()
    st.metric("客单价", f"¥{avg_order:,.0f}")

with kpi5:
    avg_margin = (filtered['profit'].sum() / filtered['revenue'].sum() * 100
                  if filtered['revenue'].sum() > 0 else 0)
    st.metric("毛利率", f"{avg_margin:.1f}%")

# --- 图表区 ---
st.divider()

# 第一行：趋势图 + 品类柱状图
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 月度销售趋势")
    monthly = (filtered
               .groupby('order_month')
               .agg(revenue=('revenue', 'sum'), orders=('order_id', 'nunique'))
               .reset_index()
               .sort_values('order_month'))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.fill_between(range(len(monthly)), monthly['revenue'], alpha=0.3, color='#1f77b4')
    ax.plot(range(len(monthly)), monthly['revenue'], 'o-', color='#1f77b4',
            linewidth=2, markersize=4)
    ax.set_xticks(range(0, len(monthly), max(1, len(monthly)//6)))
    ax.set_xticklabels(monthly['order_month'].iloc[::max(1, len(monthly)//6)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Revenue (¥)', fontsize=10)
    st.pyplot(fig)

with col2:
    st.subheader("📊 品类销售额对比")
    cat_rev = (filtered
               .groupby('category')['revenue']
               .sum()
               .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cat_rev)))
    bars = ax.barh(cat_rev.index, cat_rev.values, color=colors)
    ax.set_xlabel('Revenue (¥)', fontsize=10)
    for bar, val in zip(bars, cat_rev.values):
        ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
                f'¥{val:,.0f}', va='center', fontsize=8)
    st.pyplot(fig)

# 第二行：饼图 + 散点图
col3, col4 = st.columns(2)

with col3:
    st.subheader("🥧 用户等级分布")
    seg_dist = filtered['segment'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax.pie(
        seg_dist.values, labels=seg_dist.index,
        autopct='%1.1f%%', colors=colors_pie[:len(seg_dist)],
        startangle=90, explode=[0.02]*len(seg_dist),
    )
    for at in autotexts:
        at.set_fontweight('bold')
    st.pyplot(fig)

with col4:
    st.subheader("🔵 价格 vs 销量 散点图")
    # 按商品聚合
    product_stats = (filtered
                     .groupby(['product_name', 'category'])
                     .agg(avg_price=('price', 'mean'),
                          total_sold=('quantity', 'sum'),
                          total_revenue=('revenue', 'sum'))
                     .reset_index())

    fig, ax = plt.subplots(figsize=(8, 3.5))
    categories_unique = product_stats['category'].unique()
    cmap = plt.cm.Set1(np.linspace(0, 1, len(categories_unique)))
    for cat, color in zip(categories_unique, cmap):
        subset = product_stats[product_stats['category'] == cat]
        ax.scatter(subset['avg_price'], subset['total_sold'],
                   label=cat, c=[color], alpha=0.7, s=subset['total_revenue']/500)
    ax.set_xlabel('Average Price (¥)', fontsize=10)
    ax.set_ylabel('Total Units Sold', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    st.pyplot(fig)

# --- 数据表格 ---
st.divider()
st.subheader("📋 数据详情")

# 排序和搜索
search = st.text_input("🔍 搜索商品名/用户名", placeholder="输入关键词...")
table_df = filtered[['order_date', 'customer_name', 'segment',
                      'product_name', 'category', 'quantity', 'revenue']]
table_df['order_date'] = table_df['order_date'].dt.date

if search:
    table_df = table_df[
        table_df['product_name'].str.contains(search, case=False) |
        table_df['customer_name'].str.contains(search, case=False)
    ]

st.dataframe(
    table_df.sort_values('order_date', ascending=False),
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        'order_date': st.column_config.DateColumn('日期'),
        'customer_name': st.column_config.TextColumn('客户'),
        'segment': st.column_config.TextColumn('等级'),
        'product_name': st.column_config.TextColumn('商品'),
        'category': st.column_config.TextColumn('品类'),
        'quantity': st.column_config.NumberColumn('数量'),
        'revenue': st.column_config.NumberColumn('销售额', format='¥%.0f'),
    },
)

# --- 底部信息 ---
st.divider()
st.caption(
    "📊 项目2: 交互式数据分析看板 | "
    "数据来源: business.db | "
    f"当前筛选: {len(filtered):,} 行 | "
    "技术栈: SQLite + Pandas + Matplotlib + Streamlit"
)
