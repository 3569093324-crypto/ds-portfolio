#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 18: 多数据源整合与数据管道
CSV + JSON + Excel → 校验 → 合并 → 干净DataFrame
"""

import pandas as pd
import numpy as np
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "pipeline_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 生成3种格式的数据源
# ============================================================
print("=" * 60)
print("  1. 生成3种格式数据源")
print("=" * 60)

# --- 1a. CSV: 用户基本信息 ---
users_csv = pd.DataFrame({
    'USER_ID': range(1, 11),
    'user_name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve',
                   'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix',
              'NYC', 'LA', 'Chicago', 'Houston', None],
    'signup_date': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-01-15',
                     '2024-04-01', '2024-05-12', '2024-06-01', '2024-03-15',
                     '2024-07-01', '2024-08-10'],
})
csv_path = os.path.join(OUT_DIR, "users.csv")
users_csv.to_csv(csv_path, index=False)
print(f"  CSV:  {csv_path} — {users_csv.shape}")

# --- 1b. JSON: 用户消费记录 ---
orders_json = [
    {"user": 1, "order_amt": 150.5, "items": 3},
    {"user": 2, "order_amt": 89.9, "items": 1},
    {"user": 1, "order_amt": 234.0, "items": 5},  # Alice 2条记录
    {"user": 3, "order_amt": 567.8, "items": 8},
    {"user": 4, "order_amt": 45.0, "items": 2},
    {"user": 5, "order_amt": 1234.5, "items": 12},
    {"user": 6, "order_amt": 78.0, "items": 1},
    {"user": 8, "order_amt": 345.6, "items": 4},
    {"user": 9, "order_amt": 910.0, "items": 7},
    {"user": 2, "order_amt": 150.0, "items": 2},  # Bob重复
    {"user": 11, "order_amt": 999.0, "items": 10},  # user 11 不在CSV中
    {"user": 3, "order_amt": 200.0, "items": 3},   # Charlie重复
]
json_path = os.path.join(OUT_DIR, "orders.json")
with open(json_path, 'w') as f:
    json.dump(orders_json, f, indent=2)
print(f"  JSON: {json_path} — {len(orders_json)} 行")

# --- 1c. Excel: 用户等级信息 ---
excel_data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],  # user 12 不在CSV中
    'membership': ['Gold', 'Silver', 'Gold', 'Bronze', 'Silver',
                    'Bronze', 'Gold', 'Silver', 'Gold', 'Diamond'],
    'points': [1500, 800, 2000, 300, 900, 400, 1800, 750, 2100, 5000],
})
excel_path = os.path.join(OUT_DIR, "memberships.xlsx")
excel_data.to_excel(excel_path, index=False)
print(f"  Excel: {excel_path} — {excel_data.shape}")


# ============================================================
# 2. 分别读取
# ============================================================
print("\n" + "=" * 60)
print("  2. 用 pandas 分别读取")
print("=" * 60)

df_users = pd.read_csv(csv_path)
df_orders = pd.read_json(json_path)
df_members = pd.read_excel(excel_path)

print(f"  CSV (users):    {df_users.shape} — columns: {list(df_users.columns)}")
print(f"  JSON (orders):  {df_orders.shape} — columns: {list(df_orders.columns)}")
print(f"  Excel (members):{df_members.shape} — columns: {list(df_members.columns)}")


# ============================================================
# 3. 数据校验（合并前）
# ============================================================
print("\n" + "=" * 60)
print("  3. 数据校验 — 合并前检查")
print("=" * 60)

def validate_before_merge(dfs, key_maps):
    """
    合并前校验
    dfs: dict of {name: DataFrame}
    key_maps: dict of {name: key_column_name}
    """
    report = {}
    for name, df in dfs.items():
        key_col = key_maps[name]
        report[name] = {
            'shape': df.shape,
            'key_name': key_col,
            'key_dtype': str(df[key_col].dtype),
            'key_unique': int(df[key_col].nunique()),
            'key_duplicates': int(df.duplicated(subset=[key_col]).sum()),
            'key_nulls': int(df[key_col].isnull().sum()),
        }
        # 检查数据类型一致性
        print(f"  {name}:")
        print(f"    key={key_col}  dtype={df[key_col].dtype}  "
              f"unique={df[key_col].nunique()}/{len(df)}  "
              f"duplicates={df.duplicated(subset=[key_col]).sum()}  "
              f"nulls={df[key_col].isnull().sum()}")
    return report

key_maps = {
    'users': 'USER_ID',
    'orders': 'user',
    'members': 'id',
}

validation_report = validate_before_merge(
    {'users': df_users, 'orders': df_orders, 'members': df_members},
    key_maps
)

# 发现问题：
issues = []
issues.append("  ⚠️ orders中有user=11 (不在users中) → LEFT JOIN保留")
issues.append("  ⚠️ members中有id=12 (不在users中) → LEFT JOIN保留")
issues.append("  ⚠️ users的key名='USER_ID', orders='user', members='id' → 需统一")
issues.append("  ⚠️ orders的user=2有2条记录 → 合并会产生多行")
for issue in issues:
    print(issue)


# ============================================================
# 4 & 5. 合并3个数据源 + 处理合并问题
# ============================================================
print("\n" + "=" * 60)
print("  4 & 5. 合并 + 处理问题")
print("=" * 60)

# 策略：以 users 为左表，LEFT JOIN orders 和 members
# 处理key名称不一致、类型不匹配、重复key

# 先统一key名并确保类型一致
df_users['USER_ID'] = df_users['USER_ID'].astype(int)
df_orders['user'] = df_orders['user'].astype(int)
df_members['id'] = df_members['id'].astype(int)

# Step 1: users LEFT JOIN orders
merged = df_users.merge(
    df_orders,
    left_on='USER_ID',
    right_on='user',
    how='left',
    indicator='_merge_orders'
)
print(f"  users + orders: {merged.shape}")

# Step 2: 再 LEFT JOIN members
merged = merged.merge(
    df_members,
    left_on='USER_ID',
    right_on='id',
    how='left',
    indicator='_merge_members'
)
print(f"  + members:      {merged.shape}")

# 删除多余列
merged = merged.drop(columns=['user', 'id'])

# 标记数据来源
merged['in_users']   = ~merged['USER_ID'].isnull()
merged['in_orders']  = merged['_merge_orders'] == 'both'
merged['in_members'] = merged['_merge_members'] == 'both'
merged = merged.drop(columns=['_merge_orders', '_merge_members'])

print(f"\n  合并结果预览:")
print(merged.to_string())


# ============================================================
# 6. build_dataset() 函数
# ============================================================
print("\n" + "=" * 60)
print("  6. build_dataset() — 自动化数据管道")
print("=" * 60)

def build_dataset(file_configs, base_key='USER_ID'):
    """
    自动读取→校验→合并→返回干净DataFrame

    Parameters
    ----------
    file_configs : list of dict
        每个dict包含:
        - path: 文件路径
        - key: 该文件的join key列名
        - reader: 'csv', 'json', 'excel'
    base_key : str
        最终DataFrame中的统一key列名

    Returns
    -------
    df : pd.DataFrame
        合并后的数据
    report : dict
        处理报告
    """
    dfs = {}
    readers = {
        'csv': pd.read_csv,
        'json': pd.read_json,
        'excel': pd.read_excel,
    }

    # 读取所有文件
    for cfg in file_configs:
        reader_fn = readers[cfg['reader']]
        df = reader_fn(cfg['path'])
        # 统一key列名
        df = df.rename(columns={cfg['key']: base_key})
        # 确保key类型一致
        df[base_key] = pd.to_numeric(df[base_key], errors='coerce')
        dfs[cfg['reader']] = df
        print(f"  Read {cfg['reader']}: {cfg['path']} → {df.shape}")

    # 校验
    for name, df in dfs.items():
        dup = df.duplicated(subset=[base_key]).sum()
        nulls = df[base_key].isnull().sum()
        if dup > 0 or nulls > 0:
            print(f"  ⚠️ {name}: {dup} duplicate keys, {nulls} null keys")

    # 合并（迭代 LEFT JOIN）
    result = dfs[list(dfs.keys())[0]]
    sources_used = [list(dfs.keys())[0]]
    for name in list(dfs.keys())[1:]:
        result = result.merge(
            dfs[name],
            on=base_key,
            how='outer',  # outer 保留所有数据
            suffixes=('', f'_{name}')
        )
        sources_used.append(name)

    report = {
        'sources': sources_used,
        'final_shape': result.shape,
        'columns': list(result.columns),
        'total_rows': len(result),
    }
    return result, report


# 使用 build_dataset
configs = [
    {'path': csv_path,   'key': 'USER_ID', 'reader': 'csv'},
    {'path': json_path,  'key': 'user',    'reader': 'json'},
    {'path': excel_path, 'key': 'id',      'reader': 'excel'},
]

final_df, report = build_dataset(configs)
print(f"\n  build_dataset() 结果:")
print(f"  Final shape: {report['final_shape']}")
print(f"  Columns: {report['columns']}")
print(f"\n{final_df.head(12).to_string()}")


# ============================================================
# 7. assert 验证
# ============================================================
print("\n" + "=" * 60)
print("  7. assert 验证")
print("=" * 60)

# 验证1: 行数合理
assert len(final_df) >= 10, f"Expected >=10 rows, got {len(final_df)}"
print(f"  ✅ assert len >= 10: {len(final_df)} rows")

# 验证2: 所有原始key都有对应行
all_keys = set(range(1, 11))  # users 1-10
extra_keys = {11, 12}         # 来自orders和members
assert all_keys.issubset(set(final_df['USER_ID'].dropna())), "Missing users!"
print(f"  ✅ All 10 original users present")

# 验证3: 包含额外来源的key
assert 11 in final_df['USER_ID'].values, "Missing user 11 (from orders)"
assert 12 in final_df['USER_ID'].values, "Missing user 12 (from members)"
print(f"  ✅ Extra keys (11, 12) from orders/members present")

# 验证4: 必需列存在
required_cols = ['USER_ID', 'user_name', 'order_amt', 'membership']
for col in required_cols:
    assert col in final_df.columns, f"Missing required column: {col}"
print(f"  ✅ All required columns present: {required_cols}")

# 验证5: 无完全空行
assert final_df['USER_ID'].notna().sum() == len(final_df), "Empty key rows!"
print(f"  ✅ No empty key rows")

# 验证6: user_name 已填充（来自CSV）
csv_users = final_df[final_df['USER_ID'].between(1, 10)]
assert csv_users['user_name'].notna().all(), "CSV users missing names!"
print(f"  ✅ CSV users all have names")

print(f"\n  🎉 所有assert验证通过！")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("  Day 18 总结：数据整合管道核心")
print("=" * 60)
print("""
  1. 多格式读取: pd.read_csv / read_json / read_excel
  2. 合并前校验: key唯一性、类型一致性、NULL检查
  3. merge策略:
     - how='inner' → 只保留交集（类比INNER JOIN）
     - how='left'  → 保留左表全部（类比LEFT JOIN）
     - how='outer' → 保留全部（类比FULL OUTER JOIN）
  4. 常见问题处理:
     - key名称不一致 → rename columns
     - 类型不匹配   → astype() / pd.to_numeric()
     - 重复key      → duplicated() 检测 + 去重
     - 多对多合并    → 注意行数膨胀
  5. assert 验证: 确保合并结果的数据完整性
""")
