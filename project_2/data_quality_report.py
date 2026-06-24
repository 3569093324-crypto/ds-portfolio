#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 22: 数据质量报告自动化
generate_data_report(df) → Markdown/HTML 格式的质量评估报告
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def generate_data_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "数据质量报告",
    top_n: int = 10,
) -> str:
    """为任意 DataFrame 生成 Markdown 格式的数据质量报告。

    报告内容:
    1. 基本信息（行数列数、内存占用、列类型分布）
    2. 每列概览（类型、缺失率、唯一值数）
    3. 数值列详细统计（均值/标准差/分位数/偏度/峰度）
    4. 分类列频率分布（Top N）
    5. 异常值检测（IQR 法）
    6. 缺失值热力图数据表

    Args:
        df: 输入 DataFrame
        output_path: 报告保存路径（.md），为 None 则只返回字符串
        title: 报告标题
        top_n: 分类列展示前 N 个频次

    Returns:
        Markdown 格式的报告文本

    Examples:
        >>> df = pd.DataFrame({'a': [1,2,3], 'b': ['x','y','z']})
        >>> report = generate_data_report(df, 'report.md')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

    rows, cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # 列类型分类
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    other_cols = [c for c in df.columns
                  if c not in numeric_cols + cat_cols + date_cols]

    lines = []
    def w(s: str = "") -> None:
        lines.append(s)

    # ============================================================
    # 标题 & 生成时间
    # ============================================================
    w(f"# {title}")
    w()
    w(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**数据来源**: 内存中的 DataFrame")
    w()

    # ============================================================
    # Section 1: 基本信息
    # ============================================================
    w("## 1. 基本信息")
    w()
    w("| 指标 | 值 |")
    w("|------|----|")
    w(f"| 行数 | {rows:,} |")
    w(f"| 列数 | {cols} |")
    w(f"| 内存占用 | {memory_mb:.2f} MB |")
    w(f"| 数值列 | {len(numeric_cols)} |")
    w(f"| 分类列 | {len(cat_cols)} |")
    w(f"| 日期列 | {len(date_cols)} |")
    w(f"| 其他列 | {len(other_cols)} |")
    w(f"| 重复行 | {int(df.duplicated().sum())} |")
    w(f"| 完全空行 | {int(df.isnull().all(axis=1).sum())} |")
    w()

    # ============================================================
    # Section 2: 每列概览
    # ============================================================
    w("## 2. 列概览")
    w()
    w("| # | 列名 | 类型 | 缺失数 | 缺失率 | 唯一值 | 备注 |")
    w("|----|------|------|--------|--------|--------|------|")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        missing_pct = f"{missing / rows * 100:.1f}%"
        unique = df[col].nunique()
        notes = []
        if missing > 0:
            notes.append(f"⚠️ {missing} 个缺失")
        if unique == 1:
            notes.append("常量列")
        elif unique == rows:
            notes.append("全唯一")
        if df[col].dtype == 'object' and unique < 20:
            notes.append("可能应为分类")
        note_str = ", ".join(notes) if notes else "-"
        w(f"| {i} | `{col}` | {dtype} | {missing} | {missing_pct} | {unique} | {note_str} |")
    w()

    # ============================================================
    # Section 3: 数值列详细统计
    # ============================================================
    if numeric_cols:
        w("## 3. 数值列统计")
        w()
        stats = df[numeric_cols].describe().round(2)
        # 添加偏度和峰度
        skew = df[numeric_cols].skew().round(2)
        kurt = df[numeric_cols].kurtosis().round(2)

        w("| 统计量 | " + " | ".join(f"`{c}`" for c in numeric_cols) + " |")
        w("|--------|" + "|".join("------:" for _ in numeric_cols) + "|")
        for stat_name in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            vals = stats.loc[stat_name] if stat_name in stats.index else ['-'] * len(numeric_cols)
            row = f"| {stat_name} | " + " | ".join(
                f"{vals[c]:,.2f}" if not isinstance(vals, pd.Series) and c in vals.index
                else f"{vals[c]:,.2f}" if stat_name in stats.index
                else "-"
                for c in numeric_cols
            ) + " |"
            # 简化
            vals_dict = stats.loc[stat_name].to_dict() if stat_name in stats.index else {}
            vals_str = " | ".join(f"{vals_dict.get(c, '-'):,}" if isinstance(vals_dict.get(c), (int, float)) else str(vals_dict.get(c, '-'))
                                  for c in numeric_cols)
            w(f"| {stat_name} | {vals_str} |")

        # 偏度
        skew_str = " | ".join(f"{skew.get(c, '-'):.2f}" for c in numeric_cols)
        w(f"| skewness | {skew_str} |")
        # 峰度
        kurt_str = " | ".join(f"{kurt.get(c, '-'):.2f}" for c in numeric_cols)
        w(f"| kurtosis | {kurt_str} |")
        w()
        w("*skewness > 0 = 右偏（长尾在右），kurtosis > 3 = 厚尾分布*")
        w()

    # ============================================================
    # Section 4: 分类列频率分布
    # ============================================================
    if cat_cols:
        w("## 4. 分类列 Top {} 频率分布".format(top_n))
        w()
        for col in cat_cols:
            if df[col].nunique() == 0:
                w(f"### `{col}` — 全为空")
                continue
            w(f"### `{col}` (唯一值: {df[col].nunique()})")
            w()
            vc = df[col].value_counts().head(top_n)
            total = vc.sum()
            w("| 值 | 频次 | 占比 | 累积占比 |")
            w("|----|------|------|----------|")
            cum = 0
            for val, cnt in vc.items():
                pct = cnt / total * 100
                cum += pct
                display_val = str(val)[:50] + ('...' if len(str(val)) > 50 else '')
                w(f"| `{display_val}` | {cnt} | {pct:.1f}% | {cum:.1f}% |")
            w()
    w()

    # ============================================================
    # Section 5: 异常值检测 (IQR)
    # ============================================================
    if numeric_cols:
        w("## 5. 异常值检测 (IQR 法)")
        w()
        w("| 列名 | 下界 | 上界 | 异常值数 | 异常比例 |")
        w("|------|------|------|----------|----------|")
        outlier_cols = []
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outlier_cols.append(col)
            w(f"| `{col}` | {lower:,.2f} | {upper:,.2f} | "
              f"{'🔴' if outlier_count > 0 else '🟢'} {outlier_count} | "
              f"{outlier_count/rows*100:.2f}% |")
        w()
        if outlier_cols:
            w(f"⚠️ 检测到 {len(outlier_cols)} 列含异常值: "
              f"{', '.join(f'`{c}`' for c in outlier_cols)}")
        else:
            w("✅ 所有数值列无 IQR 异常值")
        w()

    # ============================================================
    # Section 6: 缺失值汇总
    # ============================================================
    missing_series = df.isnull().sum()
    missing_cols = missing_series[missing_series > 0]
    w("## 6. 缺失值汇总")
    w()
    if len(missing_cols) > 0:
        w("| 列名 | 缺失数 | 缺失率 |")
        w("|------|--------|--------|")
        for col, cnt in missing_cols.items():
            w(f"| `{col}` | {cnt} | {cnt/rows*100:.1f}% |")
        w()
        w(f"**总缺失率**: {missing_series.sum() / (rows * cols) * 100:.2f}%")
    else:
        w("✅ 无缺失值")
    w()

    # 组装
    report = "\n".join(lines)

    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  报告已保存到: {output_path}")

    return report


# ============================================================
# 自测
# ============================================================
if __name__ == '__main__':
    # 用 business.db 的数据测试
    import sqlite3

    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "business.db")
    conn = sqlite3.connect(DB_PATH)

    # 构造一个分析用数据集
    df = pd.read_sql("""
        SELECT c.name, c.segment, c.city, c.join_date,
               o.order_date, o.total_amount,
               p.category, p.name AS product_name,
               oi.quantity, oi.unit_price,
               oi.quantity * oi.unit_price AS line_total
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        LIMIT 500
    """, conn)
    conn.close()

    print(f"生成数据质量报告 — {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"内存: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print()

    output = os.path.join(os.path.dirname(__file__), "data_quality_report.md")
    report = generate_data_report(df, output_path=output, title="电商数据质量报告")

    # 打印前800字符预览
    print("\n--- 报告预览 (前800字符) ---")
    print(report[:800])
    print("...")

    print(f"\n✅ 完整报告已保存到: {output}")
