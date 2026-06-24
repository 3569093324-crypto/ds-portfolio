#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用数据清洗模块 — 可直接 import 复用于项目2看板的数据处理管线。

重构版 (Day 20): 添加了完整的 Type Hints、Google-style docstring、
输入验证和单一职责拆分。

用法:
    from data_cleaning import clean_data, CleaningReport
    df_clean, report = clean_data(df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class OutlierInfo:
    """单个列的异常值检测结果。"""
    count: int
    lower_bound: float
    upper_bound: float


@dataclass
class CleaningReport:
    """数据清洗报告 — 记录清洗前后的所有变更。

    通过结构化数据而非字典返回，让调用方有完整的类型提示和 IDE 自动补全。

    Attributes:
        initial_shape: 原始数据的 (行数, 列数)
        final_shape: 清洗后数据的 (行数, 列数)
        rows_removed: 删除的总行数
        duplicates_removed: 删除的重复行数
        total_missing_filled: 填充的缺失值总数
        columns_renamed: 列名变更映射 {旧名: 新名}
        missing_before: 清洗前各列缺失值计数
        missing_after: 清洗后各列缺失值计数
        outliers_detected: 检测到的异常值信息
    """
    initial_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    rows_removed: int = 0
    duplicates_removed: int = 0
    total_missing_filled: int = 0
    columns_renamed: Optional[Dict[str, str]] = None
    missing_before: Dict[str, int] = field(default_factory=dict)
    missing_after: Dict[str, int] = field(default_factory=dict)
    outliers_detected: Dict[str, OutlierInfo] = field(default_factory=dict)


# ============================================================
# 单一职责函数 — 每步只做一件事
# ============================================================

def _normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """标准化列名：去空格 → 小写 → 特殊字符替换为下划线。

    为什么这样做：不同的数据源（CSV/JSON/Excel）列名风格不同，
    统一为 snake_case 可以避免后续分析中因大小写不一致导致的 KeyError。

    Args:
        df: 输入 DataFrame

    Returns:
        (标准化后的 DataFrame, {旧列名: 新列名} 映射)
    """
    old_cols = list(df.columns)
    new_cols = (pd.Index(old_cols)
                .str.strip()
                .str.lower()
                .str.replace(r'[^a-z0-9]+', '_', regex=True)
                .str.strip('_'))
    df = df.copy()
    df.columns = new_cols
    rename_map = dict(zip(old_cols, new_cols))
    return df, rename_map


def _fill_missing_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """用中位数填充数值列的缺失值。

    选择中位数而非均值的原因：中位数对异常值稳健——
    如果列中存在极端离群值（如 99999），均值会被拉偏而中位数不变。

    Args:
        df: DataFrame（已标准化列名）
        cols: 数值列名列表

    Returns:
        填充后的 DataFrame（原地修改的副本）
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        median_val = df[col].median()
        # 兜底：如果全列都是 NaN（median 返回 NaN），用 0 填充
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)
    return df


def _fill_missing_categorical(
    df: pd.DataFrame, cols: List[str], fill_value: str = 'Unknown'
) -> pd.DataFrame:
    """用指定值填充分类列的缺失值。

    为什么用 'Unknown' 而非众数：分类列的缺失通常意味着'数据未采集'，
    用 'Unknown' 明确标记这一点，避免用众数掩盖数据质量问题。

    Args:
        df: DataFrame
        cols: 分类列名列表
        fill_value: 填充值，默认 'Unknown'

    Returns:
        填充后的 DataFrame
    """
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(fill_value)
    return df


def _remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """检测并删除完全重复的行。

    为什么只删除完全重复行而不做模糊去重：完全重复行通常是数据采集重复，
    确定性高；模糊去重可能误删合法数据（如两个同名同姓的不同用户）。

    Args:
        df: DataFrame

    Returns:
        (去重后的 DataFrame, 删除的行数)
    """
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, dup_count


def _detect_outliers_iqr(
    df: pd.DataFrame, cols: List[str], multiplier: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, OutlierInfo]]:
    """用 IQR 法检测异常值（标记为 is_outlier 列，不删除）。

    为什么标记而非删除：异常值可能是真实的极端事件（如大客户大额订单），
    直接删除会丢失信息。标记后由分析师根据业务上下文决定如何处理。

    IQR = Q3 - Q1
    异常下界 = Q1 - 1.5*IQR
    异常上界 = Q3 + 1.5*IQR

    Args:
        df: DataFrame
        cols: 数值列名列表
        multiplier: IQR 乘数，默认 1.5（标准 Tukey 法）

    Returns:
        (含 is_outlier 列的 DataFrame, 异常值信息)
    """
    df = df.copy()
    outlier_info: Dict[str, OutlierInfo] = {}

    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        is_outlier = (df[col] < lower) | (df[col] > upper)
        if is_outlier.any():
            outlier_info[col] = OutlierInfo(
                count=int(is_outlier.sum()),
                lower_bound=round(float(lower), 2),
                upper_bound=round(float(upper), 2),
            )
            # 累积标记（一个行可能在多个列都是异常值）
            if 'is_outlier' not in df.columns:
                df['is_outlier'] = False
            df['is_outlier'] = df['is_outlier'] | is_outlier
        else:
            outlier_info[col] = OutlierInfo(count=0, lower_bound=0, upper_bound=0)

    return df, outlier_info


def _parse_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """解析日期列（errors='coerce' 将无效日期转为 NaT）。

    Args:
        df: DataFrame
        cols: 日期列名列表

    Returns:
        日期列已解析的 DataFrame
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


# ============================================================
# 主函数 — 串联整个清洗管线
# ============================================================

def clean_data(
    df: pd.DataFrame,
    date_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    categorical_fill: str = 'Unknown',
    outlier_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """对 DataFrame 执行完整的清洗管线。

    清洗步骤（按顺序）：
    1. 列名标准化（snake_case）
    2. 数值列缺失值 → 中位数填充
    3. 分类列缺失值 → 'Unknown' 填充
    4. 删除完全重复行
    5. IQR 法异常值标记（不删除）
    6. 日期列解析

    如果未指定列类型，函数会自动检测：
    - 数值列: select_dtypes(include=[np.number])
    - 分类列: select_dtypes(include=['object', 'category'])
    - 日期列: 列名含 'date' 或 'time'

    Args:
        df: 待清洗的原始 DataFrame
        date_cols: 日期列名列表（自动检测）
        numeric_cols: 数值列名列表（自动检测）
        cat_cols: 分类列名列表（自动检测）
        categorical_fill: 分类列缺失值填充文本，默认 'Unknown'
        outlier_multiplier: IQR 异常值检测乘数，默认 1.5

    Returns:
        (清洗后的 DataFrame, CleaningReport)

    Raises:
        TypeError: 如果 df 不是 DataFrame

    Examples:
        >>> df = pd.DataFrame({'A': [1, None, 3], 'B': ['x', None, 'z']})
        >>> clean, report = clean_data(df)
        >>> report.total_missing_filled
        2
    """
    # --- 输入验证 ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df 必须是 pandas DataFrame，得到 {type(df).__name__}"
        )
    if df.empty:
        return df.copy(), CleaningReport(initial_shape=df.shape, final_shape=df.shape)

    df = df.copy()
    report = CleaningReport(initial_shape=df.shape)

    # Step 1: 列名标准化
    df, rename_map = _normalize_column_names(df)
    if rename_map and any(k != v for k, v in rename_map.items()):
        report.columns_renamed = {
            k: v for k, v in rename_map.items() if k != v
        }

    # 同步用户传入的列名（匹配标准化后的列名）
    if date_cols:
        date_cols = [rename_map.get(c, c) for c in date_cols]
    if numeric_cols:
        numeric_cols = [rename_map.get(c, c) for c in numeric_cols]
    if cat_cols:
        cat_cols = [rename_map.get(c, c) for c in cat_cols]

    # 自动检测列类型（在列名标准化之后）
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    if date_cols is None:
        date_cols = [
            c for c in df.columns
            if 'date' in c.lower() or 'time' in c.lower()
        ]

    # Step 2: 记录缺失值
    report.missing_before = {
        k: int(v) for k, v in df.isnull().sum().to_dict().items() if v > 0
    }

    # Step 3: 填充缺失值
    df = _fill_missing_numeric(df, numeric_cols)
    df = _fill_missing_categorical(
        df,
        [c for c in cat_cols if c not in (date_cols or [])],
        fill_value=categorical_fill,
    )

    report.missing_after = {
        k: int(v) for k, v in df.isnull().sum().to_dict().items() if v > 0
    }
    report.total_missing_filled = sum(report.missing_before.values()) - sum(
        report.missing_after.values()
    )

    # Step 4: 删除重复行
    df, dup_count = _remove_duplicates(df)
    report.duplicates_removed = dup_count

    # Step 5: 异常值检测
    df, outlier_info = _detect_outliers_iqr(df, numeric_cols, multiplier=outlier_multiplier)
    report.outliers_detected = {
        k: v for k, v in outlier_info.items() if v.count > 0
    }

    # Step 6: 日期解析
    df = _parse_dates(df, date_cols)

    # 完成
    report.final_shape = df.shape
    report.rows_removed = report.initial_shape[0] - report.final_shape[0]

    return df, report


def print_cleaning_report(report: CleaningReport) -> None:
    """格式化打印清洗报告到终端。

    Args:
        report: CleaningReport 对象（由 clean_data() 返回）
    """
    print("\n" + "=" * 60)
    print("  数据清洗报告")
    print("=" * 60)
    print(f"  原始形状: {report.initial_shape}")
    print(f"  清洗后形状: {report.final_shape}")
    print(f"  删除重复行: {report.duplicates_removed}")
    print(f"  填充缺失值: {report.total_missing_filled} 个")

    if report.outliers_detected:
        print(f"\n  📍 异常值检测 (IQR 法):")
        for col, info in report.outliers_detected.items():
            print(f"    {col}: {info.count} 个异常值 "
                  f"(范围 [{info.lower_bound}, {info.upper_bound}])")

    if report.missing_before:
        print(f"\n  📍 缺失值处理:")
        for col, cnt in report.missing_before.items():
            after = report.missing_after.get(col, 0)
            print(f"    {col}: {cnt} → {after}")

    if report.columns_renamed:
        print(f"\n  📍 列名标准化:")
        for old, new in report.columns_renamed.items():
            print(f"    '{old}' → '{new}'")


# ============================================================
# 自测
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)

    n = 200
    dirty = pd.DataFrame({
        'Customer Name': ['Alice', 'Bob', None, '  Charlie  ', 'Diana'] * 40,
        'AGE': np.random.randint(18, 90, n).astype(float),
        'Order Date': np.random.choice([
            '2024-01-15', '2024/03/20', 'invalid_date', '2024-06-01', '2024-12-25'
        ], n),
        'Amount': np.random.normal(500, 200, n),
        'City': np.random.choice(['NYC', 'LA', ' SF ', None, 'Chicago'], n),
    })

    # 注入问题
    dirty.loc[10:15, 'AGE'] = np.nan
    dirty.loc[20:22, 'Amount'] = 99999
    dirty.loc[30:32, 'Amount'] = -5000
    dirty.loc[40:42, 'City'] = None
    dirty.loc[5, :] = dirty.loc[0, :]
    dirty.loc[6, :] = dirty.loc[1, :]
    dirty.loc[50, 'AGE'] = 150

    # 类型验证：传非 DataFrame 应该抛出 TypeError
    try:
        clean_data("not a dataframe")  # type: ignore[arg-type]
        print("❌ 应该抛出 TypeError！")
    except TypeError as e:
        print(f"✅ 输入验证生效: {e}")

    # 正常清洗
    df_clean, report = clean_data(dirty)
    print_cleaning_report(report)

    print(f"\n✅ data_cleaning.py 重构完成")
    print(f"  - 4个单一职责函数 + 1个主函数")
    print(f"  - 所有函数有 Type Hints + Google-style docstring")
    print(f"  - 用 @dataclass 定义 CleaningReport 和 OutlierInfo")
    print(f"  - 输入类型验证（TypeError for non-DataFrame）")
    print(f"  - 注释解释'为什么'而非'做什么'")
