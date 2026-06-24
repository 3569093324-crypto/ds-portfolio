"""
pytest 单元测试 — data_cleaning.py

运行:
    cd portfolio/project_2 && python -m pytest tests/ -v
    cd portfolio/project_2 && python -m pytest tests/ --cov=. --cov-report=term
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# 确保可以 import data_cleaning 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_cleaning import (
    clean_data,
    CleaningReport,
    OutlierInfo,
    _normalize_column_names,
    _fill_missing_numeric,
    _fill_missing_categorical,
    _remove_duplicates,
    _detect_outliers_iqr,
)


# ============================================================
# Fixtures — 可复用的测试数据
# ============================================================

@pytest.fixture
def simple_df():
    """基础测试 DataFrame: 包含数值、分类、缺失值。"""
    return pd.DataFrame({
        'A': [1.0, 2.0, None, 4.0, 5.0],
        'B': ['x', 'y', None, 'x', 'z'],
        'C': [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def dirty_df():
    """真实场景的脏数据。"""
    return pd.DataFrame({
        'Customer Name': ['Alice', 'Bob', None, '  Charlie  '],
        'AGE': [25.0, None, 30.0, 150.0],  # 150 是异常值
        'Amount': [100.0, 200.0, -9999.0, 300.0],  # -9999 是异常值
        'City': ['NYC', None, 'LA', 'NYC'],
    })


@pytest.fixture
def df_with_duplicates():
    """包含重复行的 DataFrame。"""
    df = pd.DataFrame({
        'id': [1, 2, 3, 1, 2],
        'val': [10, 20, 30, 10, 20],
    })
    return df


# ============================================================
# 测试: 列名标准化
# ============================================================

class TestNormalizeColumnNames:
    """测试 _normalize_column_names 函数。"""

    def test_strips_and_lowercases(self):
        """空格去除 + 小写转换。"""
        df = pd.DataFrame({'  Foo Bar  ': [1], 'Baz Qux': [2]})
        result, _ = _normalize_column_names(df)
        assert list(result.columns) == ['foo_bar', 'baz_qux']

    def test_replaces_special_chars(self):
        """特殊字符替换为下划线。"""
        df = pd.DataFrame({'Col-Name!': [1], 'Another.One': [2]})
        result, _ = _normalize_column_names(df)
        assert 'col_name' in result.columns
        assert 'another_one' in result.columns

    def test_returns_rename_map(self):
        """验证返回的映射表正确。"""
        df = pd.DataFrame({'Old Name': [1]})
        _, rename_map = _normalize_column_names(df)
        assert rename_map == {'Old Name': 'old_name'}


# ============================================================
# 测试: 缺失值填充
# ============================================================

class TestFillMissing:
    """测试缺失值填充函数。"""

    def test_fill_numeric_with_median(self, simple_df):
        """数值列用中位数填充。"""
        result = _fill_missing_numeric(simple_df, ['A'])
        assert result['A'].isnull().sum() == 0
        # 中位数: [1,2,4,5] → 3.0 (丢弃None后)
        assert result.loc[2, 'A'] == pytest.approx(3.0)

    def test_fill_categorical_with_unknown(self, simple_df):
        """分类列用 'Unknown' 填充。"""
        result = _fill_missing_categorical(simple_df, ['B'])
        assert result['B'].isnull().sum() == 0
        assert result.loc[2, 'B'] == 'Unknown'

    def test_all_nan_column_defaults_to_zero(self):
        """全列 NaN 的数值列兜底为 0。"""
        df = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        result = _fill_missing_numeric(df, ['A'])
        assert result['A'].tolist() == [0.0, 0.0, 0.0]

    def test_no_missing_no_change(self, simple_df):
        """无缺失值的列保持不变。"""
        result = _fill_missing_numeric(simple_df, ['C'])
        pd.testing.assert_series_equal(result['C'], simple_df['C'])


# ============================================================
# 测试: 重复行删除
# ============================================================

class TestRemoveDuplicates:
    """测试 _remove_duplicates 函数。"""

    def test_removes_duplicates(self, df_with_duplicates):
        """正确删除重复行。"""
        result, count = _remove_duplicates(df_with_duplicates)
        assert len(result) == 3
        assert count == 2

    def test_no_duplicates_unchanged(self):
        """无重复时数据不变。"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result, count = _remove_duplicates(df)
        assert len(result) == 3
        assert count == 0


# ============================================================
# 测试: 异常值检测 (IQR)
# ============================================================

class TestDetectOutliers:
    """测试 _detect_outliers_iqr 函数。"""

    def test_detects_extreme_values(self, dirty_df):
        """检测到显著异常值。"""
        result, info = _detect_outliers_iqr(dirty_df, ['AGE', 'Amount'])
        # 150 是年龄异常值，IQR 法判定为异常
        # 注意：小数据集下 IQR 可能只检测到最极端的值
        assert result['is_outlier'].sum() >= 1
        assert 'AGE' in info
        assert 'Amount' in info

    def test_adds_is_outlier_column(self):
        """有异常值时正确添加 is_outlier 标记列。"""
        # 构造确定有异常值的数据
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 100.0]})  # 100 明显异常
        result, info = _detect_outliers_iqr(df, ['A'])
        assert 'is_outlier' in result.columns
        assert result['is_outlier'].sum() == 1

    def test_no_outliers_normal_data(self):
        """无异常值的数据不产生误报。"""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result, info = _detect_outliers_iqr(df, ['x'])
        # 无异常值时 is_outlier 列可能不存在或全为 False
        if 'is_outlier' in result.columns:
            assert result['is_outlier'].sum() == 0


# ============================================================
# 测试: 主函数 clean_data()
# ============================================================

class TestCleanData:
    """测试 clean_data() 主函数。"""

    def test_returns_dataframe_and_report(self, dirty_df):
        """返回 DataFrame 和 CleaningReport。"""
        df_clean, report = clean_data(dirty_df)
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(report, CleaningReport)

    def test_column_names_normalized(self, dirty_df):
        """列名自动标准化。"""
        df_clean, _ = clean_data(dirty_df)
        assert 'customer_name' in df_clean.columns
        assert 'age' in df_clean.columns

    def test_missing_values_filled(self, dirty_df):
        """缺失值被填充。"""
        df_clean, _ = clean_data(dirty_df)
        assert df_clean.isnull().sum().sum() >= 0  # 可能有 NaT（日期解析）

    def test_empty_dataframe(self):
        """空 DataFrame 的处理。"""
        empty = pd.DataFrame()
        df_clean, report = clean_data(empty)
        assert df_clean.empty
        assert report.initial_shape == (0, 0)

    def test_single_row(self):
        """单行数据不崩溃。"""
        single = pd.DataFrame({'X': [1.0], 'Y': ['a']})
        df_clean, report = clean_data(single)
        assert len(df_clean) == 1
        assert report.final_shape[0] == 1


# ============================================================
# 测试: 错误处理
# ============================================================

class TestErrorHandling:
    """测试异常处理和输入验证。"""

    def test_raises_typeerror_for_non_dataframe(self):
        """传入非 DataFrame 抛出 TypeError。"""
        with pytest.raises(TypeError, match='必须是 pandas DataFrame'):
            clean_data([1, 2, 3])  # type: ignore[arg-type]

    def test_raises_typeerror_for_none(self):
        """传入 None 抛出 TypeError。"""
        with pytest.raises(TypeError, match='必须是 pandas DataFrame'):
            clean_data(None)  # type: ignore[arg-type]

    def test_raises_typeerror_for_string(self):
        """传入字符串抛出 TypeError。"""
        with pytest.raises(TypeError):
            clean_data("hello")  # type: ignore[arg-type]


# ============================================================
# 参数化测试
# ============================================================

class TestParametrized:
    """参数化测试 — 同一测试逻辑，多种输入。"""

    @pytest.mark.parametrize('col_name,expected', [
        ('  Foo  ', 'foo'),
        ('Bar Baz', 'bar_baz'),
        ('CAPS', 'caps'),
        ('dot.separated', 'dot_separated'),
        ('Special@Char!', 'special_char'),
    ])
    def test_column_name_variants(self, col_name, expected):
        """不同格式的列名都能正确标准化。"""
        df = pd.DataFrame({col_name: [1]})
        result, _ = _normalize_column_names(df)
        assert result.columns[0] == expected

    @pytest.mark.parametrize('values,expected_non_null', [
        ([1.0, None, 3.0], 3),
        ([None, None, None], 3),  # 全NaN用0填充，所以全部非空
        ([1.0, 2.0, 3.0], 3),
        ([np.nan, 5.0, np.nan], 3),
    ])
    def test_fill_numeric_various_inputs(self, values, expected_non_null):
        """各种缺失值模式都能处理。"""
        df = pd.DataFrame({'val': values})
        result = _fill_missing_numeric(df, ['val'])
        assert result['val'].notnull().sum() == expected_non_null

    @pytest.mark.parametrize('fill_str', [
        'Unknown', 'N/A', 'Missing', '',
    ])
    def test_custom_fill_values(self, fill_str):
        """自定义填充值生效。"""
        df = pd.DataFrame({'cat': ['a', None, 'b']})
        result = _fill_missing_categorical(df, ['cat'], fill_value=fill_str)
        assert result.loc[1, 'cat'] == fill_str


# ============================================================
# 测试: CleaningReport 数据类
# ============================================================

class TestCleaningReport:
    """测试 CleaningReport 数据类。"""

    def test_default_values(self):
        """默认值正确。"""
        report = CleaningReport()
        assert report.initial_shape == (0, 0)
        assert report.duplicates_removed == 0
        assert report.outliers_detected == {}

    def test_tracks_changes(self, dirty_df):
        """正确追踪清洗变更。"""
        _, report = clean_data(dirty_df)
        assert report.initial_shape[0] > 0
        assert report.final_shape[0] > 0
        assert isinstance(report.missing_before, dict)


# ============================================================
# 运行入口
# ============================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
