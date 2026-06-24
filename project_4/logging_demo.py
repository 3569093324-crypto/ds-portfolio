#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 58: 日志、错误处理与配置管理
生产级代码的最佳实践
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
import traceback

# ============================================================
# 1. 配置日志 (不用 print!)
# ============================================================
def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    配置日志器 — 同时输出到控制台和文件

    Args:
        name: 日志器名称
        log_file: 日志文件路径 (None=不写文件)
        level: 日志级别

    Returns:
        配置好的 Logger 实例

    Example:
        >>> logger = setup_logger("my_module", "app.log")
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 格式: 时间 | 级别 | 模块 | 消息
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件 handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 模块级 logger
logger = setup_logger("project_4", "project_4.log")

# ============================================================
# 2. 异常处理 — 合理使用 try/except
# ============================================================
class DataLoadError(Exception):
    """自定义数据加载异常"""
    pass

class ModelNotLoadedError(Exception):
    """模型未加载异常"""
    pass

def safe_read_csv(path: str) -> Optional['pd.DataFrame']:
    """
    安全读取 CSV — 带完整错误处理

    为什么每个 except 分支分开写？
    → 不同错误需要不同的处理策略和日志信息。
    """
    import pandas as pd

    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")

    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"数据文件不存在: {path}")

    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"File is empty: {path}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        raise DataLoadError(f"无法解析CSV文件 {path}: {e}") from e
    except Exception as e:
        # 未预期的错误 — 记录完整 traceback
        logger.error(f"Unexpected error loading {path}: {e}\n{traceback.format_exc()}")
        raise

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法 — 避免除零错误"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numeric")
    if b == 0:
        logger.debug(f"Division by zero prevented: {a}/{b}")
        return default
    return a / b


# ============================================================
# 3. 配置管理演示
# ============================================================
from config import model_cfg, path_cfg, api_cfg

logger.info("=" * 50)
logger.info("Configuration Summary")
logger.info("=" * 50)
logger.info(f"Model: random_state={model_cfg.random_state}, cv_folds={model_cfg.cv_folds}")
logger.info(f"Paths: db={path_cfg.db_path}")
logger.info(f"API: host={api_cfg.host}:{api_cfg.port}, debug={api_cfg.debug}")

# ============================================================
# 4. 输入验证
# ============================================================
def validate_dataframe(df, required_cols: list, min_rows: int = 1) -> None:
    """
    验证 DataFrame 完整性

    Raises:
        ValueError: 列缺失或行数不足
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, got {len(df)}")

    logger.debug(f"DataFrame validated: {len(df)} rows, {len(df.columns)} cols")


# ============================================================
# 5. 演示
# ============================================================
if __name__ == "__main__":
    import pandas as pd

    logger.info("Starting Day 58 logging & error handling demo")

    # 演示1: 正常日志
    logger.info("This is INFO — normal operations")
    logger.warning("This is WARNING — something to check")
    logger.debug("This is DEBUG — detailed diagnostic info")

    # 演示2: 异常处理
    try:
        safe_read_csv("non_existent_file.csv")
    except FileNotFoundError as e:
        logger.error(f"Caught expected error: {e}")

    # 演示3: 输入验证
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    try:
        validate_dataframe(df, ['a', 'b', 'c'])  # 'c' 不存在
    except ValueError as e:
        logger.warning(f"Validation failed as expected: {e}")

    # 演示4: 安全除法
    result = safe_divide(10, 0)
    logger.info(f"safe_divide(10, 0) = {result} (default used)")

    # 演示5: 配置来自环境变量
    logger.info(f"DB_PATH (from env or default): {path_cfg.db_path}")

    logger.info("Day 58 demo completed")
    print(f"\n✅ Log file written to: project_4.log")
