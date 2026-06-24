"""
集中配置管理 — 避免硬编码，支持环境变量覆盖
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """模型训练配置"""
    random_state: int = 42
    test_size: float = 0.25
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"


@dataclass
class XGBoostConfig:
    """XGBoost 默认参数"""
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 20


@dataclass
class PathConfig:
    """路径配置 — 支持环境变量覆盖"""
    # 数据库路径
    db_path: str = field(default_factory=lambda: os.getenv(
        "DB_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "business.db")
    ))
    # 模型保存路径
    model_dir: str = field(default_factory=lambda: os.getenv(
        "MODEL_DIR", os.path.join(os.path.dirname(__file__))
    ))
    # 输出目录
    output_dir: str = field(default_factory=lambda: os.getenv(
        "OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "visuals")
    ))


@dataclass
class APIConfig:
    """API 服务配置"""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    model_path: str = field(default_factory=lambda: os.getenv(
        "MODEL_PATH", os.path.join(os.path.dirname(__file__), "model_xgboost.joblib")
    ))


# 默认配置实例
model_cfg = ModelConfig()
xgb_cfg = XGBoostConfig()
path_cfg = PathConfig()
api_cfg = APIConfig()
