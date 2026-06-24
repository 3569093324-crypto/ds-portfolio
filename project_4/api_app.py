"""
Project 4 Model API — FastAPI 模型服务

启动: uvicorn api_app:app --reload --port 8000
文档: http://localhost:8000/docs
"""

import joblib
import numpy as np
import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# ============================================================
# 模型加载 (应用启动时)
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_xgboost.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.json")

model = None
scaler = None
feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载模型"""
    global model, scaler, feature_names
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            feature_names = json.load(f)
    print(f"Model loaded: {type(model).__name__}, features: {len(feature_names) if feature_names else 0}")
    yield

app = FastAPI(
    title="Repurchase Prediction API",
    description="预测用户未来30天内是否复购 — XGBoost Model",
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================
# Pydantic 数据模型
# ============================================================
class PredictionInput(BaseModel):
    """单条预测输入"""
    features: List[float] = Field(
        ..., min_length=10, max_length=10,
        description="10个特征值 (feature_0 ~ feature_9)",
        example=[0.5, -1.2, 0.3, 2.1, -0.8, 0.1, 1.5, -0.3, 0.7, -0.2]
    )

class BatchPredictionInput(BaseModel):
    """批量预测输入"""
    instances: List[List[float]] = Field(
        ..., min_length=1, max_length=1000,
        description="批量特征列表，每个包含10个特征值"
    )

class PredictionOutput(BaseModel):
    """预测输出"""
    prediction: int = Field(..., description="预测类别 (0=不复购, 1=复购)")
    probability: float = Field(..., description="复购概率 (0-1)")
    features_received: int

class BatchPredictionOutput(BaseModel):
    """批量预测输出"""
    predictions: List[PredictionOutput]
    count: int

class HealthResponse(BaseModel):
    status: str
    model_name: str
    features_required: int
    model_loaded: bool

# ============================================================
# 端点
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """模型健康检查"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_name=type(model).__name__ if model else "N/A",
        features_required=len(feature_names) if feature_names else 0,
        model_loaded=model is not None,
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """单条预测"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(input_data.features).reshape(1, -1)
    if scaler:
        X = scaler.transform(X)

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    return PredictionOutput(
        prediction=pred,
        probability=round(prob, 4),
        features_received=len(input_data.features),
    )

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """批量预测"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for instance in input_data.instances:
        X = np.array(instance).reshape(1, -1)
        if scaler:
            X = scaler.transform(X)
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        results.append(PredictionOutput(
            prediction=pred, probability=round(prob, 4),
            features_received=len(instance)
        ))

    return BatchPredictionOutput(predictions=results, count=len(results))

@app.get("/")
async def root():
    return {"message": "Repurchase Prediction API", "docs": "/docs"}

# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
