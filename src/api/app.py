"""FastAPI application for multilingual sentiment analysis."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.middleware import RequestLoggingMiddleware, setup_cors
from src.api.schemas import (
    BatchPredictionRequest,
    BatchSentimentResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    SentimentResponse,
)
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_predictor: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    try:
        from src.model.predict import SentimentPredictor

        _predictor = SentimentPredictor()
        logger.info("Predictor loaded")
    except Exception as exc:
        logger.warning("Could not load model: %s", exc)
    yield
    _predictor = None


app = FastAPI(
    title="Multilingual Sentiment API",
    description="Sentiment analysis for 10+ languages using XLM-RoBERTa",
    version="0.1.0",
    lifespan=lifespan,
)

setup_cors(app)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/", tags=["root"])
async def root():
    return {"message": "Welcome to the Multilingual Sentiment API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(status="ok", model_loaded=_predictor is not None)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["info"])
async def model_info():
    return ModelInfoResponse(
        model_name=settings.MODEL_NAME,
        languages=settings.SUPPORTED_LANGUAGES,
        labels=settings.SENTIMENT_LABELS,
    )


@app.post("/predict", response_model=SentimentResponse, tags=["predict"])
async def predict(request: PredictionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = _predictor.predict(request.text)
    return SentimentResponse(**result)


@app.post("/predict/batch", response_model=BatchSentimentResponse, tags=["predict"])
async def predict_batch(request: BatchPredictionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = _predictor.predict_batch(request.texts)
    return BatchSentimentResponse(predictions=[SentimentResponse(**r) for r in results])
