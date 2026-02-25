"""Pydantic schemas for the sentiment API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyse")
    language: Optional[str] = Field(
        None, description="ISO 639-1 language code (optional)"
    )


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to analyse")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]


class BatchSentimentResponse(BaseModel):
    predictions: List[SentimentResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    languages: List[str]
    labels: List[str]
