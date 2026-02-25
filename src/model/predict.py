"""SentimentPredictor: load model and run inference."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPredictor:
    def __init__(self, model_path: str = settings.MODEL_PATH) -> None:
        logger.info("Loading model from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._pipeline = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            top_k=None,
        )
        logger.info("Model loaded successfully")

    def predict(self, text: str) -> dict[str, Any]:
        """Return sentiment label and confidence scores for a single text."""
        results = self._pipeline(text, truncation=True, max_length=128)[0]
        scores = {r["label"]: round(r["score"], 4) for r in results}
        top = max(results, key=lambda r: r["score"])
        return {
            "text": text,
            "sentiment": top["label"],
            "confidence": round(top["score"], 4),
            "scores": scores,
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return predictions for a list of texts."""
        return [self.predict(t) for t in texts]
