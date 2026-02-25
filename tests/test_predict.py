"""Tests for prediction output format (without loading a real model)."""

import sys
import types
from unittest.mock import MagicMock


def _make_predictor():
    """Return a SentimentPredictor with mocked heavy dependencies."""
    # Stub out transformers before importing predict
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoModelForSequenceClassification = MagicMock()

    mock_pipe_output = [
        [
            {"label": "positive", "score": 0.9},
            {"label": "neutral", "score": 0.07},
            {"label": "negative", "score": 0.03},
        ]
    ]
    mock_pipe_instance = MagicMock(return_value=mock_pipe_output)
    fake_transformers.pipeline = MagicMock(return_value=mock_pipe_instance)

    # Also stub datasets if needed by other imports
    if "datasets" not in sys.modules:
        sys.modules["datasets"] = types.ModuleType("datasets")

    orig = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers

    # Remove cached predict module so it re-imports with our stub
    for key in list(sys.modules.keys()):
        if "src.model.predict" in key or key == "src.model.predict":
            del sys.modules[key]

    try:
        from src.model.predict import SentimentPredictor

        predictor = SentimentPredictor.__new__(SentimentPredictor)
        predictor._pipeline = mock_pipe_instance
    finally:
        if orig is None:
            del sys.modules["transformers"]
        else:
            sys.modules["transformers"] = orig

    return predictor


def test_predict_returns_required_keys():
    predictor = _make_predictor()
    result = predictor.predict("I love this!")
    assert "text" in result
    assert "sentiment" in result
    assert "confidence" in result
    assert "scores" in result


def test_predict_sentiment_is_valid_label():
    from src.utils.config import settings

    predictor = _make_predictor()
    result = predictor.predict("Great product!")
    assert result["sentiment"] in settings.SENTIMENT_LABELS


def test_predict_confidence_between_0_and_1():
    predictor = _make_predictor()
    result = predictor.predict("Okay experience")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_batch_returns_list():
    predictor = _make_predictor()
    results = predictor.predict_batch(["Good", "Bad"])
    assert isinstance(results, list)
    assert len(results) == 2
