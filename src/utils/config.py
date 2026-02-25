import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_NAME: str = os.getenv("MODEL_NAME", "xlm-roberta-base")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/sentiment-model")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SUPPORTED_LANGUAGES: List[str] = [
        "en",
        "fr",
        "de",
        "es",
        "ar",
        "zh",
        "ja",
        "ko",
        "ru",
        "tr",
    ]
    SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]


settings = Settings()
