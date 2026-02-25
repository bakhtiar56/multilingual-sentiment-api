"""Preprocessing utilities for multilingual sentiment data."""

import argparse
import os
import re

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = "data/raw/multilingual-sentiments"
PROCESSED_DATA_DIR = "data/processed/multilingual-sentiments"


def clean_text(text: str) -> str:
    """Remove URLs, special chars and normalise whitespace."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_data(
    dataset: DatasetDict, model_name: str = settings.MODEL_NAME
) -> DatasetDict:
    """Tokenize dataset using the xlm-roberta-base tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(batch):
        cleaned = [clean_text(t) for t in batch["text"]]
        return tokenizer(cleaned, truncation=True, padding="max_length", max_length=128)

    logger.info("Tokenizing with %s", model_name)
    return dataset.map(_tokenize, batched=True)


def create_splits(dataset: DatasetDict) -> DatasetDict:
    """Ensure train/validation/test splits exist."""
    if "validation" not in dataset and "train" in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
                "test": dataset.get("test", split["test"]),
            }
        )
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess multilingual sentiment data"
    )
    parser.add_argument(
        "--input", default=RAW_DATA_DIR, help="Path to raw dataset on disk"
    )
    parser.add_argument("--output", default=PROCESSED_DATA_DIR, help="Output path")
    args = parser.parse_args()

    logger.info("Loading raw data from %s", args.input)
    ds = load_from_disk(args.input)
    ds = create_splits(ds)
    ds = tokenize_data(ds)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ds.save_to_disk(args.output)
    logger.info("Processed dataset saved to %s", args.output)
