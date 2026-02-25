"""Download multilingual sentiment dataset from HuggingFace Hub."""

import argparse
import os

from datasets import load_dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = "data/raw"


def download_dataset(languages: list[str] | None = None) -> None:
    """Download the multilingual-sentiments dataset and save splits to disk."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    logger.info("Downloading dataset: tydjohnson/multilingual-sentiments")
    dataset = load_dataset("tydjohnson/multilingual-sentiments")

    if languages:
        logger.info("Filtering for languages: %s", languages)
        for split in dataset:
            dataset[split] = dataset[split].filter(
                lambda ex: ex.get("language") in languages
            )

    out_path = os.path.join(RAW_DATA_DIR, "multilingual-sentiments")
    dataset.save_to_disk(out_path)
    logger.info("Dataset saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download multilingual sentiment data")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="List of language codes to keep (e.g. en fr de). Default: all.",
    )
    args = parser.parse_args()
    download_dataset(languages=args.languages)
