"""Evaluate the trained sentiment model on the test split."""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DATA_DIR = "data/processed/multilingual-sentiments"


def evaluate(model_path: str) -> None:
    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

    logger.info("Loading test data from %s", PROCESSED_DATA_DIR)
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    test_ds = dataset["test"]

    texts = test_ds["text"]
    true_labels = test_ds["label"]

    logger.info("Running inference on %d examples...", len(texts))
    raw_preds = clf(texts, truncation=True, max_length=128, batch_size=32)
    pred_labels = [max(p, key=lambda x: x["score"])["label"] for p in raw_preds]

    label_names = settings.SENTIMENT_LABELS
    print("\n=== Classification Report ===")
    print(classification_report(true_labels, pred_labels, target_names=label_names))

    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    out_path = os.path.join(model_path, "confusion_matrix.png")
    fig.savefig(out_path, bbox_inches="tight")
    logger.info("Confusion matrix saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--model-path", default=settings.MODEL_PATH)
    args = parser.parse_args()
    evaluate(model_path=args.model_path)
