"""Fine-tune XLM-RoBERTa for 3-class sentiment classification."""

import argparse
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DATA_DIR = "data/processed/multilingual-sentiments"
MODEL_OUTPUT_DIR = settings.MODEL_PATH
NUM_LABELS = len(settings.SENTIMENT_LABELS)
ID2LABEL = {i: lbl for i, lbl in enumerate(settings.SENTIMENT_LABELS)}
LABEL2ID = {lbl: i for i, lbl in ID2LABEL.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


def train(epochs: int, batch_size: int, learning_rate: float) -> None:
    logger.info("Loading processed dataset from %s", PROCESSED_DATA_DIR)
    dataset = load_from_disk(PROCESSED_DATA_DIR)

    logger.info("Loading model: %s", settings.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    logger.info("Model saved to %s", MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    train(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
    )
