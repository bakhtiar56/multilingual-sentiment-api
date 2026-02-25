# Multilingual Sentiment API

![CI](https://github.com/bakhtiar56/multilingual-sentiment-api/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)

A production-ready REST API for **multilingual sentiment analysis** powered by a fine-tuned `xlm-roberta-base` model. Classifies text as **positive**, **neutral**, or **negative** across 10+ languages.

---

## Problem Statement

Sentiment analysis tools typically focus on English. This project addresses the gap by fine-tuning a multilingual transformer (XLM-RoBERTa) on a diverse, multilingual dataset, then exposing the model through a scalable FastAPI service.

---

## Architecture

```
Client
  │  HTTP POST /predict
  ▼
FastAPI (uvicorn)
  │
  ├── Middleware: CORS, Request Logging
  │
  ├── SentimentPredictor
  │     └── xlm-roberta-base (fine-tuned, 3-class)
  │
  └── JSON Response: {sentiment, confidence, scores}
```

---

## Results

| Metric        | Value |
|---------------|-------|
| Accuracy      | ~0.88 |
| F1 Macro      | ~0.87 |
| F1 Weighted   | ~0.88 |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/bakhtiar56/multilingual-sentiment-api.git
cd multilingual-sentiment-api
make install-dev

# Download and preprocess data
make download-data
make preprocess

# Train
make train

# Start the API
make serve
```

---

## Docker

```bash
cp .env.example .env
make docker-build
make docker-run
# or
docker compose up
```

---

## API Usage

### Predict single text

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "language": "en"}'
```

```json
{
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 0.9512,
  "scores": {"positive": 0.9512, "neutral": 0.0341, "negative": 0.0147}
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Not bad", "Terrible experience"]}'
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Testing

```bash
make test
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | XLM-RoBERTa (HuggingFace Transformers) |
| API | FastAPI + Uvicorn |
| Data | HuggingFace Datasets |
| Training | HuggingFace Trainer |
| Metrics | scikit-learn |
| Containerisation | Docker / Docker Compose |

---

## Project Structure

```
multilingual-sentiment-api/
├── src/
│   ├── api/          # FastAPI app, schemas, middleware
│   ├── data/         # download.py, preprocess.py
│   ├── model/        # train.py, evaluate.py, predict.py
│   └── utils/        # config.py, logger.py
├── tests/            # pytest test suite
├── data/
│   ├── raw/          # raw downloaded datasets
│   └── processed/    # tokenized datasets
├── models/           # saved model checkpoints
├── notebooks/        # EDA notebook
├── docs/             # architecture docs
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## License

MIT License © 2026 Bakhtiar Rasheed
