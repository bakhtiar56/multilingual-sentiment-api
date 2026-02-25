# Architecture Overview

## Overview

The **Multilingual Sentiment API** is a production-ready REST API that classifies text into positive, neutral, or negative sentiment across 10+ languages. It is built on a fine-tuned `xlm-roberta-base` model served via FastAPI.

## Components

| Component | Technology | Description |
|-----------|------------|-------------|
| Model | XLM-RoBERTa | Multilingual transformer fine-tuned for 3-class sentiment |
| API | FastAPI + Uvicorn | Async REST API serving predictions |
| Data pipeline | HuggingFace Datasets | Download, preprocess and tokenize raw data |
| Training | HuggingFace Trainer | Fine-tuning loop with early stopping |
| Evaluation | scikit-learn | Metrics, confusion matrix |
| Containerisation | Docker / Compose | Reproducible deployment |

## Data Flow

```
Raw text
   │
   ▼
clean_text()          ← remove URLs, special chars, normalise whitespace
   │
   ▼
AutoTokenizer         ← xlm-roberta-base tokenizer (max_length=128)
   │
   ▼
XLM-RoBERTa model     ← fine-tuned classifier (3 labels)
   │
   ▼
SentimentPredictor    ← returns label + confidence + per-class scores
   │
   ▼
FastAPI endpoint      ← JSON response
```

## Deployment

### Local

```bash
make install
make serve
```

### Docker

```bash
make docker-build
make docker-run
```

### Docker Compose

```bash
docker compose up
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.
