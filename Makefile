.PHONY: install install-dev download-data preprocess train evaluate serve test lint format docker-build docker-run clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

download-data:
	python src/data/download.py

preprocess:
	python src/data/preprocess.py

train:
	python src/model/train.py

evaluate:
	python src/model/evaluate.py

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	black src/ tests/

docker-build:
	docker build -t multilingual-sentiment-api .

docker-run:
	docker run -p 8000:8000 --env-file .env multilingual-sentiment-api

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
