"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "languages" in data
    assert "labels" in data
    assert isinstance(data["languages"], list)
    assert isinstance(data["labels"], list)
