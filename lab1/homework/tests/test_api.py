import pytest
from fastapi.testclient import TestClient
from src.app import app
from src.model.PredictionModel import PredictionModel


client = TestClient(app)


def test_health_and_root():
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "message" in r_root.json()


def test_model_loading():
    loaded_model = PredictionModel()
    assert loaded_model.sentence_transformer is not None
    assert loaded_model.classifier is not None


@pytest.mark.parametrize(
    "text",
    [
        "I love MLOps!",
        "This is terrible...",
        "It's okay, not great but fine.",
    ]
)
def test_predict_valid_input(text: str):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in ["negative", "neutral", "positive"]



def test_predict_invalid_input():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # FastAPI validation error
    data = response.json()
    assert "detail" in data
