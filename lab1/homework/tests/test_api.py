import pytest
from fastapi.testclient import TestClient
from src.app import app
from src.model.PredictionModel import PredictionModel


client = TestClient(app)


def test_health_and_root():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    root = client.get("/")
    assert root.status_code == 200
    assert "message" in root.json()


def test_model_loading():
    loaded_model = PredictionModel()
    assert loaded_model.sentence_transformer is not None
    assert loaded_model.classifier is not None


@pytest.mark.parametrize(
    "text",
    [
        "What a beatiful evening",
        "I feel like jumping out the window",
        "Not great, not terrible",
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
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
