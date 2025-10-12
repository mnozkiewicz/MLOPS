from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_welcome_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML API"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text

    data = response.json()
    assert "prediction" in data, "Missing 'prediction' field in response"
    assert isinstance(data["prediction"], str), "Prediction should be a string"
