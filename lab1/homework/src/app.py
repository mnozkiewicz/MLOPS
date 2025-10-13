from fastapi import FastAPI
from .api.models.request import PredictRequest, PredictResponse
from .model.PredictionModel import PredictionModel

app = FastAPI(title="Sentiment Inference API")
model = PredictionModel()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    prediction = model.predict(request.model_dump()["text"])
    return PredictResponse(prediction=prediction)
