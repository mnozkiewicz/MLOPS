from fastapi import FastAPI
from .api.models.request import PredictRequest, PredictResponse

app = FastAPI(title="Sentiment Inference API")


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


def dummy_inference(text: str) -> str:
    if "great" in text.lower():
        return "positive"
    return "negative"


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    prediction = dummy_inference(request.text)
    return PredictResponse(prediction=prediction)
