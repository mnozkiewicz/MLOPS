from fastapi import FastAPI
import pandas as pd
from .model.inference import predict, load_model
from .api.models.iris import PredictRequest, PredictResponse

app = FastAPI()
model = load_model()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    features_df = pd.DataFrame([request.model_dump()])
    prediction_values = predict(model, features_df)
    prediction = str(prediction_values[0])

    return PredictResponse(prediction=prediction)
