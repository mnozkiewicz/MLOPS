from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from api.models.request import PredictRequest, PredictResponse


app = FastAPI(title="ONNX model serving api")
session = ort.InferenceSession("models/model.onnx")
tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")


def prepare_input(text: str) -> dict[str, np.ndarray]:
    sample_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="np",
    )

    inputs_onnx = {
        "input_ids": sample_input["input_ids"],
        "attention_mask": sample_input["attention_mask"],
    }

    return inputs_onnx

@app.get("/")
def welcome_root():
    return {"message": "Welcome from ONNX"}


@app.post("/inference", response_model=PredictResponse)
def inference(request: PredictRequest):
    inputs_onnx = prepare_input(request.text)
    outputs_onnx = session.run(None, inputs_onnx)

    return PredictResponse(
        token_embeddings=outputs_onnx[0].tolist(),
        sentence_embedding=outputs_onnx[1].tolist()
    )