from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
from typing import List

app = FastAPI(title="ONNX model serving api")
session = ort.InferenceSession("models/model.onnx")

@app.get("/")
def welcome_root():
    return {"message": "Welcome from ONNX"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/inference")
def inference(input_ids: List[List[int]], attention_mask: List[List[int]]):
    inputs_onnx = {
        "input_ids": np.array(input_ids),
        "attention_mask": np.array(attention_mask)
    }
    
    outputs_onnx = session.run(None, inputs_onnx)
    return {
        "token_embeddings": outputs_onnx[0].tolist(),
        "sentence_embedding": outputs_onnx[1].tolist()
    }