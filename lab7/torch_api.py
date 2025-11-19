from fastapi import FastAPI
import torch
from typing import List
from transformers import AutoModel


@torch.inference_mode()
def run_inference(inputs: dict[str, torch.Tensor]):
    output = compiled_model.forward(**inputs)
    return output


app = FastAPI(title="ONNX model serving api")
model = AutoModel.from_pretrained("models/mpnet")
compiled_model = torch.compile(model)

run_inference({
    "input_ids": torch.ones(1, 100, dtype=torch.int32), 
    "attention_mask": torch.ones(1, 100, dtype=torch.int32)
})

@app.get("/")
def welcome_root():
    return {"message": "Welcome from ONNX"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/inference")
def inference(input_ids: List[List[int]], attention_mask: List[List[int]]):
    inputs_torch = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask)
    }

    outputs_torch = run_inference(inputs_torch)
    return {
        "token_embeddings": outputs_torch.last_hidden_state.tolist(),
        "sentence_embedding": outputs_torch.pooler_output.tolist()
    }