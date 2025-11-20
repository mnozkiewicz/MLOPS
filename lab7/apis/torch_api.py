from fastapi import FastAPI
import torch
from transformers import AutoModel, AutoTokenizer
from api.models.request import PredictRequest, PredictResponse

@torch.inference_mode()
def run_inference(inputs: dict[str, torch.Tensor]):
    output = compiled_model.forward(**inputs)
    return output

def prepare_input(text: str) -> dict[str, torch.Tensor]:
    sample_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    inputs_torch = {
        "input_ids": sample_input["input_ids"],
        "attention_mask": sample_input["attention_mask"],
    }

    return inputs_torch


app = FastAPI(title="ONNX model serving api")
model = AutoModel.from_pretrained("models/mpnet")
tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")

compiled_model = torch.compile(model)

# Run initial inference through the compiled torch model, on the server startup
run_inference({
    "input_ids": torch.ones(1, 100, dtype=torch.int32), 
    "attention_mask": torch.ones(1, 100, dtype=torch.int32)
})

@app.get("/")
def welcome_root():
    return {"message": "Welcome from Torch"}


@app.post("/inference", response_model=PredictResponse)
def inference(request: PredictRequest):
    inputs_torch = prepare_input(request.text)
    outputs_torch = run_inference(inputs_torch)

    return PredictResponse(
        token_embeddings=outputs_torch.last_hidden_state.tolist(),
        sentence_embedding=outputs_torch.pooler_output.tolist()
    )