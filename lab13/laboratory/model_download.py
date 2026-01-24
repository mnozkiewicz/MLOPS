import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"

# use half-precision inference and GPU if available
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)