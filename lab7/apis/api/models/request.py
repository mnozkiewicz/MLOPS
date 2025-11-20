from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    text: str = Field(min_length=1)

class PredictResponse(BaseModel):
    token_embeddings: List[List[List[float]]]
    sentence_embedding: List[List[float]]