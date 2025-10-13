from pydantic import BaseModel
from pydantic import Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


class PredictResponse(BaseModel):
    prediction: str
