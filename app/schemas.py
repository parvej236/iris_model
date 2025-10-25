# app/schemas.py

from pydantic import BaseModel
from typing import List

# Input schema for /predict
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output schema for /predict
class PredictionOutput(BaseModel):
    predicted_class: str