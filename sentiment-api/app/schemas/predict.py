from typing import Any, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel
from sentiment_model.utilities.validation import SentimentDataInputSchema


class ClassificationResults(BaseModel):
    reviews: Optional[List[Tuple[str, str]]]
    predictions: Optional[List[List[float]]]
    version: str
    errors: Optional[Any]


class MultipleSentimentDataInputs(BaseModel):
    inputs: List[SentimentDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {"Review": "This book had descent character development."},
                    {"Review": "I thought the plot was shallow and underdeveloped."}
                ]
            }
        }
