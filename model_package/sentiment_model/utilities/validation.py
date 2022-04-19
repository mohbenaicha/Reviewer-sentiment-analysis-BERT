from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

from sentiment_model.config.base import config


def validate_inputs(
    *,
    input_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[dict]]:

    """Validate inputs are as expected according to a defined
    Pydantic schema."""
    validated_data = input_data[[config.model_config.feature_col]].copy()
    print(validated_data.head(3))
    errors = None

    try:
        MultipleSentimentDataInputs(inputs=validated_data.to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class SentimentDataInputSchema(BaseModel):
    """Single-record schema"""

    Review: str


class MultipleSentimentDataInputs(BaseModel):
    """Applying schema to input data structure type
    In this case, it's a list of strings
    """

    inputs: List[SentimentDataInputSchema]
