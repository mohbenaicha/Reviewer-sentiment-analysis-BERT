import json
from typing import Any
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from sentiment_model import __version__ as model_version
from sentiment_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = (
    APIRouter()
)  # instantiate a vanilla APIRouter object for creating end points


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Health end point return model and api version
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.ClassificationResults, status_code=200)
async def predict(input_data: schemas.MultipleSentimentDataInputs) -> Any:
    """
    Post request as test data and receive repsonse as predictions
    """

    _input_data = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Predicting review sentiment for review: {input_data.inputs}")
    results = make_prediction(input_data=_input_data)

    if results["errors"] is not None:
        logger.warning(f"Classification validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(
        f"Reviews and their corresponding sentiment: {results.get('reviews')}"
    )

    return results
