import pandas as pd

from sentiment_model import __version__ as _version
from sentiment_model.config.base import config
from sentiment_model.pipeline import Pipeline
from sentiment_model.utilities.validation import validate_inputs

pipe = Pipeline(data=None)


def make_prediction(*, input_data: pd.DataFrame, test: bool = False) -> dict:
    """Make a prediction using a saved model pipeline."""

    validated_data, errors = validate_inputs(input_data=input_data)
    results = {"predictions": None, "version": _version, "errors": errors}

    labels = config.model_config.labels
    if not errors:
        preds = pipe.predict(query=validated_data, test=test)
        results = {
            "reviews": [
                (sent, labels[i.argmax()])
                for sent, i in zip(
                    validated_data[(config.model_config.feature_col)].to_list(), preds
                )
            ],
            "predictions": preds.tolist(),
            "version": _version,
            "errors": errors,
        }

    return results
